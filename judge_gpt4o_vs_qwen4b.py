import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import stdev
from typing import Any
from xml.sax.saxutils import escape

from openai import AsyncOpenAI

from api_keys import resolve_openrouter_api_key


DEFAULT_PROMPTS = "train_1000_first_user_prompts_random_unique.txt"
DEFAULT_GPT4O = "train_1000_first_user_prompts_random_unique_gpt4o_messages.jsonl"
DEFAULT_QWEN4B = "train_1000_first_user_prompts_random_unique_qwen4b_messages.jsonl"
DEFAULT_JUDGE_PROMPT = "JUDGE_PROMPT.txt"
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.6"
DEFAULT_CONCURRENCY = 6
DEFAULT_MAX_RETRIES = 6
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_SEED = 0
DEFAULT_VERBOSITY = "max"
DEFAULT_PROGRESS_EVERY = 5

LABEL_GPT4O = "gpt-4o"
LABEL_QWEN4B = "qwen4b"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Judge GPT-4o vs Qwen 4B responses with Claude Opus 4.6 via OpenRouter."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=repo_root / DEFAULT_PROMPTS,
        help="TXT file with one first-user prompt per line.",
    )
    parser.add_argument(
        "--model-a",
        type=Path,
        default=repo_root / DEFAULT_GPT4O,
        help="JSONL file with prompt/answer pairs in messages format for model A.",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=repo_root / DEFAULT_QWEN4B,
        help="JSONL file with prompt/answer pairs in messages format for model B.",
    )
    parser.add_argument(
        "--model-a-label",
        default=LABEL_GPT4O,
        help="Human-readable label stored for model A.",
    )
    parser.add_argument(
        "--model-b-label",
        default=LABEL_QWEN4B,
        help="Human-readable label stored for model B.",
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=repo_root / DEFAULT_JUDGE_PROMPT,
        help="Template prompt with $QUESTION_TEXT / $RESPONSE_A_TEXT / $RESPONSE_B_TEXT.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter judge model slug.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "model_a_vs_model_b_claude46_judgments.jsonl",
        help="Final ordered JSONL output path.",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=repo_root / "model_a_vs_model_b_claude46_partial.jsonl",
        help="Append-only checkpoint JSONL path used for resume.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "model_a_vs_model_b_claude46_summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of in-flight judge requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry count for failed or malformed judgments.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="max_tokens passed to the judge model.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress after this many completed judgments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used to deterministically randomize whether GPT-4o is A or B.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of examples to judge. When set below the dataset size, "
            "the script selects a deterministic random sample."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic random sampling when --limit is used.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help=(
            "Optional OpenRouter reasoning.max_tokens budget. Leave unset to use "
            "Claude 4.6 adaptive reasoning."
        ),
    )
    parser.add_argument(
        "--verbosity",
        default=DEFAULT_VERBOSITY,
        help=(
            "Claude 4.6 effort level as exposed by OpenRouter chat completions. "
            "Use max for highest-effort judging."
        ),
    )
    return parser.parse_args()

def load_prompt_order(path: Path) -> list[str]:
    prompts = [line.replace("\\n", "\n") for line in path.read_text(encoding="utf-8").splitlines()]
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"Prompt file contains duplicates: {path}")
    return prompts


def extract_first_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"Could not find role={role!r} in messages payload")


def load_messages_by_prompt(path: Path) -> dict[str, str]:
    by_prompt: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, 1):
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number} is missing a messages list")

            prompt = extract_first_content(messages, "user")
            response = extract_first_content(messages, "assistant")
            if prompt in by_prompt:
                raise ValueError(f"Duplicate prompt found in {path}: {prompt[:120]!r}")
            by_prompt[prompt] = response
    return by_prompt


def build_examples(
    prompt_order: list[str],
    model_a_by_prompt: dict[str, str],
    model_b_by_prompt: dict[str, str],
    model_a_label: str,
    model_b_label: str,
    seed: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    missing_model_a = [prompt for prompt in prompt_order if prompt not in model_a_by_prompt]
    missing_model_b = [prompt for prompt in prompt_order if prompt not in model_b_by_prompt]

    if missing_model_a:
        raise ValueError(f"Missing {len(missing_model_a)} prompts in model A file")
    if missing_model_b:
        raise ValueError(f"Missing {len(missing_model_b)} prompts in model B file")

    extra_model_a = sorted(set(model_a_by_prompt) - set(prompt_order))
    extra_model_b = sorted(set(model_b_by_prompt) - set(prompt_order))
    if extra_model_a:
        raise ValueError(f"Model A file contains {len(extra_model_a)} extra prompts")
    if extra_model_b:
        raise ValueError(f"Model B file contains {len(extra_model_b)} extra prompts")

    for index, prompt in enumerate(prompt_order):
        model_a_response = model_a_by_prompt[prompt]
        model_b_response = model_b_by_prompt[prompt]
        digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
        model_a_is_a = digest[0] % 2 == 0

        if model_a_is_a:
            response_a_model = model_a_label
            response_a = model_a_response
            response_b_model = model_b_label
            response_b = model_b_response
        else:
            response_a_model = model_b_label
            response_a = model_b_response
            response_b_model = model_a_label
            response_b = model_a_response

        examples.append(
            {
                "index": index,
                "prompt": prompt,
                "model_a_label": model_a_label,
                "model_b_label": model_b_label,
                model_a_label: model_a_response,
                model_b_label: model_b_response,
                "response_a_model": response_a_model,
                "response_a": response_a,
                "response_b_model": response_b_model,
                "response_b": response_b,
            }
        )

    return examples


def render_judge_prompt(
    template: str,
    question: str,
    response_a: str,
    response_b: str,
) -> str:
    rendered = template
    rendered = rendered.replace("$QUESTION_TEXT", escape(question))
    rendered = rendered.replace("$RESPONSE_A_TEXT", escape(response_a))
    rendered = rendered.replace("$RESPONSE_B_TEXT", escape(response_b))
    return rendered


def maybe_limit_examples(
    examples: list[dict[str, Any]],
    limit: int | None,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(examples):
        return examples
    if limit <= 0:
        raise ValueError("--limit must be positive")

    rng = random.Random(sample_seed)
    selected = rng.sample(examples, limit)
    return sorted(selected, key=lambda example: example["index"])


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def extract_xml_tag(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def parse_judgment_xml(raw_text: str) -> dict[str, Any]:
    text = strip_code_fences(raw_text)
    judge_reasoning = extract_xml_tag(
        text,
        r"<reasoning>\s*(.*?)\s*</reasoning>\s*<judg(?:e)?ment>",
    )
    judgment = extract_xml_tag(text, r"<judg(?:e)?ment>\s*(A|B|Tie)\s*</judg(?:e)?ment>")
    genre = extract_xml_tag(
        text,
        r"<genre>\s*(STEM|General Knowledge|Creative|Chat|Roleplay|Other|Unknown)\s*</genre>",
    )
    difficulty_block = extract_xml_tag(text, r"<difficulty>\s*(.*?)\s*</difficulty>")
    knowledge = None
    reasoning_difficulty = None
    if difficulty_block is not None:
        knowledge = extract_xml_tag(
            difficulty_block,
            r"<knowledge>\s*(Low|Medium|High)\s*</knowledge>",
        )
        reasoning_difficulty = extract_xml_tag(
            difficulty_block,
            r"<reasoning>\s*(Low|Medium|High)\s*</reasoning>",
        )
    language = extract_xml_tag(text, r"<language>\s*([A-Za-z]{2}|Unknown)\s*</language>")

    if not all([judge_reasoning, judgment, genre, knowledge, reasoning_difficulty, language]):
        raise ValueError(f"Judge output did not match the required XML format: {raw_text!r}")

    return {
        "judge_reasoning": judge_reasoning,
        "judgement": judgment,
        "genre": genre,
        "difficulty": {
            "knowledge": knowledge,
            "reasoning": reasoning_difficulty,
        },
        "language": language,
    }


def winner_from_judgement(
    judgement: str,
    response_a_model: str,
    response_b_model: str,
) -> str:
    if judgement == "A":
        return response_a_model
    if judgement == "B":
        return response_b_model
    return "Tie"


def load_partial_results(partial_output: Path, examples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if not partial_output.exists():
        return results

    expected_by_index = {example["index"]: example for example in examples}
    with partial_output.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            row = json.loads(raw_line)
            index = row["index"]
            expected = expected_by_index[index]
            for key in ("prompt", "response_a_model", "response_b_model"):
                if row.get(key) != expected.get(key):
                    raise ValueError(
                        f"Partial output mismatch for index {index} field {key}: {partial_output}"
                    )
            results[index] = row
    return results


def build_extra_body(verbosity: str, reasoning_max_tokens: int | None) -> dict[str, Any]:
    # Anthropic exposes Claude 4.6 effort as output_config.effort. OpenRouter's
    # current Claude 4.6 chat-completions mapping exposes that control as
    # `verbosity`, while adaptive reasoning remains under `reasoning`.
    reasoning: dict[str, Any] = {
        "enabled": True,
        "exclude": True,
    }
    if reasoning_max_tokens is not None:
        reasoning["max_tokens"] = reasoning_max_tokens

    return {
        "reasoning": reasoning,
        "verbosity": verbosity,
    }


def serialize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return json.loads(json.dumps(usage, default=lambda value: getattr(value, "__dict__", str(value))))


async def request_judgment(
    client: AsyncOpenAI,
    prompt_text: str,
    judge_model: str,
    max_output_tokens: int,
    extra_body: dict[str, Any],
    max_retries: int,
    index: int,
) -> tuple[str, dict[str, Any], dict[str, Any] | None, str | None]:
    for attempt in range(1, max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0,
                max_tokens=max_output_tokens,
                extra_body=extra_body,
            )
            raw_text = completion.choices[0].message.content or ""
            parsed = parse_judgment_xml(raw_text)
            usage = serialize_usage(getattr(completion, "usage", None))
            generation_id = getattr(completion, "id", None)
            return raw_text, parsed, usage, generation_id
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Judge request failed for example {index} after {max_retries} attempts"
                ) from exc

            delay = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"[retry] example={index} attempt={attempt}/{max_retries} "
                f"sleep={delay:.1f}s error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)


async def judge_one(
    client: AsyncOpenAI,
    example: dict[str, Any],
    template: str,
    judge_model: str,
    max_output_tokens: int,
    extra_body: dict[str, Any],
    max_retries: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt_text = render_judge_prompt(
        template=template,
        question=example["prompt"],
        response_a=example["response_a"],
        response_b=example["response_b"],
    )

    async with semaphore:
        raw_judgment_xml, parsed, usage, generation_id = await request_judgment(
            client=client,
            prompt_text=prompt_text,
            judge_model=judge_model,
            max_output_tokens=max_output_tokens,
            extra_body=extra_body,
            max_retries=max_retries,
            index=example["index"],
        )

    judgement = parsed["judgement"]
    cost = None
    if isinstance(usage, dict) and usage.get("cost") is not None:
        cost = float(usage["cost"])
    return {
        **example,
        "judge_model": judge_model,
        "judge_generation_id": generation_id,
        "usage": usage,
        "cost": cost,
        "raw_judgment_xml": raw_judgment_xml,
        "judge_reasoning": parsed["judge_reasoning"],
        "judgement": judgement,
        "winner": winner_from_judgement(
            judgement=judgement,
            response_a_model=example["response_a_model"],
            response_b_model=example["response_b_model"],
        ),
        "genre": parsed["genre"],
        "difficulty": parsed["difficulty"],
        "language": parsed["language"],
    }


async def generate_missing_results(
    client: AsyncOpenAI,
    examples: list[dict[str, Any]],
    existing_results: dict[int, dict[str, Any]],
    partial_output: Path,
    template: str,
    judge_model: str,
    max_output_tokens: int,
    extra_body: dict[str, Any],
    concurrency: int,
    max_retries: int,
    progress_every: int,
) -> dict[int, dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    pending = [example for example in examples if example["index"] not in existing_results]
    running_cost = sum(
        float(row["cost"])
        for row in existing_results.values()
        if row.get("cost") is not None
    )

    print(
        f"Loaded {len(existing_results)} existing judgments, requesting {len(pending)} more. "
        f"Running cost so far: {running_cost:.6f} credits.",
        flush=True,
    )

    if not pending:
        return existing_results

    async def run_and_persist(example: dict[str, Any]) -> dict[str, Any] | None:
        try:
            result = await judge_one(
                client=client,
                example=example,
                template=template,
                judge_model=judge_model,
                max_output_tokens=max_output_tokens,
                extra_body=extra_body,
                max_retries=max_retries,
                semaphore=semaphore,
            )
        except RuntimeError as exc:
            print(
                f"[skip] example={example['index']} permanently failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None
        async with write_lock:
            with partial_output.open("a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    tasks = [asyncio.create_task(run_and_persist(example)) for example in pending]
    total = len(examples)
    completed = len(existing_results)
    skipped = 0
    progress_every = max(1, progress_every)

    for future in asyncio.as_completed(tasks):
        result = await future
        if result is None:
            skipped += 1
        else:
            existing_results[result["index"]] = result
            if result.get("cost") is not None:
                running_cost += float(result["cost"])
        completed += 1
        if completed % progress_every == 0 or completed == total:
            average_cost = running_cost / completed if completed else 0.0
            estimated_total_cost = average_cost * total if completed else 0.0
            print(
                f"Completed {completed}/{total} | "
                f"running_cost={running_cost:.6f} credits | "
                f"avg_cost={average_cost:.6f} | "
                f"est_total={estimated_total_cost:.6f} | "
                f"skipped={skipped}",
                flush=True,
            )

    if skipped:
        print(f"Warning: {skipped} examples skipped due to persistent failures.", flush=True)

    return existing_results


def write_final_outputs(
    results: dict[int, dict[str, Any]],
    output_path: Path,
    summary_path: Path,
    full_example_count: int,
) -> None:
    ordered = [results[index] for index in sorted(results)]
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in ordered:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    winner_counts = Counter(row["winner"] for row in ordered)
    genre_counts = Counter(row["genre"] for row in ordered)
    language_counts = Counter(row["language"] for row in ordered)
    knowledge_counts = Counter(row["difficulty"]["knowledge"] for row in ordered)
    reasoning_counts = Counter(row["difficulty"]["reasoning"] for row in ordered)
    costs = [row["cost"] for row in ordered if row.get("cost") is not None]
    sampled_total_cost = sum(costs)
    average_cost = sampled_total_cost / len(costs) if costs else None
    estimated_total_cost = (
        average_cost * full_example_count if average_cost is not None else None
    )
    estimate_95ci = None
    if costs:
        if len(costs) == full_example_count:
            estimate_95ci = {
                "low": sampled_total_cost,
                "high": sampled_total_cost,
            }
        elif len(costs) > 1:
            sample_std = stdev(costs)
            finite_population_correction = math.sqrt(
                (full_example_count - len(costs)) / (full_example_count - 1)
            )
            standard_error_mean = (
                sample_std / math.sqrt(len(costs)) * finite_population_correction
            )
            margin = 1.96 * standard_error_mean * full_example_count
            estimate_95ci = {
                "low": estimated_total_cost - margin,
                "high": estimated_total_cost + margin,
            }

    summary = {
        "sample_size": len(ordered),
        "full_set_size": full_example_count,
        "winner_counts": dict(winner_counts),
        "genre_counts": dict(genre_counts),
        "language_counts": dict(language_counts),
        "knowledge_difficulty_counts": dict(knowledge_counts),
        "reasoning_difficulty_counts": dict(reasoning_counts),
        "sampled_total_cost": sampled_total_cost,
        "average_cost": average_cost,
        "estimated_total_cost": estimated_total_cost,
        "estimated_total_cost_95ci": estimate_95ci,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    prompt_order = load_prompt_order(args.prompts)
    if args.model_a_label == args.model_b_label:
        raise ValueError("--model-a-label and --model-b-label must be different")

    model_a_by_prompt = load_messages_by_prompt(args.model_a)
    model_b_by_prompt = load_messages_by_prompt(args.model_b)
    examples = build_examples(
        prompt_order=prompt_order,
        model_a_by_prompt=model_a_by_prompt,
        model_b_by_prompt=model_b_by_prompt,
        model_a_label=args.model_a_label,
        model_b_label=args.model_b_label,
        seed=args.seed,
    )
    full_example_count = len(examples)
    examples = maybe_limit_examples(
        examples=examples,
        limit=args.limit,
        sample_seed=args.sample_seed,
    )

    template = args.judge_prompt.read_text(encoding="utf-8")
    api_key = resolve_openrouter_api_key(repo_root)
    extra_body = build_extra_body(
        verbosity=args.verbosity,
        reasoning_max_tokens=args.reasoning_max_tokens,
    )

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    existing_results = load_partial_results(args.partial_output, examples)
    results = await generate_missing_results(
        client=client,
        examples=examples,
        existing_results=existing_results,
        partial_output=args.partial_output,
        template=template,
        judge_model=args.judge_model,
        max_output_tokens=args.max_output_tokens,
        extra_body=extra_body,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        progress_every=args.progress_every,
    )

    if len(results) != len(examples):
        missing = sorted(set(range(len(examples))) - set(results))
        print(f"Note: {len(missing)} examples skipped (indices: {missing[:10]})", flush=True)

    write_final_outputs(
        results=results,
        output_path=args.output,
        summary_path=args.summary_output,
        full_example_count=full_example_count,
    )
    print(f"Judgments JSONL: {args.output}")
    print(f"Partial JSONL: {args.partial_output}")
    print(f"Summary JSON: {args.summary_output}")


if __name__ == "__main__":
    asyncio.run(main())
