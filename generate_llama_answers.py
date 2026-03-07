import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

from api_keys import resolve_runpod_api_key


DEFAULT_INPUT = "train_1000_first_user_prompts_random_unique.txt"
DEFAULT_MODEL = "nousresearch/meta-llama-3.1-8b-instruct"
DEFAULT_BASE_URL = "https://api.runpod.ai/v2/3rm9a0h147q3nz/openai/v1"
DEFAULT_CONCURRENCY = 20
DEFAULT_MAX_RETRIES = 6
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096
DEFAULT_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Llama 3.1 8B Instruct answer for each prompt in a txt file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name(DEFAULT_INPUT),
        help="Path to the one-prompt-per-line txt file.",
    )
    parser.add_argument(
        "--messages-output",
        type=Path,
        default=None,
        help="Optional output path for prompt/answer messages JSONL.",
    )
    parser.add_argument(
        "--answers-output",
        type=Path,
        default=None,
        help="Optional output path for line-aligned assistant answers txt.",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=None,
        help="Optional checkpoint JSONL path used for resume.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL for the deployed endpoint.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name exposed by the endpoint.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of in-flight API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="How many times to retry a failed request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. Meta's chat examples use 0.6.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p sampling value. Meta's chat examples use 0.9.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Optional seed passed to the OpenAI-compatible API for reproducibility.",
    )
    return parser.parse_args()


def derive_output_paths(input_path: Path) -> tuple[Path, Path, Path]:
    stem = input_path.stem
    parent = input_path.parent
    return (
        parent / f"{stem}_llama31_8b_messages.jsonl",
        parent / f"{stem}_llama31_8b_answers.txt",
        parent / f"{stem}_llama31_8b_partial.jsonl",
    )

def load_prompts(input_path: Path) -> list[str]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    return [line.replace("\\n", "\n") for line in lines]


def load_partial_results(partial_output: Path, prompts: list[str]) -> dict[int, dict]:
    results: dict[int, dict] = {}
    if not partial_output.exists():
        return results

    with partial_output.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            row = json.loads(raw_line)
            index = row["index"]
            if prompts[index] != row["prompt"]:
                raise ValueError(
                    f"Partial output prompt mismatch at index {index}: "
                    f"{partial_output}"
                )
            results[index] = row
    return results


async def request_completion(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    max_retries: int,
    index: int,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                request_kwargs["seed"] = seed

            completion = await client.chat.completions.create(**request_kwargs)
            return completion.choices[0].message.content or ""
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Request failed for prompt {index} after {max_retries} attempts"
                ) from exc

            delay = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"[retry] prompt={index} attempt={attempt}/{max_retries} "
                f"sleep={delay:.1f}s error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)


async def fetch_one(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    index: int,
) -> dict:
    async with semaphore:
        response = await request_completion(
            client=client,
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            max_retries=max_retries,
            index=index,
        )
        return {"index": index, "prompt": prompt, "response": response}


async def generate_missing_results(
    client: AsyncOpenAI,
    prompts: list[str],
    existing_results: dict[int, dict],
    partial_output: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    concurrency: int,
    max_retries: int,
) -> dict[int, dict]:
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    pending_indices = [idx for idx in range(len(prompts)) if idx not in existing_results]

    print(
        f"Loaded {len(existing_results)} existing responses, "
        f"requesting {len(pending_indices)} more.",
        flush=True,
    )

    if not pending_indices:
        return existing_results

    async def run_and_persist(index: int) -> dict:
        result = await fetch_one(
            client=client,
            prompt=prompts[index],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            max_retries=max_retries,
            semaphore=semaphore,
            index=index,
        )
        async with write_lock:
            with partial_output.open("a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    tasks = [asyncio.create_task(run_and_persist(index)) for index in pending_indices]
    total = len(prompts)
    completed = len(existing_results)

    for future in asyncio.as_completed(tasks):
        result = await future
        existing_results[result["index"]] = result
        completed += 1
        if completed % 25 == 0 or completed == total:
            print(f"Completed {completed}/{total}", flush=True)

    return existing_results


def write_outputs(
    prompts: list[str],
    results: dict[int, dict],
    messages_output: Path,
    answers_output: Path,
) -> None:
    ordered = [results[idx] for idx in range(len(prompts))]

    with messages_output.open("w", encoding="utf-8") as messages_file:
        for row in ordered:
            record = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
            messages_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with answers_output.open("w", encoding="utf-8") as answers_file:
        for row in ordered:
            escaped = (
                row["response"]
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n", "\\n")
            )
            answers_file.write(escaped + "\n")


async def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    default_messages_output, default_answers_output, default_partial_output = (
        derive_output_paths(args.input)
    )
    messages_output = args.messages_output or default_messages_output
    answers_output = args.answers_output or default_answers_output
    partial_output = args.partial_output or default_partial_output

    prompts = load_prompts(args.input)
    api_key = resolve_runpod_api_key(repo_root)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )

    existing_results = load_partial_results(partial_output, prompts)
    results = await generate_missing_results(
        client=client,
        prompts=prompts,
        existing_results=existing_results,
        partial_output=partial_output,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    if len(results) != len(prompts):
        missing = sorted(set(range(len(prompts))) - set(results))
        raise RuntimeError(f"Missing responses for indices: {missing[:10]}")

    write_outputs(
        prompts=prompts,
        results=results,
        messages_output=messages_output,
        answers_output=answers_output,
    )
    print(f"Messages JSONL: {messages_output}")
    print(f"Answers TXT: {answers_output}")
    print(f"Partial JSONL: {partial_output}")


if __name__ == "__main__":
    asyncio.run(main())
