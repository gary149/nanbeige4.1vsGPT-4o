# Nanbeige 4.1-3B vs GPT-4o Evaluation

Pairwise benchmark comparing **Nanbeige 4.1-3B** against **GPT-4o** on ~1000 real user prompts from [WildChat](https://huggingface.co/datasets/allenai/WildChat), judged by **Claude Opus 4.6** via OpenRouter.

Fork of [N8Programs' Qwen 4B vs GPT-4o benchmark](https://x.com/N8Programs), applied to Nanbeige 4.1-3B.

## Setup

```bash
uv sync
```

### API keys

**OpenRouter** (for GPT-4o generation and Claude judging):

- `OPENROUTER_API_KEY` env var, or
- `openrouter-key.txt` in the repo root (see `openrouter-key.txt.example`)

**Nanbeige endpoint** (HuggingFace Inference Endpoint or any OpenAI-compatible API):

- Set the `--base-url` and `--model` flags in `generate_nanbeige_answers.py`
- See [Deploying the Nanbeige endpoint](#deploying-the-nanbeige-endpoint) below

**Runpod / Llama endpoint** (optional, for control experiment):

- `RUNPOD_API_KEY` env var, or
- `llama-key.txt` in the repo root (see `llama-key.txt.example`)

## Pipeline

### 1. Prepare prompts

Download a train parquet shard from [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat), then sample unique first-user prompts:

```bash
uv run python to_json.py
```

This produces `train_1000_first_user_prompts_random_unique.txt` (one prompt per line, newlines escaped as `\n`).

### 2. Generate answers

```bash
# GPT-4o via OpenRouter
uv run python generate_gpt4o_answers.py

# Nanbeige 4.1-3B via HuggingFace endpoint
# (strips <think> tags, uses model-card params: temp=0.6, top_p=0.95)
uv run python generate_nanbeige_answers.py

# (Optional) Llama 3.1 8B via Runpod
uv run python generate_llama_answers.py --base-url <endpoint>
```

All generation scripts support **resume** via `*_partial.jsonl` checkpoints.

### 3. Judge with Claude Opus 4.6

```bash
uv run python judge_gpt4o_vs_qwen4b.py \
  --model-a train_1000_first_user_prompts_random_unique_gpt4o_messages.jsonl \
  --model-b train_1000_first_user_prompts_random_unique_nanbeige_messages.jsonl \
  --model-a-label gpt-4o --model-b-label nanbeige4.1-3b
```

The judge randomizes A/B order per prompt, uses extended thinking, and outputs structured XML with reasoning, verdict, genre, difficulty, and language tags.

### 4. Generate report and plots

```bash
uv run python make_judgment_pdf_report.py --judgments <judgments.jsonl>
uv run python plot_qwen_winrate_by_difficulty.py
uv run python plot_experiment_summary.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `to_json.py` | Convert source parquet to JSONL |
| `generate_gpt4o_answers.py` | Batch generation for GPT-4o via OpenRouter |
| `generate_nanbeige_answers.py` | Batch generation for Nanbeige 4.1-3B (strips `<think>` tags) |
| `generate_llama_answers.py` | Batch generation for Llama 3.1 8B (control experiment) |
| `judge_gpt4o_vs_qwen4b.py` | Generic pairwise judge runner using Claude Opus 4.6 |
| `make_judgment_pdf_report.py` | PDF report with statistics, plots, and subgroup analysis |
| `plot_qwen_winrate_by_difficulty.py` | Win rate heatmap by knowledge x reasoning difficulty |
| `plot_experiment_summary.py` | Side-by-side summary figure for multiple experiments |

## Model parameters

| Model | Temperature | top_p | Other |
|-------|------------|-------|-------|
| GPT-4o | default | default | via OpenRouter |
| Nanbeige 4.1-3B | 0.6 | 0.95 | thinking enabled, `<think>` stripped before judging |
| Llama 3.1 8B | 0.6 | 0.9 | seed=0, max_tokens=4096 |
| Claude Opus 4.6 (judge) | 0 | - | reasoning enabled, verbosity=max, max_tokens=32768 |

## Deploying the Nanbeige endpoint

The Nanbeige 4.1-3B answers were generated using a [HuggingFace Inference Endpoint](https://huggingface.co/docs/inference-endpoints) with the following configuration:

| Setting | Value |
|---------|-------|
| Model | [Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) |
| Revision | `6f3b2c3` (latest) |
| Engine | vLLM (`vllm/vllm-openai:v0.16.0`) |
| Task | `text-generation` |
| GPU | Nvidia A10G (1x GPU, 24 GB VRAM, 6x vCPUs, 30 GB RAM) |
| Region | AWS `us-east-1` (N. Virginia) |
| Security | Public (no authentication required) |
| Autoscaling | Min 1 / Max 4 replicas, hardware usage strategy (80% threshold) |
| Scale-to-zero | Never |
| KV cache dtype | Auto |
| Cost | ~$1.00/h per replica |

All engine parameters (max batched tokens, max sequences, tensor/data parallel size) were left at vLLM defaults. No custom container arguments, commands, or environment variables were set.

The endpoint exposes an OpenAI-compatible `/v1/chat/completions` API, so the generation script uses the standard `openai` Python client with a custom `--base-url`.

To deploy your own:

1. Go to [HuggingFace Inference Endpoints](https://endpoints.huggingface.co)
2. Click **Deploy** → select `Nanbeige/Nanbeige4.1-3B`
3. Pick **Nvidia A10G** (24 GB VRAM is sufficient for the 3B model)
4. Engine will auto-select **vLLM** with task **text-generation**
5. Set min replicas to 1 and disable scale-to-zero to avoid cold starts during batch jobs
6. Set security to **Public** if you don't need authentication (the generation script uses `api_key="none"`)

Generated datasets, judgments, reports, and plots are gitignored so the repo stays source-only.
