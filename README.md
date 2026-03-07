# Qwen 4B vs GPT-4o Evaluation

Small scripts for:

- sampling prompts from a training set
- generating model responses
- judging pairwise outputs with Claude via OpenRouter
- plotting and reporting the results

## Setup

Install dependencies in your preferred environment, then provide API credentials either as environment variables or untracked local key files.

### OpenRouter

Use either:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `openrouter-key.txt` in the repo root

### Runpod / Llama endpoint

Use either:

- `RUNPOD_API_KEY`
- `OPENAI_API_KEY`
- `llama-key.txt` in the repo root

The real key files are ignored by `.gitignore`. Templates are included:

- `openrouter-key.txt.example`
- `llama-key.txt.example`

## Main scripts

- `to_json.py`: convert source data into chat-format JSONL
- `generate_gpt4o_answers.py`: batch OpenRouter generation for GPT-4o
- `generate_llama_answers.py`: batch Runpod/OpenAI-compatible generation for Llama 3.1 8B
- `judge_gpt4o_vs_qwen4b.py`: generic pairwise judge runner
- `make_judgment_pdf_report.py`: render a PDF report from judgments
- `plot_qwen_winrate_by_difficulty.py`: heatmap for Qwen win rate by difficulty
- `plot_experiment_summary.py`: compact summary figure for the benchmark

Generated datasets, judgments, reports, and plots are intentionally ignored so the repo stays source-only by default.
