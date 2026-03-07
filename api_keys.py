from __future__ import annotations

import os
from pathlib import Path


def _read_first_nonempty_line(path: Path) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            return value
    return None


def resolve_openrouter_api_key(repo_root: Path) -> str:
    key_file = repo_root / "openrouter-key.txt"
    value = _read_first_nonempty_line(key_file)
    if value:
        return value

    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_var)
        if value:
            return value

    raise RuntimeError(
        "Could not resolve an OpenRouter API key from openrouter-key.txt, "
        "OPENROUTER_API_KEY, or OPENAI_API_KEY."
    )


def resolve_runpod_api_key(repo_root: Path) -> str:
    key_file = repo_root / "llama-key.txt"
    value = _read_first_nonempty_line(key_file)
    if value:
        return value

    for env_var in ("RUNPOD_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_var)
        if value:
            return value

    raise RuntimeError(
        "Could not resolve a Runpod API key from llama-key.txt, "
        "RUNPOD_API_KEY, or OPENAI_API_KEY."
    )
