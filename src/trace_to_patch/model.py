"""Ollama over HTTP, with an on-disk cache.

One model call per hunk, with a careful prompt and strict extraction. That is the whole
requirement, so there is no framework here - a chat abstraction would add indirection
without adding anything the caller needs.

The cache is keyed on everything that changes the answer. It matters more than it looks:
a migration is re-run constantly while the verification ladder is being tuned, and without
a cache every run pays for 300 generations to test a change in the differential checker.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CACHE = Path(os.environ.get("TRACE_TO_PATCH_CACHE", Path.home() / ".cache" / "trace-to-patch"))

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _key(model: str, prompt: str, temperature: float, seed: int | None) -> str:
    raw = json.dumps([model, prompt, temperature, seed], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def generate(
    prompt: str,
    model: str = "qwen2.5-coder:14b",
    temperature: float = 0.0,
    seed: int | None = None,
    num_predict: int = 1024,
    timeout: int = 300,
    use_cache: bool = True,
) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{_key(model, prompt, temperature, seed)}.txt"
    if use_cache and path.is_file():
        return path.read_text(encoding="utf-8")

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "keep_alive": "30m",
    }
    if seed is not None:
        body["options"]["seed"] = seed

    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            out = json.loads(fh.read()).get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    if use_cache:
        path.write_text(out, encoding="utf-8", newline="")
    return out


def generate_many(
    prompts: list[str],
    model: str = "qwen2.5-coder:14b",
    temperature: float = 0.0,
    workers: int = 8,
    num_predict: int = 1024,
    progress: str = "",
) -> list[str | None]:
    """Concurrent generation. One request at a time leaves the GPU mostly idle waiting
    on HTTP round trips rather than decoding."""
    out: list[str | None] = [None] * len(prompts)
    done = 0

    def one(i: int) -> None:
        nonlocal done
        out[i] = generate(prompts[i], model, temperature, num_predict=num_predict)
        done += 1
        if progress and done % 10 == 0:
            print(f"    {progress}: {done}/{len(prompts)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(len(prompts))))
    return out


def extract_code(raw: str) -> str:
    """The code out of a response, whether or not it came fenced."""
    blocks = _FENCE.findall(raw or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (raw or "").strip()


def available(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=10) as fh:
            names = {m["name"] for m in json.loads(fh.read())["models"]}
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
        return False
    return model in names or f"{model}:latest" in names
