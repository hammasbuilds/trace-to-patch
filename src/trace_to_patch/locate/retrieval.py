"""Find the function at fault when the traceback does not name it.

This is the hard path, and the measurement says it is the common one: 60% of failures are a
bare assertion with zero library frames. All you know is which test went red.

Three signals, fused. None of them is reliable alone and they fail differently, which is the
only reason combining them helps:

- **The traceback**, when there is one. Nearly free and nearly always right - but present
  for only 40% of failures, so it cannot be the whole answer.
- **BM25 over function source**, queried with the test's own name and assertion text. A test
  called `test_merge_sorted` is a strong hint about a function called `merge_sorted`, and
  lexical overlap catches that without a model.
- **Embeddings**, for the case where the test's vocabulary and the function's do not
  overlap - the situation `swebench-localization` measured as the hard half of the problem.

Fused with reciprocal rank fusion, which needs no score calibration between the three. That
matters because BM25 scores, cosine similarities and "it was in the traceback" are not on
any common scale, and normalising them would mean inventing one.
"""

from __future__ import annotations

import ast
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from trace_to_patch.types import Candidate, Failure

HOST = "http://localhost:11434"
SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
    ".ruff_cache",
    "node_modules",
    "site-packages",
}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Words, plus the pieces of snake_case and camelCase names.

    `test_merge_sorted` has to match `merge_sorted`, and neither matches the other as a
    whole token. Splitting on both conventions is what makes the lexical arm work at all.
    """
    out: list[str] = []
    # Case is split BEFORE lowercasing. Doing it the other way round - lowercase the text
    # and then look for capitals - means the camelCase branch can never match anything,
    # which is a rule that silently does nothing rather than one that fails.
    for w in _WORD.findall(text):
        lower = w.lower()
        out.append(lower)
        out.extend(p.lower() for p in w.split("_") if p)
        out.extend(p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", w) if p)
    return out


def index_functions(repo: Path) -> list[tuple[str, str, str]]:
    """(path, name, source) for every function outside the tests, methods included.

    Methods are indexed as `Class.method`, and they are indexed at all because this tool
    can act on them. Its sibling `repo-surgeon` skips methods, since verifying a rewrite
    there means *calling* the function standalone and a method needs an instance. Here
    verification runs the repository's own test suite, which constructs whatever it needs
    by itself - so the constraint simply does not apply.

    Leaving them out was a real bug: the benchmark broke methods and recorded them as the
    answer while the index could never contain them, so those cases were unfindable by
    construction and quietly deflated recall.
    """
    out: list[tuple[str, str, str]] = []
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo)
        if any(part in SKIP for part in rel.parts):
            continue
        if "test" in p.name or "tests" in rel.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        lines = src.splitlines()
        rel_s = str(rel).replace("\\", "/")

        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                out.append((rel_s, node.name, body))
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                        body = "\n".join(lines[sub.lineno - 1 : sub.end_lineno])
                        out.append((rel_s, f"{node.name}.{sub.name}", body))
    return out


class BM25:
    """Okapi BM25. Pure Python, no dependency - the corpus is a few hundred functions."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avg = sum(len(d) for d in docs) / self.n if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for i, tf in enumerate(self.tf):
            length = len(self.docs[i]) or 1
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += (
                    self.idf.get(t, 0.0)
                    * (f * (self.k1 + 1))
                    / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
                )
            out[i] = s
        return out


def embed(texts: list[str], model: str = "nomic-embed-text", timeout: int = 300):
    """Batched embeddings. One request per item costs seconds each when a 14B holds VRAM."""
    try:
        req = urllib.request.Request(
            f"{HOST}/api/embed",
            data=json.dumps({"model": model, "input": texts, "keep_alive": "30m"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read()).get("embeddings")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def query_from(failure: Failure) -> str:
    """What to search with: the test's name, its assertion, and the error message.

    The test name is the strongest part - `test_merge_sorted` all but names its target -
    which is also why it is not used alone. A test named `test_edge_cases` says nothing,
    and those are exactly the failures where the other signals have to carry it.
    """
    test_name = failure.test_id.rsplit("::", 1)[-1]
    return f"{test_name} {failure.assertion} {failure.message} {failure.exc_type}"


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: no score calibration needed between different signals."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            out[key] = out.get(key, 0.0) + 1.0 / (k + rank)
    return out


def locate(
    repo: Path,
    failure: Failure,
    top_k: int = 10,
    use_embeddings: bool = True,
) -> list[Candidate]:
    """Ranked guesses at the function responsible."""
    corpus = index_functions(repo)
    if not corpus:
        return []
    keys = [f"{p}::{f}" for p, f, _ in corpus]

    rankings: list[list[str]] = []

    # 1. The traceback, when it exists. Cheap, precise, and available 40% of the time.
    tb = [
        f"{fr.path}::{fr.func}"
        for fr in failure.library_frames
        if f"{fr.path}::{fr.func}" in set(keys)
    ]
    if tb:
        rankings.append(tb)

    # 2. Lexical.
    query = tokenize(query_from(failure))
    bm = BM25([tokenize(f"{f} {src}") for _, f, src in corpus])
    scores = bm.scores(query)
    order = sorted(range(len(keys)), key=lambda i: -scores[i])[: top_k * 3]
    rankings.append([keys[i] for i in order])

    # 3. Semantic, for when the test's vocabulary and the function's do not overlap.
    if use_embeddings:
        vecs = embed([f"{f}\n{src[:1500]}" for _, f, src in corpus])
        qvec = embed([query_from(failure)])
        if vecs and qvec:
            sims = [_cosine(qvec[0], v) for v in vecs]
            order = sorted(range(len(keys)), key=lambda i: -sims[i])[: top_k * 3]
            rankings.append([keys[i] for i in order])

    fused = rrf(rankings)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    lookup = {f"{p}::{f}": (p, f) for p, f, _ in corpus}
    return [
        Candidate(path=lookup[k][0], func=lookup[k][1], score=s, source="fused")
        for k, s in ranked
        if k in lookup
    ]
