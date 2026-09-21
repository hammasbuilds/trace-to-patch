"""Inject a bug, keep the answer, and measure every stage against it.

The labels here are free and exact, which is the reason this benchmark is worth trusting:
the bug is introduced by mutating one function, so the function at fault is known by
construction rather than inferred. Nothing is annotated and nothing is guessed.

A case is only kept when the mutation **actually makes the suite red**. A mutation the
tests do not notice is not a bug anyone would be debugging, and scoring a locator against a
failure that never happened would be inventing a number.

Four things are measured, and every one is reported split by signal:

    localised    is the real culprit in the top-k candidates
    reproduced   did a generated test fail against the broken code, for the right reason
    patched      did the fix pass the reproduction and regress nothing
    end-to-end   all three

The split is the point. A traceback names a file; a bare assertion names a test. Reporting
one average over both would hide the only interesting thing in the data.
"""

from __future__ import annotations

import ast
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from trace_to_patch.failure import parse, run_suite
from trace_to_patch.types import Failure, Signal

SKIP = {".git", "__pycache__", ".venv", ".tox", ".nox", ".pytest_cache", ".ruff_cache"}


class Mutator(ast.NodeTransformer):
    """One single-point change, chosen by index, on comparisons and constants."""

    SWAP = {
        ast.Lt: ast.GtE,
        ast.Gt: ast.LtE,
        ast.LtE: ast.Gt,
        ast.GtE: ast.Lt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }
    ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add}

    def __init__(self, target: int) -> None:
        self.target, self.seen, self.hit = target, 0, False
        self.func: str | None = None
        self._stack: list[str] = []
        self._class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._class = node.name
        self.generic_visit(node)
        self._class = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Qualify a method as `Class.method`, matching how the locator indexes it. Two
        # classes in one file both defining `__init__` are different functions, and
        # labelling both as `__init__` would score a hit on the wrong one.
        qualified = f"{self._class}.{node.name}" if self._class and not self._stack else node.name
        self._stack.append(qualified)
        self.generic_visit(node)
        self._stack.pop()
        return node

    def _take(self) -> bool:
        hit = self.seen == self.target
        self.seen += 1
        if hit and self._stack:
            self.func = self._stack[0]  # attribute to the outermost enclosing definition
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if node.ops and type(node.ops[0]) in self.SWAP and self._take():
            node.ops[0] = self.SWAP[type(node.ops[0])]()
            self.hit = True
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in self.ARITH and self._take():
            node.op = self.ARITH[type(node.op)]()
            self.hit = True
        return node


@dataclass
class Case:
    """One injected bug, its resulting failure, and the answer."""

    repo: Path
    rel: str
    truth_func: str
    failure: Failure
    blast: int = 0
    """How many distinct tests this one bug breaks.

    A proxy for how deep the broken function sits. A leaf that only its own test calls
    breaks one test; a utility that everything calls breaks dozens - and then the name of
    whichever test happened to fail first says nothing at all about the cause.
    """

    @property
    def truth(self) -> str:
        return f"{self.rel}::{self.truth_func}"

    @property
    def signal(self) -> Signal:
        return self.failure.signal


@dataclass
class Tally:
    """Counts per signal type, so the two cases are never averaged together."""

    n: int = 0
    localised: int = 0
    reproduced: int = 0
    patched: int = 0
    ranks: list[int] = field(default_factory=list)

    def recall_at(self, k: int) -> float | None:
        return (sum(r <= k for r in self.ranks) / self.n) if self.n else None


def copy_repo(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SKIP))


def library_files(repo: Path) -> list[Path]:
    """Files that are actually part of the shipped package.

    Membership is decided by the directory having an `__init__.py`, not by a name
    blacklist. `examples/`, `docs/` and `bench/` are not the library, and a bug injected
    into one of them produces a failure that no reasonable locator should be scored on -
    it was quietly costing this benchmark three cases out of twenty-eight.
    """
    out = []
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo)
        if any(part in SKIP for part in rel.parts):
            continue
        if "test" in p.name or "tests" in rel.parts:
            continue
        if not (p.parent / "__init__.py").is_file():
            continue
        out.append(p)
    return out


def build(
    source_repo: Path,
    workdir: Path,
    limit: int = 20,
    seed: int = 0,
    test_target: str = "",
) -> list[Case]:
    """Injected bugs that the suite actually notices, with the culprit recorded."""
    rnd = random.Random(seed)
    files = library_files(source_repo)
    rnd.shuffle(files)
    cases: list[Case] = []

    for path in files:
        if len(cases) >= limit:
            break
        rel = str(path.relative_to(source_repo)).replace("\\", "/")
        try:
            src = path.read_text(encoding="utf-8")
            ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue

        for target in range(0, 40, 3):
            if len(cases) >= limit:
                break
            m = Mutator(target)
            try:
                tree = m.visit(ast.parse(src))
            except (SyntaxError, RecursionError):
                continue
            if not m.hit or not m.func:
                continue
            ast.fix_missing_locations(tree)
            try:
                mutated = ast.unparse(tree)
            except (AttributeError, ValueError):
                continue

            work = Path(tempfile.mkdtemp(dir=workdir))
            try:
                copy_repo(source_repo, work / "repo")
                (work / "repo" / rel).write_text(mutated, encoding="utf-8", newline="")
                ok, output = run_suite(work / "repo", test_target, maxfail=1)
                if ok:
                    # The suite does not notice: not a bug anyone would be debugging.
                    continue
                failures = parse(output)
                if not failures:
                    continue
                blast = len(baseline_failures(work / "repo", test_target))
                cases.append(
                    Case(
                        repo=work / "repo",
                        rel=rel,
                        truth_func=m.func,
                        failure=failures[0],
                        blast=blast,
                    )
                )
            except Exception:  # noqa: BLE001
                shutil.rmtree(work, ignore_errors=True)
                continue
    return cases


def signal_breakdown(cases: list[Case]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        out[str(c.signal)] = out.get(str(c.signal), 0) + 1
    return out


def baseline_failures(repo: Path, test_target: str = "") -> set[str]:
    """Which tests are already red, before any patch. The bug makes some fail by design."""
    _, output = run_suite(repo, test_target, maxfail=0)
    return set(re.findall(r"^FAILED\s+(\S+)", output, re.M))
