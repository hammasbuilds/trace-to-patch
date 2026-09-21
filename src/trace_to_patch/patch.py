"""Propose a fix, then let three checks decide whether it was one.

One attempt. `code-llm-lab` ran self-debug loops to five rounds and found rounds 1-2
captured 100% of everything the loop ever achieved, and pitted repair against rewrite on
first-attempt failures with 93.3% surviving both. A second attempt at the same broken
function buys close to nothing and costs a full generation, so a refused patch stays
refused and the reason is reported.

The three checks, in order, and each one is there because the other two miss something:

1. **The reproduction now passes.** Necessary, and nowhere near sufficient - deleting the
   function's body satisfies it more often than one would like.
2. **The rest of the suite is no worse than the baseline.** Catches the deletion, and
   catches a fix that trades this bug for another one.
3. **The patch stayed inside the located function.** A model asked to fix one function will
   sometimes rewrite its neighbours too, and a change nobody reviewed is not a fix even
   when the tests are green.

The baseline for check 2 is taken against the *broken* tree, before any patch. The bug is
already making tests fail; comparing against a green suite that does not exist would mark
every patch as a regression.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from trace_to_patch.failure import run_suite
from trace_to_patch.model import extract_code, generate
from trace_to_patch.types import Candidate, Failure

PROMPT = """This Python function has a bug.

The failing test:
{test_id}

What it asserted:
{assertion}

What happened:
{message}

```python
{source}
```

Fix the bug. Change as little as possible: do not rename the function, do not change its
parameters, do not restructure code that is not part of the bug.

Output ONLY the corrected function and any imports it needs. No explanation.
"""


def read_function(repo: Path, cand: Candidate) -> tuple[str, int, int] | None:
    """(source, start line, end line) for a candidate, which may be `Class.method`.

    Methods are in scope here. Verification runs the repository's own test suite, which
    builds whatever instances it needs, so a method is no harder to check than a free
    function - unlike in `repo-surgeon`, where verifying means calling the thing directly.
    """
    path = repo / cand.path
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    lines = src.splitlines()

    def span(node) -> tuple[str, int, int]:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno or node.lineno
        return "\n".join(lines[start - 1 : end]), start, end

    cls, _, bare = cand.func.rpartition(".")
    for node in tree.body:
        if not cls and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name == bare:
                return span(node)
        elif cls and isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef) and sub.name == bare:
                    return span(sub)
    return None


def indent_of(source: str) -> str:
    """The leading whitespace on a block's first non-empty line."""
    for line in source.splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


def reindent(proposed: str, to: str) -> str:
    """Put a proposal back at the indentation of the block it replaces.

    A method's source is indented inside its class, but the model is shown that block and
    answers with a dedented `def`. Writing that straight back into the class body is a
    SyntaxError, and the patch would be refused for a reason that has nothing to do with
    whether it was correct.
    """
    body = textwrap.dedent(proposed)
    if not to:
        return body
    return "\n".join(to + line if line.strip() else line for line in body.splitlines())


def apply_patch(repo: Path, cand: Candidate, start: int, end: int, new_source: str) -> str:
    """Write the patch in and return the file's previous contents, so it can be undone."""
    path = repo / cand.path
    before = path.read_text(encoding="utf-8")
    lines = before.splitlines()
    original = "\n".join(lines[start - 1 : end])
    lines[start - 1 : end] = reindent(new_source, indent_of(original)).splitlines()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return before


def restore(repo: Path, cand: Candidate, before: str) -> None:
    (repo / cand.path).write_text(before, encoding="utf-8", newline="")


def touched_only(original: str, proposed: str, func: str) -> tuple[bool, str]:
    """Did the proposal stay inside the one function it was given?

    A model asked to fix `merge_sorted` will occasionally return `merge_sorted` plus a
    rewritten helper beside it. Tests may well pass. An unreviewed change is still not a
    fix, so it is refused and named.
    """
    # Both sides dedented: a method arrives indented inside its class, and parsing it as
    # written would fail for reasons that say nothing about the patch.
    proposed = textwrap.dedent(proposed)
    original = textwrap.dedent(original)
    try:
        tree = ast.parse(proposed)
    except SyntaxError as exc:
        return False, f"the patch does not parse: {exc.msg}"

    bare = func.rpartition(".")[2]
    defined = [n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if bare not in defined:
        return False, f"the patch does not define `{bare}`"
    extra = [d for d in defined if d != bare]
    if extra:
        return False, f"the patch also redefines {', '.join(extra)}"

    try:
        old_tree = ast.parse(original)
    except SyntaxError:
        return True, ""
    old_fn = next(
        (n for n in old_tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)), None
    )
    new_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)), None
    )
    if old_fn and new_fn:
        old_args = [a.arg for a in old_fn.args.args]
        new_args = [a.arg for a in new_fn.args.args]
        if old_args != new_args:
            return False, f"the patch changed the signature: {old_args} -> {new_args}"
    return True, ""


def attempt(
    repo: Path,
    failure: Failure,
    cand: Candidate,
    reproduction: str,
    baseline_failed: set[str],
    model: str = "qwen2.5-coder:14b",
) -> tuple[bool, str, str]:
    """(verified, patch source, verdict). Leaves the tree as it found it unless verified."""
    found = read_function(repo, cand)
    if found is None:
        return False, "", f"could not read {cand.key}"
    source, start, end = found

    raw = generate(
        PROMPT.format(
            test_id=failure.test_id,
            assertion=failure.assertion or "(none)",
            message=failure.message or "(none)",
            source=source,
        ),
        model=model,
        temperature=0.0,
    )
    proposed = extract_code(raw or "")
    if not proposed.strip():
        return False, "", "the model returned nothing"

    ok, why = touched_only(source, proposed, cand.func)
    if not ok:
        return False, proposed, why

    before = apply_patch(repo, cand, start, end, proposed)
    try:
        from trace_to_patch.reproduce import run_snippet

        still_fails, _ = run_snippet(repo, reproduction)
        if still_fails:
            restore(repo, cand, before)
            return False, proposed, "the reproduction still fails - the bug is not fixed"

        _, output = run_suite(repo, maxfail=0)
        import re

        now_failed = set(re.findall(r"^FAILED\s+(\S+)", output, re.M))
        new_breaks = now_failed - baseline_failed
        if new_breaks:
            restore(repo, cand, before)
            listed = ", ".join(sorted(new_breaks)[:3])
            return False, proposed, f"the patch broke {len(new_breaks)} other test(s): {listed}"
    except Exception as exc:  # noqa: BLE001 - never leave the tree patched on an error
        restore(repo, cand, before)
        return False, proposed, f"verification errored: {type(exc).__name__}: {exc}"

    return True, proposed, "reproduction passes and no other test regressed"
