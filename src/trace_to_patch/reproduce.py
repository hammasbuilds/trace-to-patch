"""Write a test that fails, before attempting any fix.

This runs first, and it is the only reason a verdict from this tool means anything.

Without it, "the suite is green after the patch" has two readings and no way to tell them
apart: the bug is fixed, or the patch removed the code path that was failing. Deleting the
body of a function makes a surprising number of tests stop failing.

So a reproduction has to clear two bars, in this order:

1. It **fails** against the current, broken code. A reproduction that passes now is not a
   reproduction of anything - it is a test that happens to be true.
2. It fails **for the stated reason**. A test that errors on a typo in its own setup also
   "fails", and would then "pass" after any edit at all.

Only a reproduction that clears both is allowed to judge a patch. Everything else is
reported as `could not reproduce`, which is a first-class outcome here rather than a
failure of the tool - a bug nobody can reproduce is not a bug anybody should be patching.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from trace_to_patch.model import extract_code, generate
from trace_to_patch.types import Failure

PROMPT = """A test in this project is failing. Write the smallest possible standalone test
that reproduces the same failure.

Failing test: {test_id}
{exc}
What the test asserted:
{assertion}

What happened:
{message}

The code under test is imported like this:
{imports}

Write ONE test function named `test_reproduces`. It must fail against the current code for
the reason above. Use a plain `assert`. Do not import pytest. Do not use fixtures.

Output ONLY the test function and the imports it needs.
"""

_FAIL_RE = re.compile(r"\b(\d+) failed")


def build_prompt(failure: Failure, imports: str) -> str:
    exc = f"Exception raised: {failure.exc_type}\n" if failure.exc_type else ""
    return PROMPT.format(
        test_id=failure.test_id,
        exc=exc,
        assertion=failure.assertion or "(no assertion recorded)",
        message=failure.message or "(no message recorded)",
        imports=imports,
    )


def run_snippet(repo: Path, code: str, timeout: float = 120.0) -> tuple[bool, str]:
    """Run one test file inside the repo. Returns (it failed, output).

    Note the return value is "did it FAIL" rather than "did it pass" - at this stage
    failing is the desired outcome, and naming the flag for what is wanted keeps the
    call sites from reading backwards.
    """
    with tempfile.TemporaryDirectory(dir=repo) as tmp:
        path = Path(tmp) / "test_ttp_reproduction.py"
        # newline="" or Windows turns the newlines into CR CR LF and breaks any line
        # continuation inside the generated test.
        path.write_text(code, encoding="utf-8", newline="")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    str(path),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, str(exc)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode != 0 and bool(_FAIL_RE.search(out)), out


def same_reason(output: str, failure: Failure) -> bool:
    """Did it fail for the stated reason, rather than for some reason of its own?

    A generated test that raises `NameError` because it imported something that does not
    exist is not a reproduction, and would be "fixed" by any edit whatsoever.
    """
    if failure.exc_type and failure.exc_type != "AssertionError":
        return failure.exc_type in output
    # An assertion failure is the right shape if it, too, failed on an assert rather than
    # blowing up on its own scaffolding.
    broken_scaffolding = (
        "NameError",
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "TypeError: test_reproduces()",
    )
    return "assert" in output.lower() and not any(b in output for b in broken_scaffolding)


def reproduce(
    repo: Path,
    failure: Failure,
    imports: str,
    model: str = "qwen2.5-coder:14b",
    attempts: int = 2,
) -> tuple[bool, str, str]:
    """(reproduced, test source, why not).

    Two attempts rather than one, and that is a deliberate exception to this portfolio's
    usual rule. `code-llm-lab` measured retries as near worthless for *solving* a task -
    93.3% of first-attempt failures survive both repair and rewrite. But writing a
    reproduction is not solving anything, and the common failure is a mechanical one: a
    wrong import path, a fixture it was told not to use. A second draw fixes that class of
    mistake, and the harness can *verify* which draw worked, so the extra attempt costs a
    generation and risks nothing.
    """
    why = "no attempt made"
    for seed in range(attempts):
        raw = generate(build_prompt(failure, imports), model=model, temperature=0.0, seed=seed)
        code = extract_code(raw or "")
        if not code.strip():
            why = "the model returned nothing"
            continue
        failed, output = run_snippet(repo, code)
        if not failed:
            why = "the generated test passes against the broken code, so it reproduces nothing"
            continue
        if not same_reason(output, failure):
            why = "the generated test fails for a different reason than the original"
            continue
        return True, code, ""
    return False, "", why
