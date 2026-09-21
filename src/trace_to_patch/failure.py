"""Run a test suite and read what actually failed.

Parsing pytest output is unglamorous and it is where this tool either gets a real signal or
invents one. The trap that got me first: classifying a failure as a crash by matching
`/\\w*(Error|Exception)/` on pytest's `E` lines. `AssertionError` contains "Error", so every
ordinary assertion failure was counted as a crash and the measured crash rate came out at
52% instead of 40%.

That is why `AssertionError` is excluded explicitly below rather than by a pattern that
happens to work. A wrong value and a raised exception are the two cases this whole tool is
organised around, and mixing them up at the parsing step corrupts everything downstream.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from trace_to_patch.types import Failure, Frame

# "FAILED toolz/tests/test_x.py::test_y - assert [1] == [2]"
_FAILED = re.compile(r"^FAILED\s+(\S+?)(?:\s+-\s+(.*))?$", re.MULTILINE)
# "E       TypeError: unsupported operand" - the type on an error line.
_EXC = re.compile(
    r"^E\s+([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning))\b:?(.*)$", re.MULTILINE
)
# "toolz\itertoolz.py:351: in merge_sorted"  /  "toolz/itertoolz.py:351: AssertionError"
_FRAME = re.compile(r"^([\w./\\-]+\.py):(\d+): (?:in (\S+)|(\w+))", re.MULTILINE)
# The line pytest marks with ">" is the statement that failed.
_MARKED = re.compile(r"^>\s+(.+)$", re.MULTILINE)


def run_suite(
    repo: Path, target: str = "", timeout: float = 900.0, maxfail: int = 1
) -> tuple[bool, str]:
    """(everything passed, combined output)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "--tb=long",
        f"--maxfail={maxfail}",
        "-rf",
    ]
    if target:
        cmd.append(target)
    try:
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"the suite did not finish within {timeout}s"
    except OSError as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def parse(output: str) -> list[Failure]:
    """Every failure pytest reported, with whatever location information exists."""
    out: list[Failure] = []
    for test_id, summary in _FAILED.findall(output):
        exc_type = ""
        # The exception type, excluding AssertionError - which is not a crash, it is the
        # test's own assert firing, and conflating the two was a real bug here.
        for name, _ in _EXC.findall(output):
            if name != "AssertionError":
                exc_type = name
                break
        if not exc_type and "AssertionError" in output:
            exc_type = "AssertionError"

        frames = [
            Frame(path=p.replace("\\", "/"), lineno=int(n), func=(fn or kind or "?"))
            for p, n, fn, kind in _FRAME.findall(output)
        ]
        marked = _MARKED.findall(output)

        out.append(
            Failure(
                test_id=test_id,
                message=(summary or "").strip(),
                exc_type=exc_type,
                frames=frames,
                assertion=marked[-1].strip() if marked else "",
            )
        )
    return out


def first_failure(repo: Path, target: str = "", timeout: float = 900.0) -> Failure | None:
    ok, output = run_suite(repo, target, timeout)
    if ok:
        return None
    failures = parse(output)
    return failures[0] if failures else None
