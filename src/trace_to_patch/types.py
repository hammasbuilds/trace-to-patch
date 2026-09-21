"""What moves through the pipeline: a failure, a guess at its cause, and a verdict.

The shape of this tool is set by one measurement, taken before any of it was written.
Twenty-five single-point bugs were injected into a real library and its own test suite run
against each:

    60%  the test fails on a bare assertion - a wrong value, and ZERO library frames
    40%  something raises, and the traceback names exactly one library file

So for three failures in five there is no trace to follow at all. The only signal is which
test went red. That is why `Failure` carries `frames` as a *possibly empty* list and why
localisation has two paths rather than one: an easy case that is over in a step, and a hard
case that is the majority.

A tool that assumed the traceback would be there would work 40% of the time and quietly
guess the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Signal(StrEnum):
    """How much the failure tells you about where to look."""

    TRACEBACK = "traceback"
    """Something raised. The traceback names at least one library file."""

    ASSERTION = "assertion"
    """A bare assert in the test. Names the symptom, not the cause."""

    def __str__(self) -> str:
        return self.value


@dataclass
class Frame:
    path: str
    lineno: int
    func: str

    @property
    def is_test(self) -> bool:
        return "test" in self.path.lower()


@dataclass
class Failure:
    """One red test, and everything recoverable about why."""

    test_id: str
    message: str
    exc_type: str = ""
    frames: list[Frame] = field(default_factory=list)
    assertion: str = ""

    @property
    def signal(self) -> Signal:
        return (
            Signal.TRACEBACK
            if self.exc_type and self.exc_type != "AssertionError" and self.library_frames
            else Signal.ASSERTION
        )

    @property
    def library_frames(self) -> list[Frame]:
        """Frames outside the test files - the only ones that point at a cause."""
        return [f for f in self.frames if not f.is_test]


@dataclass
class Candidate:
    """A function that might be at fault, and why it was suggested."""

    path: str
    func: str
    score: float
    source: str  # "traceback" | "bm25" | "embedding" | "llm" | "fused"

    @property
    def key(self) -> str:
        return f"{self.path}::{self.func}"


@dataclass
class Attempt:
    """One end-to-end run against one failure."""

    failure: Failure
    candidates: list[Candidate] = field(default_factory=list)
    truth: str = ""  # "path::func" of the function actually broken, when known

    reproduced: bool = False
    reproduction: str = ""
    """A test written from the failure alone, which must fail before the patch.

    Written *first* and on purpose. Without it there is no way to tell a fix from a
    coincidence: the suite going green after a patch could mean the bug is gone, or that
    the patch deleted the code path the test exercised.
    """

    patched: str = ""
    verified: bool = False
    verdict: str = ""
    seconds: float = 0.0

    def rank_of_truth(self) -> int | None:
        """1-based position of the real culprit among the candidates, or None."""
        for i, c in enumerate(self.candidates, 1):
            if c.key == self.truth:
                return i
        return None
