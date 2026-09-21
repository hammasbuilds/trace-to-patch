"""Command line entry point.

    trace-to-patch fix   <repo> [--test tests/test_x.py]
    trace-to-patch bench <repo> --limit 20

`fix` runs the pipeline against whatever is currently red in a repo. `bench` injects known
bugs and measures each stage against the answer, split by whether a traceback existed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from trace_to_patch import bench as bench_mod
from trace_to_patch import patch as patch_mod
from trace_to_patch import reproduce as repro_mod
from trace_to_patch.failure import first_failure
from trace_to_patch.locate.retrieval import locate
from trace_to_patch.model import available
from trace_to_patch.types import Signal


def _imports_hint(repo: Path, rel: str) -> str:
    """How a generated test should import the code under test."""
    parts = Path(rel).with_suffix("").parts
    return f"from {'.'.join(parts)} import *"


def cmd_fix(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    t0 = time.time()

    print(f"running {repo.name}'s test suite...")
    failure = first_failure(repo, args.test)
    if failure is None:
        print("  nothing is failing. There is nothing here to fix.")
        return 0

    print(f"  FAILED {failure.test_id}")
    print(f"  signal: {failure.signal}", end="")
    if failure.signal is Signal.TRACEBACK:
        print(f" ({failure.exc_type}, {len(failure.library_frames)} library frame(s))")
    else:
        print("  <- a wrong value; the traceback names the test, not the cause")

    print("\nlocating...")
    candidates = locate(repo, failure, top_k=args.top_k, use_embeddings=not args.no_embeddings)
    if not candidates:
        print("  no candidates. Cannot proceed.")
        return 1
    for i, c in enumerate(candidates[:5], 1):
        print(f"  {i}. {c.key}")

    print("\nwriting a reproduction, before attempting any fix...")
    ok, reproduction, why = repro_mod.reproduce(
        repo, failure, _imports_hint(repo, candidates[0].path), model=args.model
    )
    if not ok:
        print(f"  COULD NOT REPRODUCE: {why}")
        print("\n  That is the honest outcome, not a failure of the tool. A bug that")
        print("  cannot be reproduced is a bug nobody should be patching blind.")
        return 1
    print("  reproduced: it fails against the current code, for the stated reason")

    baseline = bench_mod.baseline_failures(repo, args.test)
    print(f"\npatching (one attempt, {len(candidates[: args.tries])} candidate(s))...")
    for c in candidates[: args.tries]:
        verified, proposed, verdict = patch_mod.attempt(
            repo, failure, c, reproduction, baseline, model=args.model
        )
        mark = "VERIFIED" if verified else "refused "
        print(f"  {mark} {c.key}: {verdict}")
        if verified:
            print(f"\n  patch applied to {c.path} - `git diff` to review")
            print(f"  took {time.time() - t0:.0f}s")
            return 0

    print("\n  no candidate produced a verified patch. Nothing was changed.")
    print(f"  took {time.time() - t0:.0f}s")
    return 1


def cmd_bench(args: argparse.Namespace) -> int:
    source = Path(args.repo).resolve()
    t0 = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="ttp-bench-"))

    try:
        print(f"injecting bugs into {source.name} and keeping the ones its suite notices...")
        cases = bench_mod.build(
            source, workdir, limit=args.limit, seed=args.seed, test_target=args.test
        )
        if not cases:
            print("  no case produced a failing suite; nothing to measure")
            return 1
        breakdown = bench_mod.signal_breakdown(cases)
        print(f"  {len(cases)} cases: " + ", ".join(f"{v} {k}" for k, v in breakdown.items()))

        tallies = {s: bench_mod.Tally() for s in (Signal.TRACEBACK, Signal.ASSERTION)}
        rows = []

        for i, case in enumerate(cases, 1):
            t = tallies[case.signal]
            t.n += 1
            cands = locate(
                case.repo, case.failure, top_k=args.top_k, use_embeddings=not args.no_embeddings
            )
            rank = next((j for j, c in enumerate(cands, 1) if c.key == case.truth), None)
            if rank:
                t.ranks.append(rank)
                t.localised += 1

            reproduced = patched = False
            if rank and not args.locate_only:
                ok, reproduction, _ = repro_mod.reproduce(
                    case.repo, case.failure, _imports_hint(case.repo, case.rel), model=args.model
                )
                reproduced = ok
                t.reproduced += ok
                if ok:
                    baseline = bench_mod.baseline_failures(case.repo, args.test)
                    cand = next(c for c in cands if c.key == case.truth)
                    verified, _, _ = patch_mod.attempt(
                        case.repo, case.failure, cand, reproduction, baseline, model=args.model
                    )
                    patched = verified
                    t.patched += verified

            rows.append(
                {
                    "truth": case.truth,
                    "signal": str(case.signal),
                    "blast": case.blast,
                    "rank": rank,
                    "reproduced": reproduced,
                    "patched": patched,
                }
            )
            print(
                f"  [{i}/{len(cases)}] {case.truth:44} {case.signal:9} "
                f"rank={rank if rank else '-':>3} repro={'y' if reproduced else 'n'} "
                f"patch={'y' if patched else 'n'}"
            )

        print("\n" + "=" * 78)
        print(f"TRACE TO PATCH - {len(cases)} injected bugs in {source.name}")
        print("=" * 78)
        print(
            f"  {'signal':12} {'n':>4} {'r@1':>7} {'r@3':>7} {'r@10':>7} "
            f"{'repro':>7} {'patched':>8}"
        )
        for sig, t in tallies.items():
            if not t.n:
                continue
            f = lambda v: "   -  " if v is None else f"{v:6.1%}"  # noqa: E731
            print(
                f"  {str(sig):12} {t.n:4} {f(t.recall_at(1))} {f(t.recall_at(3))} "
                f"{f(t.recall_at(10))} {f(t.reproduced / t.n)} {f(t.patched / t.n):>8}"
            )

        # Blast radius: how many tests each bug breaks, split by signal. This is the
        # mechanism behind the recall numbers rather than a footnote to them.
        for sig in (Signal.TRACEBACK, Signal.ASSERTION):
            group = [c for c in cases if c.signal is sig]
            if not group:
                continue
            blasts = sorted(c.blast for c in group)
            mid = blasts[len(blasts) // 2]
            print(
                f"  {str(sig):12} breaks a median of {mid} test(s) (range {blasts[0]}-{blasts[-1]})"
            )

        tb, asrt = tallies[Signal.TRACEBACK], tallies[Signal.ASSERTION]
        if tb.n and asrt.n:
            d = (tb.recall_at(1) or 0) - (asrt.recall_at(1) or 0)
            print(f"\n  a traceback is worth {d:+.1%} of recall@1 over a bare assertion,")
            print(
                f"  and it is present for {tb.n}/{len(cases)} ({tb.n / len(cases):.0%}) "
                "of these failures."
            )

        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "repo": source.name,
                        "cases": len(cases),
                        "signals": breakdown,
                        "by_signal": {
                            str(s): {
                                "n": t.n,
                                "localised": t.localised,
                                "reproduced": t.reproduced,
                                "patched": t.patched,
                                "recall_at_1": t.recall_at(1),
                                "recall_at_3": t.recall_at(3),
                                "recall_at_10": t.recall_at(10),
                            }
                            for s, t in tallies.items()
                            if t.n
                        },
                        "rows": rows,
                        "seconds": round(time.time() - t0, 1),
                    },
                    indent=2,
                ),
                encoding="utf-8",
                newline="",
            )
            print(f"\n  wrote {args.json}")
        print(f"  took {time.time() - t0:.0f}s")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trace-to-patch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("repo")
        p.add_argument("--test", default="", help="restrict to one test file or node id")
        p.add_argument("--model", default="qwen2.5-coder:14b")
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--no-embeddings", action="store_true")

    f = sub.add_parser("fix", help="find and fix whatever is currently red")
    common(f)
    f.add_argument("--tries", type=int, default=3, help="candidates to attempt, best first")
    f.set_defaults(fn=cmd_fix)

    b = sub.add_parser("bench", help="inject known bugs and measure every stage")
    common(b)
    b.add_argument("--limit", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--locate-only", action="store_true", help="skip the model entirely")
    b.add_argument("--json")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    if getattr(args, "model", None) and args.cmd == "fix" and not available(args.model):
        print(f"{args.model} is not available from ollama")
        return 1
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
