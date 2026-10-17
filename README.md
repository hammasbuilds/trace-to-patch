# trace-to-patch

[![ci](https://github.com/hammasbuilds/trace-to-patch/actions/workflows/ci.yml/badge.svg)](https://github.com/hammasbuilds/trace-to-patch/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![runtime deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

We built trace-to-patch to take a red test in a Python repository to a verified patch, or to
an explicit "could not reproduce". It locates the faulty function, writes a failing
reproduction test before any fix is tried, makes one patch attempt with a local model, and
keeps the patch only if the reproduction now passes, nothing that was green went red, and the
edit stayed inside the one function. It is for developers who want an automated fixer whose
"fixed" means something, and for anyone measuring how much a traceback really helps
localisation.

## What it does

- **Locate**: ranks candidate functions with three signals fused by reciprocal rank: frames
  from the traceback, BM25 over function source, and embeddings (`nomic-embed-text`).
- **Reproduce first**: a generated test must fail on the current code, and fail for the
  stated reason (a test that dies on its own bad import does not count). Otherwise the
  result is `could not reproduce`, reported as an outcome.
- **Patch once**: one attempt with `qwen2.5-coder:14b`. A refused patch stays refused and says why.
- **Verify three ways**: the reproduction passes, no previously green test is red, and the
  patch touched only the target function. The tree is restored unless all three hold.
- **Bench**: injects single-point bugs into a real library, keeps the ones its suite
  notices, and scores every stage against the known answer.

## Sample input / output

`python demo.py` (localisation only, no model; `toolz` is vendored under `targets/`):

```
trace-to-patch: how much is a traceback worth when locating a bug?

injecting bugs into toolz and keeping the ones its suite notices...
  5 cases: 3 traceback, 2 assertion
  [1/5] toolz/itertoolz.py::accumulate               traceback rank=  1 repro=n patch=n
  [2/5] toolz/itertoolz.py::_merge_sorted_binary     traceback rank=  1 repro=n patch=n
  [3/5] toolz/itertoolz.py::_merge_sorted_binary     assertion rank=  2 repro=n patch=n
  [4/5] toolz/itertoolz.py::_merge_sorted_binary_key assertion rank=  3 repro=n patch=n
  [5/5] toolz/itertoolz.py::get                      traceback rank=  1 repro=n patch=n

==============================================================================
TRACE TO PATCH - 5 injected bugs in toolz
==============================================================================
  signal          n     r@1     r@3    r@10   repro  patched
  traceback       3 100.0% 100.0% 100.0%   0.0%     0.0%
  assertion       2   0.0% 100.0% 100.0%   0.0%     0.0%
  traceback    breaks a median of 2 test(s) (range 2-3)
  assertion    breaks a median of 2 test(s) (range 2-2)

  a traceback is worth +100.0% of recall@1 over a bare assertion,
  and it is present for 3/5 (60%) of these failures.
  took 26s
```

Five cases is a smoke test of the pipeline; the measured result is the 25-bug run below.
`repro` and `patched` are 0% here because the demo runs without a model.

## Results

25 single-point bugs injected into `toolz`, localisation with both retrieval arms
(full run in [docs/RESULTS.md](docs/RESULTS.md)):

- 56% of the failures came with a traceback naming a library file; 44% were a bare assertion
  in the test with no library frame at all.

| signal | n | r@1 | r@3 | r@10 |
|---|---:|---:|---:|---:|
| traceback | 14 | 35.7% | 42.9% | 64.3% |
| bare assertion | 11 | **45.5%** | 63.6% | **90.9%** |

- Failures with a traceback were localised worse: a traceback names where execution stopped,
  which is often not where the mistake was made.
- Depth did not explain the misses (median tests broken: 2 for found and missed bugs alike; the
  deepest bug broke 27 tests and was found at rank 3). The missed functions (`curry.__eq__`,
  `num_pos_args`, `check_partial`, `juxt.__init__`) have no test named after them.
- The published run is `--locate-only`; the patch stage has no published rate yet.

## Tech stack

- Python 3.11+, zero runtime dependencies
- `ast` and `subprocess` (stdlib) to read functions and run pytest; BM25 written in plain Python
- A local Ollama server over HTTP: `qwen2.5-coder:14b` for reproduction tests and patches,
  `nomic-embed-text` for the embedding arm
- pytest, ruff, hatchling, GitHub Actions

## Run it

```bash
git clone https://github.com/hammasbuilds/trace-to-patch
cd trace-to-patch
uv sync --all-groups

uv run python demo.py

# fix whatever is currently red
uv run trace-to-patch fix /path/to/repo

# measure every stage against known answers
uv run trace-to-patch bench /path/to/repo --limit 45

# localisation only - no language model at all
uv run trace-to-patch bench /path/to/repo --locate-only --no-embeddings
```

`--locate-only --no-embeddings` is pure standard library and needs nothing running. The full
pipeline needs Ollama with `qwen2.5-coder:14b`, plus `nomic-embed-text` for the semantic arm.

Code layout: `src/trace_to_patch/` holds `failure.py` (run the suite, read the traceback),
`locate/retrieval.py` (the three signals and their fusion), `reproduce.py`, `patch.py`,
`bench.py` and `types.py`.

Related: [pr-referee](https://github.com/hammasbuilds/pr-referee) and
[repo-surgeon](https://github.com/hammasbuilds/repo-surgeon) apply the same "only what can be
proven" rule to review and to migrations;
[swebench-localization](https://github.com/hammasbuilds/swebench-localization) measures
issue-text-to-file retrieval.

## Licence

MIT, see [LICENSE](LICENSE). `targets/toolz` is a vendored copy of
[toolz](https://github.com/pytoolz/toolz) (BSD-3-Clause).
