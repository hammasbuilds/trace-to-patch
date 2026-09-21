# Results

One run, reproduced by:

```bash
git clone --depth 1 https://github.com/pytoolz/toolz targets/toolz
trace-to-patch bench targets/toolz --limit 45 --locate-only --test toolz/tests
```

`--locate-only` needs no language model at all; the embedding arm uses `nomic-embed-text`.
Every number below is read out of `docs/bench-*.json`, not typed by hand.

## The set

25 single-point bugs injected into `toolz`, keeping only the ones its own test
suite actually notices. The broken function is known by construction, so the labels are
exact rather than annotated.

| | count | share |
|---|---:|---:|
| produced a traceback naming a library file | 14 | 56% |
| produced only a bare assertion in the test | 11 | 44% |

**44% of failures give you nothing but a red
test name.** A logic bug returns a wrong value; it does not raise, so no frame inside the
library ever appears.

## Localisation

| signal | n | r@1 | r@3 | r@10 |
|---|---:|---:|---:|---:|
| traceback, lexical only | 14 | 35.7% | 50.0% | 50.0% |
| traceback + embeddings | 14 | 35.7% | 42.9% | **64.3%** |
| assertion, lexical only | 11 | 36.4% | 72.7% | 81.8% |
| assertion + embeddings | 11 | **45.5%** | 63.6% | **90.9%** |

### A traceback does not help

This is the result worth the whole project. With both arms running, a failure that produced
a traceback is localised **35.7%** of the time at rank 1
against **45.5%** for one that produced nothing but an
assertion, and at rank 10 the gap widens to
**64.3% against 90.9%**.

Having the extra information leaves you worse off. The reason is not that the traceback is
noise - it is that a traceback names **where execution stopped**, and that is a different
place from where the mistake is. A wrong value, by contrast, usually surfaces in the test
written for the very function that computed it.

### What does not explain it

The obvious story is that crashes come from deep utility functions, whose breakage surfaces
far from the cause. It was measured and it is **wrong**:

| | n | median tests broken | max |
|---|---:|---:|---:|
| localised | 16 | 2 | 27 |
| missed | 9 | 2 | 4 |

The single deepest bug in the set - `curry.__init__`, which breaks 27 tests - was found at
rank 3. The misses sit at a blast radius of 1 to 4, the same as the hits.

What the misses do share is a **name**: `curry.__eq__`, `curry.__signature__`,
`num_pos_args`, `check_partial`, `juxt.__init__`. No test in the suite is named after any of
them. Lexical retrieval is, to a first approximation, a test-naming-convention detector, and
the embedding arm exists to cover exactly the functions that convention misses - which is
what the `+9.1`
points it adds at rank 10 are actually buying.

### Embeddings deepen and blur

They add recall at k=10 and **cost** it at k=3
(72.7% to 63.6% on the
assertion arm). Semantic similarity surfaces functions lexical search never reaches, and it
also promotes plausible-looking neighbours above the exact match. Worth it when a human reads
ten candidates; not worth it if something downstream only ever takes the first.

## Limits

- **One repository.** `toolz` is small, functional and unusually well named, which flatters
  the lexical arm. A codebase with generic test names would look different.
- **Injected bugs are not human mistakes.** A flipped comparison is a proxy for a real bug.
- **25 cases**, and split two ways, so each arm is a dozen or so. The direction
  of the traceback result is consistent across every configuration run; the exact figures
  are not tight.
- **The patch and reproduction stages are not scored here.** `--locate-only` skips them, and
  they need a GPU that was busy. Localisation is the part this run establishes.
