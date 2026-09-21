"""Tests for the parts that decide. No model, no network.

Most of these guard against the tool *inventing a signal it does not have*. That is the
characteristic failure here: a locator that scores well because its labels leaked, a
reproduction that fails for its own reasons, a patch declared verified because the test it
was checked against never worked. Each one produces a confident wrong answer rather than
an error.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from trace_to_patch import bench
from trace_to_patch.failure import parse
from trace_to_patch.locate.retrieval import BM25, index_functions, query_from, rrf, tokenize
from trace_to_patch.patch import indent_of, reindent, touched_only
from trace_to_patch.reproduce import same_reason
from trace_to_patch.types import Candidate, Failure, Frame, Signal

# Real pytest output: a wrong value, so the only frame is the test file itself.
ASSERTION_OUTPUT = """\
=================================== FAILURES ===================================
______________________________ test_merge_sorted ______________________________
    def test_merge_sorted():
>       assert list(merge_sorted([1, 2, 3], [1, 2, 3])) == [1, 1, 2, 2, 3, 3]
E       assert [1, 2, 3, 1, 2, 3] == [1, 1, 2, 2, 3, 3]
E         At index 1 diff: 2 != 1

toolz/tests/test_itertoolz.py:69: AssertionError
=========================== short test summary info ============================
FAILED toolz/tests/test_itertoolz.py::test_merge_sorted - assert [1, 2, 3, 1,...
"""

# Real pytest output for a crash: a library frame appears.
TRACEBACK_OUTPUT = """\
=================================== FAILURES ===================================
__________________________________ test_get ___________________________________
    def test_get():
>       assert get(1, [1, 2, 3]) == 2

toolz/tests/test_itertoolz.py:120: in test_get
    assert get(1, [1, 2, 3]) == 2
toolz/itertoolz.py:351: in get
    return seq[ind]
E   IndexError: list index out of range
=========================== short test summary info ============================
FAILED toolz/tests/test_itertoolz.py::test_get - IndexError: list index out of range
"""


# --- reading the failure ---------------------------------------------------------------


def test_assertion_failure_is_not_read_as_a_crash():
    # The bug this pins: classifying by matching /\\w*(Error|Exception)/ on pytest's `E`
    # lines counts AssertionError as a crash, because it contains "Error". That single
    # mistake put the measured crash rate at 52% when the real figure is 40%, and the
    # whole tool is designed around that split.
    failures = parse(ASSERTION_OUTPUT)
    assert len(failures) == 1
    f = failures[0]
    assert f.exc_type == "AssertionError"
    assert f.signal is Signal.ASSERTION
    assert f.library_frames == []


def test_a_real_crash_is_read_as_a_traceback():
    f = parse(TRACEBACK_OUTPUT)[0]
    assert f.exc_type == "IndexError"
    assert f.signal is Signal.TRACEBACK
    assert any(fr.path.endswith("itertoolz.py") for fr in f.library_frames)


def test_the_marked_statement_is_captured():
    f = parse(ASSERTION_OUTPUT)[0]
    assert "merge_sorted" in f.assertion


def test_test_frames_are_not_library_frames():
    f = Failure(
        test_id="t",
        message="",
        exc_type="TypeError",
        frames=[
            Frame("toolz/tests/test_x.py", 1, "test_a"),
            Frame("toolz/itertoolz.py", 2, "nth"),
        ],
    )
    assert [fr.func for fr in f.library_frames] == ["nth"]


def test_a_crash_with_no_library_frame_is_not_a_usable_traceback():
    # It raised, but every frame is in the test. There is nothing to follow, so it has to
    # take the hard path rather than be treated as located.
    f = Failure(
        test_id="t",
        message="",
        exc_type="TypeError",
        frames=[Frame("tests/test_x.py", 1, "test_a")],
    )
    assert f.signal is Signal.ASSERTION


# --- retrieval -------------------------------------------------------------------------


def test_tokenizer_splits_snake_case_so_a_test_name_matches_its_target():
    # `test_merge_sorted` has to reach `merge_sorted`; neither matches the other whole.
    toks = tokenize("test_merge_sorted")
    assert "merge" in toks and "sorted" in toks


def test_tokenizer_splits_camel_case():
    assert "sorted" in tokenize("mergeSorted")


def test_bm25_ranks_the_matching_function_first():
    docs = [
        tokenize("merge_sorted def merge_sorted(*seqs): heapq merge"),
        tokenize("unrelated def concat(seqs): chain from iterable"),
        tokenize("also_unrelated def frequencies(seq): counter"),
    ]
    bm = BM25(docs)
    scores = bm.scores(tokenize("test_merge_sorted"))
    assert scores[0] == max(scores)


def test_rrf_needs_no_shared_scale():
    # BM25 scores, cosine similarities and "it was in the traceback" share no units, which
    # is exactly why fusion is by rank rather than by value.
    fused = rrf([["a", "b", "c"], ["b", "a", "c"]])
    assert fused["a"] > fused["c"] and fused["b"] > fused["c"]


def test_a_signal_present_in_both_rankings_outranks_one_in_only_a_single_ranking():
    fused = rrf([["x", "y"], ["y", "z"]])
    assert fused["y"] > fused["x"]


def test_query_uses_the_test_name_and_the_assertion():
    f = parse(ASSERTION_OUTPUT)[0]
    q = query_from(f)
    assert "test_merge_sorted" in q
    assert "merge_sorted" in q


def test_methods_are_indexed_as_class_dot_method(tmp_path):
    # Leaving methods out was a real bug: the benchmark broke them and recorded them as
    # the answer while the index could never contain them, so those cases were unfindable
    # by construction and quietly deflated recall.
    (tmp_path / "lib.py").write_text(
        "class Curry:\n    def __eq__(self, o):\n        return True\n\ndef free():\n    pass\n",
        encoding="utf-8",
    )
    found = {f for _, f, _ in index_functions(tmp_path)}
    assert found == {"Curry.__eq__", "free"}


def test_two_classes_with_the_same_method_name_stay_distinct(tmp_path):
    # Labelling both as `__init__` would score a hit on whichever came first.
    (tmp_path / "lib.py").write_text(
        "class A:\n    def __init__(self):\n        pass\n\n"
        "class B:\n    def __init__(self):\n        pass\n",
        encoding="utf-8",
    )
    found = {f for _, f, _ in index_functions(tmp_path)}
    assert found == {"A.__init__", "B.__init__"}


def test_index_skips_tests_and_finds_module_level_functions(tmp_path):
    (tmp_path / "lib.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text(
        "def test_alpha():\n    pass\n", encoding="utf-8"
    )
    found = {f for _, f, _ in index_functions(tmp_path)}
    assert found == {"alpha"}


# --- the reproduction gate -------------------------------------------------------------


def test_a_reproduction_failing_for_its_own_reasons_is_rejected():
    # A generated test that raises NameError because it imported something wrong also
    # "fails", and would then "pass" after literally any edit to the target.
    f = Failure(test_id="t", message="", exc_type="AssertionError")
    assert same_reason("E   NameError: name 'merge_sorted' is not defined", f) is False
    assert same_reason("E   ImportError: cannot import name 'x'", f) is False


def test_a_reproduction_failing_on_an_assert_is_accepted():
    f = Failure(test_id="t", message="", exc_type="AssertionError")
    assert same_reason("E       assert [1, 2] == [2, 1]", f) is True


def test_a_reproduction_must_raise_the_same_exception_type():
    f = Failure(test_id="t", message="", exc_type="IndexError")
    assert same_reason("E   IndexError: list index out of range", f) is True
    assert same_reason("E   KeyError: 'a'", f) is False


# --- the patch gate --------------------------------------------------------------------


def test_a_patch_that_rewrites_a_neighbour_is_refused():
    # Tests may well pass. A change nobody asked for and nobody reviewed is not a fix.
    original = "def target(x):\n    return x\n"
    proposed = "def target(x):\n    return x + 1\n\ndef helper(y):\n    return y\n"
    ok, why = touched_only(original, proposed, "target")
    assert ok is False and "helper" in why


def test_a_patch_that_changes_the_signature_is_refused():
    ok, why = touched_only("def f(a, b):\n    return a\n", "def f(a, b, c=1):\n    return a\n", "f")
    assert ok is False and "signature" in why


def test_a_patch_that_drops_the_function_is_refused():
    ok, why = touched_only("def f(a):\n    return a\n", "def g(a):\n    return a\n", "f")
    assert ok is False and "does not define" in why


def test_a_minimal_in_place_fix_is_accepted():
    ok, why = touched_only("def f(a):\n    return a - 1\n", "def f(a):\n    return a + 1\n", "f")
    assert ok is True and why == ""


def test_a_method_patch_is_put_back_at_the_right_indentation():
    # The model is shown an indented method and answers with a dedented `def`. Writing
    # that straight into the class body is a SyntaxError, and the patch would be refused
    # for a reason that says nothing about whether it was correct.
    method = "    def __eq__(self, o):\n        return self.x == o.x\n"
    proposed = "def __eq__(self, o):\n    return self.x != o.x\n"
    assert indent_of(method) == "    "
    out = reindent(proposed, indent_of(method))
    assert out.startswith("    def __eq__")
    assert "        return" in out


def test_reindent_leaves_a_module_level_function_alone():
    src = "def f(a):\n    return a\n"
    assert reindent(src, "") == src


def test_touched_only_accepts_a_dedented_method_patch():
    method = "    def __eq__(self, o):\n        return self.x == o.x\n"
    proposed = "def __eq__(self, o):\n    return self.x != o.x\n"
    ok, why = touched_only(method, proposed, "Curry.__eq__")
    assert ok is True, why


def test_an_unparseable_patch_is_refused():
    ok, why = touched_only("def f(a):\n    return a\n", "def f(a:\n", "f")
    assert ok is False and "does not parse" in why


# --- the benchmark's labels ------------------------------------------------------------


def test_the_mutator_records_which_function_it_broke():
    src = textwrap.dedent("""
        def alpha(n):
            if n > 0:
                return n
            return 0

        def beta(n):
            return n + 1
    """).strip()
    found = set()
    for target in range(6):
        m = bench.Mutator(target)
        m.visit(ast.parse(src))
        if m.hit and m.func:
            found.add(m.func)
    # Both functions have a mutable site, and each mutation is attributed to its own.
    assert found == {"alpha", "beta"}


def test_a_mutation_is_attributed_to_the_top_level_function_not_a_nested_one():
    src = "def outer(n):\n    def inner(m):\n        return m + 1\n    return inner(n)\n"
    for target in range(4):
        m = bench.Mutator(target)
        m.visit(ast.parse(src))
        if m.hit:
            # The file is patched at top-level function granularity, so that is the unit
            # the locator is scored against.
            assert m.func == "outer"
            return
    pytest.fail("no mutation was produced")


def test_tally_recall_is_over_all_cases_not_just_the_found_ones():
    # Dividing by len(ranks) instead of n would report recall over the cases it already
    # succeeded on, which is always flattering and always wrong.
    t = bench.Tally(n=4, ranks=[1, 3])
    assert t.recall_at(1) == 0.25
    assert t.recall_at(3) == 0.5
    assert t.recall_at(10) == 0.5


def test_tally_of_nothing_is_none_not_zero():
    assert bench.Tally().recall_at(1) is None


# --- candidates ------------------------------------------------------------------------


def test_candidate_key_matches_the_truth_format():
    c = Candidate(path="toolz/itertoolz.py", func="nth", score=1.0, source="fused")
    assert c.key == "toolz/itertoolz.py::nth"
