"""Golden cases for the description checker.

The reviewer is a language model, so the checker is the only thing standing between a fluent
rewrite and a changed fact. Each case is a rewrite a reviewer could plausibly return, and
whether the checker must accept or reject it.

Run: python3 .github/review/test_review.py -v
"""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review import check  # noqa: E402

TITLE = "feat(vllm): replay PR 62240 in the live lab"
BODY = """## What and why

We hot-swap `sparse_attn_indexer` from vllm-project/vllm#62240 on base `0124e16a6c06`.
Install took 22 ms; the 32K point is not measured yet.

## Loop record

- Engine: vllm
- Upstream PRs: vllm-project/vllm#62240
- Function groups: OP
- Rule added: swap `__wrapped__.__code__` for decorated ops
- Overheads measured: install 22 ms (receipt r1)
- Receipts: none
- Not verified: FULL cudagraph batches
"""


def rewrite(old, new, body=BODY):
    assert old in body, old
    return body.replace(old, new)


class Accepts(unittest.TestCase):
    def test_clearer_wording_same_facts(self):
        """Plain rewording that keeps every number, reference and code span is accepted."""
        new = rewrite("We hot-swap `sparse_attn_indexer` from", "This PR hot-swaps `sparse_attn_indexer`, from")
        self.assertEqual(check(TITLE, BODY, "feat(vllm): replay vllm PR 62240 in the live lab", new), [])

    def test_unchanged(self):
        self.assertEqual(check(TITLE, BODY, TITLE, BODY), [])


    def test_backticks_around_existing_text(self):
        """Formatting existing text as code is not a new fact (seen on PR 6: the checker
        rejected a rewrite that only added backticks)."""
        new = rewrite("Install took 22 ms; the 32K point", "Install took `22 ms`; the `32K` point")
        self.assertEqual(check(TITLE, BODY, TITLE, new), [])

    def test_code_span_unwrapped(self):
        new = rewrite("We hot-swap `sparse_attn_indexer` from", "We hot-swap sparse_attn_indexer from")
        self.assertEqual(check(TITLE, BODY, TITLE, new), [])


class Rejects(unittest.TestCase):
    def test_changed_number(self):
        """The failure we fear most: a fluent rewrite that quietly changes a measurement."""
        new = rewrite("Install took 22 ms;", "Install took 2.2 ms;")
        p = check(TITLE, BODY, TITLE, new)
        self.assertTrue(any("numbers" in x for x in p), p)

    def test_invented_number(self):
        new = rewrite("the 32K point is not measured yet.", "the 32K point improves by 15%.")
        self.assertTrue(any("added numbers" in x for x in check(TITLE, BODY, TITLE, new)))

    def test_dropped_base_commit(self):
        """Dropping the base commit loses the one thing that makes the result reproducible."""
        new = rewrite(" on base `0124e16a6c06`", "")
        self.assertTrue(any("dropped code" in x for x in check(TITLE, BODY, TITLE, new)))

    def test_invented_code_span(self):
        """A new flag or command the author never wrote must not appear."""
        new = rewrite("Install took 22 ms;", "Install took 22 ms with `VLLM_USE_FASTRSI_JIT`;")
        self.assertTrue(any("added code" in x for x in check(TITLE, BODY, TITLE, new)))

    def test_changed_number_inside_code(self):
        """Moving a number into backticks must not hide a change to it."""
        new = rewrite("Install took 22 ms;", "Install took `2.2 ms`;")
        p = check(TITLE, BODY, TITLE, new)
        self.assertTrue(any("numbers" in x for x in p), p)

    def test_title_scope_changed(self):
        p = check(TITLE, BODY, "feat(core): replay PR 62240 in the live lab", BODY)
        self.assertIn("title changed its type or scope", p)

    def test_title_breaks_rule(self):
        p = check(TITLE, BODY, "Replay PR 62240 in the live lab", BODY)
        self.assertTrue(any("does not match" in x for x in p), p)

    def test_loop_record_field_removed(self):
        new = rewrite("- Receipts: none\n", "")
        self.assertIn("Loop record field missing: Receipts", check(TITLE, BODY, TITLE, new))

    def test_ai_attribution_added(self):
        new = BODY + "\nGenerated with [Claude Code](https://claude.com/claude-code)\n"
        self.assertIn("adds AI attribution", check(TITLE, BODY, TITLE, new))


if __name__ == "__main__":
    unittest.main()
