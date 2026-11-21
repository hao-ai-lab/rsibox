"""Golden cases for ci_timing_comment.py: API records in, the comment's numbers out.

Run: python3 -m unittest discover -s .github/scripts -p 'test_*.py' -v
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ci_timing_comment", os.path.join(HERE, "ci_timing_comment.py"))
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def step(name, start, end, conclusion="success"):
    return {"name": name, "started_at": start, "completed_at": end, "conclusion": conclusion}


def job(name, created, started, completed, steps, conclusion="success", runner="runner-a"):
    return {"name": name, "created_at": created, "started_at": started, "completed_at": completed,
            "conclusion": conclusion, "runner_name": runner, "steps": steps}


RUNS = [
    {"run": {"name": "CPU tests", "html_url": "https://example.test/runs/1", "conclusion": "failure",
             "created_at": "2026-01-01T00:00:00Z"},
     "jobs": [
         job("fast suite", "2026-01-01T00:00:00Z", "2026-01-01T00:00:05Z", "2026-01-01T00:00:15Z",
             [step("Set up job", "2026-01-01T00:00:05Z", "2026-01-01T00:00:07Z"),
              step("Run the unit cases", "2026-01-01T00:00:07Z", "2026-01-01T00:00:15Z")]),
         job("slow suite <engine> | toy", "2026-01-01T00:00:00Z", "2026-01-01T00:00:40Z",
             "2026-01-01T00:02:10Z",
             [step("Run the conformance cases", "2026-01-01T00:00:40Z", "2026-01-01T00:02:05Z", "failure"),
              step("Run setup-python", "2026-01-01T00:00:40Z", "2026-01-01T00:01:40Z", "skipped"),
              step("Complete job", "2026-01-01T00:02:05Z", "2026-01-01T00:02:10Z")],
             conclusion="failure", runner="runner-b"),
     ]},
    {"run": {"name": "PR rules", "html_url": "https://example.test/runs/2", "conclusion": "success",
             "created_at": "2026-01-01T00:00:02Z"},
     "jobs": [job("title", "2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z", "2026-01-01T00:00:08Z", [])]},
]


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.body = timing.render("abcdef1234567890", RUNS)
        self.lines = self.body.splitlines()

    def test_headline_counts_wall_clock_from_first_queue_to_last_job(self):
        """Input: two workflows; the first run is created at 00:00:00, the last job ends at
        00:02:10; jobs ran 10 s, 90 s and 5 s; one job failed. Expect: 2 min 10 s wall-clock,
        3 jobs, 1 min 45 s of runner time, "1 not passed". Proves: the headline is the span a
        person waits, not the sum of jobs. Not: that the API's timestamps are right."""
        self.assertEqual(self.lines[0], timing.MARKER)
        self.assertEqual(self.lines[1], "### CI timing for `abcdef1`")
        self.assertEqual(
            self.lines[3],
            "**2 min 10 s wall-clock** from the first queue to the last job done; "
            "3 jobs, 1 min 45 s of runner time, 1 not passed.")

    def test_jobs_are_sorted_by_run_time_with_queue_and_slowest_step(self):
        """Input: the slow job waited 40 s for a runner and ran 90 s, 85 s of it in one failed
        step; a skipped step of 60 s sits beside it. Expect: it is the first job row, with
        queued 40 s, ran 1 min 30 s, and that 85 s step as slowest; the skipped step is never
        shown. Proves: the costliest job leads and a skipped step is not counted as time spent.
        Not: per-test-case time (a step is a whole suite)."""
        header = self.lines.index("| Job | Result | Runner | Queued | Ran | Slowest step |")
        self.assertEqual(
            self.lines[header + 2],
            "| slow suite &lt;engine&gt; \\| toy | failure | runner-b | 40 s | 1 min 30 s "
            "| Run the conformance cases (1 min 25 s) |")
        self.assertTrue(self.lines[header + 3].startswith("| fast suite | success | runner-a | 5 s | 10 s |"))
        self.assertNotIn("setup-python", self.body)
        self.assertIn("  - 1 min 25 s: Run the conformance cases, failure", self.body)

    def test_workflow_rows_report_each_run(self):
        """Input: CPU tests (2 jobs, longest queue 40 s, ends 2 min 10 s after creation) and PR
        rules (1 job, queue 1 s, ends 6 s after creation). Expect: one row each with those
        figures and the run link. Proves: a slow queue is visible apart from slow tests."""
        self.assertIn("| [CPU tests](https://example.test/runs/1) | failure | 2 min 10 s | 40 s | 2 |", self.lines)
        self.assertIn("| [PR rules](https://example.test/runs/2) | success | 6 s | 1 s | 1 |", self.lines)

    def test_a_job_still_running_is_shown_without_inventing_a_time(self):
        """Input: a job with no completed_at (still running when the comment is built).
        Expect: its run time reads n/a and its status is shown; no exception. Proves: missing
        timestamps are reported as missing, not as 0 s."""
        running = {"run": {"name": "CPU tests", "html_url": "u", "status": "in_progress", "created_at": "2026-01-01T00:00:00Z"},
                   "jobs": [{"name": "pending suite", "status": "in_progress", "created_at": "2026-01-01T00:00:00Z",
                             "started_at": "2026-01-01T00:00:04Z", "completed_at": None, "steps": []}]}
        body = timing.render("abcdef1", [running])
        self.assertIn("| pending suite | in_progress | n/a | 4 s | n/a | n/a |", body)
        self.assertIn("**n/a wall-clock**", body)


if __name__ == "__main__":
    unittest.main()
