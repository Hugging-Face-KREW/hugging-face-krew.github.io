"""Offline regression of the actual event expression; no GitHub/model calls."""
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def routes_review(event_name, action="", labels=(), label=""):
    text = (ROOT / ".github/workflows/hf-agent-review.yml").read_text()
    match = re.search(r"    if: >-\n((?: {6}.*\n)+)", text)
    expression = match[1].strip().removeprefix("${{").removesuffix("}}").strip()
    for name, value in {
        "github.event.pull_request.labels.*.name": list(labels),
        "github.event.label.name": label,
        "github.event.action": action,
        "github.event_name": event_name,
    }.items():
        expression = expression.replace(name, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = expression.replace("!contains", "not contains")
    return eval(f"({expression})", {"__builtins__": {}, "contains": lambda values, item: item in values})


class ReviewEventTests(unittest.TestCase):
    def test_opened_before_managed_label_does_not_start_paid_review(self):
        self.assertFalse(routes_review("pull_request", "opened", ["hf-agent:experiment"]))

    def test_internal_labels_do_not_restart_review(self):
        for label in ("hf-agent:needs-human", "hf-agent:experiment", "documentation"):
            with self.subTest(label=label):
                self.assertFalse(routes_review("pull_request", "labeled", ["hf-agent:managed", label], label))

    def test_managed_label_and_real_content_events_start_review(self):
        self.assertTrue(routes_review("pull_request", "labeled", ["hf-agent:managed"], "hf-agent:managed"))
        for action in ("opened", "reopened", "ready_for_review", "synchronize"):
            with self.subTest(action=action):
                self.assertTrue(routes_review("pull_request", action, ["hf-agent:managed"]))

    def test_paused_event_does_not_enter_snapshot_or_review(self):
        self.assertFalse(routes_review("pull_request", "synchronize", ["hf-agent:managed", "hf-agent:paused"]))

    def test_dispatch_still_uses_live_snapshot_authorization(self):
        self.assertTrue(routes_review("workflow_dispatch"))

    def test_review_and_feedback_mutation_share_non_cancelling_pr_group(self):
        for name in ("hf-agent-review.yml", "hf-agent-feedback.yml"):
            text = (ROOT / ".github/workflows" / name).read_text()
            self.assertIn("group: hf-agent-pr-${{ github.repository }}-${{ needs.", text)
            self.assertIn("cancel-in-progress: false", text)


if __name__ == "__main__":
    unittest.main()
