"""Runs only in the Inspect environment: .venv-inspect/bin/python -m unittest tests.test_inspect_replay"""
import importlib.util
import unittest

HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None


@unittest.skipUnless(HAS_INSPECT, "inspect-ai lives in .venv-inspect")
class InspectReplayTests(unittest.TestCase):
    def test_outcome_separates_visible_from_silent(self):
        from toolslip.inspect_replay import EXCLUDED, outcome

        self.assertEqual(outcome({"call_correct": True, "silent_error": False}), "richtig")
        self.assertEqual(outcome({"call_correct": False, "silent_error": False}), "sichtbar")
        self.assertEqual(outcome({"call_correct": False, "silent_error": True}), "still")
        self.assertEqual(outcome({"call_correct": False, "silent_error": True, "reference_issue": "x"}), EXCLUDED)
        self.assertEqual(outcome({"call_correct": True, "silent_error": False, "usage": {"refused": True}}), EXCLUDED)

    def test_replay_keeps_every_stored_answer_and_its_verdict(self):
        """The rebuilt prompt must use the stored tool list; the sample carries the stored verdict unchanged."""
        from toolslip import toolchoice as tc
        from toolslip.inspect_replay import samples

        run = tc.ROOT / "runs" / "tc-exp2-l12v2-luna-low"
        if not run.exists():
            self.skipTest("run not present")
        by_state = samples(run)
        self.assertEqual({k: len(v) for k, v in by_state.items()}, {"complete": 489, "tool_missing": 489})
        missing = by_state["tool_missing"]
        self.assertEqual(sum(s.metadata["ergebnis"] == "still" for s in missing), 16)
        example = next(s for s in missing if s.id == "l1-ModifyAlarm-AddAgenda-GetUserToken-level-2-2-2:tool_missing")
        self.assertEqual(example.metadata["gewaehlt_tool"], "DeleteAlarm")
        self.assertEqual(example.target, "NoSuitableTool({})")


if __name__ == "__main__":
    unittest.main()
