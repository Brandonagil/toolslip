import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toolslip import toolchoice_execute as ex

TOKEN = "a9s8d7f6g5h4j3k2l1"  # JohnDoe in init_database/Account.json
ALARM = "2023-03-16 07:00:00"  # JohnDoe's alarm in init_database/Alarm.json


try:
    import nltk, numpy, rank_bm25, requests  # noqa: F401  (extra "apibank-tools")
    HAS_APIBANK_TOOLS = True
except ImportError:
    HAS_APIBANK_TOOLS = False


@unittest.skipUnless(HAS_APIBANK_TOOLS, 'needs pip install -e ".[apibank-tools]"')
class RealToolTests(unittest.TestCase):
    def test_call_that_matches_the_start_data_runs(self):
        result = ex.execute("level12", "DeleteAlarm", {"token": TOKEN, "time": ALARM})
        self.assertEqual(result["status"], ex.RUNS)
        self.assertEqual(result["output"], "success")

    def test_call_without_matching_data_raises(self):
        result = ex.execute("level12", "DeleteAlarm", {"token": TOKEN, "time": "2023-03-16 08:00:00"})
        self.assertEqual(result["status"], ex.RAISES)
        self.assertIn("no alarm clock", result["exception"])

    def test_every_call_starts_from_fresh_start_data(self):
        first = ex.execute("level12", "DeleteAlarm", {"token": TOKEN, "time": ALARM})
        second = ex.execute("level12", "DeleteAlarm", {"token": TOKEN, "time": ALARM})
        self.assertEqual((first["status"], second["status"]), (ex.RUNS, ex.RUNS))

    def test_unknown_parameter_raises_like_the_tool_manager(self):
        result = ex.execute("level12", "DeleteAlarm", {"token": TOKEN, "time": ALARM, "label": "x"})
        self.assertEqual(result["status"], ex.RAISES)
        self.assertIn("invalid parameter name", result["exception"])

    def test_missing_parameter_raises(self):
        result = ex.execute("level12", "DeleteAlarm", {"token": TOKEN})
        self.assertEqual(result["status"], ex.RAISES)

    def test_level3_tool_runs(self):
        result = ex.execute("level3", "Geocoding", {"address": "Paris"})
        self.assertEqual(result["status"], ex.RUNS)

    def test_search_engine_runs_with_its_packages(self):
        result = ex.execute("level12", "SearchEngine", {"keyword": "weather"})
        self.assertEqual(result["status"], ex.RUNS)
        self.assertEqual(len(result["output"]), 2)

    def test_level3_tools_from_both_folders_load(self):
        self.assertEqual(ex.execute("level3", "Calculator", {"formula": "(1 + 2) * 3"})["status"], ex.RUNS)
        self.assertEqual(len(ex.tools("level3")), 21)

    def test_same_name_resolves_per_benchmark(self):
        self.assertIsNot(ex.tools("level12")["AddMeeting"], ex.tools("level3").get("AddMeeting"))


class StandInTests(unittest.TestCase):
    def test_look_alike_without_code_is_a_stand_in(self):
        result = ex.execute("level3", "AccountDelete", {"username": "JohnDoe", "password": "pass123"})
        self.assertEqual(result["status"], ex.STAND_IN)

    def test_stand_in_still_checks_parameter_names(self):
        result = ex.execute("level3", "AccountDelete", {"username": "JohnDoe", "pin": "1"})
        self.assertEqual(result["status"], ex.RAISES)

    def test_unknown_tool_is_reported(self):
        self.assertEqual(ex.execute("level12", "Teleport", {})["status"], ex.UNKNOWN)


class PoolTests(unittest.TestCase):
    def test_pool_counts_each_call_once(self):
        rows = [
            {"task_id": "t", "state": "complete", "chosen_tool": "A", "chosen_arguments": {"x": 1},
             "silent_error": True, "reference_issue": None, "artefact": False},
            {"task_id": "t", "state": "complete", "chosen_tool": "A", "chosen_arguments": {"x": 1},
             "silent_error": True, "reference_issue": None, "artefact": False},
            {"task_id": "t", "state": "complete", "chosen_tool": "A", "chosen_arguments": {"x": 2},
             "silent_error": True, "reference_issue": "Mangel", "artefact": False},
            {"task_id": "t", "state": "complete", "chosen_tool": "B", "chosen_arguments": {},
             "silent_error": False, "reference_issue": None, "artefact": False},
        ]
        pool = ex.distinct_silent(rows)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["seen"], 2)


class OverwriteTests(unittest.TestCase):
    def test_existing_label_list_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = Path(folder) / "labels.csv"
            labels.write_text("nr,label\n1,falsch\n")
            with mock.patch.object(sys, "argv", ["toolchoice_execute", "--out", folder]):
                with self.assertRaises(SystemExit):
                    ex.main()
            self.assertEqual(labels.read_text(), "nr,label\n1,falsch\n")
            self.assertFalse((Path(folder) / "pool.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
