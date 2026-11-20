import json
import random
import tempfile
import unittest
from pathlib import Path

from toolslip import toolchoice as tc
from toolslip.toolchoice_distractors import DISTRACTORS
from toolslip import toolchoice_report as report


class CatalogAndTasksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = tc.load_episodes()
        cls.catalog = tc.build_catalog(cls.episodes)
        cls.tasks = tc.build_tasks(cls.episodes, cls.catalog)
        cls.by_id = {t["task_id"]: t for t in cls.tasks}

    def test_catalog_contains_every_gold_tool(self):
        self.assertEqual(len(self.catalog), 21)
        self.assertTrue({t["gold"]["tool"] for t in self.tasks} <= set(self.catalog))

    def test_one_task_per_real_call(self):
        self.assertEqual(len(self.tasks), 131)
        for task in self.tasks:
            self.assertNotIn("ToolSearcher", [h["tool"] for h in task["history"]])

    def test_history_holds_only_earlier_calls(self):
        task = self.by_id["ep12-k3"]
        self.assertEqual([h["tool"] for h in task["history"]],
                         ["OrganizationMembers", "TravelStatus", "TravelStatus"])
        self.assertEqual(task["gold"]["tool"], "AddMeeting")

    def test_independent_later_call_counts_as_acceptable(self):
        task = self.by_id["ep20-k0"]
        self.assertEqual(task["gold"]["tool"], "UserMoviePreferences")
        self.assertEqual(task["acceptable_tools"], ["UserMoviePreferences", "UserWatchedMovies"])

    def test_later_call_with_unknown_arguments_is_not_acceptable(self):
        task = self.by_id["ep12-k1"]
        self.assertEqual(task["acceptable_tools"], ["TravelStatus"])

    def test_numeric_values_match_despite_formatting(self):
        self.assertTrue(tc.derivable("74.0060", "longitude=74.006"))
        self.assertTrue(tc.derivable(200000, '{"salary": 200000.0}'))
        self.assertFalse(tc.derivable("Conference Room A", "Book a meeting at 14:00"))
        self.assertTrue(tc.derivable(["Jack", "Rose"], '["Jack", "Rose", "Tom"]'))


class DistractorTests(unittest.TestCase):
    def test_five_distractors_per_real_tool(self):
        catalog = tc.build_catalog(tc.load_episodes())
        self.assertEqual(set(DISTRACTORS), set(catalog))
        for family in DISTRACTORS.values():
            self.assertEqual(len(family), 5)

    def test_distractor_names_are_unique_and_never_real(self):
        catalog = tc.build_catalog(tc.load_episodes())
        names = [d["name"] for family in DISTRACTORS.values() for d in family]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(set(names) & set(catalog))

    def test_every_distractor_explains_why_it_is_wrong(self):
        for family in DISTRACTORS.values():
            for d in family:
                self.assertTrue(d["why_wrong"].strip())
                self.assertTrue(d["description"].strip())
                self.assertIsInstance(d["input_parameters"], dict)


class ToolListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        episodes = tc.load_episodes()
        cls.catalog = tc.build_catalog(episodes)
        cls.tasks = tc.build_tasks(episodes, cls.catalog)

    def test_sizes_per_level(self):
        task = self.tasks[0]
        sizes = [len(tc.tool_list(task, level, self.catalog, seed=1)) for level in (0, 1, 2)]
        self.assertEqual(sizes, [23, 65, 128])  # 21 / 63 / 126 plus AskUser and NoSuitableTool

    def test_gold_offered_exactly_once(self):
        for task in self.tasks[:20]:
            for level in (0, 1, 2):
                names = [t["name"] for t in tc.tool_list(task, level, self.catalog, seed=1)]
                self.assertEqual(names.count(task["gold"]["tool"]), 1)

    def test_order_is_reproducible_but_differs_between_tasks(self):
        a = [t["name"] for t in tc.tool_list(self.tasks[0], 2, self.catalog, seed=7)]
        b = [t["name"] for t in tc.tool_list(self.tasks[0], 2, self.catalog, seed=7)]
        c = [t["name"] for t in tc.tool_list(self.tasks[1], 2, self.catalog, seed=7)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_prompt_lists_every_offered_tool(self):
        tools = tc.tool_list(self.tasks[0], 1, self.catalog, seed=1)
        messages = tc.prompt(self.tasks[0], tools)
        text = messages[-1]["content"]
        for tool in tools:
            self.assertIn(tool["name"] + "(", text)
        self.assertIn(self.tasks[0]["requirement"].strip(), text)


class ParseAndEvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        episodes = tc.load_episodes()
        cls.catalog = tc.build_catalog(episodes)
        cls.tasks = tc.build_tasks(episodes, cls.catalog)
        cls.task = next(t for t in cls.tasks if t["task_id"] == "ep33-k0")

    def test_parse_plain_fenced_and_thinking_output(self):
        plain = '{"tool": "Geocoding", "arguments": {"address": "Paris"}}'
        fenced = "```json\n" + plain + "\n```"
        thinking = "<think>Paris needs coordinates</think>\nAnswer: " + plain + " done"
        for text in (plain, fenced, thinking):
            self.assertEqual(tc.parse_choice(text), {"tool": "Geocoding", "arguments": {"address": "Paris"}})

    def test_parse_structured_api_variant(self):
        text = json.dumps({"tool": "Geocoding", "arguments_json": '{"address": "Paris"}'})
        self.assertEqual(tc.parse_choice(text), {"tool": "Geocoding", "arguments": {"address": "Paris"}})

    def test_unreadable_output(self):
        self.assertIsNone(tc.parse_choice("I would call the geocoding tool."))

    def test_broken_json_still_names_the_first_tool(self):
        text = '{"tool": "Geocoding(address: \'Paris\')}, {"tool": "Calculator(formula: \'x\')"}'
        self.assertEqual(tc.parse_choice(text), {"tool": "Geocoding", "arguments": {}, "lenient": True})

    def test_signature_inside_tool_field_is_reduced_to_the_name(self):
        text = '{"tool": "Geocoding(address)", "arguments": {"address": "Paris"}}'
        self.assertEqual(tc.parse_choice(text)["tool"], "Geocoding")

    def test_oracle_is_always_right(self):
        for task in self.tasks:
            for level in (0, 1, 2):
                offered = [t["name"] for t in tc.tool_list(task, level, self.catalog, seed=3)]
                verdict = tc.evaluate(task, dict(task["gold"]), offered)
                self.assertTrue(verdict["tool_correct"], task["task_id"])
                self.assertEqual(verdict["error_kind"], "richtig")

    def test_confusion_with_similar_tool_is_named(self):
        similar = DISTRACTORS["Geocoding"][0]["name"]
        offered = [t["name"] for t in tc.tool_list(self.task, 2, self.catalog, seed=3)]
        verdict = tc.evaluate(self.task, {"tool": similar, "arguments": {}}, offered)
        self.assertFalse(verdict["tool_correct"])
        self.assertEqual(verdict["error_kind"], "ähnliches Werkzeug verwechselt")
        self.assertEqual(verdict["chosen_family"], "Geocoding")

    def test_other_error_kinds(self):
        offered = [t["name"] for t in tc.tool_list(self.task, 0, self.catalog, seed=3)]
        cases = [({"tool": "Calculator", "arguments": {}}, "anderes echtes Werkzeug"),
                 ({"tool": "MadeUpTool", "arguments": {}}, "nicht angebotenes Werkzeug"),
                 (None, "Antwort nicht lesbar")]
        for choice, kind in cases:
            self.assertEqual(tc.evaluate(self.task, choice, offered)["error_kind"], kind)

    def test_arguments_checked_only_where_derivable(self):
        offered = [t["name"] for t in tc.tool_list(self.task, 0, self.catalog, seed=3)]
        right = tc.evaluate(self.task, {"tool": "geocoding", "arguments": {"address": " paris "}}, offered)
        wrong = tc.evaluate(self.task, {"tool": "Geocoding", "arguments": {"address": "Lyon"}}, offered)
        self.assertTrue(right["tool_correct"])
        self.assertTrue(right["args_correct"])
        self.assertFalse(wrong["args_correct"])
        meeting = next(t for t in self.tasks if t["task_id"] == "ep12-k3")
        offered = [t["name"] for t in tc.tool_list(meeting, 0, self.catalog, seed=3)]
        # topic and room are invented by the reference, so they are not checked; the named times are (since 19.11.)
        times = {"start_time": "2023-06-08 14:00:00", "end_time": "2023-06-08 15:00:00"}
        verdict = tc.evaluate(meeting, {"tool": "AddMeeting", "arguments": {"attendees": ["Rose", "Jack"], **times}},
                              offered)
        self.assertTrue(verdict["args_correct"])
        verdict = tc.evaluate(meeting, {"tool": "AddMeeting", "arguments": {"attendees": ["Jack", "Rose"]}}, offered)
        self.assertFalse(verdict["args_correct"])

    def test_random_baseline_is_near_chance(self):
        rng = random.Random(0)
        for level, expected in ((0, 1 / 21), (2, 1 / 126)):
            hits, n = 0, 0
            for task in self.tasks:
                offered = [t["name"] for t in tc.tool_list(task, level, self.catalog, seed=3)]
                choice = {"tool": rng.choice(offered), "arguments": {}}
                hits += tc.evaluate(task, choice, offered)["tool_correct"]
                n += 1
            self.assertLess(hits / n, expected * 3 + 0.03)


class StateTests(unittest.TestCase):
    """Experiment 2 (Stand 18.11. abends): complete, tool missing, info missing."""

    @classmethod
    def setUpClass(cls):
        episodes = tc.load_episodes()
        cls.catalog = tc.build_catalog(episodes)
        cls.tasks = tc.build_tasks(episodes, cls.catalog)
        cls.by_id = {t["task_id"]: t for t in cls.tasks}
        cls.paris = cls.by_id["ep33-k0"]
        cls.missing = tc.variant(cls.paris, "info_missing",
                                 rewrite={"key": "address", "value": "Paris",
                                          "requirement": "Check the current weather conditions in the city."})

    def offered(self, task, level=1):
        return [t["name"] for t in tc.tool_list(task, level, self.catalog, seed=5)]

    def test_extra_tools_always_offered(self):
        for state in ("complete", "tool_missing"):
            names = self.offered(tc.variant(self.paris, state), level=0)
            self.assertIn(tc.ASK, names)
            self.assertIn(tc.NONE, names)
        self.assertEqual(len(self.offered(tc.variant(self.paris, "complete"), level=0)), 23)

    def test_tool_missing_removes_every_acceptable_tool_but_keeps_lookalikes(self):
        task = tc.variant(self.by_id["ep20-k0"], "tool_missing")
        names = self.offered(task, level=2)
        self.assertNotIn("UserMoviePreferences", names)
        self.assertNotIn("UserWatchedMovies", names)
        self.assertIn(DISTRACTORS["UserMoviePreferences"][0]["name"], names)
        self.assertEqual(task["acceptable_tools"], [tc.NONE])

    def test_info_missing_expects_asking(self):
        self.assertEqual(self.missing["acceptable_tools"], [tc.ASK])
        self.assertNotIn("Paris", self.missing["context"])

    def test_missing_value_candidates_come_from_the_request_not_the_history(self):
        self.assertEqual(tc.missing_candidate(self.paris), ("address", "Paris"))
        self.assertIsNone(tc.missing_candidate(self.by_id["ep12-k1"]))

    def test_rewrite_must_really_remove_the_value(self):
        self.assertTrue(tc.rewrite_ok(self.paris, "address", "Paris", "Check the weather in the city."))
        self.assertFalse(tc.rewrite_ok(self.paris, "address", "Paris", "Check the weather in Paris."))

    def test_baselines(self):
        """Reference answers are always right, asking is right only where info is missing."""
        for task in self.tasks[:40]:
            for state in ("complete", "tool_missing"):
                v = tc.variant(task, state)
                ref = {"tool": v["acceptable"][0]["tool"], "arguments": dict(v["acceptable"][0].get("arguments", {}))}
                self.assertTrue(tc.evaluate(v, ref, self.offered(v))["call_correct"], (task["task_id"], state))
                self.assertFalse(tc.evaluate(v, {"tool": tc.ASK, "arguments": {}}, self.offered(v))["tool_correct"])
        ask = tc.evaluate(self.missing, {"tool": tc.ASK, "arguments": {"question": "Which city?"}},
                          self.offered(self.missing))
        self.assertTrue(ask["call_correct"])

    def test_error_kinds_per_state(self):
        acted = tc.evaluate(self.missing, {"tool": "Geocoding", "arguments": {"address": "Berlin"}},
                            self.offered(self.missing))
        self.assertEqual(acted["error_kind"], "vorschnell gehandelt statt gefragt")
        self.assertTrue(acted["silent_error"])
        self.assertEqual(acted["invented_args"], ["address"])
        gone = tc.variant(self.paris, "tool_missing")
        lookalike = DISTRACTORS["Geocoding"][1]["name"]
        verdict = tc.evaluate(gone, {"tool": lookalike, "arguments": {}}, self.offered(gone, level=2))
        self.assertEqual(verdict["error_kind"], "ähnliches Werkzeug statt ‚keines‘")
        complete = tc.variant(self.paris, "complete")
        asked = tc.evaluate(complete, {"tool": tc.ASK, "arguments": {}}, self.offered(complete))
        self.assertEqual(asked["error_kind"], "unnötig gefragt")
        self.assertFalse(asked["silent_error"])

    def test_two_levels_and_world_knowledge(self):
        weather = next(t for t in self.tasks if t["task_id"] == "ep5-k0")
        self.assertTrue(weather["artefact"])
        v = tc.variant(weather, "complete")
        flipped = tc.evaluate(v, {"tool": "GetWeatherForCoordinates",
                                  "arguments": {"latitude": 40.7128, "longitude": -74.006}}, self.offered(v))
        self.assertTrue(flipped["tool_correct"])
        self.assertFalse(flipped["call_correct"])
        self.assertEqual(flipped["error_kind"], "Weltwissen gegen Vorgabe")
        self.assertFalse(self.paris["artefact"])
        # API-Bank gives New York, San Francisco and London positive longitudes (all are west).
        flagged = {t["task_id"] for t in self.tasks if t["artefact"]}
        self.assertEqual(len(flagged), 12)  # 3 cities × (2 weather-by-coordinates, 1 restaurant, 1 weather-by-city)
        self.assertTrue({"ep6-k0", "ep16-k1", "ep32-k1"} <= flagged)
        self.assertFalse({"ep8-k0", "ep34-k1"} & flagged)  # Paris, Tokyo are east: correct

    def test_qualifier_after_the_reference_value_is_accepted(self):
        """Found in a run on 18.11.2026: 'Paris, France' for 'Paris' is a right argument."""
        v = tc.variant(self.paris, "complete")
        for value in ("Paris, France", "paris", " Paris "):
            verdict = tc.evaluate(v, {"tool": "Geocoding", "arguments": {"address": value}}, self.offered(v))
            self.assertTrue(verdict["call_correct"], value)
        for value in ("Lyon", "Parisienne Street"):
            verdict = tc.evaluate(v, {"tool": "Geocoding", "arguments": {"address": value}}, self.offered(v))
            self.assertFalse(verdict["call_correct"], value)

    def test_prompt_allows_asking_and_declining(self):
        text = tc.prompt(self.missing, tc.tool_list(self.missing, 0, self.catalog, seed=1))[0]["content"]
        self.assertIn(tc.ASK, text)
        self.assertIn(tc.NONE, text)


class ReportTests(unittest.TestCase):
    def test_report_summarises_levels_and_lists_cases(self):
        rows = [
            {"task_id": "ep1-k0", "level": 0, "tool_correct": True, "args_correct": True,
             "error_kind": "richtig", "gold_tool": "Geocoding", "chosen_tool": "Geocoding",
             "latency_s": 1.0, "gold_position": 3, "n_offered": 21, "requirement": "Weather in Paris",
             "history": [], "offered": ["Geocoding"], "raw_output": "{}", "chosen_family": "Geocoding",
             "acceptable_tools": ["Geocoding"]},
            {"task_id": "ep1-k0", "level": 2, "tool_correct": False, "args_correct": None,
             "error_kind": "ähnliches Werkzeug verwechselt", "gold_tool": "Geocoding",
             "chosen_tool": "ReverseGeocoding", "latency_s": 2.0, "gold_position": 50, "n_offered": 126,
             "requirement": "Weather in Paris", "history": [], "offered": ["Geocoding", "ReverseGeocoding"],
             "raw_output": "{}", "chosen_family": "Geocoding", "acceptable_tools": ["Geocoding"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "manifest.json").write_text(json.dumps({"model": "test-model", "levels": [0, 2]}))
            (run / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            html = report.build(run).read_text()
        self.assertIn("test-model", html)
        self.assertIn("1/1", html)
        self.assertIn("0/1", html)
        self.assertIn("ReverseGeocoding", html)
        self.assertIn("ähnliches Werkzeug verwechselt", html)


class Level3ScoringV5Tests(unittest.TestCase):
    """Level 3 gets the same v5 rules as level 1/2 (19.11.) through the shared loader."""

    @classmethod
    def setUpClass(cls):
        from toolslip.toolchoice_run import load_benchmark

        cls.catalog, tasks = load_benchmark("level3")
        cls.by_id = {t["task_id"]: t for t in tasks}

    def test_spec_date_format_is_visible(self):
        meeting = self.by_id["ep12-k3"]
        self.assertEqual(meeting["date_formats"]["AddMeeting"]["start_time"], r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        offered = [t["name"] for t in tc.tool_list(meeting, 0, self.catalog, seed=3)]
        args = dict(meeting["gold"]["arguments"], start_time="2023-06-08 2:00 PM")
        verdict = tc.evaluate(meeting, {"tool": "AddMeeting", "arguments": args}, offered)
        self.assertEqual(verdict["error_kind"], "Datumsformat verletzt")
        self.assertFalse(verdict["silent_error"])
        self.assertTrue(tc.evaluate(meeting, dict(meeting["gold"]), offered)["call_correct"])

    def test_every_task_has_question_topics(self):
        self.assertTrue(all(isinstance(t["ask_topics"], list) for t in self.by_id.values()))


if __name__ == "__main__":
    unittest.main()
