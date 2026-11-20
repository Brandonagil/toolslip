import unittest

from toolslip import toolchoice as tc
from toolslip import toolchoice_levels12 as l12


class Levels12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = l12.load_samples()
        cls.catalog = l12.build_catalog(cls.samples)
        cls.tasks = l12.build_tasks(cls.samples)
        cls.by_id = {t["task_id"]: t for t in cls.tasks}

    def test_parse_call_reads_literal_arguments_only(self):
        name, args = l12.parse_call("API-Request: [ModifyRegistration(appointment_id='34567890', "
                                    "new_appointment_date='2023-03-26', amount=12.5, tags=['a', \"b\"])]")
        self.assertEqual(name, "ModifyRegistration")
        self.assertEqual(args, {"appointment_id": "34567890", "new_appointment_date": "2023-03-26",
                                "amount": 12.5, "tags": ["a", "b"]})
        self.assertIsNone(l12.parse_call("API-Request: [Evil(x=__import__('os'))]"))

    def test_parse_call_tolerates_api_bank_nested_quotes(self):
        """41 API-Bank references put quoted lists inside single quotes, which is not valid Python."""
        name, args = l12.parse_call(
            "API-Request: [RecordHealthData(user_id='12345', time='2023-10-13 09:30:00', "
            "health_data='[{'name': 'blood_pressure', 'value': '120/80'}]')]")
        self.assertEqual(name, "RecordHealthData")
        self.assertEqual(args["user_id"], "12345")
        self.assertEqual(args["health_data"], [{"name": "blood_pressure", "value": "120/80"}])
        _, args = l12.parse_call("API-Request: [AddMeeting(meeting_topic='Sync', attendees='['John', 'Mary']')]")
        self.assertEqual(args["attendees"], ["John", "Mary"])

    def test_counts(self):
        self.assertEqual(len(self.catalog), 49)  # ToolSearcher is not a target tool
        self.assertNotIn("ToolSearcher", self.catalog)
        self.assertEqual(len(self.tasks), 489)
        self.assertEqual(len({t["sequence"] for t in self.tasks}), 89)

    def test_task_ids_are_unique(self):
        """API-Bank ids restart in every file; resume, rescore and pairing all key on task_id."""
        self.assertEqual(len({t["task_id"] for t in self.tasks}), len(self.tasks))
        self.assertIn("l1-CancelTimedSwitch-level-1-2-0", self.by_id)

    def test_reference_issues_are_flagged(self):
        issues = [t["reference_issue"] for t in self.tasks if t["reference_issue"]]
        self.assertEqual(issues.count(l12.ISSUE_UNKNOWN_ARGS), 8)  # CancelTimedSwitch gold uses device_id, spec says name
        self.assertEqual(issues.count(l12.ISSUE_REPEAT), 6)  # gold repeats an identical earlier call
        self.assertEqual(self.by_id["l1-CancelTimedSwitch-level-1-2-0"]["reference_issue"], l12.ISSUE_UNKNOWN_ARGS)
        self.assertEqual(self.by_id["l1-SymptomSearch-AppointmentRegistration-level-2-1-2"]["reference_issue"],
                         l12.ISSUE_REPEAT)
        self.assertIsNone(next(t for t in self.tasks if t["gold"]["tool"] == "ModifyAgenda")["reference_issue"])

    def test_duplicate_tools_both_count_as_right(self):
        """EmergencyKnowledge and SymptomSearch have the same signature and nearly the same description."""
        task = next(t for t in self.tasks if t["gold"]["tool"] == "EmergencyKnowledge")
        offered = [t["name"] for t in tc.tool_list(task, 0, self.catalog, seed=3)]
        verdict = tc.evaluate(task, {"tool": "SymptomSearch", "arguments": dict(task["gold"]["arguments"])}, offered)
        self.assertTrue(verdict["call_correct"])

    def test_tool_missing_removes_the_whole_capability(self):
        """If a look-up tool is missing, SearchEngine and Wiki would still do the job, so they go too."""
        stock = tc.variant(next(t for t in self.tasks if t["gold"]["tool"] == "QueryStock"), "tool_missing")
        offered = {t["name"] for t in tc.tool_list(stock, 0, self.catalog, seed=3)}
        self.assertFalse({"QueryStock", "SearchEngine", "Wiki"} & offered)
        self.assertIn("Dictionary", offered)
        symptom = tc.variant(next(t for t in self.tasks if t["gold"]["tool"] == "SymptomSearch"), "tool_missing")
        offered = {t["name"] for t in tc.tool_list(symptom, 0, self.catalog, seed=3)}
        self.assertFalse({"SymptomSearch", "EmergencyKnowledge", "SearchEngine", "Wiki"} & offered)
        # action tools keep their siblings: confusing them is exactly the error we want to see
        agenda = tc.variant(next(t for t in self.tasks if t["gold"]["tool"] == "ModifyAgenda"), "tool_missing")
        offered = {t["name"] for t in tc.tool_list(agenda, 0, self.catalog, seed=3)}
        self.assertEqual({"ModifyAgenda"} & offered, set())
        self.assertTrue({"QueryAgenda", "SearchEngine", "Wiki"} <= offered)

    def test_get_today_first_is_not_a_silent_error(self):
        """Reading the date (e.g. to resolve 'March 26th' without a year) changes nothing."""
        task = next(t for t in self.tasks if t["gold"]["tool"] == "ModifyRegistration")
        offered = [t["name"] for t in tc.tool_list(task, 0, self.catalog, seed=3)]
        verdict = tc.evaluate(task, {"tool": "GetToday", "arguments": {}}, offered)
        self.assertEqual(verdict["error_kind"], "harmloser Vorabaufruf")
        self.assertFalse(verdict["silent_error"])
        self.assertFalse(verdict["call_correct"])

    def _verdict(self, task_id: str, tool: str, **arguments) -> dict:
        task = self.by_id[task_id]
        offered = [t["name"] for t in tc.tool_list(task, 0, self.catalog, seed=3)]
        return tc.evaluate(task, {"tool": tool, "arguments": {**task["gold"]["arguments"], **arguments}}, offered)

    def test_free_text_may_add_words_to_the_reference(self):
        """'rash on arm' still searches for the rash; adding words is fine, replacing them is not."""
        task_id = "l2-SymptomSearch-level-3-2-1"
        self.assertTrue(self._verdict(task_id, "SymptomSearch", symptom="red, bumpy rash on the arm")["call_correct"])
        self.assertFalse(self._verdict(task_id, "SymptomSearch", symptom="contact dermatitis and eczema")["call_correct"])
        agenda = next(t for t in self.tasks
                      if t["gold"]["tool"] == "DeleteAgenda" and t["gold"]["arguments"].get("location") == "Cleaners")
        self.assertTrue(self._verdict(agenda["task_id"], "DeleteAgenda", location="the Cleaners")["call_correct"])

    def test_names_and_secrets_stay_strict(self):
        task_id = "l1-QueryRegistration-EmergencyKnowledge-level-2-2-0"
        self.assertFalse(self._verdict(task_id, "QueryRegistration", patient_name="34567890")["call_correct"])
        self.assertFalse(self._verdict(task_id, "QueryRegistration", patient_name="Robert Lee Jr")["call_correct"])
        token = next(t for t in self.tasks if t["gold"]["arguments"].get("password") == "user1newpass")
        self.assertFalse(self._verdict(token["task_id"], "GetUserToken", password="user1pass")["call_correct"])

    def test_language_name_counts_as_its_code(self):
        """The Translate spec itself says 'english/en'."""
        self.assertTrue(self._verdict("l1-Translate-level-1-4-0", "Translate", tgt_lang="French")["call_correct"])
        self.assertFalse(self._verdict("l1-Translate-level-1-4-0", "Translate", tgt_lang="Spanish")["call_correct"])

    def test_asking_is_right_when_it_asks_for_a_fact_nobody_gave(self):
        """The reference invents credentials (JohnDoe/pass123); asking for them is right, a mere 'shall I?' is not."""
        task = self.by_id["l1-AddAgenda-AddMeeting-GetUserToken-level-2-4-0"]
        self.assertIn("password", task["ask_topics"])
        verdict = self._verdict(task["task_id"], tc.ASK, question="What are your username and password?")
        self.assertEqual(verdict["error_kind"], "berechtigt gefragt")
        self.assertTrue(verdict["call_correct"])
        self.assertFalse(verdict["silent_error"])
        confirm = self._verdict(task["task_id"], tc.ASK, question="Should I add 'Meeting with John' tomorrow at 9?")
        self.assertEqual(confirm["error_kind"], "unnötig gefragt")
        year = next(t for t in self.tasks if t["gold"]["arguments"].get("new_appointment_date") == "2023-10-15")
        self.assertIn("year", year["ask_topics"])  # 'October 15th' gives no year
        gone = tc.variant(task, "tool_missing")
        offered = [t["name"] for t in tc.tool_list(gone, 0, self.catalog, seed=3)]
        self.assertEqual(tc.evaluate(gone, {"tool": tc.ASK, "arguments": {"question": "Your password?"}},
                                     offered)["error_kind"], "gefragt statt ‚keines‘")

    def test_an_invented_time_is_a_fact_nobody_gave(self):
        """The reference sets 09:00–11:00, but the dialogue never names a time."""
        self.assertIn("time", self.by_id["l1-ModifyMeeting-level-1-3-2"]["ask_topics"])

    def test_asking_is_unnecessary_when_everything_is_given(self):
        # '2022-05-06 at 2 PM' gives the full date; 'Forgot Password' is a constant from the tool spec
        for task_id in ("l1-AddReminder-level-1-3-1", "l1-ForgotPassword-level-1-1-0"):
            self.assertEqual(self.by_id[task_id]["ask_topics"], [], task_id)
            self.assertEqual(self._verdict(task_id, tc.ASK, question="What year?")["error_kind"], "unnötig gefragt")

    def test_dates_are_checked_for_format_and_content(self):
        """Spec: '%Y-%m-%d'. Wrong format fails loudly (visible); a wrong day in the right format runs silently."""
        task_id = next(t["task_id"] for t in self.tasks
                       if t["gold"]["arguments"].get("new_appointment_date") == "2023-10-15")
        loud = self._verdict(task_id, "ModifyRegistration", new_appointment_date="October 15th")
        self.assertEqual(loud["error_kind"], "Datumsformat verletzt")
        self.assertFalse(loud["call_correct"])
        self.assertFalse(loud["silent_error"])
        silent = self._verdict(task_id, "ModifyRegistration", new_appointment_date="2023-10-16")
        self.assertEqual(silent["error_kind"], "Argument falsch übernommen")
        self.assertTrue(silent["silent_error"])
        # the year is not in the dialogue, so any year is fine
        self.assertTrue(self._verdict(task_id, "ModifyRegistration", new_appointment_date="2026-10-15")["call_correct"])
        # '2022-05-06 at 2 PM': the hour is given, so 15:00 is wrong
        reminder = "l1-AddReminder-level-1-3-1"
        self.assertTrue(self._verdict(reminder, "AddReminder", time="2022-05-06 14:00:00")["call_correct"])
        self.assertTrue(self._verdict(reminder, "AddReminder", time="2022-05-06 15:00:00")["silent_error"])
        self.assertTrue(self._verdict(reminder, "AddReminder", time="2021-05-06 14:00:00")["silent_error"])

    def test_date_format_is_visible_for_any_tool(self):
        """A wrong tool with a malformed date is refused by the API just like the right one (review 19.11.)."""
        task = next(t for t in self.tasks if t["gold"]["tool"] == "ModifyAlarm" and not t["reference_issue"])
        gone = tc.variant(task, "tool_missing")
        offered = [t["name"] for t in tc.tool_list(gone, 0, self.catalog, seed=3)]
        loud = tc.evaluate(gone, {"tool": "DeleteAlarm", "arguments": {"token": "x", "time": "2023-03-16 7:00 AM"}},
                           offered)
        self.assertEqual(loud["error_kind"], "Datumsformat verletzt")
        self.assertFalse(loud["silent_error"])
        silent = tc.evaluate(gone, {"tool": "DeleteAlarm", "arguments": {"token": "x", "time": "2023-03-16 07:00:00"}},
                             offered)
        self.assertEqual(silent["error_kind"], "ähnliches Werkzeug statt ‚keines‘")
        self.assertTrue(silent["silent_error"])
        self.assertEqual(task["date_formats"]["QueryHistoryToday"]["date"], r"\d{2}-\d{2}")  # spec says %m-%d

    def test_negation_is_not_an_added_word(self):
        self.assertFalse(self._verdict("l2-SymptomSearch-level-3-2-1", "SymptomSearch", symptom="no rash")["call_correct"])

    def test_truncated_and_contradicting_references_are_flagged(self):
        """API-Bank quotes values in single quotes, so "Don't forget …" became 'Don'."""
        self.assertEqual(self.by_id["l1-SendEmail-level-1-1-0"]["reference_issue"], l12.ISSUE_TRUNCATED)
        for task_id in l12.KNOWN_DEFECTS:  # checked by hand against the dialogue on 19.11.
            self.assertEqual(self.by_id[task_id]["reference_issue"], l12.ISSUE_CONTRADICTS)

    def test_conversation_and_history(self):
        task = next(t for t in self.tasks if t["gold"]["tool"] == "QueryHealthData" and t["history"])
        self.assertNotIn("Generate API Request", task["conversation"])
        self.assertIn("API-Request: [ModifyRegistration(", task["conversation"])
        self.assertEqual(task["history"][0]["tool"], "ModifyRegistration")
        self.assertEqual(task["history"][0]["result"], "success")

    def test_oracle_is_always_right(self):
        for task in self.tasks:
            offered = [t["name"] for t in tc.tool_list(task, 0, self.catalog, seed=3)]
            verdict = tc.evaluate(task, dict(task["gold"]), offered)
            self.assertTrue(verdict["call_correct"], task["task_id"])

    def test_natural_siblings_count_as_similar(self):
        task = next(t for t in self.tasks if t["gold"]["tool"] == "ModifyAgenda")
        offered = [t["name"] for t in tc.tool_list(task, 0, self.catalog, seed=3)]
        self.assertEqual(len(offered), 49 + 2)
        verdict = tc.evaluate(task, {"tool": "QueryAgenda", "arguments": {}}, offered)
        self.assertEqual(verdict["error_kind"], "ähnliches Werkzeug verwechselt")
        gone = tc.variant(task, "tool_missing")
        offered = [t["name"] for t in tc.tool_list(gone, 0, self.catalog, seed=3)]
        self.assertNotIn("ModifyAgenda", offered)
        self.assertIn("QueryAgenda", offered)
        verdict = tc.evaluate(gone, {"tool": "AddAgenda", "arguments": {}}, offered)
        self.assertEqual(verdict["error_kind"], "ähnliches Werkzeug statt ‚keines‘")
        self.assertTrue(verdict["silent_error"])
        verdict = tc.evaluate(gone, {"tool": "PlayMusic", "arguments": {}}, offered)
        self.assertEqual(verdict["error_kind"], "anderes Werkzeug statt ‚keines‘")

    def test_prompt_shows_the_conversation(self):
        task = self.tasks[0]
        text = tc.prompt(task, tc.tool_list(task, 0, self.catalog, seed=1))[1]["content"]
        self.assertIn("Conversation so far:", text)
        self.assertIn(task["conversation"].splitlines()[0], text)


if __name__ == "__main__":
    unittest.main()
