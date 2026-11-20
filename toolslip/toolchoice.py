"""Tool-choice tasks on API-Bank level 3 (Experiment 2, Stand 18.11. abends).

Each real tool call of the 50 level-3 episodes becomes one task: the agent sees
the user request, the earlier calls with their results and a list of tools, and
must name the next call. Every list also offers AskUser and NoSuitableTool.

Factors:
- state: complete · tool_missing (every acceptable tool removed, look-alikes stay,
  right answer NoSuitableTool) · info_missing (one needed value removed from the
  request by a frozen, checked rewrite; right answer AskUser)
- level: look-alike tools per real tool (0, 2, 5) → 21 / 63 / 126 real tools

A call counts as right if it names the gold tool or another later call of the
same episode whose arguments are already known (independent calls may come in
any order). Level 1 = right tool; level 2 = right tool and every argument whose
reference value can be read from request or history. Several API-Bank reference
values are invented by the dataset; those are never checked, but values the agent
makes up are flagged as invented.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .toolchoice_distractors import DISTRACTORS

ROOT = Path(__file__).resolve().parent.parent
LEVEL3 = ROOT / "data/apibank-handoff-20261113/hf/level-3.json"
LOOKALIKES_PER_TOOL = {0: 0, 1: 2, 2: 5}
FAMILY_OF = {d["name"]: real for real, family in DISTRACTORS.items() for d in family}
WHY_WRONG = {d["name"]: d["why_wrong"] for family in DISTRACTORS.values() for d in family}
ASK, NONE = "AskUser", "NoSuitableTool"
# Reads the date and changes nothing, often a sensible first step: wrong, but not a silent error.
HARMLESS_LOOKUPS = ("GetToday",)
EXTRA_TOOLS = [
    {"name": ASK, "description": "Ask the user for information that the next tool call needs but that is "
                                "neither in the request nor in the results so far.",
     "input_parameters": {"question": {"type": "str", "description": "Question to the user."}}},
    {"name": NONE, "description": "Use when none of the available tools can perform the next required step.",
     "input_parameters": {"reason": {"type": "str", "description": "Why no tool fits."}}},
]
STATES = ("complete", "tool_missing", "info_missing")
# API-Bank stores New York, San Francisco and London with positive longitudes (all lie west of Greenwich).
ARTEFACT_LONGITUDES = (74.006, 122.4194, 0.1278)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FULL_NUMBER = re.compile(r"\s*-?\d+(?:\.\d+)?\s*")

SYSTEM = (
    "You are a tool-using assistant. You get a user request, the tool calls made so far with their "
    "results, and the list of available tools. Decide the single next tool call. Use exactly one tool "
    f"name from the list. If the next call needs a value that is neither in the request nor in the results, "
    f"use {ASK}. If no available tool can perform the next required step, use {NONE}. Reply with JSON only: "
    '{"tool": "<tool name>", "arguments": {"<argument>": <value>}}'
)


def load_episodes(path: Path = LEVEL3) -> list[dict]:
    return json.loads(path.read_text())


def build_catalog(episodes: list[dict]) -> dict[str, dict]:
    catalog = {}
    for episode in episodes:
        for call in episode["apis"]:
            spec = call["output"].get("output") if call["api_name"] == "ToolSearcher" else None
            if isinstance(spec, dict) and "name" in spec:
                catalog[spec["name"]] = {key: spec[key] for key in ("name", "description", "input_parameters")}
    return dict(sorted(catalog.items()))


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _FULL_NUMBER.fullmatch(value):
        return float(value)
    return None


def derivable(value: object, context: str) -> bool:
    """True if the value can be read from the context text."""
    if isinstance(value, dict):
        return all(derivable(v, context) for v in value.values())
    if isinstance(value, list):
        return all(derivable(v, context) for v in value)
    number = _as_number(value)
    if number is not None:
        return any(abs(float(m) - number) < 1e-6 for m in _NUMBER.findall(context))
    return str(value).strip().lower() in context.lower()


def _history_entry(call: dict) -> dict:
    output = call["output"]
    return {"tool": call["api_name"], "arguments": call["input"],
            "result": output.get("output"), "exception": output.get("exception")}


def _context(requirement: str, history: list[dict]) -> str:
    return requirement + "\n" + json.dumps(history, ensure_ascii=False)


def build_tasks(episodes: list[dict], catalog: dict[str, dict]) -> list[dict]:
    tasks = []
    for e, episode in enumerate(episodes):
        calls = [c for c in episode["apis"] if c["api_name"] != "ToolSearcher"]
        sequence = " → ".join(c["api_name"] for c in calls)
        for k, call in enumerate(calls):
            history = [_history_entry(c) for c in calls[:k]]
            context = _context(episode["requirement"].strip(), history)
            acceptable = [{"tool": call["api_name"], "arguments": call["input"]}]
            acceptable += [{"tool": c["api_name"], "arguments": c["input"]}
                           for c in calls[k + 1:] if derivable(c["input"], context)]
            gold = {"tool": call["api_name"], "arguments": call["input"]}
            tasks.append({
                "task_id": f"ep{e}-k{k}", "episode": e, "sequence": sequence, "state": "complete",
                "requirement": episode["requirement"].strip(), "history": history, "context": context,
                "gold": gold, "acceptable": acceptable, "references": acceptable,
                "acceptable_tools": list(dict.fromkeys(a["tool"] for a in acceptable)),
                "removed_tools": [],
                "artefact": any(derivable(lon, context) or derivable(lon, json.dumps(gold))
                                for lon in ARTEFACT_LONGITUDES),
            })
    return tasks


def missing_candidate(task: dict) -> tuple[str, object] | None:
    """First gold argument that only the request supplies (not the history)."""
    history = json.dumps(task["history"], ensure_ascii=False)
    for key, value in task["gold"]["arguments"].items():
        if derivable(value, task["requirement"]) and not derivable(value, history):
            return key, value
    return None


def rewrite_ok(task: dict, key: str, value: object, requirement: str) -> bool:
    """The removed value is gone, every other value the call needs from the request is still there."""
    if derivable(value, requirement):
        return False
    others = [v for k, v in task["gold"]["arguments"].items()
              if k != key and derivable(v, task["requirement"]) and not derivable(v, json.dumps(task["history"]))]
    return all(derivable(v, requirement) for v in others)


def variant(task: dict, state: str, rewrite: dict | None = None) -> dict:
    if state == "complete":
        return dict(task, state=state)
    if state == "tool_missing":
        # "substitutes": tools that could do the same job (only set for API-Bank level 1/2).
        removed = list(dict.fromkeys(task["acceptable_tools"] + task.get("substitutes", [])))
        return dict(task, state=state, acceptable=[{"tool": NONE}], acceptable_tools=[NONE], removed_tools=removed)
    if state != "info_missing" or rewrite is None:
        raise ValueError(f"unknown state or missing rewrite: {state}")
    context = _context(rewrite["requirement"], task["history"])
    still = [a for a in task["acceptable"] if derivable(a["arguments"], context)]
    acceptable = [{"tool": ASK}] + still
    return dict(task, state=state, requirement=rewrite["requirement"], context=context,
                acceptable=acceptable, acceptable_tools=list(dict.fromkeys(a["tool"] for a in acceptable)),
                removed_value={"key": rewrite["key"], "value": rewrite["value"]})


def tool_list(task: dict, level: int, catalog: dict[str, dict], seed: int) -> list[dict]:
    per_tool = LOOKALIKES_PER_TOOL[level]
    removed = set(task.get("removed_tools", []))
    tools = [dict(spec) for name, spec in catalog.items() if name not in removed]
    for real in catalog:
        tools += [{key: d[key] for key in ("name", "description", "input_parameters")}
                  for d in DISTRACTORS.get(real, [])[:per_tool]]
    tools += [dict(t) for t in EXTRA_TOOLS]
    random.Random(f"{seed}:{task['task_id']}:{level}").shuffle(tools)
    return tools


def _signature(tool: dict) -> str:
    params = ", ".join(f"{name}: {spec.get('type', '')}" for name, spec in tool["input_parameters"].items())
    return f"- {tool['name']}({params}): {tool['description']}"


def prompt(task: dict, tools: list[dict]) -> list[dict]:
    if "conversation" in task:  # API-Bank level 1/2: the dialogue already holds earlier calls and results
        user = (f"Conversation so far:\n{task['conversation']}\n\n"
                "Available tools:\n" + "\n".join(_signature(t) for t in tools))
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    history = "\n".join(
        f"{i}. {h['tool']}({json.dumps(h['arguments'], ensure_ascii=False)}) -> "
        f"{json.dumps(h['result'] if h['exception'] is None else {'exception': h['exception']}, ensure_ascii=False)}"
        for i, h in enumerate(task["history"], 1)) or "(none yet)"
    user = (f"User request: {task['requirement']}\n\nTool calls so far:\n{history}\n\n"
            "Available tools:\n" + "\n".join(_signature(t) for t in tools))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_TOOL_KEY = re.compile(r'"tool"\s*:\s*"\s*([A-Za-z_]\w*)')


def parse_choice(text: object) -> dict | None:
    """The first "tool" field of the answer; tolerates fences, thinking and trailing text.

    If the JSON around that field is broken, the tool name is still taken and the
    result is marked lenient (arguments then count as missing).
    """
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = _TOOL_KEY.search(text)
    if match is None:
        return None
    start = text.rfind("{", 0, match.start())
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else (None, 0)
    except json.JSONDecodeError:
        obj = None
    if not (isinstance(obj, dict) and isinstance(obj.get("tool"), str)):
        return {"tool": match.group(1), "arguments": {}, "lenient": True}
    args = obj.get("arguments")
    if args is None and isinstance(obj.get("arguments_json"), str):
        try:
            args = json.loads(obj["arguments_json"])
        except json.JSONDecodeError:
            args = None
    return {"tool": match.group(1), "arguments": args if isinstance(args, dict) else {}}


def _norm(value: object) -> object:
    number = _as_number(value)
    if number is not None:
        return round(number, 6)
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return sorted(json.dumps(_norm(v), sort_keys=True) for v in value)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


# Search terms, questions, titles, places: extra words keep the meaning ('rash' → 'rash on arm').
# Ids, names of patients, passwords and dates stay exact.
FREE_TEXT_KEYS = frozenset({"symptom", "keyword", "question", "music_name", "content", "location", "doctor_name"})
FILLER = frozenset({"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their"})
LANGUAGE_KEYS = frozenset({"src_lang", "tgt_lang"})
LANGUAGES = {"english": "en", "french": "fr", "spanish": "es", "german": "de", "italian": "it", "portuguese": "pt",
             "chinese": "zh", "japanese": "ja", "korean": "ko", "russian": "ru"}


def words(text: str) -> set[str]:
    """Lower-case words and digit runs ('21st' → 21, st)."""
    return set(re.findall(r"[a-z]+|\d+", text.lower()))


def _same(value: object, reference: object, key: str | None = None) -> bool:
    """Equal after normalisation; a string may add a qualifier after the reference ('Paris, France')."""
    a, r = _norm(value), _norm(reference)
    if key in LANGUAGE_KEYS and isinstance(a, str) and isinstance(r, str):
        a, r = LANGUAGES.get(a, a), LANGUAGES.get(r, r)
    if isinstance(a, str) and isinstance(r, str) and r:
        if a == r or re.match(rf"{re.escape(r)}(,| \()", a):
            return True
        needed, given = words(r) - FILLER, words(a)
        if key not in FREE_TEXT_KEYS or not needed or (given & NEGATIONS and not needed & NEGATIONS):
            return False
        return needed <= given
    return a == r


NEGATIONS = frozenset({"no", "not", "without", "never", "non"})
# The API-Bank specs ask for '%Y-%m-%d' or '%Y-%m-%d %H:%M:%S'.
ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?: (\d{2}):(\d{2}):(\d{2}))?")
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")


def date_given(month: int, day: int, context: str) -> bool:
    """The dialogue names this month and day ('March 26th', '26 Mar', '03-26')."""
    if f"{month:02d}-{day:02d}" in context:
        return True
    name = _MONTHS[month - 1]
    return (re.search(rf"\b(?:{name}|{name[:3]})\b", context, re.I) is not None
            and re.search(rf"(?<!\d)0?{day}(?:st|nd|rd|th)?(?!\d)", context) is not None)


def time_given(hour: int, minute: int, context: str) -> bool:
    """The dialogue names this clock time ('2 PM', '2:30pm', '14:30', 'noon')."""
    minutes = rf":{minute:02d}" if minute else r"(?::00)?"
    twelve = rf"(?<!\d){hour % 12 or 12}{minutes}\s*{'p' if hour >= 12 else 'a'}\.?m\b"
    clock = rf"(?<!\d)0?{hour}:{minute:02d}(?!\d)"
    named = {12: "noon", 0: "midnight"}.get(hour) if minute == 0 else None
    return bool(re.search(twelve, context, re.I) or re.search(clock, context)
                or (named and named in context.lower()))


def _date_verdict(value: object, reference: str, context: str) -> str:
    """'format' if the spec's format is broken (the API would refuse), else check what the dialogue names."""
    ref = ISO_DATE.fullmatch(reference)
    named = (ref.group(1) in context,  # a year nobody named may be anything
             date_given(int(ref.group(2)), int(ref.group(3)), context),
             bool(ref.group(4)) and time_given(int(ref.group(4)), int(ref.group(5)), context))
    if value is None:  # left out: wrong only if the dialogue names the date
        return "wrong" if any(named) else "ok"
    got = ISO_DATE.fullmatch(value) if isinstance(value, str) else None
    if got is None or bool(got.group(4)) != bool(ref.group(4)):
        return "format"
    parts = (got.group(1) == ref.group(1), got.group(2, 3) == ref.group(2, 3), got.group(4, 5) == ref.group(4, 5))
    return "ok" if all(same for same, check in zip(parts, named) if check) else "wrong"


def _format_broken(args: dict, formats: dict[str, str]) -> bool:
    """A date or time argument that does not match the format the tool's spec prescribes."""
    return any(k in args and not (isinstance(args[k], str) and re.fullmatch(pattern, args[k]))
               for k, pattern in formats.items())


def _candidate_verdict(reference: dict, args: dict, context: str) -> str | None:
    verdicts = []
    for k, v in reference.items():
        if isinstance(v, str) and ISO_DATE.fullmatch(v):
            verdicts.append(_date_verdict(args.get(k), v, context))
        elif derivable(v, context):
            verdicts.append("ok" if _same(args.get(k), v, k) else "wrong")
    if not verdicts:
        return None
    return "format" if "format" in verdicts else ("wrong" if "wrong" in verdicts else "ok")


def _args_verdict(candidates: list[dict], args: dict, context: str) -> str | None:
    """Best verdict over all acceptable calls: ok, then format (fails loudly), then wrong (runs silently)."""
    verdicts = {_candidate_verdict(c["arguments"], args, context) for c in candidates}
    return next((v for v in ("ok", "format", "wrong") if v in verdicts), None)


def _invented(args: dict, references: list[dict], context: str) -> list[str]:
    """Argument values that are neither readable from the context nor equal to a reference value."""
    known = [r["arguments"] for r in references]
    return [k for k, v in args.items()
            if not derivable(v, context) and all(_norm(v) != _norm(r.get(k)) for r in known)]


def _sign_flipped(args: dict, candidates: list[dict]) -> bool:
    for candidate in candidates:
        for k, ref in candidate["arguments"].items():
            a, b = _as_number(args.get(k)), _as_number(ref)
            if a is not None and b and abs(a + b) < 1e-6:
                return True
    return False


_VERBS = ("Add", "Modify", "Delete", "Query", "Cancel", "Get", "Update", "Record", "Search", "Set", "Open",
          "Book", "Check", "Send", "Play", "Create")


def _stem(name: str) -> str:
    """Object part of a tool name ('ModifyAgenda' → 'Agenda'); names without a leading verb stay whole."""
    for verb in _VERBS:
        if name.startswith(verb) and name[len(verb):len(verb) + 1].isupper():
            return name[len(verb):]
    return name


def _sibling(name: str, targets: list[str]) -> bool:
    """Same object, other verb: API-Bank's own look-alike families (AddAgenda, QueryAgenda, …)."""
    return _stem(name) != name and any(t != name and _stem(t) == _stem(name) for t in targets)


def _wrong_tool_kind(task: dict, name: str) -> str:
    state = task.get("state", "complete")
    if state == "info_missing":
        return "fälschlich ‚keines‘" if name == NONE else (
            "gefragt, aber anders" if name == ASK else "vorschnell gehandelt statt gefragt")
    if state == "tool_missing":
        if name == ASK:
            return "gefragt statt ‚keines‘"
        similar = name in FAMILY_OF or _sibling(name, task.get("removed_tools", []))
        return "ähnliches Werkzeug statt ‚keines‘" if similar else "anderes Werkzeug statt ‚keines‘"
    if name == ASK:
        return "unnötig gefragt"
    if name == NONE:
        return "fälschlich ‚keines‘"
    if name in FAMILY_OF:
        if FAMILY_OF[name] in task["acceptable_tools"]:
            return "ähnliches Werkzeug verwechselt"
        return "ähnliches Werkzeug eines anderen"
    if _sibling(name, task["acceptable_tools"]):
        return "ähnliches Werkzeug verwechselt"
    return "anderes echtes Werkzeug"


def evaluate(task: dict, choice: dict | None, offered: list[str]) -> dict:
    result = {"tool_correct": False, "call_correct": False, "args_correct": None, "chosen_tool": None,
              "chosen_family": None, "invented_args": [], "silent_error": False}
    if choice is None:
        return {**result, "error_kind": "Antwort nicht lesbar"}
    name = {n.lower(): n for n in offered}.get(choice["tool"].strip().lower())
    if name is None:
        return {**result, "chosen_tool": choice["tool"], "error_kind": "nicht angebotenes Werkzeug"}
    acts = name not in (ASK, NONE)
    references = task.get("references", task["acceptable"])
    result.update(chosen_tool=name, chosen_family=FAMILY_OF.get(name, name),
                  invented_args=_invented(choice["arguments"], [r for r in references if r["tool"] == name],
                                          task["context"]) if acts else [])
    if acts and _format_broken(choice["arguments"], task.get("date_formats", {}).get(name, {})):
        # the API refuses a date in the wrong format, whichever tool was called, so the error is visible
        return {**result, "tool_correct": name in task["acceptable_tools"], "args_correct": False,
                "error_kind": "Datumsformat verletzt"}
    asked = words(" ".join(str(v) for v in choice["arguments"].values()))
    if name == ASK and task.get("state", "complete") == "complete" and asked & set(task.get("ask_topics", [])):
        # the reference needs a fact nobody gave the agent (a password, a year), and the question asks for it
        return {**result, "tool_correct": True, "call_correct": True, "error_kind": "berechtigt gefragt"}
    if name not in task["acceptable_tools"]:
        if name in HARMLESS_LOOKUPS:
            return {**result, "error_kind": "harmloser Vorabaufruf"}
        return {**result, "silent_error": acts, "error_kind": _wrong_tool_kind(task, name)}
    if not acts:
        return {**result, "tool_correct": True, "call_correct": True, "error_kind": "richtig"}
    candidates = [a for a in task["acceptable"] if a["tool"] == name]
    verdict = _args_verdict(candidates, choice["arguments"], task["context"])
    if verdict == "format":  # the API refuses the call, so the error is visible, not silent
        return {**result, "tool_correct": True, "args_correct": False, "error_kind": "Datumsformat verletzt"}
    args_ok = None if verdict is None else verdict == "ok"
    call_ok = args_ok is not False
    kind = "richtig" if call_ok else (
        "Weltwissen gegen Vorgabe" if _sign_flipped(choice["arguments"], candidates) else "Argument falsch übernommen")
    return {**result, "tool_correct": True, "call_correct": call_ok, "args_correct": args_ok,
            "silent_error": not call_ok, "error_kind": kind}
