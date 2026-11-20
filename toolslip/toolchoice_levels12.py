"""API-Bank level 1 and 2 as tool-choice tasks (loaded 18.11.2026, pinned commit 12e8158).

Each sample is a dialogue whose expected output is the next API request. Earlier calls
appear inline as "API-Request: [Tool(k='v')]->result" and are parsed into the history.
Samples whose target is ToolSearcher are skipped (that is tool search, not tool choice).
The 49 real tools form one catalog; API-Bank's own families (Add/Modify/Delete/Query…)
act as natural look-alikes, so no hand-written distractors are added.

Source: huggingface.co/datasets/liminghao1630/API-Bank, test-data/level-1-api.json
(sha256 0d3155c5…), test-data/level-2-api.json (sha256 c6196644…).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from . import toolchoice as tc

DIR = tc.ROOT / "data/apibank-levels-12-20261118"
FILES = {"l1": "level-1-api.json", "l2": "level-2-api.json"}
_CALL = re.compile(r"\[(\w+\(.*\))\]", re.S)
# Defects in API-Bank's own references; such tasks are reported separately, like the longitude artefact.
ISSUE_UNKNOWN_ARGS = "Soll-Argumente fehlen in der Werkzeugbeschreibung"
ISSUE_REPEAT = "Sollaufruf wiederholt einen identischen früheren Aufruf"
ISSUE_TRUNCATED = "Sollwert am Apostroph abgeschnitten"
ISSUE_CONTRADICTS = "Sollwert widerspricht dem Gespräch"
# Found in a review on 19.11.2026 and checked by hand against the dialogue.
KNOWN_DEFECTS = {
    "l1-ModifyAlarm-level-1-2-2": ISSUE_CONTRADICTS,  # dialogue 8→9 and 9→10 AM, reference 7→8
    "l2-AddAgenda-level-3-1-4": ISSUE_CONTRADICTS,  # dialogue 'at their office', reference location 'Client'
    # the user says 'my password is user1pass' after resetting it to user1newpass; the reference takes the new one
    "l1-ModifyPassword-ForgotPassword-DeleteAccount-GetUserToken-level-2-3-4": ISSUE_CONTRADICTS,
}
# Same signature, nearly the same description: either one is the right tool.
DUPLICATES = {"SymptomSearch": "EmergencyKnowledge", "EmergencyKnowledge": "SymptomSearch"}
# Pure look-ups about the world. If one is missing, a web or Wikipedia search still does the job,
# so "tool missing" removes those too; otherwise the capability is not really missing.
LOOKUPS = {"SymptomSearch", "EmergencyKnowledge", "Dictionary", "QueryStock", "QueryHistoryToday", "GetToday",
           "SearchEngine", "Wiki"}
FALLBACKS = ("SearchEngine", "Wiki")


def load_samples(directory: Path = DIR) -> list[dict]:
    return [{**x, "prefix": prefix} for prefix, name in FILES.items()
            for x in json.loads((directory / name).read_text())]


def _descriptions(sample: dict) -> list[dict]:
    part = sample["instruction"].split("API descriptions:")[-1]
    return [json.loads(line) for line in part.strip().splitlines() if line.strip().startswith("{")]


def build_catalog(samples: list[dict]) -> dict[str, dict]:
    catalog = {}
    for sample in samples:
        for d in _descriptions(sample):
            if d["name"] != "ToolSearcher":
                catalog[d["name"]] = {"name": d["name"], "description": d["description"],
                                      "input_parameters": d.get("input_parameters", {})}
    return dict(sorted(catalog.items()))


_ARG_START = re.compile(r"(?:^|,\s*)(\w+)=")


def _loose_args(inner: str) -> dict | None:
    """Fallback for API-Bank's nested quotes (key='[{'a': 'b'}]'): split at ', key=' and read literals only."""
    starts = list(_ARG_START.finditer(inner))
    if not starts or starts[0].start() != 0:
        return None
    args = {}
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(inner)
        raw = inner[match.end():end].strip()
        for candidate in (raw, raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] == "'" else None):
            if candidate is None:
                continue
            try:
                args[match.group(1)] = ast.literal_eval(candidate)
                break
            except (SyntaxError, ValueError):
                continue
        else:
            if not (len(raw) >= 2 and raw[0] == raw[-1] == "'"):
                return None
            args[match.group(1)] = raw[1:-1]
    return args


def parse_call(text: str) -> tuple[str, dict] | None:
    """Tool name and literal keyword arguments; anything that is not a plain literal is rejected."""
    match = _CALL.search(text)
    if match is None:
        return None
    try:
        node = ast.parse(match.group(1), mode="eval").body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.args:
            return None
        return node.func.id, {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except ValueError:
        return None
    except SyntaxError:
        head = re.match(r"(\w+)\((.*)\)$", match.group(1), re.S)
        args = _loose_args(head.group(2)) if head else None
        return (head.group(1), args) if args is not None else None


def _result(text: str) -> object:
    try:
        return ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        return text.strip()


_SPEC_FORMATS = {"%Y-%m-%d %H:%M:%S": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d": r"\d{4}-\d{2}-\d{2}",
                 "%m-%d": r"\d{2}-\d{2}"}


def date_formats(catalog: dict[str, dict]) -> dict[str, dict[str, str]]:
    """Per tool and parameter, the date pattern its description prescribes ('Format: %Y-%m-%d …')."""
    formats: dict[str, dict[str, str]] = {}
    for name, tool in catalog.items():
        for param, spec in tool["input_parameters"].items():
            match = re.search("|".join(_SPEC_FORMATS), spec.get("description", ""))
            if match:
                formats.setdefault(name, {})[param] = _SPEC_FORMATS[match.group(0)]
    return formats


def _truncated(value: object, conversation: str) -> bool:
    """API-Bank quotes values in single quotes, so "Don't forget …" became 'Don'."""
    if not isinstance(value, str) or not value.strip():
        return False
    ends = [m.end() for m in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", conversation)]
    return bool(ends) and all(re.match(r"'[A-Za-z]", conversation[i:]) for i in ends)


def _reference_issue(catalog: dict, call: dict, history: list[dict], conversation: str) -> str | None:
    # Level-2 samples list only some tools in their own prompt, so the spec comes from the full catalog.
    if set(call["arguments"]) - set(catalog[call["tool"]]["input_parameters"]):
        return ISSUE_UNKNOWN_ARGS
    if any(h["tool"] == call["tool"] and h["arguments"] == call["arguments"] for h in history):
        return ISSUE_REPEAT
    if any(_truncated(v, conversation) for v in call["arguments"].values()):
        return ISSUE_TRUNCATED
    return None


_MONTH_DAY = re.compile(r"\d{2}-\d{2}")
_MIN_SPEC_CONSTANT = 4  # shorter strings would match the spec text by accident
_STEM = 5  # letters compared, enough for 'nauseous'/'nausea'
_LANGUAGE_NAMES = {code: name for name, code in tc.LANGUAGES.items()}
_TIME_TOPICS = {"time", "hour", "when", "start", "end", "until", "duration", "long"}
_SYNONYMS = {"stock_code": {"ticker", "symbol"}, "username": {"user", "login"}, "password": {"login", "credentials"}}


def _missing(key: str, value: object, conversation: str, spec: str) -> set[str]:
    """Words a question must contain to ask for this value; empty if the dialogue or the spec gives it."""
    topics = tc.words(key) | _SYNONYMS.get(key, set())
    if isinstance(value, (dict, list)):
        parts = value.values() if isinstance(value, dict) else value
        if any(_missing(key, v, conversation, spec) for v in parts):
            return topics | {w for w in tc.words(json.dumps(value)) if not w.isdigit()}
        return set()
    if not isinstance(value, str) or tc._as_number(value) is not None or value.lower() in ("true", "false"):
        return set()  # numbers and switches are conversions: 'two rooms' → 2, 'turn it off' → False
    if tc.derivable(value, conversation) or (len(value) >= _MIN_SPEC_CONSTANT and value.lower() in spec.lower()):
        return set()
    date = tc.ISO_DATE.fullmatch(value)
    if date:  # month and day are a conversion; a year or a clock time nobody named is a fact
        year = {"year"} if date.group(1) not in conversation else set()
        clock = date.group(4) and not tc.time_given(int(date.group(4)), int(date.group(5)), conversation)
        return year | (_TIME_TOPICS if clock else set())
    if _MONTH_DAY.fullmatch(value) or _LANGUAGE_NAMES.get(value.lower(), "\0") in conversation.lower():
        return set()
    # Rephrasing shares word stems with the dialogue ('nauseous' → 'nausea'); new facts share none.
    stems = {w[:_STEM] for w in tc.words(value) - tc.FILLER}
    return set() if stems & {w[:_STEM] for w in tc.words(conversation)} else topics


def ask_topics(arguments: dict, conversation: str, spec: str) -> list[str]:
    return sorted(set().union(*(_missing(k, v, conversation, spec) for k, v in arguments.items())))


def build_tasks(samples: list[dict]) -> list[dict]:
    catalog = build_catalog(samples)
    formats = date_formats(catalog)  # one shared dict, read-only
    tasks = []
    for sample in samples:
        gold = parse_call(sample["expected_output"])
        if gold is None or gold[0] == "ToolSearcher":
            continue
        lines = [line for line in sample["input"].splitlines() if line.strip() != "Generate API Request:"]
        history = []
        for line in lines:
            if line.startswith("API-Request:"):
                call, _, result = line.partition("]->")
                parsed = parse_call(call + "]")
                if parsed:
                    history.append({"tool": parsed[0], "arguments": parsed[1], "result": _result(result),
                                    "exception": None})
        conversation = "\n".join(lines).strip()
        turns = "\n".join(line for line in lines if not line.startswith("API-Request:")).strip()
        call = {"tool": gold[0], "arguments": gold[1]}
        acceptable = [call] + ([{"tool": DUPLICATES[gold[0]], "arguments": gold[1]}] if gold[0] in DUPLICATES else [])
        tools = [a["tool"] for a in acceptable]
        substitutes = [f for f in FALLBACKS if f not in tools] if gold[0] in LOOKUPS else []
        # API-Bank ids restart in every file, so the file name is part of the id.
        task_id = f"{sample['prefix']}-{sample['file'].removesuffix('.jsonl')}-{sample['id']}"
        tasks.append({
            "task_id": task_id, "episode": sample["file"],
            "reference_issue": _reference_issue(catalog, call, history, conversation) or KNOWN_DEFECTS.get(task_id),
            # Facts the reference needs but nobody gave (a password, a year): a question about them is right too.
            "ask_topics": ask_topics(call["arguments"], conversation, json.dumps(catalog[call["tool"]])),
            "date_formats": formats,
            "sequence": " → ".join(re.sub(r"-level-\d+-\d+\.jsonl$", "", sample["file"]).split("-")),
            "state": "complete", "requirement": turns, "conversation": conversation, "history": history,
            "context": conversation, "gold": call, "acceptable": acceptable, "references": acceptable,
            "acceptable_tools": tools, "substitutes": substitutes, "removed_tools": [],
            "artefact": any(tc.derivable(lon, conversation) for lon in tc.ARTEFACT_LONGITUDES),
        })
    return tasks
