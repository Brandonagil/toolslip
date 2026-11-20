"""Run each silent tool-choice error once with the unchanged API-Bank tools.

"Silent" is then defined as: the call runs without an exception. Like API-Bank's own
evaluator (upstream/evaluator.py, Evaluator.evaluate), every call starts from fresh start
data; earlier calls of the dialogue are not replayed, and tokens are fixed in Account.json.
Parameter handling follows ToolManager.api_call (upstream/tool_manager.py), with one change:
a JSON boolean true counts as True (upstream compares with the string 'True' only).
Hand-written look-alikes have no code. They run as stand-ins that check parameter names,
types and completeness, and return no result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from . import toolchoice as tc
from .toolchoice_distractors import DISTRACTORS

TOOLS = tc.ROOT / "data/apibank-tools-20261119"
HANDOFF = tc.ROOT / "data/apibank-handoff-20261113"
NLTK_DATA = TOOLS / "nltk_data"  # punkt_tab for SearchEngine's word_tokenize
SEED = 20261119  # three upstream tools draw random ids or codes
RUNS, RAISES, STAND_IN, NOT_RUN, UNKNOWN = "läuft", "Ausnahme", "Attrappe", "nicht ausführbar", "unbekannt"
NETWORK = {"Dictionary": "ruft eine Web-API auf"}
POOL_RUNS = {
    "level12": ["tc-exp2-l12v2-luna-low", "tc-exp2-l12v2-luna-none"],
    "level3": ["tc-exp2-luna-low", "tc-exp2-luna-none"],
}
ROUNDS = ("", "-w2", "-w3")
_CLASS = re.compile(r"^class (\w+)\(API\)", re.M)


def _verify(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"API-Bank-Quelle verändert: {directory.name}/{relative}")


def _tool_files(benchmark: str) -> list[Path]:
    if benchmark == "level12":
        return sorted(p for p in (TOOLS / "upstream/apis").glob("*.py") if p.name not in ("__init__.py", "api.py"))
    seen, files = set(), []
    for directory in (HANDOFF / "upstream/lv3_apis", TOOLS / "upstream/lv3_apis"):
        for path in sorted(directory.glob("*.py")) if directory.exists() else []:
            if path.name not in seen and path.name != "tool_search.py":
                seen.add(path.name)
                files.append(path)
    return files


@lru_cache(maxsize=None)
def _load(benchmark: str) -> tuple[dict[str, type], dict[str, str]]:
    """Tool classes, and tools whose file cannot be imported here (missing package) with the reason."""
    _verify(TOOLS)
    _verify(HANDOFF)
    try:
        import nltk
        if str(NLTK_DATA) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_DATA))
    except ImportError:
        pass  # SearchEngine is then reported as not runnable
    missing = object()
    old = {name: sys.modules.get(name, missing) for name in ("apis", "apis.api")}
    package = ModuleType("apis")
    package.__path__ = [str(TOOLS / "upstream/apis")]
    base = ModuleType("apis.api")
    base_path = TOOLS / "upstream/apis/api.py"
    exec(compile(base_path.read_bytes(), str(base_path), "exec"), base.__dict__)
    package.API = base.API  # two upstream files write 'from apis import API'
    sys.modules.update({"apis": package, "apis.api": base})
    classes, unavailable = {}, {}
    try:
        for path in _tool_files(benchmark):
            name = f"apis.{path.stem}" if benchmark == "level12" else f"apis_lv3_{path.stem}"
            module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
            module.__package__ = "apis"
            try:
                exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            except ImportError as error:
                unavailable |= {tool: f"Paket fehlt: {error.name}" for tool in _CLASS.findall(path.read_text())}
                continue
            classes |= {k: v for k, v in vars(module).items()
                        if isinstance(v, type) and issubclass(v, base.API) and v is not base.API}
    finally:
        for name, value in old.items():
            sys.modules.pop(name, None) if value is missing else sys.modules.__setitem__(name, value)
    return classes, unavailable


def tools(benchmark: str) -> dict[str, type]:
    return _load(benchmark)[0]


@lru_cache(maxsize=1)
def _start_data() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (TOOLS / "upstream/init_database").glob("*.json")}


def _process(spec: dict, arguments: dict) -> dict:
    """ToolManager.api_call: unknown names and unreadable numbers raise before the tool runs."""
    out = {}
    for key, value in arguments.items():
        assert key in spec, f"invalid parameter name. parameter: {key}"
        kind = spec[key]["type"]
        if kind in ("int", "float") and isinstance(value, str):
            assert value.replace(".", "", 1 if kind == "float" else 0).isdigit(), f"invalid parameter type. parameter: {value}"
        if kind == "int":
            out[key] = int(value)
        elif kind == "float":
            out[key] = float(value)
        elif kind == "bool":
            out[key] = value is True or value == "True"
        elif kind in ("str", "list", "list(str)", "list(float)"):  # list(float) only in a hand-written look-alike
            out[key] = value
        else:
            raise Exception("invalid parameter type.")
    return out


def _run(cls: type, arguments: dict) -> dict:
    data = copy.deepcopy(_start_data())  # fresh per call, shared between tool and token check as upstream
    database = getattr(cls, "database_name", None)
    if database and database not in data:
        return {"status": NOT_RUN, "output": None, "exception": f"Startdaten {database} nicht geladen"}
    try:
        params = _process(cls.input_parameters, arguments)
        args = [data[database]] if database else []
        if cls.__name__ != "CheckToken" and "token" in cls.input_parameters:
            args.append(tools("level12")["CheckToken"](data["Account"]))
        random.seed(SEED)
        result = cls(*args).call(**params)
    except Exception as error:
        return {"status": RAISES, "output": None, "exception": f"{type(error).__name__}: {error}"}
    status = RUNS if result.get("exception") is None else RAISES
    return {"status": status, "output": result.get("output"), "exception": result.get("exception")}


@lru_cache(maxsize=1)
def _look_alikes() -> dict[str, dict]:
    return {d["name"]: d for family in DISTRACTORS.values() for d in family}


def _stand_in(spec: dict, arguments: dict) -> dict:
    try:
        _process(spec["input_parameters"], arguments)
        absent = sorted(set(spec["input_parameters"]) - set(arguments))
        assert not absent, f"missing parameters: {absent}"
    except Exception as error:
        return {"status": RAISES, "output": None, "exception": f"{type(error).__name__}: {error}"}
    return {"status": STAND_IN, "output": None, "exception": None}


def execute(benchmark: str, tool: str, arguments: dict) -> dict:
    classes, unavailable = _load(benchmark)
    if tool in NETWORK or tool in unavailable:
        return {"status": NOT_RUN, "output": None, "exception": NETWORK.get(tool) or unavailable[tool]}
    if tool in classes:
        return _run(classes[tool], arguments)
    if benchmark == "level3" and tool in _look_alikes():
        return _stand_in(_look_alikes()[tool], arguments)
    return {"status": UNKNOWN, "output": None, "exception": f"kein Werkzeug namens {tool}"}


def distinct_silent(rows: list[dict]) -> list[dict]:
    """Each silent call once per task, state, tool and arguments; reference defects and the artefact left out."""
    pool: dict[tuple, dict] = {}
    for row in rows:
        if not row["silent_error"] or row.get("reference_issue") or row.get("artefact"):
            continue
        key = (row["task_id"], row["state"], row["chosen_tool"], json.dumps(row["chosen_arguments"], sort_keys=True))
        pool[key] = {**row, "seen": pool[key]["seen"] + 1} if key in pool else {**row, "seen": 1}
    return list(pool.values())


def _rows(benchmark: str) -> list[dict]:
    return [json.loads(line) for run in POOL_RUNS[benchmark] for suffix in ROUNDS
            for line in (tc.ROOT / "runs" / f"{run}{suffix}" / "results.jsonl").read_text().splitlines() if line]


LABEL_HELP = "falsch = wirklich falscher Aufruf | umweg = führt auch zum Ziel | unklar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="exp3-pool-20261119")
    out = tc.ROOT / "runs" / parser.parse_args().out
    if (out / "labels.csv").exists():  # holds proposals and manual labels
        raise SystemExit(f"{out / 'labels.csv'} gibt es schon; bitte --out mit einem neuen Ordner wählen")
    out.mkdir(parents=True, exist_ok=True)
    pool = [{**row, "benchmark": b, "execution": execute(b, row["chosen_tool"], row["chosen_arguments"])}
            for b in POOL_RUNS for row in distinct_silent(_rows(b))]
    (out / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool))
    with (out / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nr", "level", "zustand", "folge", "entfernt", "richtig wäre", "Modell ruft auf",
                         "ausführung", "ergebnis oder ausnahme", "gespräch", f"label ({LABEL_HELP})", "notiz"])
        for nr, r in enumerate(pool, 1):
            e = r["execution"]
            writer.writerow([nr, "L1/2" if r["benchmark"] == "level12" else "L3", r["state"], r["sequence"],
                             ", ".join(r.get("removed_tools") or []), ", ".join(r["acceptable_tools"]),
                             f"{r['chosen_tool']}({json.dumps(r['chosen_arguments'], ensure_ascii=False)})",
                             e["status"], json.dumps(e["exception"] or e["output"], ensure_ascii=False)[:300],
                             r["requirement"], "", ""])
    counts = Counter((r["benchmark"], r["execution"]["status"]) for r in pool)
    for (benchmark, status), n in sorted(counts.items()):
        print(f"{benchmark:8s} {status:17s} {n:4d}")
    print(f"{len(pool)} Aufrufe → {out}")


if __name__ == "__main__":
    main()
