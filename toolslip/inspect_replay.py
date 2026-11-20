"""Experiment 2 im Inspect-Log-Viewer: gespeicherte Antworten nachgespielt, ohne Modellaufrufe.

Jede gespeicherte Antwort wird ein Inspect-Sample: der genaue Prompt, den der Agent sah (System, Gespräch,
angebotene Werkzeuge), seine gespeicherte Wahl als Werkzeugaufruf und die gespeicherte Wertung als Score.
Ein Log je Lauf und Zustand. Läuft in der eigenen Umgebung (inspect-ai 0.3.289, siehe docs/inspect.md):

    .venv-inspect/bin/python -m toolslip.inspect_replay tc-exp2-l12v2-luna-low tc-exp2-l12v2-luna-none
    .venv-inspect/bin/inspect view --log-dir runs/inspect-logs --host 127.0.0.1 --port 7576
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.viewer import TaskSamplesColumn, TaskSamplesSort, TaskSamplesView, ViewerConfig

from . import toolchoice as tc
from .toolchoice_run import SEED, load_benchmark

LOG_DIR = tc.ROOT / "runs" / "inspect-logs"
STATES = {"complete": "vollstaendig", "tool_missing": "werkzeug-fehlt"}
EXCLUDED = "ausgeschlossen"
COLORS = {"richtig": "good", "sichtbar": "warn", "still": "bad", EXCLUDED: "muted"}


def outcome(row: dict) -> str:
    """richtig, sichtbar (falsch, aber man merkt es) oder still (falscher Aufruf läuft durch)."""
    if (row.get("usage") or {}).get("refused") or row.get("artefact") or row.get("reference_issue"):
        return EXCLUDED
    if row["call_correct"]:
        return "richtig"
    return "still" if row["silent_error"] else "sichtbar"


def _call(tool: str | None, arguments: dict | None) -> str:
    return f"{tool}({json.dumps(arguments or {}, ensure_ascii=False)})" if tool else "(keine lesbare Wahl)"


def _sample(row: dict, task: dict, tools: list[dict], model: str) -> Sample:
    system, user = tc.prompt(task, tools)
    gold = task["gold"]
    reasons = [r for r in (row.get("reference_issue"), (row.get("usage") or {}).get("refused") and "Anbieter lehnte ab")
               if r]
    return Sample(
        id=f"{row['task_id']}:{row['state']}",
        input=[ChatMessageSystem(content=system["content"]), ChatMessageUser(content=user["content"])],
        target=_call(task["acceptable"][0]["tool"], task["acceptable"][0].get("arguments")),
        metadata={"model": model, "ergebnis": outcome(row), "fehlerart": row["error_kind"],
                  "zustand": row["state"], "werkzeugfolge": row["sequence"],
                  "soll_original": _call(gold["tool"], gold["arguments"]),
                  "entfernt": ", ".join(row.get("removed_tools") or []) or "–",
                  "gewaehlt_tool": row.get("chosen_tool"), "gewaehlt_args": row.get("chosen_arguments"),
                  "rohantwort": row.get("raw_output") if isinstance(row.get("raw_output"), str) else json.dumps(
                      row.get("raw_output"), ensure_ascii=False),
                  "ausschlussgrund": "; ".join(reasons) or "–"},
    )


def samples(run_dir: Path) -> dict[str, list[Sample]]:
    """Samples je Zustand; die Werkzeugliste wird nachgebaut und muss der gespeicherten gleichen."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("benchmark") != "level12":
        raise ValueError("Nur Level-1/2-Läufe (Experiment 2) werden nachgespielt.")
    catalog, tasks = load_benchmark("level12")
    by_id = {t["task_id"]: t for t in tasks}
    out: dict[str, list[Sample]] = {state: [] for state in STATES}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = tc.variant(by_id[row["task_id"]], row["state"])
        tools = tc.tool_list(task, row["level"], catalog, SEED)
        if [t["name"] for t in tools] != row["offered"]:
            raise ValueError(f"{row['task_id']}: nachgebaute Werkzeugliste weicht von der gespeicherten ab")
        out[row["state"]].append(_sample(row, task, tools, manifest["model"]))
    return out


@solver
def replay():
    """Setzt die gespeicherte Wahl als Antwort des Agenten; ruft kein Modell auf."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        if meta["gewaehlt_tool"]:
            output = ModelOutput.for_tool_call(model=meta["model"], tool_name=meta["gewaehlt_tool"],
                                               tool_arguments=meta["gewaehlt_args"] or {})
        else:
            output = ModelOutput.from_content(model=meta["model"], content=meta["rohantwort"] or "")
        state.output = output
        state.messages.append(output.message)
        return state
    return solve


def _share(label: str) -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        counted = [s for s in scores if s.score.value != EXCLUDED]
        return sum(s.score.value == label for s in counted) / len(counted) if counted else 0.0
    return compute


@metric(scores="unreduced")
def ganz_richtig() -> Metric:
    return _share("richtig")


@metric(scores="unreduced")
def still_falsch() -> Metric:
    return _share("still")


@scorer(metrics=[ganz_richtig(), still_falsch()])
def ergebnis():
    """Übernimmt die gespeicherte Wertung unverändert (Wertung laut results.jsonl)."""
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        why = meta["fehlerart"] if meta["ergebnis"] != EXCLUDED else f"ausgeschlossen: {meta['ausschlussgrund']}"
        return Score(value=meta["ergebnis"], answer=_call(meta["gewaehlt_tool"], meta["gewaehlt_args"]),
                     explanation=f"{why}. Soll: {target.text}. Original-Soll: {meta['soll_original']}.")
    return score


VIEW = ViewerConfig(task_samples_view=TaskSamplesView(
    name="Werkzeugwahl",
    columns=[TaskSamplesColumn(id="sampleId"), TaskSamplesColumn(id="target"), TaskSamplesColumn(id="answer"),
             TaskSamplesColumn.score("ergebnis"), TaskSamplesColumn(id="input", visible=False)],
    sort=[TaskSamplesSort.score("ergebnis", dir="desc"), TaskSamplesSort(column="sampleId", dir="asc")],
    score_labels={"ergebnis": "Ergebnis"},
    score_color_scales={"ergebnis": COLORS},
    color_scales_enabled=True,
))


def build(run_name: str) -> list[Task]:
    run_dir = tc.ROOT / "runs" / run_name
    return [Task(name=f"{run_name}-{STATES[state]}", dataset=items, solver=replay(), scorer=ergebnis(),
                 viewer=VIEW, metadata={"lauf": run_name, "zustand": state})
            for state, items in samples(run_dir).items() if items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Ordnernamen unter runs/")
    args = parser.parse_args()
    tasks = [t for name in args.runs for t in build(name)]
    inspect_eval(tasks, model=None, log_dir=str(LOG_DIR), display="plain", max_tasks=1)


if __name__ == "__main__":
    main()
