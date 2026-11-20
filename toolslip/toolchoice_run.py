"""Run Experiment 2 (tool choice without a checker): one selection agent over states × tool levels.

Local model (mlx_lm.server, OpenAI-compatible chat API):
    .venv/bin/python -m toolslip.toolchoice_run --name tc-x --backend chat --endpoint http://127.0.0.1:18790 \
        --model openbmb/MiniCPM5-1B-MLX
OpenAI Responses API (key only from the environment, never written anywhere):
    OPENAI_API_KEY=... .venv/bin/python -m toolslip.toolchoice_run --name tc-luna --backend openai --model gpt-6-luna

Every answer is appended to runs/<name>/results.jsonl right away; a rerun skips finished cases.
runs/<name>/report.html shows the summary, all confusions and every single case.
The info_missing state needs the frozen rewrites (python -m toolslip.toolchoice_missing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import toolchoice as tc
from . import toolchoice_report

log = logging.getLogger("toolchoice")
SEED = 20261118
CALL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["tool", "arguments_json"],
               "properties": {"tool": {"type": "string"},
                              "arguments_json": {"type": "string", "description": "JSON object with the arguments"}}}
MISSING_FILE = tc.ROOT / "data/toolchoice/missing_v1.json"


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")[:300]
            if error.code not in (429, 500, 502, 503) or attempt == 3:
                raise RuntimeError(f"HTTP {error.code}: {body}") from error
            log.warning("HTTP %s, neuer Versuch in %d s", error.code, 5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise
            log.warning("Aufruf fehlgeschlagen (%s), neuer Versuch", error)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def call_chat(args: argparse.Namespace, messages: list[dict]) -> tuple[str, dict, str]:
    payload = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens}
    if args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = _post(args.endpoint.rstrip("/") + "/v1/chat/completions", payload, {}, args.timeout)
    choice = body["choices"][0]
    usage = {**body.get("usage", {}), "finish_reason": choice.get("finish_reason")}
    return choice["message"].get("content") or "", usage, choice["message"].get("reasoning") or ""


def call_openai(args: argparse.Namespace, messages: list[dict], schema: dict = CALL_SCHEMA,
                name: str = "tool_choice") -> tuple[str, dict, str]:
    payload = {"model": args.model, "instructions": messages[0]["content"], "input": messages[1]["content"],
               "store": False, "max_output_tokens": args.max_tokens, "reasoning": {"effort": args.reasoning},
               "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}}
    body = _post("https://api.openai.com/v1/responses", payload,
                 {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, args.timeout)
    texts = [(item.get("phase"), c["text"]) for item in body.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    final = [t for phase, t in texts if phase in (None, "final_answer")] or [t for _, t in texts]
    usage = {**body.get("usage", {}), "status": body.get("status")}
    return (final[-1] if final else ""), usage, ""


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _manifest(args: argparse.Namespace, n_tasks: int) -> dict:
    return {"experiment": "2 (Stand 18.11. abends)", "benchmark": args.benchmark, "model": args.model,
            "backend": args.backend,
            "endpoint": args.endpoint if args.backend == "chat" else "https://api.openai.com/v1/responses",
            "states": args.states, "levels": args.levels, "seed": SEED, "tasks": n_tasks,
            "max_tokens": args.max_tokens, "thinking": not args.no_thinking,
            "reasoning": args.reasoning if args.backend == "openai" else None,
            "system_prompt_sha256": hashlib.sha256(tc.SYSTEM.encode()).hexdigest(),
            "distractors_sha256": _sha(Path(tc.__file__).parent / "toolchoice_distractors.py"),
            "missing_file": str(MISSING_FILE.relative_to(tc.ROOT)), "missing_sha256": _sha(MISSING_FILE),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _select(tasks: list[dict], per_tool: int | None) -> list[dict]:
    if not per_tool:
        return tasks
    seen: dict[str, int] = {}
    chosen = []
    for task in tasks:
        tool = task["gold"]["tool"]
        if seen.get(tool, 0) < per_tool:
            chosen.append(task)
            seen[tool] = seen.get(tool, 0) + 1
    return chosen


def _jobs(tasks: list[dict], args: argparse.Namespace, rewrites: dict[str, dict]) -> list[tuple[dict, int]]:
    jobs = []
    for task in tasks:
        for state in args.states:
            if state == "info_missing" and task["task_id"] not in rewrites:
                continue
            v = tc.variant(task, state, rewrites.get(task["task_id"]) if state == "info_missing" else None)
            jobs += [(v, level) for level in args.levels]
    return jobs


def _one(args: argparse.Namespace, call, task: dict, level: int, catalog: dict) -> dict:
    tools = tc.tool_list(task, level, catalog, SEED)
    offered = [t["name"] for t in tools]
    start = time.perf_counter()
    try:
        raw, usage, reasoning = call(args, tc.prompt(task, tools))
    except RuntimeError as error:
        if "flagged" not in str(error):
            raise
        # Provider refused the prompt (policy false alarm): a technical outage, not a wrong choice.
        raw, usage, reasoning = "", {"refused": str(error)[:300]}, ""
    latency = time.perf_counter() - start
    choice = tc.parse_choice(raw)
    verdict = tc.evaluate(task, choice, offered)
    if "refused" in usage:
        verdict = {**verdict, "error_kind": "vom Anbieter abgelehnt"}
    return {"task_id": task["task_id"], "episode": task["episode"], "sequence": task["sequence"],
            "state": task["state"], "level": level, "artefact": task["artefact"],
            "reference_issue": task.get("reference_issue"), "n_offered": len(offered), "offered": offered, "requirement": task["requirement"],
            "removed_value": task.get("removed_value"), "removed_tools": task.get("removed_tools", []),
            "history": task["history"], "gold_tool": task["gold"]["tool"],
            "acceptable_tools": task["acceptable_tools"],
            "gold_position": offered.index(task["acceptable_tools"][0]) + 1,
            "raw_output": raw, "reasoning": reasoning, "lenient": bool(choice and choice.get("lenient")),
            "chosen_arguments": choice["arguments"] if choice else None, "usage": usage,
            "latency_s": round(latency, 3), **verdict}


def load_benchmark(name: str) -> tuple[dict, list[dict]]:
    """Catalog and tasks: 'level3' (API-Bank level 3, 131 calls) or 'level12' (level 1 and 2, 489 calls)."""
    from . import toolchoice_levels12 as l12

    if name == "level12":
        samples = l12.load_samples()
        return l12.build_catalog(samples), l12.build_tasks(samples)
    episodes = tc.load_episodes()
    catalog = tc.build_catalog(episodes)
    formats = l12.date_formats(catalog)  # same v5 rules as level 1/2 (19.11.): spec date format, justified questions
    return catalog, [dict(t, date_formats=formats,
                          ask_topics=l12.ask_topics(t["gold"]["arguments"], t["context"],
                                                    json.dumps(catalog[t["gold"]["tool"]])))
                     for t in tc.build_tasks(episodes, catalog)]


def run(args: argparse.Namespace) -> Path:
    from . import toolchoice_missing

    if args.benchmark == "level12" and (args.levels != [0] or "info_missing" in args.states):
        raise SystemExit("level12: nur --levels 0 und die Zustände complete, tool_missing (keine Umschreibungen).")
    catalog, tasks = load_benchmark(args.benchmark)
    tasks = _select(tasks, args.per_tool)
    rewrites = toolchoice_missing.load(MISSING_FILE) if args.benchmark == "level3" else {}
    if "info_missing" in args.states and not rewrites:
        raise SystemExit("Keine Umschreibungen für info_missing. Erst: python -m toolslip.toolchoice_missing")
    run_dir = tc.ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = _manifest(args, len(tasks))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        keys = ("benchmark", "model", "backend", "states", "levels", "seed", "thinking", "reasoning",
                "system_prompt_sha256", "distractors_sha256", "missing_sha256")
        changed = [k for k in keys if old.get(k) != manifest[k]]
        if changed:
            raise SystemExit(f"Lauf {args.name} existiert mit anderen Einstellungen: {changed}. Neuen Namen wählen.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    results = run_dir / "results.jsonl"
    done = set()
    if results.exists():
        done = {(r["task_id"], r.get("state", "complete"), r["level"])
                for r in map(json.loads, results.read_text().splitlines()) if r}
    jobs = [(t, lv) for t, lv in _jobs(tasks, args, rewrites) if (t["task_id"], t["state"], lv) not in done]
    call = call_openai if args.backend == "openai" else call_chat
    total, count = len(jobs) + len(done), len(done)
    log.info("%d Aufrufe offen, %d schon erledigt, %d parallel", len(jobs), len(done), args.workers)
    with results.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, args, call, task, level, catalog) for task, level in jobs]
        for future in as_completed(futures):
            row = future.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            count += 1
            mark = "✓" if row["call_correct"] else "✗"
            log.info("[%d/%d] %-12s Stufe %d %-10s richtig=%-26s gewählt=%-28s %s %s (%.1f s)", count, total,
                     row["state"], row["level"], row["task_id"], ",".join(row["acceptable_tools"])[:26],
                     row["chosen_tool"] or "–", mark, "" if row["call_correct"] else row["error_kind"],
                     row["latency_s"])
            if count % 50 == 0:
                toolchoice_report.build(run_dir)
    path = toolchoice_report.build(run_dir)
    log.info("Bericht: %s", path)
    return path


SCORING = "v5 (19.11.: wie v4, Datumsformat laut Werkzeugbeschreibung gilt für jedes gewählte Werkzeug gleich)"


def rescore(run_dir: Path) -> Path:
    """Re-evaluate stored answers with the current rules; no model calls. The old file is kept."""
    from . import toolchoice_missing

    benchmark = json.loads((run_dir / "manifest.json").read_text()).get("benchmark", "level3")
    tasks = {t["task_id"]: t for t in load_benchmark(benchmark)[1]}
    rewrites = toolchoice_missing.load(MISSING_FILE)
    results = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    backup = run_dir / "results-scoring-v1.jsonl"
    if not backup.exists():
        backup.write_text(results.read_text())
    updated = []
    for row in rows:
        state = row.get("state", "complete")
        task = tc.variant(tasks[row["task_id"]], state, rewrites.get(row["task_id"]) if state == "info_missing" else None)
        verdict = tc.evaluate(task, tc.parse_choice(row["raw_output"]), row["offered"])
        updated.append({**row, **verdict, "reference_issue": task.get("reference_issue"), "scoring": SCORING})
    results.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated))
    changed = sum(a.get("call_correct") != b["call_correct"] for a, b in zip(rows, updated))
    log.info("%s: %d Antworten neu bewertet, %d Urteile geändert", run_dir.name, len(rows), changed)
    return toolchoice_report.build(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Ordnername unter runs/")
    parser.add_argument("--backend", choices=["chat", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18790")
    parser.add_argument("--benchmark", choices=["level3", "level12"], default="level3")
    parser.add_argument("--states", nargs="+", choices=tc.STATES, default=list(tc.STATES))
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-tool", type=int, default=None, help="höchstens so viele Aufgaben je richtigem Werkzeug")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--no-thinking", action="store_true", help="Denkmodus lokaler Modelle ausschalten")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--rescore", action="store_true", help="gespeicherte Antworten neu bewerten, ohne Modellaufrufe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.rescore:
        rescore(tc.ROOT / "runs" / args.name)
        return
    if not (args.backend and args.model):
        parser.error("--backend und --model sind nötig, außer bei --rescore")
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY fehlt in der Umgebung.")
    run(args)


if __name__ == "__main__":
    main()
