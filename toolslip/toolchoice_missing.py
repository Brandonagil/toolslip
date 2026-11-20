"""Frozen rewrites for the info_missing state: one value the next call needs is removed from the request.

The value comes from tc.missing_candidate (a gold argument that only the request supplies).
A language model rewrites the request once; tc.rewrite_ok checks that the value is gone and every
other needed value is still there. Failed rewrites are kept in the file but not used.
The file is written once and then frozen; runs record its hash. A sample is checked by hand.

    OPENAI_API_KEY=... .venv/bin/python -m toolslip.toolchoice_missing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from . import toolchoice as tc

log = logging.getLogger("toolchoice")
INSTRUCTIONS = (
    "You edit user requests for a test set. Rewrite the request so that the given value is no longer stated "
    "and cannot be inferred from the text. Keep every other detail and keep the wording as close to the "
    "original as possible. Do not mention that information is missing, do not add a question, do not add "
    'placeholders like [name]. Return JSON: {"request": "<rewritten request>"}'
)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["request"],
          "properties": {"request": {"type": "string"}}}


def load(path: Path) -> dict[str, dict]:
    """task_id -> usable rewrite."""
    if not path.exists():
        return {}
    return {e["task_id"]: e for e in json.loads(path.read_text())["rewrites"] if e["ok"]}


def build(args: argparse.Namespace, path: Path) -> None:
    from .toolchoice_run import call_openai

    if path.exists():
        raise SystemExit(f"{path} existiert bereits und ist eingefroren. Für eine neue Fassung neuen Namen wählen.")
    episodes = tc.load_episodes()
    tasks = tc.build_tasks(episodes, tc.build_catalog(episodes))
    entries = []
    for task in tasks:
        candidate = tc.missing_candidate(task)
        if candidate is None:
            continue
        key, value = candidate
        request = json.dumps({"request": task["requirement"], "value_to_remove": value, "parameter": key},
                             ensure_ascii=False)
        text, usage, _ = call_openai(args, [{"role": "system", "content": INSTRUCTIONS},
                                            {"role": "user", "content": request}], schema=SCHEMA, name="rewrite")
        rewritten = json.loads(text)["request"].strip() if text else ""
        ok = bool(rewritten) and tc.rewrite_ok(task, key, value, rewritten)
        entries.append({"task_id": task["task_id"], "key": key, "value": value, "original": task["requirement"],
                        "requirement": rewritten, "ok": ok, "usage": usage})
        log.info("%-8s %-12s %-22s %s  %s", task["task_id"], key, str(value)[:22], "✓" if ok else "✗", rewritten)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "model": args.model, "reasoning": args.reasoning, "instructions": INSTRUCTIONS,
                                "hand_checked": False, "rewrites": entries}, indent=2, ensure_ascii=False))
    log.info("%d Umschreibungen, %d brauchbar → %s", len(entries), sum(e["ok"] for e in entries), path)


def main() -> None:
    from .toolchoice_run import MISSING_FILE

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY fehlt in der Umgebung.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args, MISSING_FILE)


if __name__ == "__main__":
    main()
