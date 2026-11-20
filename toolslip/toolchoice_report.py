"""Static HTML report for a tool-choice run: summary per state and level, error classes, every single case."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from .toolchoice import FAMILY_OF, ROOT, WHY_WRONG

STATE_NAMES = {"complete": "vollständig", "tool_missing": "Werkzeug fehlt", "info_missing": "Angabe fehlt"}

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e2dfd6;--ok:#1f7a4d;--bad:#b3261e;--warn:#a15c00;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--ok:#5cc58f;--bad:#f2867d;--warn:#e7a54a;--card:#1c1c1a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;max-width:1150px;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.band{background:rgba(31,122,77,.10)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:6px 0;padding:8px 12px}
summary{cursor:pointer}pre{white-space:pre-wrap;font-size:12px;background:rgba(127,127,127,.08);padding:8px;border-radius:6px;overflow-x:auto}
.tools{columns:3;font-size:12px}.tools li{break-inside:avoid}.gold{font-weight:700;color:var(--ok)}.chosen{text-decoration:underline}.look{color:var(--muted)}
label{font-size:14px}body.only-errors details.right{display:none}
"""


def _rows(run: Path) -> list[dict]:
    lines = (run / "results.jsonl").read_text().splitlines() if (run / "results.jsonl").exists() else []
    return [json.loads(line) for line in lines if line.strip()]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _call_ok(r: dict) -> bool:
    return r.get("call_correct", r["tool_correct"])


def _frac(k: int, n: int) -> str:
    return f"<b>{k}/{n}</b> ({100 * k / n:.0f} %)" if n else "–"


def _summary(rows: list[dict]) -> str:
    out = ["<table><tr><th>Zustand</th><th>Stufe</th><th>Werkzeuge</th><th>Fälle</th><th>Ebene 1: Werkzeug</th>"
           "<th>Ebene 2: ganzer Aufruf</th><th>95-%-KI Ebene 2</th><th>stille Fehler</th><th>häufigste Fehler</th>"
           "<th>Median s</th></tr>"]
    states = [s for s in STATE_NAMES if any(r.get("state", "complete") == s for r in rows)]
    for state in states:
        for level in sorted({r["level"] for r in rows}):
            rs = [r for r in rows if r.get("state", "complete") == state and r["level"] == level]
            if not rs:
                continue
            n = len(rs)
            k1, k2 = sum(r["tool_correct"] for r in rs), sum(_call_ok(r) for r in rs)
            lo, hi = wilson(k2, n)
            band = " class=band" if 0.5 <= k2 / n <= 0.85 else ""
            kinds = Counter(r["error_kind"] for r in rs if not _call_ok(r)).most_common(3)
            out.append(f"<tr{band}><td>{STATE_NAMES[state]}</td><td>{level}</td><td class=num>{rs[0]['n_offered']}</td>"
                       f"<td class=num>{n}</td><td class=num>{_frac(k1, n)}</td><td class=num>{_frac(k2, n)}</td>"
                       f"<td class=num>{100*lo:.0f}–{100*hi:.0f} %</td>"
                       f"<td class=num>{sum(bool(r.get('silent_error')) for r in rs)}</td>"
                       f"<td>{'<br>'.join(f'{escape(k)}: {c}' for k, c in kinds) or '–'}</td>"
                       f"<td class=num>{statistics.median(r['latency_s'] for r in rs):.1f}</td></tr>")
    out.append("</table><p class=muted>Grün hinterlegt: Ebene 2 im Zielband 50–85 %. Stille Fehler: Der Agent führt "
               "einen falschen Aufruf aus, statt zu fragen oder abzulehnen – genau das muss ein Prüfer finden. "
               "Aufrufpunkte derselben Episode sind nicht unabhängig. Fälle mit API-Bank-Längengrad-Artefakt sind "
               "hier ausgeschlossen und stehen unten getrennt.</p>")
    return "".join(out)


def _by_sequence(rows: list[dict]) -> str:
    cell = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        c = cell[r.get("sequence", "–")]
        c[0] += 1
        c[1] += not _call_ok(r)
        c[2] += bool(r.get("silent_error"))
    total = sum(c[1] for c in cell.values()) or 1
    out = ["<table><tr><th>Werkzeugfolge der Episode</th><th>Fälle</th><th>Fehler Ebene 2</th>"
           "<th>Anteil an allen Fehlern</th><th>stille Fehler</th></tr>"]
    for seq, (n, err, silent) in sorted(cell.items(), key=lambda x: -x[1][1]):
        out.append(f"<tr><td>{escape(seq)}</td><td class=num>{n}</td><td class=num>{err}</td>"
                   f"<td class=num>{100 * err / total:.0f} %</td><td class=num>{silent}</td></tr>")
    return "".join(out) + "</table>"


def _confusions(rows: list[dict]) -> str:
    counts = Counter((STATE_NAMES.get(r.get("state", "complete")), r["level"], ",".join(r["acceptable_tools"]),
                      r.get("chosen_tool") or "–", r["error_kind"]) for r in rows if not _call_ok(r))
    if not counts:
        return "<p>Keine Fehler.</p>"
    out = ["<table><tr><th>Zustand</th><th>Stufe</th><th>richtig wäre</th><th>gewählt</th><th>Art</th>"
           "<th>warum falsch</th><th>Anzahl</th></tr>"]
    for (state, level, gold, chosen, kind), n in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        out.append(f"<tr><td>{state}</td><td>{level}</td><td>{escape(gold)}</td><td>{escape(chosen)}</td>"
                   f"<td>{escape(kind)}</td><td class=muted>{escape(WHY_WRONG.get(chosen, ''))}</td>"
                   f"<td class=num>{n}</td></tr>")
    return "".join(out) + "</table>"


def _tools(r: dict) -> str:
    items = []
    for name in r["offered"]:
        cls = ["gold"] if name in r["acceptable_tools"] else (["look"] if name in FAMILY_OF else [])
        if name == r.get("chosen_tool"):
            cls.append("chosen")
        tag = f" ≈ {FAMILY_OF[name]}" if name in FAMILY_OF else ""
        items.append(f"<li class='{' '.join(cls)}'>{escape(name)}{escape(tag)}</li>")
    return "<ul class=tools>" + "".join(items) + "</ul>"


def _cases(rows: list[dict]) -> str:
    out = []
    order = sorted(rows, key=lambda r: (_call_ok(r), r.get("state", ""), r["level"], r["task_id"]))
    for r in order:
        ok = _call_ok(r)
        mark = "<span class=ok>✓</span>" if ok else "<span class=bad>✗</span>"
        why = WHY_WRONG.get(r.get("chosen_tool") or "", "")
        history = "\n".join(f"{i}. {h['tool']}({json.dumps(h['arguments'], ensure_ascii=False)}) → "
                            f"{json.dumps(h['result'], ensure_ascii=False)[:400]}"
                            for i, h in enumerate(r["history"], 1)) or "(noch keine Aufrufe)"
        notes = []
        if r.get("removed_value"):
            notes.append(f"entfernte Angabe: {r['removed_value']['key']} = {r['removed_value']['value']}")
        if r.get("removed_tools"):
            notes.append("entfernte Werkzeuge: " + ", ".join(r["removed_tools"]))
        if r.get("invented_args"):
            notes.append("erfundene Argumente: " + ", ".join(r["invented_args"]))
        out.append(
            f"<details class='{'right' if ok else 'wrong'}'><summary>{mark} <b>{escape(r['task_id'])}</b> · "
            f"{STATE_NAMES.get(r.get('state', 'complete'))} · Stufe {r['level']} · richtig: "
            f"{escape(', '.join(r['acceptable_tools']))} · gewählt: {escape(r.get('chosen_tool') or '–')} · "
            f"<span class=muted>{escape(r['error_kind'])}</span></summary>"
            f"<p><b>Anfrage:</b> {escape(r['requirement'])}</p>"
            + (f"<p class=muted>{escape(' · '.join(notes))}</p>" if notes else "")
            + (f"<p class=warn><b>Warum falsch:</b> {escape(why)}</p>" if why and not ok else "")
            + f"<p><b>Bisherige Aufrufe</b></p><pre>{escape(history)}</pre>"
            + (f"<details><summary>Denken des Modells</summary><pre>{escape(r['reasoning'])}</pre></details>"
               if r.get("reasoning") else "")
            + f"<p><b>Antwort des Modells</b> ({r['latency_s']:.1f} s"
            f"{', Format unsauber' if r.get('lenient') else ''})</p><pre>{escape(str(r['raw_output']))}</pre>"
            f"<p><b>Angebotene Werkzeuge</b> ({r['n_offered']}; grün = richtig, grau = ähnliches Werkzeug, "
            f"unterstrichen = gewählt)</p>{_tools(r)}</details>")
    return "".join(out)


def _rewrites(manifest: dict) -> str:
    path = ROOT / manifest["missing_file"] if manifest.get("missing_file") else None
    if not path or not path.exists():
        return ""
    doc = json.loads(path.read_text())
    out = [f"<h2>Umschreibungen „Angabe fehlt“ zur Handkontrolle</h2><p class=muted>{len(doc['rewrites'])} "
           f"Umschreibungen von {escape(doc['model'])}, {sum(e['ok'] for e in doc['rewrites'])} automatisch geprüft "
           f"brauchbar. Von Hand geprüft: {'ja' if doc.get('hand_checked') else '<b>noch nicht</b>'}.</p>"
           "<table><tr><th>Aufgabe</th><th>entfernt</th><th>Original</th><th>Umschreibung</th><th>Prüfung</th></tr>"]
    for e in doc["rewrites"]:
        out.append(f"<tr><td>{escape(e['task_id'])}</td><td>{escape(e['key'])} = {escape(str(e['value']))}</td>"
                   f"<td>{escape(e['original'])}</td><td>{escape(e['requirement'])}</td>"
                   f"<td class={'ok' if e['ok'] else 'bad'}>{'✓' if e['ok'] else '✗'}</td></tr>")
    return "".join(out) + "</table>"


def build(run: Path) -> Path:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    all_rows = _rows(run)
    refused = [r for r in all_rows if (r.get("usage") or {}).get("refused")]
    rows = [r for r in all_rows if not (r.get("usage") or {}).get("refused")]
    main = [r for r in rows if not r.get("artefact") and not r.get("reference_issue")]
    artefact = [r for r in rows if r.get("artefact")]
    flawed = [r for r in rows if r.get("reference_issue") and not r.get("artefact")]
    head = (f"<h1>Werkzeugwahl ohne Prüfer (Experiment 2)</h1><p class=muted>Modell <b>{escape(str(manifest.get('model')))}</b>"
            f" · Reasoning {escape(str(manifest.get('reasoning')))} · {len(rows)} Antworten · Lauf {escape(run.name)} · "
            f"{escape(str(manifest.get('started_utc', '')))}</p>"
            "<p>Frage: In welchen Zuständen macht der Agent bei 50–85 % richtigen Aufrufen genug stille Fehler, "
            "damit ein Prüfer etwas zu finden hat?</p>")
    art = ""
    if artefact:
        k = sum(_call_ok(r) for r in artefact)
        kinds = Counter(r["error_kind"] for r in artefact if not _call_ok(r))
        art = (f"<h2>Längengrad-Artefakt (getrennt)</h2><p>{len(artefact)} Antworten auf Aufgaben, in denen API-Bank "
               f"New York, San Francisco oder London einen positiven Längengrad gibt: {k} ganz richtig. "
               f"Fehler: {escape(', '.join(f'{a} {b}' for a, b in kinds.items())) or 'keine'}.</p>")
    if flawed:
        k = sum(_call_ok(r) for r in flawed)
        reasons = Counter(r["reference_issue"] for r in flawed)
        art += (f"<h2>Fehlerhafte API-Bank-Referenzen (getrennt)</h2><p>{len(flawed)} Antworten, {k} ganz richtig. "
                f"Gründe: {escape(', '.join(f'{a} ({b})' for a, b in reasons.items()))}.</p>")
    if refused:
        head += (f"<p class=warn>{len(refused)} Anfragen hat der Anbieter abgelehnt (Richtlinien-Fehlalarm). "
                 f"Sie fehlen in allen Quoten: {escape(', '.join(sorted({r['task_id'] for r in refused})))}</p>")
    body = (head + "<h2>Überblick je Zustand und Stufe</h2>" + _summary(main) + art
            + "<h2>Fehler je Werkzeugfolge</h2>" + _by_sequence(main)
            + "<h2>Fehler im Einzelnen</h2>" + _confusions(main)
            + _rewrites(manifest)
            + "<h2>Alle Fälle</h2><label><input type=checkbox "
              "onchange=\"document.body.classList.toggle('only-errors',this.checked)\"> nur Fehler zeigen</label>"
            + _cases(rows))
    html = (f"<!doctype html><html lang=de><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>Werkzeugwahl-Vorversuch</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    path = run / "report.html"
    path.write_text(html)
    return path
