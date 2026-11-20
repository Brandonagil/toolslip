# Toolslip

**Find silent wrong tool calls of LLM agents before they reach real data.**

An agent that picks the wrong tool usually does not crash. The call runs, returns something plausible, and nobody notices. Toolslip measures how often that happens: it shows an agent a tool list, removes the right tool or a needed value, adds look-alike tools, and checks whether the agent calls a wrong tool instead of saying "no suitable tool" or asking the user.

> Status: research prototype from a master's thesis (Hochschule Karlsruhe, 2026/27). It currently runs on the API-Bank benchmark. Testing your own MCP tool catalog is the next step (see [Roadmap](#roadmap)). Some report labels are still in German.

## What it tests

Each real tool call of an API-Bank dialogue becomes one task: the agent sees the user request, the earlier calls with their results, and a tool list that always also offers `AskUser` and `NoSuitableTool`. It must name the next call.

| State | What changes | Right answer |
|---|---|---|
| `complete` | nothing | the gold tool with the right arguments |
| `tool_missing` | every acceptable tool is removed, look-alikes stay | `NoSuitableTool` |
| `info_missing` | one value the call needs is removed from the request | `AskUser` |

Look-alike levels: 0, 2 or 5 hand-written look-alikes per real tool (`toolslip/toolchoice_distractors.py`). Each look-alike does something different from what the task needs, so choosing one is always wrong. Every look-alike carries a `why_wrong` note that the report shows.

Outcomes are split into **right**, **visible errors** (e.g. a tool that was not offered, or asking when not needed) and **silent errors** (a real or look-alike tool is called instead of reporting the gap). `toolslip.toolchoice_execute` runs each silent error once against the unchanged API-Bank tools to confirm it executes without an exception.

### First result

One current model, API-Bank level 3, state `tool_missing`, 131 tasks per level:

| Look-alikes per real tool | Silent wrong tool call |
|---|---|
| 0 | 7 / 131 (5 %) |
| 5 | 20 / 131 (15 %) |

Preliminary: one model, one benchmark. Not a model ranking.

## Quickstart

```sh
uv venv --python 3.13
uv pip install -e .
python -m unittest discover -s tests
```

Run against a local OpenAI-compatible chat endpoint (e.g. `mlx_lm.server`, Ollama, vLLM):

```sh
python -m toolslip.toolchoice_run --name my-run --backend chat \
    --endpoint http://127.0.0.1:11434 --model <model> --levels 0 2
```

Or the OpenAI Responses API (the key is read from the environment only and never written to disk):

```sh
OPENAI_API_KEY=... python -m toolslip.toolchoice_run --name my-run --backend openai --model <model>
```

Every answer is appended to `runs/<name>/results.jsonl` right away; a rerun skips finished tasks. `runs/<name>/report.html` shows the summary, all confusions and every single case. `--rescore` scores stored answers again without new model calls.

Confirm silent errors against the real API-Bank tools:

```sh
uv pip install -e ".[apibank-tools]"
python -m toolslip.toolchoice_execute --out my-pool
```

It currently reads the run names listed in `POOL_RUNS` in `toolslip/toolchoice_execute.py`; edit them to point at your runs.

Replay stored runs in the [Inspect](https://inspect.aisi.org.uk/) log viewer (no model calls):

```sh
uv pip install -e ".[inspect]"
python -m toolslip.inspect_replay <run-name>
inspect view --log-dir runs/inspect-logs
```

## Reproducibility

- API-Bank files are pinned to a commit and checked against SHA-256 hashes in each `data/*/manifest.json` before use.
- The `info_missing` rewrites (`data/toolchoice/missing_v1.json`) were generated once, checked automatically and by hand, and frozen. Runs record the hash of the rewrites and of the look-alike file.
- Tool lists are rebuilt with a fixed seed; replay aborts if they differ from the stored list.

## Roadmap

- Read tool catalogs from any MCP server (`tools/list`) or OpenAPI / JSON Schema
- Generate and review look-alike tools for your own catalog, frozen by hash
- Stand-in tools that check parameters without side effects
- Local models via Ollama by default, so tool catalogs and data stay on your machine
- GitHub Action that fails a build when the silent-error rate passes a threshold
- English report labels
- Packaging as an Inspect extension

## Related work

Toolslip builds on [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) and [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai). Benchmarks such as the Berkeley Function Calling Leaderboard, τ-bench or MCP-Bench compare models on fixed tool sets. Scanners such as mcp-scan check tool descriptions statically. Toolslip focuses on agent behaviour when the right tool is missing and similar tools are present, and on separating silent from visible errors.

## License

Apache License 2.0, see [LICENSE](LICENSE). Includes API-Bank data and source files (Apache-2.0) and the English NLTK Punkt model (Apache-2.0); see [NOTICE](NOTICE).
