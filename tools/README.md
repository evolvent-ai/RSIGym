# rsiwatch

Live dashboard for RSI experiments: watch Harbor jobs and Claude Code agent
trajectories **while they run**.

```sh
cd tools
uv run rsiwatch                 # defaults to ../rsi_task/jobs
uv run rsiwatch <jobs-dir> --port 8420
```

Then open <http://127.0.0.1:8420>.

## Why this exists

Harbor ships `harbor view`, which is good — use it for finished jobs. But it
reads `agent/trajectory.json`, and the Claude Code adapter only writes that file
in `populate_context_post_run()`, *after* the trial ends
(`harbor/agents/installed/claude_code.py:1493`). While a trial is running the
endpoint returns `null`, so there is nothing to watch.

The native session log underneath it *is* written live. `rsiwatch` tails that
instead, so a running experiment is visible from its first tool call.

| | `harbor view` | `rsiwatch` |
|---|---|---|
| Finished trials | ✅ full ATIF viewer | ✅ |
| **Running trials** | ❌ `trajectory` returns `null` | ✅ live via SSE |
| Token/cost breakdown | job-level totals | per-trial, cache-aware |
| Stall detection | — | ✅ flags quiet sessions |

## What it shows

**Header** — running/stalled/errored counts, total tokens, cache hit rate, and
cost across all watched jobs.

**Sidebar** — jobs with per-status rollups; each trial is one line: status dot,
tool count, tokens, and the number that matters for its state (idle time,
reward, or error type).

**Detail** — the trajectory as it streams. A tool call and its result are one
step card (duration, ok/failed, stderr); assistant text stands out as the
narrative; thinking is dimmed and collapsed until clicked. Quiet stretches over
five minutes are marked inline. Toolbar: kind filters, errors-only, plain-text
search.

**Rail** — the whole run as one vertical strip at the right edge, one tick per
step colored by kind, errors in red, the visible window highlighted. Click
anywhere to jump. Long runs load the newest 400 steps first; "earlier steps"
pages backwards, and the rail jumps straight to any point.

Keys: `j`/`k` switch trials · `f` follow · `e` expand/collapse all · `/` search.

Status is inferred, not read: a trial with no `result.json` whose session file
was touched recently is running; one whose `verifier/test-stdout.txt` exists is
**verifying** (the agent is done, the verifier is running); one that has gone
quiet past `--stall-after` (default 1h) is flagged **stalled**. That distinction is the point — silence
looks identical to progress until you measure it.

The default is deliberately generous. RSI agents go quiet for long stretches
while a rollout batch, training run, or benchmark sweep finishes, so a
minutes-scale threshold would report healthy runs as wedged. Lower it with
`--stall-after` for workloads that should tick more often.

## pi trials

Harbor runs pi with `--session-dir /logs/agent/pi/sessions`, so pi's own session
file, `agent/pi/sessions/<stamp>_<id>.jsonl`, is on the host from the first
message and is what gets tailed. Every entry is timestamped; a `message` entry
is one whole user prompt, assistant turn (thinking, text and `toolCall` blocks
with usage) or `toolResult`, so step cards carry durations and tokens update
per turn. Compaction and branch summaries count the usage of the call that made
them. Cost is the figure pi computed from the task image's `models.json`
prices — the driver model is whatever that file declares (`qwen/qwen3.8-27b`
in the Qwen3.8 tasks), not something in `PRICING`.

## Codex trials

Harbor's Codex adapter keeps the session under `$CODEX_HOME` inside the
container and only copies it out when the trial ends. What is on the host while
it runs is `agent/codex.txt`, the `codex exec --json` event stream (stdout and
stderr teed together), so that is what gets tailed for a trial without a Claude
Code session. Each `item.started` / `item.completed` pair becomes one step card;
`agent_message` is the narrative. That stream carries no timestamps and reports
usage only on `turn.completed`, so while a Codex trial runs its step durations
are blank and its tokens and cost read zero.

When the agent finishes, harbor copies the session out as
`agent/sessions/<date>/rollout-*.jsonl` and the trial switches to it: every
record is timestamped, each command carries its start and end, a `token_count`
follows every model response, and reasoning summaries show as thinking. Cost
uses the `gpt-6-astra` list price in `PRICING`, cache writes at 1.25x input.

## Two traps in the session format

Both are handled here, and both are worth knowing if you ever parse these files
yourself:

1. **One content block per line, not one message.** A single assistant message
   is split across as many lines as it has blocks, all sharing `message.id`.
2. **`usage` is repeated on every one of those lines.** Summing per line
   overcounts. On a real trial here it inflated output tokens 3.2× (215,313 vs
   the true 66,447).

`TrajectoryReader` keys on `message.id` and counts usage once per id.

## Notes

- **Cost is an estimate.** Prices are Claude list rates in `trajectory.py`
  (`PRICING`), matched against gateway-mangled model names like
  `vibe-claude-sub2api-opus-5[1m]`. It does not know your actual contract.
- Reads are incremental — byte offset plus inode, so a rotated or restarted
  session rebuilds cleanly and a half-written line is buffered until complete.
- Polling is one background task shared by all connected browsers; clients get
  a content-free SSE tick and re-fetch, so a missed tick is self-healing.
- Read-only: it never writes to `jobs/`.

## Tests

```sh
uv run pytest
```
