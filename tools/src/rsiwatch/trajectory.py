"""Incremental parser for Claude Code session JSONL.

Harbor only writes `agent/trajectory.json` (ATIF) once a trial finishes, so a
running trial is invisible to `harbor view`. The native session log underneath
it, however, is appended live — this module reads that.

Two things about the format bite anyone who treats it as one-message-per-line:

* Each line carries exactly ONE content block. A single assistant message is
  split across as many lines as it has blocks, all sharing `message.id`.
* `message.usage` is repeated verbatim on every one of those lines. Summing it
  per line overcounts tokens 3-4x.

Both are handled by keying on `message.id` and counting usage once per id.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Claude Opus 5 list price, USD per million tokens. Cache writes are 1.25x input
# (5m TTL) and 2x (1h); reads are 0.1x.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write_5m": 6.25,
                      "cache_write_1h": 10.0, "cache_read": 0.5},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_write_5m": 6.25,
                        "cache_write_1h": 10.0, "cache_read": 0.5},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_write_5m": 3.75,
                        "cache_write_1h": 6.0, "cache_read": 0.3},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write_5m": 1.25,
                         "cache_write_1h": 2.0, "cache_read": 0.1},
    # OpenAI list price; cache writes are 1.25x input and have no TTL tiers.
    "gpt-6-astra": {"input": 10.0, "output": 50.0, "cache_write_5m": 12.5,
                    "cache_write_1h": 12.5, "cache_read": 1.0},
}
_DEFAULT_PRICE = PRICING["claude-opus-5"]


def price_for(model: str | None) -> dict[str, float]:
    """Price table for a model name, tolerating gateway suffixes.

    Trial model names are gateway-mangled (`vibe-claude-sub2api-opus-5[1m]`),
    so match on substring rather than equality.
    """
    if not model:
        return _DEFAULT_PRICE
    name = model.lower()
    for key, table in PRICING.items():
        if key in name:
            return table
    # `opus-5` inside a gateway alias won't match `claude-opus-5` above.
    for fragment, key in (("opus-5", "claude-opus-5"), ("opus-4-8", "claude-opus-4-8"),
                          ("sonnet-5", "claude-sonnet-5"), ("haiku", "claude-haiku-4-5")):
        if fragment in name:
            return PRICING[key]
    return _DEFAULT_PRICE


@dataclass
class Usage:
    """Token totals, deduplicated by message id."""

    input: int = 0
    output: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0

    def add(self, usage: dict[str, Any]) -> None:
        self.input += usage.get("input_tokens") or 0
        self.output += usage.get("output_tokens") or 0
        self.cache_read += usage.get("cache_read_input_tokens") or 0
        creation = usage.get("cache_creation") or {}
        write_5m = creation.get("ephemeral_5m_input_tokens")
        write_1h = creation.get("ephemeral_1h_input_tokens")
        if write_5m is None and write_1h is None:
            # Older sessions only carry the flat total; bill it at the 5m rate.
            self.cache_write_5m += usage.get("cache_creation_input_tokens") or 0
        else:
            self.cache_write_5m += write_5m or 0
            self.cache_write_1h += write_1h or 0

    def add_openai(self, usage: dict[str, Any]) -> None:
        """OpenAI counts cache reads and writes inside input_tokens; split them out."""
        cached = usage.get("cached_input_tokens") or 0
        written = usage.get("cache_write_input_tokens") or 0
        self.input += (usage.get("input_tokens") or 0) - cached - written
        self.cache_read += cached
        self.cache_write_5m += written
        self.output += usage.get("output_tokens") or 0

    @property
    def total(self) -> int:
        return (self.input + self.output + self.cache_write_5m
                + self.cache_write_1h + self.cache_read)

    def cost_usd(self, model: str | None) -> float:
        p = price_for(model)
        return (
            self.input / 1e6 * p["input"]
            + self.output / 1e6 * p["output"]
            + self.cache_write_5m / 1e6 * p["cache_write_5m"]
            + self.cache_write_1h / 1e6 * p["cache_write_1h"]
            + self.cache_read / 1e6 * p["cache_read"]
        ) if self.total else 0.0

    def as_dict(self, model: str | None = None) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_write": self.cache_write_5m + self.cache_write_1h,
            "cache_read": self.cache_read,
            "total": self.total,
            "cost_usd": round(self.cost_usd(model), 4),
            # Share of billable input served from cache — the headline number for
            # whether a long agent run is being cached well.
            "cache_hit_rate": round(
                self.cache_read / (self.cache_read + self.input + self.cache_write_5m
                                   + self.cache_write_1h), 4)
            if (self.cache_read + self.input + self.cache_write_5m + self.cache_write_1h)
            else 0.0,
        }


def _text_of(block: dict[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text") or ""
    if kind == "thinking":
        return block.get("thinking") or ""
    return ""


def _preview(value: Any, limit: int = 2000) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else text[:limit] + f"\n… (+{len(text) - limit} chars)"


# Tool inputs read far better as raw text than as escaped JSON — a Bash command
# rendered as "python3 - <<'EOF'\nimport json\n…" is unreadable at a glance.
# These are each tool's principal field, shown verbatim with the rest appended.
_PRIMARY_INPUT: dict[str, str] = {
    "Bash": "command",
    "Write": "content",
    "Read": "file_path",
    "Edit": "new_string",
    "Skill": "skill",
    "Task": "prompt",
    "Agent": "prompt",
    "WebFetch": "url",
    "Grep": "pattern",
    "Glob": "pattern",
}


def _tool_input_preview(name: str | None, value: Any, limit: int = 2000) -> str:
    """Render a tool_use input for reading, not for round-tripping."""
    if not isinstance(value, dict):
        return _preview(value, limit)

    key = _PRIMARY_INPUT.get(name or "")
    primary = value.get(key) if key else None
    if not isinstance(primary, str) or not primary.strip():
        return _preview(value, limit)

    rest = {k: v for k, v in value.items() if k != key}
    text = primary
    if rest:
        # Keep the secondary args, but out of the way of the main payload.
        text += "\n\n· " + json.dumps(rest, ensure_ascii=False)[:400]
    return _preview(text, limit)


@dataclass
class Event:
    """One renderable step in the trajectory."""

    seq: int
    kind: str  # thinking | text | tool_use | tool_result | user | attachment
    timestamp: str | None = None
    message_id: str | None = None
    name: str | None = None       # tool name
    tool_use_id: str | None = None
    body: str = ""
    is_error: bool = False
    duration_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "name": self.name,
            "tool_use_id": self.tool_use_id,
            "body": self.body,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
        }


class TrajectoryReader:
    """Tails one session JSONL, parsing only newly appended bytes.

    Holds a byte offset and the file's inode. A shrunken file or a changed inode
    means the log was rotated or the trial restarted, so state is rebuilt from
    scratch. A trailing partial line (these are ~35KB each, so reading mid-write
    is routine) is buffered until its newline arrives.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._reset()

    def _reset(self) -> None:
        self._offset = 0
        self._inode: int | None = None
        self._buffer = ""
        self._seq = 0
        self._seen_usage: set[str] = set()
        self._pending_tools: dict[str, Event] = {}

        self.events: list[Event] = []
        self.usage = Usage()
        self.model: str | None = None
        self.session_id: str | None = None
        self.cwd: str | None = None
        self.version: str | None = None
        self.tool_counts: dict[str, int] = {}
        self.tool_errors: dict[str, int] = {}
        self.first_timestamp: str | None = None
        self.last_timestamp: str | None = None
        self.message_ids: set[str] = set()
        self.first_prompt: str = ""

    def poll(self) -> bool:
        """Read appended bytes. Returns True if anything new was parsed."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False

        if self._inode is None:
            self._inode = stat.st_ino
        elif stat.st_ino != self._inode or stat.st_size < self._offset:
            self._reset()
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                return False
            self._inode = stat.st_ino

        if stat.st_size == self._offset:
            return False

        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()

        if not chunk:
            return False

        self._buffer += chunk
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()  # trailing partial line, if any

        parsed = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._ingest(record)
            parsed = True
        return parsed

    def _ingest(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        timestamp = record.get("timestamp")
        if timestamp:
            self.first_timestamp = self.first_timestamp or timestamp
            self.last_timestamp = timestamp

        self.session_id = record.get("sessionId") or self.session_id
        self.cwd = record.get("cwd") or self.cwd
        self.version = record.get("version") or self.version

        # Subagent transcripts are a separate stream; the main thread is what a
        # top-level trajectory view should show.
        if record.get("isSidechain"):
            return

        if kind == "assistant":
            self._ingest_assistant(record, timestamp)
        elif kind == "user":
            self._ingest_user(record, timestamp)

    def _ingest_assistant(self, record: dict[str, Any], timestamp: str | None) -> None:
        message = record.get("message") or {}
        self.model = message.get("model") or self.model

        message_id = message.get("id")
        if message_id:
            self.message_ids.add(message_id)
            # Usage repeats on every line of a message — count it once.
            if message_id not in self._seen_usage:
                self._seen_usage.add(message_id)
                self.usage.add(message.get("usage") or {})

        content = message.get("content")
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("text", "thinking"):
                body = _text_of(block)
                if not body.strip():
                    continue
                self._append(Event(seq=self._next(), kind=block_type, timestamp=timestamp,
                                   message_id=message_id, body=_preview(body, 4000)))
            elif block_type == "tool_use":
                name = block.get("name") or "?"
                self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
                event = Event(seq=self._next(), kind="tool_use", timestamp=timestamp,
                              message_id=message_id, name=name,
                              tool_use_id=block.get("id"),
                              body=_tool_input_preview(name, block.get("input")))
                self._append(event)
                if event.tool_use_id:
                    self._pending_tools[event.tool_use_id] = event

    def _ingest_user(self, record: dict[str, Any], timestamp: str | None) -> None:
        message = record.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            if content.strip():
                self.first_prompt = self.first_prompt or content
                self._append(Event(seq=self._next(), kind="user", timestamp=timestamp,
                                   body=_preview(content, 6000)))
            return
        if not isinstance(content, list):
            return

        result = record.get("toolUseResult")
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                self._ingest_tool_result(block, result, timestamp)
            elif block_type == "text":
                body = block.get("text") or ""
                if body.strip():
                    self.first_prompt = self.first_prompt or body
                    self._append(Event(seq=self._next(), kind="user", timestamp=timestamp,
                                       body=_preview(body, 6000)))

    def _ingest_tool_result(self, block: dict[str, Any], result: Any,
                            timestamp: str | None) -> None:
        tool_use_id = block.get("tool_use_id")
        origin = self._pending_tools.pop(tool_use_id, None) if tool_use_id else None
        name = origin.name if origin else None

        body = self._result_body(block, result)
        is_error = bool(block.get("is_error"))
        if isinstance(result, dict):
            # Bash reports failure via stderr rather than is_error.
            if result.get("stderr") and not result.get("stdout"):
                is_error = True
            if result.get("interrupted"):
                is_error = True
        if is_error and name:
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1

        duration = None
        if origin and origin.timestamp and timestamp:
            duration = _delta_ms(origin.timestamp, timestamp)

        self._append(Event(seq=self._next(), kind="tool_result", timestamp=timestamp,
                           name=name, tool_use_id=tool_use_id, body=body,
                           is_error=is_error, duration_ms=duration))

    @staticmethod
    def _result_body(block: dict[str, Any], result: Any) -> str:
        if isinstance(result, dict):
            parts = []
            for key in ("stdout", "stderr"):
                value = result.get(key)
                if value:
                    label = "" if key == "stdout" else "[stderr] "
                    parts.append(f"{label}{value}")
            if parts:
                return _preview("\n".join(parts), 3000)
            if "file" in result:
                return _preview(result.get("file"), 1500)
            return _preview(result, 1500)
        if isinstance(result, str) and result.strip():
            return _preview(result, 3000)
        return _preview(block.get("content"), 3000)

    def _append(self, event: Event) -> None:
        self.events.append(event)

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "cwd": self.cwd,
            "version": self.version,
            "n_events": len(self.events),
            "n_messages": len(self.message_ids),
            "tool_counts": dict(sorted(self.tool_counts.items(),
                                       key=lambda kv: kv[1], reverse=True)),
            "tool_errors": self.tool_errors,
            "n_tool_calls": sum(self.tool_counts.values()),
            "n_tool_errors": sum(self.tool_errors.values()),
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "usage": self.usage.as_dict(self.model),
            "first_prompt": self.first_prompt[:2000],
        }


# Codex names items by what they are, not by a tool name; these read better in a
# tool-count sidebar.
_CODEX_TOOL_NAMES: dict[str, str] = {
    "command_execution": "shell",
    "file_change": "apply_patch",
    "web_search": "web_search",
}
_CODEX_PROSE_ITEMS = {"agent_message", "reasoning", "error", "todo_list"}


class CodexReader(TrajectoryReader):
    """Tails `agent/codex.txt`, the `codex exec --json` event stream harbor tees.

    Every line is one event: `thread.started`, `turn.started`, `turn.completed`
    (the only place usage appears), `item.started` / `item.completed` carrying an
    item whose `type` says what it is, and top-level `error`. Events carry no
    timestamps. Harbor tees stderr into the same file, so non-JSON lines (Codex's
    own tracing) are routine and skipped by the tailer.
    """

    def __init__(self, path: Path, model: str | None = None) -> None:
        super().__init__(path)
        self.model = model

    def _ingest(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        if kind == "thread.started":
            self.session_id = record.get("thread_id") or self.session_id
        elif kind == "turn.completed":
            self.usage.add_openai(record.get("usage") or {})
        elif kind == "error":
            self._append(Event(seq=self._next(), kind="text", is_error=True,
                               body=_preview(record.get("message"), 4000)))
        elif kind in ("item.started", "item.completed"):
            item = record.get("item")
            if isinstance(item, dict):
                self._ingest_item(kind == "item.completed", item)

    def _ingest_item(self, completed: bool, item: dict[str, Any]) -> None:
        item_type = item.get("type") or "?"
        item_id = item.get("id")
        if item_type in _CODEX_PROSE_ITEMS:
            if not completed:
                return
            if item_type == "reasoning":
                self._append(Event(seq=self._next(), kind="thinking", message_id=item_id,
                                   body=_preview(item.get("text"), 4000)))
            elif item_type == "agent_message":
                if item_id:
                    self.message_ids.add(item_id)
                self._append(Event(seq=self._next(), kind="text", message_id=item_id,
                                   body=_preview(item.get("text"), 4000)))
            elif item_type == "error":
                self._append(Event(seq=self._next(), kind="text", message_id=item_id,
                                   is_error=True, body=_preview(item.get("message"), 4000)))
            else:
                self._append(Event(seq=self._next(), kind="text", message_id=item_id,
                                   body=_preview(item.get("items"), 4000)))
            return

        name = self._tool_name(item_type, item)
        # An item the tailer first sees completed still gets its call card.
        if not completed or item_id not in self._pending_tools:
            self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
            call = Event(seq=self._next(), kind="tool_use", name=name, tool_use_id=item_id,
                         body=self._tool_input(item_type, item))
            self._append(call)
            if item_id:
                self._pending_tools[item_id] = call
        if not completed:
            return

        if item_id:
            self._pending_tools.pop(item_id, None)
        is_error = item.get("status") == "failed" or (item.get("exit_code") or 0) != 0
        if is_error:
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1
        self._append(Event(seq=self._next(), kind="tool_result", name=name,
                           tool_use_id=item_id, is_error=is_error,
                           body=self._tool_output(item_type, item)))

    @staticmethod
    def _tool_name(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "mcp_tool_call":
            return f"{item.get('server') or 'mcp'}.{item.get('tool') or '?'}"
        return _CODEX_TOOL_NAMES.get(item_type, item_type)

    @staticmethod
    def _tool_input(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "command_execution":
            return _preview(item.get("command"))
        if item_type == "file_change":
            changes = item.get("changes") or []
            return _preview("\n".join(
                f"{c.get('kind', '?')} {c.get('path', '?')}" for c in changes if isinstance(c, dict)
            ))
        if item_type == "web_search":
            return _preview(item.get("query"))
        if item_type == "mcp_tool_call":
            return _preview(item.get("arguments"))
        return _preview({k: v for k, v in item.items() if k not in ("id", "type")})

    @staticmethod
    def _tool_output(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "command_execution":
            output = item.get("aggregated_output") or ""
            code = item.get("exit_code")
            if code not in (None, 0):
                output = f"{output}\n[exit code {code}]" if output else f"[exit code {code}]"
            return _preview(output, 3000)
        if item_type == "mcp_tool_call":
            return _preview(item.get("result") if item.get("result") is not None
                            else item.get("error"), 3000)
        return _preview(item.get("status"), 3000)


def _iso_from_ms(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat(timespec="milliseconds")


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text") or "" for block in content if isinstance(block, dict)
    )


class CodexRolloutReader(TrajectoryReader):
    """Reads a Codex session rollout, `$CODEX_HOME/sessions/**/rollout-*.jsonl`.

    Harbor copies it into `agent/sessions/` when the trial ends, so this is the
    finished-trial view of a Codex run: every record is timestamped, each
    `item_completed` carries the item's start and end, and a `token_count` event
    follows every model response. The transport stream (`codex.txt`) has none of
    that, so a trial switches to this file as soon as it exists.
    """

    def __init__(self, path: Path, model: str | None = None) -> None:
        super().__init__(path)
        self.model = model

    def _ingest(self, record: dict[str, Any]) -> None:
        timestamp = record.get("timestamp")
        if timestamp:
            self.first_timestamp = self.first_timestamp or timestamp
            self.last_timestamp = timestamp
        kind = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return
        if kind == "session_meta":
            self.session_id = payload.get("session_id") or self.session_id
            self.cwd = payload.get("cwd") or self.cwd
            self.version = payload.get("cli_version") or self.version
        elif kind == "turn_context":
            self.model = payload.get("model") or self.model
        elif kind == "event_msg":
            event_type = payload.get("type")
            if event_type == "token_count":
                info = payload.get("info") or {}
                self.usage.add_openai(info.get("last_token_usage") or {})
            elif event_type == "item_completed":
                item = payload.get("item")
                if isinstance(item, dict):
                    self._ingest_item(item, payload, timestamp)

    def _ingest_item(self, item: dict[str, Any], payload: dict[str, Any],
                     timestamp: str | None) -> None:
        item_type = item.get("type") or "?"
        item_id = item.get("id")
        if item_type == "UserMessage":
            body = _content_text(item.get("content"))
            self.first_prompt = self.first_prompt or body
            self._append(Event(seq=self._next(), kind="user", timestamp=timestamp,
                               body=_preview(body, 6000)))
            return
        if item_type == "AgentMessage":
            if item_id:
                self.message_ids.add(item_id)
            self._append(Event(seq=self._next(), kind="text", timestamp=timestamp,
                               message_id=item_id,
                               body=_preview(_content_text(item.get("content")), 4000)))
            return
        if item_type == "Reasoning":
            # Reasoning comes back encrypted; only the summary, when there is one, is readable.
            summary = "\n".join(s for s in item.get("summary_text") or [] if isinstance(s, str))
            if summary.strip():
                self._append(Event(seq=self._next(), kind="thinking", timestamp=timestamp,
                                   message_id=item_id, body=_preview(summary, 4000)))
            return
        if item_type == "ContextCompaction":
            self._append(Event(seq=self._next(), kind="text", timestamp=timestamp,
                               message_id=item_id, body="context compacted"))
            return

        name = self._tool_name(item_type, item)
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        started, completed = payload.get("started_at_ms"), payload.get("completed_at_ms")
        duration = None
        if isinstance(started, (int, float)) and isinstance(completed, (int, float)):
            duration = int(completed - started)
        self._append(Event(seq=self._next(), kind="tool_use",
                           timestamp=_iso_from_ms(started) or timestamp,
                           name=name, tool_use_id=item_id,
                           body=self._tool_input(item_type, item)))
        is_error = item.get("status") == "failed" or (item.get("exit_code") or 0) != 0
        if is_error:
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1
        self._append(Event(seq=self._next(), kind="tool_result", timestamp=timestamp,
                           name=name, tool_use_id=item_id, is_error=is_error,
                           duration_ms=duration, body=self._tool_output(item_type, item)))

    @staticmethod
    def _tool_name(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "CommandExecution":
            return "shell"
        if item_type == "FileChange":
            return "apply_patch"
        if item_type == "Extension":
            return str(item.get("kind") or "extension")
        return item_type

    @staticmethod
    def _tool_input(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "CommandExecution":
            command = item.get("command")
            if isinstance(command, list):
                # `/bin/bash -lc <script>`: the script is the part worth reading.
                command = command[-1] if len(command) == 3 and command[1] == "-lc" \
                    else " ".join(str(part) for part in command)
            return _preview(command)
        if item_type == "FileChange":
            changes = item.get("changes")
            if isinstance(changes, dict):
                return _preview("\n".join(
                    f"{(change or {}).get('type', '?')} {path}" for path, change in changes.items()
                ))
            return _preview(changes)
        return _preview({k: v for k, v in item.items() if k not in ("type", "id")})

    @staticmethod
    def _tool_output(item_type: str, item: dict[str, Any]) -> str:
        if item_type == "CommandExecution":
            output = item.get("aggregated_output") or ""
            code = item.get("exit_code")
            if code not in (None, 0):
                output = f"{output}\n[exit code {code}]" if output else f"[exit code {code}]"
            return _preview(output, 3000)
        if item_type == "FileChange":
            return _preview(item.get("stderr") or item.get("stdout") or item.get("status"), 3000)
        return _preview(item.get("status") or "done", 3000)


# pi's tools take lower-case names; the field worth reading first for each.
_PI_PRIMARY_INPUT: dict[str, str] = {
    "bash": "command",
    "read": "path",
    "write": "content",
    "edit": "newText",
    "grep": "pattern",
    "find": "pattern",
    "ls": "path",
}


class PiReader(TrajectoryReader):
    """Reads a pi session, `agent/pi/sessions/<stamp>_<id>.jsonl`.

    Harbor runs pi with `--session-dir /logs/agent/pi/sessions`, so the session
    is written on the host while the trial runs. Every entry is timestamped; a
    `message` entry carries one whole message: a user prompt, an assistant turn
    (thinking / text / toolCall blocks plus usage and pi's own cost), or a
    toolResult. Compaction and branch summaries carry the usage of the call that
    produced them. Cost is what pi computed from its models.json prices, not the
    PRICING table above, because the driver model is whatever the task's
    models.json declares.
    """

    def __init__(self, path: Path, model: str | None = None) -> None:
        super().__init__(path)
        self.model = model
        self.pi_cost = 0.0

    def _reset(self) -> None:
        super()._reset()
        self.pi_cost = 0.0

    def _ingest(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        timestamp = record.get("timestamp")
        if timestamp:
            self.first_timestamp = self.first_timestamp or timestamp
            self.last_timestamp = timestamp

        if kind == "session":
            self.session_id = record.get("id") or self.session_id
            self.cwd = record.get("cwd") or self.cwd
            version = record.get("version")
            self.version = str(version) if version is not None else self.version
        elif kind == "model_change":
            self.model = record.get("modelId") or self.model
        elif kind == "message":
            message = record.get("message")
            if isinstance(message, dict):
                self._ingest_message(record.get("id"), message, timestamp)
        elif kind in ("compaction", "branch_summary"):
            self._add_usage(record.get("usage"))
            label = "context compacted" if kind == "compaction" else "branch summarized"
            self._append(Event(seq=self._next(), kind="text", timestamp=timestamp,
                               message_id=record.get("id"), body=label))
        elif kind == "custom_message":
            body = record.get("content")
            if isinstance(body, str) and body.strip():
                self._append(Event(seq=self._next(), kind="user", timestamp=timestamp,
                                   body=_preview(body, 6000)))

    def _add_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        # pi's `input` already excludes the cached share.
        self.usage.input += usage.get("input") or 0
        self.usage.output += usage.get("output") or 0
        self.usage.cache_read += usage.get("cacheRead") or 0
        self.usage.cache_write_5m += usage.get("cacheWrite") or 0
        cost = usage.get("cost")
        if isinstance(cost, dict):
            self.pi_cost += cost.get("total") or 0.0

    def _ingest_message(self, entry_id: str | None, message: dict[str, Any],
                        timestamp: str | None) -> None:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            body = content if isinstance(content, str) else _content_text(content)
            if body.strip():
                self.first_prompt = self.first_prompt or body
                self._append(Event(seq=self._next(), kind="user", timestamp=timestamp,
                                   body=_preview(body, 6000)))
        elif role == "assistant":
            self._ingest_assistant_message(entry_id, message, content, timestamp)
        elif role == "toolResult":
            self._ingest_pi_tool_result(message, content, timestamp)
        elif role in ("bashExecution", "custom", "compactionSummary", "branchSummary"):
            body = content if isinstance(content, str) else _content_text(content)
            if body.strip():
                self._append(Event(seq=self._next(), kind="text", timestamp=timestamp,
                                   message_id=entry_id, body=_preview(body, 4000)))

    def _ingest_assistant_message(self, entry_id: str | None, message: dict[str, Any],
                                  content: Any, timestamp: str | None) -> None:
        self.model = message.get("model") or self.model
        if entry_id:
            self.message_ids.add(entry_id)
        self._add_usage(message.get("usage"))

        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("text", "thinking"):
                body = _text_of(block)
                if body.strip():
                    self._append(Event(seq=self._next(), kind=block_type, timestamp=timestamp,
                                       message_id=entry_id, body=_preview(body, 4000)))
            elif block_type == "toolCall":
                name = block.get("name") or "?"
                self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
                event = Event(seq=self._next(), kind="tool_use", timestamp=timestamp,
                              message_id=entry_id, name=name, tool_use_id=block.get("id"),
                              body=self._tool_input(name, block.get("arguments")))
                self._append(event)
                if event.tool_use_id:
                    self._pending_tools[event.tool_use_id] = event

        error = message.get("errorMessage")
        if message.get("stopReason") == "error" or (isinstance(error, str) and error.strip()):
            self._append(Event(seq=self._next(), kind="text", timestamp=timestamp,
                               message_id=entry_id, is_error=True,
                               body=_preview(error or message.get("stopReason"), 4000)))

    def _ingest_pi_tool_result(self, message: dict[str, Any], content: Any,
                               timestamp: str | None) -> None:
        tool_use_id = message.get("toolCallId")
        origin = self._pending_tools.pop(tool_use_id, None) if tool_use_id else None
        name = message.get("toolName") or (origin.name if origin else None)
        is_error = bool(message.get("isError"))
        if is_error and name:
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1
        duration = None
        if origin and origin.timestamp and timestamp:
            duration = _delta_ms(origin.timestamp, timestamp)
        self._append(Event(seq=self._next(), kind="tool_result", timestamp=timestamp,
                           name=name, tool_use_id=tool_use_id, is_error=is_error,
                           duration_ms=duration, body=_preview(_content_text(content), 3000)))

    @staticmethod
    def _tool_input(name: str, value: Any) -> str:
        if not isinstance(value, dict):
            return _preview(value)
        key = _PI_PRIMARY_INPUT.get(name)
        primary = value.get(key) if key else None
        if not isinstance(primary, str) or not primary.strip():
            return _preview(value)
        rest = {k: v for k, v in value.items() if k != key}
        text = primary
        if rest:
            text += "\n\n· " + json.dumps(rest, ensure_ascii=False)[:400]
        return _preview(text)

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["usage"]["cost_usd"] = round(self.pi_cost, 4)
        return result


def _delta_ms(start: str, end: str) -> int | None:
    try:
        a = datetime.fromisoformat(start)
        b = datetime.fromisoformat(end)
    except ValueError:
        return None
    return int((b - a).total_seconds() * 1000)


def find_session_file(agent_dir: Path) -> Path | None:
    """Newest non-subagent session JSONL under a trial's agent/ directory."""
    root = agent_dir / "sessions" / "projects"
    if not root.is_dir():
        return None
    candidates = [
        path for path in root.rglob("*.jsonl")
        if "subagents" not in path.parts
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_codex_rollout(agent_dir: Path) -> Path | None:
    """Newest Codex session rollout under a trial's agent/ directory, present only
    once harbor has copied the session out at the end of the trial."""
    root = agent_dir / "sessions"
    if not root.is_dir():
        return None
    candidates = list(root.rglob("rollout-*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_pi_session(agent_dir: Path) -> Path | None:
    """Newest pi session under a trial's agent/ directory. Harbor points pi's
    `--session-dir` at /logs/agent/pi/sessions, so it is on the host from the
    first message."""
    root = agent_dir / "pi" / "sessions"
    if not root.is_dir():
        return None
    candidates = list(root.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def file_age_seconds(path: Path) -> float | None:
    """Seconds since the file was last written, or None if it is gone."""
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None
