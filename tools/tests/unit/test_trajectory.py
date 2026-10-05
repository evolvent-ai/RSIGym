"""Parser tests, focused on the two ways this format misleads a naive reader:
one content block per line, and usage repeated on every line of a message.
"""

from __future__ import annotations

import json

import pytest

from rsiwatch.trajectory import TrajectoryReader, Usage, price_for


def line(**fields):
    base = {"isSidechain": False, "timestamp": "2026-08-15T08:20:10.000Z",
            "sessionId": "s1", "cwd": "/workspace", "version": "2.1.0"}
    return json.dumps({**base, **fields})


def assistant(message_id, block, usage=None, **kw):
    message = {"id": message_id, "model": "claude-opus-5", "role": "assistant",
               "content": [block]}
    if usage is not None:
        message["usage"] = usage
    return line(type="assistant", message=message, **kw)


USAGE = {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 900,
         "cache_creation": {"ephemeral_5m_input_tokens": 500,
                            "ephemeral_1h_input_tokens": 0}}


def write(path, *records):
    path.write_text("".join(r + "\n" for r in records), encoding="utf-8")


def test_usage_counted_once_per_message(tmp_path):
    """Claude Code repeats `usage` on every line of a message; summing per line
    is the 3-4x overcount this parser exists to avoid."""
    path = tmp_path / "s.jsonl"
    write(
        path,
        assistant("m1", {"type": "thinking", "thinking": "hm"}, USAGE),
        assistant("m1", {"type": "text", "text": "hello"}, USAGE),
        assistant("m1", {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "ls"}}, USAGE),
    )
    reader = TrajectoryReader(path)
    assert reader.poll()

    assert reader.usage.output == 40      # not 120
    assert reader.usage.input == 100
    assert reader.usage.cache_read == 900
    assert len(reader.message_ids) == 1
    assert len(reader.events) == 3        # blocks are still distinct events


def test_incremental_matches_one_shot(tmp_path):
    path = tmp_path / "s.jsonl"
    records = [
        assistant("m1", {"type": "text", "text": "a"}, USAGE),
        assistant("m2", {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "ls"}}, USAGE),
        line(type="user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            toolUseResult={"stdout": "ok", "stderr": "", "interrupted": False}),
    ]

    incremental = TrajectoryReader(path)
    for i in range(1, len(records) + 1):
        write(path, *records[:i])
        incremental.poll()

    write(path, *records)
    one_shot = TrajectoryReader(path)
    one_shot.poll()

    assert len(incremental.events) == len(one_shot.events)
    assert incremental.usage.total == one_shot.usage.total
    assert incremental.tool_counts == one_shot.tool_counts


def test_partial_trailing_line_is_buffered(tmp_path):
    """Session lines are tens of KB; reading one mid-write must not lose it."""
    path = tmp_path / "s.jsonl"
    full = assistant("m1", {"type": "text", "text": "complete"}, USAGE)
    path.write_text(full[: len(full) // 2], encoding="utf-8")

    reader = TrajectoryReader(path)
    reader.poll()
    assert reader.events == []            # nothing emitted from half a line

    path.write_text(full + "\n", encoding="utf-8")
    reader.poll()
    assert len(reader.events) == 1
    assert reader.events[0].body == "complete"


def test_truncation_rebuilds_state(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, assistant("m1", {"type": "text", "text": "one"}, USAGE),
          assistant("m2", {"type": "text", "text": "two"}, USAGE))
    reader = TrajectoryReader(path)
    reader.poll()
    assert len(reader.events) == 2

    write(path, assistant("m9", {"type": "text", "text": "fresh"}, USAGE))
    reader.poll()
    assert len(reader.events) == 1
    assert reader.usage.output == 40      # not carried over from the old file


def test_tool_result_pairs_and_flags_errors(tmp_path):
    path = tmp_path / "s.jsonl"
    write(
        path,
        assistant("m1", {"type": "tool_use", "id": "t1", "name": "Read",
                         "input": {"file_path": "/nope"}}, USAGE),
        line(type="user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "Error: File does not exist", "is_error": True}]},
            toolUseResult="Error: File does not exist"),
    )
    reader = TrajectoryReader(path)
    reader.poll()

    result = reader.events[-1]
    assert result.kind == "tool_result"
    assert result.name == "Read"          # name carried over from the tool_use
    assert result.is_error
    assert reader.tool_errors == {"Read": 1}


def test_bash_stderr_only_counts_as_error(tmp_path):
    path = tmp_path / "s.jsonl"
    write(
        path,
        assistant("m1", {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "false"}}, USAGE),
        line(type="user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "boom"}]},
            toolUseResult={"stdout": "", "stderr": "boom", "interrupted": False}),
    )
    reader = TrajectoryReader(path)
    reader.poll()
    assert reader.events[-1].is_error
    assert "boom" in reader.events[-1].body


def test_bash_command_rendered_as_raw_text(tmp_path):
    """A shell command shown as escaped JSON is unreadable; show it verbatim."""
    path = tmp_path / "s.jsonl"
    write(path, assistant("m1", {"type": "tool_use", "id": "t1", "name": "Bash",
                                 "input": {"command": "python3 -c 'x'\nls -la",
                                           "timeout": 900}}, USAGE))
    reader = TrajectoryReader(path)
    reader.poll()

    body = reader.events[0].body
    assert body.startswith("python3 -c 'x'\nls -la")   # real newline, not \\n
    assert '\\n' not in body
    assert "900" in body                                # secondary args retained


def test_unknown_tool_falls_back_to_json(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, assistant("m1", {"type": "tool_use", "id": "t1", "name": "Mystery",
                                 "input": {"a": 1}}, USAGE))
    reader = TrajectoryReader(path)
    reader.poll()
    assert '"a": 1' in reader.events[0].body


def test_sidechain_excluded(tmp_path):
    """Subagent transcripts are a separate stream from the main trajectory."""
    path = tmp_path / "s.jsonl"
    write(path,
          assistant("m1", {"type": "text", "text": "main"}, USAGE),
          assistant("m2", {"type": "text", "text": "sub"}, USAGE, isSidechain=True))
    reader = TrajectoryReader(path)
    reader.poll()
    assert len(reader.events) == 1
    assert reader.usage.output == 40


def test_malformed_line_is_skipped(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("{not json\n" + assistant("m1", {"type": "text", "text": "ok"}, USAGE)
                    + "\n", encoding="utf-8")
    reader = TrajectoryReader(path)
    reader.poll()
    assert len(reader.events) == 1


@pytest.mark.parametrize("model,expected_input", [
    ("claude-opus-5", 5.0),
    ("vibe-claude-sub2api-opus-5[1m]", 5.0),   # the gateway alias these trials use
    ("claude-sonnet-5", 3.0),
    ("claude-haiku-4-5", 1.0),
    (None, 5.0),
])
def test_pricing_tolerates_gateway_aliases(model, expected_input):
    assert price_for(model)["input"] == expected_input


def test_cost_and_cache_hit_rate():
    usage = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000)
    assert usage.cost_usd("claude-opus-5") == pytest.approx(5.0 + 25.0 + 0.5)
    # 1M cache-read against 1M uncached input.
    assert usage.as_dict()["cache_hit_rate"] == pytest.approx(0.5)


def test_empty_usage_is_free():
    assert Usage().cost_usd("claude-opus-5") == 0.0
    assert Usage().as_dict()["cache_hit_rate"] == 0.0
