"""Tool-call coercion, reasoning folding, and tool-spec prefixing (the pure-logic pieces
of training.py; the loop itself is exercised end-to-end against real Tinker, not here)."""

from __future__ import annotations

from typing import Any

from tinker_cookbook.renderers.base import ToolCall

from train_server.training import _render_messages, _to_messages


class _FakeRenderer:
    """Stands in for a real renderer: records the tools/system_prompt it was handed."""

    seen_tools: list[dict[str, Any]]

    def create_conversation_prefix_with_tools(
        self, tools: list[dict[str, Any]], system_prompt: str = ""
    ) -> list[dict[str, Any]]:
        self.seen_tools = tools
        return [{"role": "system", "content": f"sys={system_prompt!r} tools={len(tools)}"}]


WRAPPED_TOOL = {"type": "function", "function": {"name": "t", "description": "", "parameters": {}}}


def test_plain_messages_pass_through() -> None:
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert _to_messages(msgs) == msgs


def test_dict_tool_calls_become_toolcall_objects() -> None:
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "id": "c1",
                         "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}]},
    ]
    call = _to_messages(msgs)[1]["tool_calls"][0]
    assert isinstance(call, ToolCall)
    assert call.function.name == "get_weather"
    assert call.function.arguments == '{"city": "SF"}'
    assert call.id == "c1"


def test_dict_arguments_are_serialized_to_json_string() -> None:
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}]}]
    call = _to_messages(msgs)[0]["tool_calls"][0]
    assert call.function.arguments == '{"a": 2, "b": 3}'


def test_reasoning_content_becomes_a_leading_thinking_part() -> None:
    msgs = [{"role": "assistant", "reasoning_content": "check the tests first",
             "content": "Fixed the writer."}]
    assert _to_messages(msgs) == [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "check the tests first"},
        {"type": "text", "text": "Fixed the writer."},
    ]}]


def test_reasoning_with_empty_content_is_thinking_only() -> None:
    msgs = [{"role": "assistant", "reasoning_content": "look around", "content": "",
             "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]}]
    out = _to_messages(msgs)[0]
    assert out["content"] == [{"type": "thinking", "thinking": "look around"}]
    assert isinstance(out["tool_calls"][0], ToolCall)


def test_null_content_on_a_tool_call_turn_becomes_empty() -> None:
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]}]
    out = _to_messages(msgs)[0]
    assert out["content"] == []
    assert isinstance(out["tool_calls"][0], ToolCall)


def test_a_missing_content_key_becomes_empty() -> None:
    msgs = [{"role": "assistant",
             "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]}]
    out = _to_messages(msgs)[0]
    assert out["content"] == []
    assert isinstance(out["tool_calls"][0], ToolCall)


def test_empty_content_becomes_empty_parts() -> None:
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]}]
    assert _to_messages(msgs)[0]["content"] == []


def test_null_content_with_reasoning_is_thinking_only() -> None:
    msgs = [{"role": "assistant", "reasoning_content": "look around", "content": None,
             "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]}]
    assert _to_messages(msgs)[0]["content"] == [{"type": "thinking", "thinking": "look around"}]


def test_null_content_on_other_roles_is_left_alone() -> None:
    msgs = [{"role": "tool", "content": None}]
    assert _to_messages(msgs) == msgs


def test_empty_reasoning_is_left_alone() -> None:
    msgs = [{"role": "assistant", "reasoning_content": "", "content": "hi"}]
    assert _to_messages(msgs) == msgs


def test_empty_tool_calls_key_is_dropped() -> None:
    msgs = [{"role": "assistant", "content": "Done.", "tool_calls": []}]
    assert _to_messages(msgs) == [{"role": "assistant", "content": "Done."}]


def test_non_assistant_reasoning_content_is_left_alone() -> None:
    msgs = [{"role": "user", "reasoning_content": "not a thing", "content": "hi"}]
    assert _to_messages(msgs) == msgs


def test_preserve_thinking_renderer_is_registered() -> None:
    from types import SimpleNamespace

    from tinker_cookbook.renderers import get_renderer

    renderer = get_renderer("qwen3_5_preserve_thinking", SimpleNamespace(name_or_path="x"))
    assert renderer.strip_thinking_from_history is False


def test_no_tools_is_plain_messages() -> None:
    record = {"messages": [{"role": "user", "content": "hi"}]}
    assert _render_messages(record, _FakeRenderer()) == [{"role": "user", "content": "hi"}]


def test_openai_tools_pass_through_verbatim_then_prefixed() -> None:
    renderer = _FakeRenderer()
    record = {"messages": [{"role": "user", "content": "hi"}], "tools": [WRAPPED_TOOL]}
    out = _render_messages(record, renderer)
    assert renderer.seen_tools == [WRAPPED_TOOL]  # OpenAI wire form, wrapper kept
    assert out[0] == {"role": "system", "content": "sys='' tools=1"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_tools_fold_a_leading_system_message_into_the_prefix() -> None:
    record = {
        "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}],
        "tools": [WRAPPED_TOOL],
    }
    out = _render_messages(record, _FakeRenderer())
    assert out[0] == {"role": "system", "content": "sys='be nice' tools=1"}
    assert [m["role"] for m in out] == ["system", "user"]
