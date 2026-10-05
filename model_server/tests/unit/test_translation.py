"""Translation layer, tested without tinker installed.

The parsing tests run the real transformers ResponseParser against
QWEN3_5_RESPONSE_TEMPLATE, so a template that stops matching Qwen's grammar fails here
rather than mid-evaluation.
"""

from __future__ import annotations

import json

from model_server.translation import (
    build_chat_completion,
    coerce_tool_call_arguments,
    normalize_model_ref,
    parse_completion,
    sampling_params,
)

#: What apply_chat_template(add_generation_prompt=True) leaves the model sitting at.
PROMPT = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"

QWEN_TOOL_CALL = """\
I'll list the files.
</think>

Sure.

<tool_call>
<function=bash>
<parameter=command>
ls -la
</parameter>
</function>
</tool_call>"""


def _parse(completion: str) -> dict:
    return parse_completion(PROMPT, completion)


# ----------------------------------------------------------------- model ref


def test_model_ref_strips_litellm_provider_prefix() -> None:
    assert normalize_model_ref("openai/tinker://svc/ckpt") == "tinker://svc/ckpt"
    assert normalize_model_ref("Qwen/Qwen3.6-35B-A3B") == "Qwen/Qwen3.6-35B-A3B"


# ------------------------------------------------- sampling parameter forwarding


def test_only_provided_params_are_forwarded() -> None:
    assert sampling_params({"messages": [], "temperature": 0.7}) == {"temperature": 0.7}


def test_nothing_provided_forwards_nothing() -> None:
    """No fallback constants: an empty dict means the model's own defaults apply."""
    assert sampling_params({"messages": []}) == {}


def test_all_supported_params_pass_through_unchanged() -> None:
    knobs = {
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "stop": ["</s>"],
        "seed": 7,
    }
    assert sampling_params(knobs) == knobs


def test_unsupported_params_are_dropped() -> None:
    """SamplingParams would reject unknown keys, taking down the request."""
    assert sampling_params({"frequency_penalty": 0.5, "temperature": 0.3}) == {
        "temperature": 0.3
    }


def test_explicit_null_counts_as_absent() -> None:
    """OpenAI clients routinely send `"temperature": null`; forwarding it would
    override the model default with None."""
    assert sampling_params({"temperature": None, "top_p": 0.9}) == {"top_p": 0.9}


def test_max_completion_tokens_is_max_tokens() -> None:
    """OpenAI's current name for the knob (the OpenAI SDK and langchain send it);
    it wins over the legacy name when both are sent."""
    assert sampling_params({"max_completion_tokens": 8192}) == {"max_tokens": 8192}
    assert sampling_params({"max_tokens": 4096, "max_completion_tokens": 8192}) == {
        "max_tokens": 8192
    }


# -------------------------------------------------------------- request adaptation


def test_string_arguments_become_a_dict() -> None:
    """The chat template iterates arguments as a mapping; the wire carries a JSON
    string. Without this every second agent turn dies in the template."""
    messages = [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        }
    ]

    adapted = coerce_tool_call_arguments(messages)

    assert adapted[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_dict_arguments_pass_through() -> None:
    """Both shapes arrive in practice; dicts are already what the template wants."""
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "b", "arguments": {"x": 1}}}]}
    ]

    assert coerce_tool_call_arguments(messages)[0]["tool_calls"][0]["function"][
        "arguments"
    ] == {"x": 1}


def test_unparseable_arguments_are_left_alone() -> None:
    """Better the template's own error than a guessed shape."""
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": "{{{"}}]}
    ]

    assert coerce_tool_call_arguments(messages)[0]["tool_calls"][0]["function"][
        "arguments"
    ] == "{{{"


def test_messages_without_tool_calls_are_untouched() -> None:
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
    ]

    assert coerce_tool_call_arguments(messages) == messages


def test_caller_messages_are_not_mutated() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "b", "arguments": '{"a": 1}'}}]}
    ]

    coerce_tool_call_arguments(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


# ------------------------------------------------------------ completion parsing


def test_parses_the_shape_the_template_asks_for() -> None:
    parsed = _parse(QWEN_TOOL_CALL)

    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls -la"}}]
    assert parsed["reasoning_content"].strip() == "I'll list the files."
    assert parsed["content"].strip() == "Sure."


def test_parameter_whitespace_is_preserved() -> None:
    """Only the template's own wrapping newline is markup; the rest belongs to the
    argument. Eating it corrupts file contents and heredocs with no error anywhere."""
    parsed = _parse(
        "</think>\n<tool_call>\n<function=edit>\n<parameter=content>\n"
        "    indented\n\ntrailing blank follows\n\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"]["content"] == (
        "    indented\n\ntrailing blank follows\n"
    )


def test_recovers_from_missing_parameter_close_tag() -> None:
    """A dropped </parameter> must not silently lose the argument."""
    parsed = _parse(
        "</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"] == {"command": "ls"}


def test_recovers_when_an_earlier_parameter_is_unclosed() -> None:
    parsed = _parse(
        "</think>\n<tool_call>\n<function=edit>\n<parameter=path>\n/a.py\n"
        "<parameter=content>\nprint(1)\n</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"] == {"path": "/a.py", "content": "print(1)"}


def test_multiple_calls_and_empty_arguments() -> None:
    parsed = _parse(
        "</think>\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\nfirst\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=done>\n</function>\n</tool_call>"
    )

    assert [c["name"] for c in parsed["tool_calls"]] == ["bash", "done"]
    assert parsed["tool_calls"][1]["arguments"] == {}


def test_tool_call_ends_reasoning_without_a_closing_think_tag() -> None:
    """Models go straight from thinking into a call; the call must not be swallowed."""
    parsed = _parse(
        "thinking...\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls"}}]
    assert parsed["reasoning_content"].strip() == "thinking..."


def test_plain_text_yields_no_tool_calls() -> None:
    parsed = _parse("</think>\n\nThe answer is 42.")

    assert "tool_calls" not in parsed
    assert parsed["content"].strip() == "The answer is 42."


def test_text_keeps_its_original_bytes() -> None:
    """No trimming, no invented separators: content is the model's own bytes with the
    grammar tags cut out, the way vLLM's buffer accumulates it."""
    parsed = _parse(
        "</think>\nBefore.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>\nAfter."
    )

    assert parsed["content"] == "\nBefore.\n\nAfter."


def test_whitespace_only_text_fields_are_omitted() -> None:
    parsed = _parse(
        "</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    assert "content" not in parsed
    assert "reasoning_content" not in parsed


def test_content_on_both_sides_of_a_call_is_all_kept() -> None:
    parsed = _parse(
        "</think>\nBefore.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>\nAfter."
    )

    assert parsed["content"].strip() == "Before.\n\nAfter."
    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls"}}]


def test_stray_duplicate_think_close_is_dropped() -> None:
    parsed = _parse("thinking\n</think>\n</think>\n\nThe answer is 42.")

    assert parsed["reasoning_content"].strip() == "thinking"
    assert parsed["content"].strip() == "The answer is 42."


def test_stray_think_open_inside_reasoning_is_dropped() -> None:
    parsed = _parse("thinking <think> more\n</think>\n\nanswer")

    assert "<think>" not in parsed["reasoning_content"]
    assert parsed["content"].strip() == "answer"


def test_consecutive_calls_with_unclosed_first_are_both_recovered() -> None:
    parsed = _parse(
        "</think>\n<tool_call>\n<function=first_tool>\n<parameter=a>\n1\n</parameter>\n"
        "</function>\n"
        "<tool_call>\n<function=second_tool>\n<parameter=b>\n2\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    assert parsed["tool_calls"] == [
        {"name": "first_tool", "arguments": {"a": "1"}},
        {"name": "second_tool", "arguments": {"b": "2"}},
    ]


def test_bare_function_without_wrapper_is_recognized() -> None:
    parsed = _parse(
        "</think>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>"
    )

    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls"}}]


def test_crlf_bytes_survive_in_values() -> None:
    """An XML-parser implementation would normalize \\r\\n to \\n; bytes must survive
    for the re-render round trip."""
    parsed = _parse(
        "</think>\n<tool_call>\n<function=bash>\n<parameter=command>\ndir\r\nls\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"] == {"command": "dir\r\nls"}


def test_json_looking_values_stay_strings() -> None:
    """No schema-driven coercion: "null"/"true"/"123" are the model's literal text."""
    parsed = _parse(
        "</think>\n<tool_call>\n<function=bash>\n"
        "<parameter=a>\nnull\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n"
        "<parameter=c>\n123\n</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"] == {"a": "null", "b": "true", "c": "123"}


def test_value_containing_tool_call_literal_is_kept_whole() -> None:
    parsed = _parse(
        "</think>\n<tool_call>\n<function=edit_file>\n<parameter=content>\n"
        "a <tool_call> b\n</parameter>\n</function>\n</tool_call>"
    )

    assert parsed["tool_calls"][0]["arguments"] == {"content": "a <tool_call> b"}


def test_params_after_a_tag_literal_value_are_still_parsed() -> None:
    """A value quoting </parameter> ends early (format limitation, same as vLLM), but
    the params after it must not be abandoned."""
    parsed = _parse(
        "</think>\n<tool_call>\n<function=bash>\n"
        "<parameter=p0>\ntext </parameter> inside\n</parameter>\n"
        "<parameter=p1>\n3.14\n</parameter>\n</function>\n</tool_call>"
    )

    arguments = parsed["tool_calls"][0]["arguments"]
    assert arguments["p1"] == "3.14"


def test_call_inside_think_block_is_extracted() -> None:
    parsed = _parse(
        "hmm\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call>\nmore think\n</think>\n\nvisible"
    )

    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls"}}]
    assert parsed["reasoning_content"].strip() == "hmm"


def test_generation_truncated_mid_call_salvages_the_partial_call() -> None:
    parsed = _parse("</think>\n<tool_call>\n<function=bash>\n<parameter=command>\ngrep -r foo")

    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "grep -r foo"}}]


def test_never_leaving_the_think_block_is_all_reasoning() -> None:
    """What running out of max_tokens mid-thought looks like. Matches vLLM/SGLang;
    passing half-formed reasoning off as an answer would have the agent act on it."""
    parsed = _parse("still thinking about it")

    assert parsed["reasoning_content"] == "still thinking about it"
    assert not parsed.get("content")


# ---------------------------------------------------------------- response body


def test_tool_calls_shape_and_finish_reason() -> None:
    body = build_chat_completion(
        model="tinker://ckpt",
        parsed=_parse(QWEN_TOOL_CALL),
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=None,
        stop_reason="stop",
    )

    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    # Back to a JSON string on the wire.
    assert json.loads(call["function"]["arguments"]) == {"command": "ls -la"}
    assert choice["message"]["reasoning_content"].strip() == "I'll list the files."


def test_plain_response_shape_and_usage() -> None:
    body = build_chat_completion(
        model="Qwen/Qwen3.6-35B-A3B",
        parsed={"content": "done"},
        prompt_tokens=10,
        completion_tokens=2,
        cost_usd=0.0123,
        stop_reason="stop",
    )

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == "done"
    assert "tool_calls" not in body["choices"][0]["message"]
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
        "cost_usd": 0.0123,
    }


def test_usage_reports_prompt_cache_hits() -> None:
    body = build_chat_completion(
        model="m", parsed={"content": "x"}, prompt_tokens=45_000,
        completion_tokens=100, cached_tokens=43_000, cost_usd=None, stop_reason="stop",
    )

    assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 43_000}


def test_cost_usd_is_carried_into_usage() -> None:
    """A None cost (model absent from the price list) stays a present, null field."""
    priced = build_chat_completion(
        model="m", parsed={"content": "x"}, prompt_tokens=1,
        completion_tokens=1, cost_usd=0.5, stop_reason="stop",
    )
    unpriced = build_chat_completion(
        model="m", parsed={"content": "x"}, prompt_tokens=1,
        completion_tokens=1, cost_usd=None, stop_reason="stop",
    )

    assert priced["usage"]["cost_usd"] == 0.5
    assert unpriced["usage"]["cost_usd"] is None


def test_length_is_reported_when_cut_off() -> None:
    body = build_chat_completion(
        model="m", parsed={"content": "trunc"}, prompt_tokens=1,
        completion_tokens=4096, cost_usd=None, stop_reason="length",
    )

    assert body["choices"][0]["finish_reason"] == "length"


def test_tool_calls_outrank_length() -> None:
    """Reporting "length" would make the agent drop an action it should have run."""
    body = build_chat_completion(
        model="m", parsed=_parse(QWEN_TOOL_CALL), prompt_tokens=1,
        completion_tokens=10, cost_usd=None, stop_reason="length",
    )

    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_call_ids_are_unique() -> None:
    parsed = _parse(
        QWEN_TOOL_CALL + "\n<tool_call>\n<function=done>\n</function>\n</tool_call>"
    )

    body = build_chat_completion(
        model="m", parsed=parsed, prompt_tokens=1, completion_tokens=2,
        cost_usd=None, stop_reason="stop",
    )

    ids = [c["id"] for c in body["choices"][0]["message"]["tool_calls"]]
    assert len(ids) == 2
    assert len(set(ids)) == len(ids)
