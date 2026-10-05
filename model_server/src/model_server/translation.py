"""OpenAI chat-completions <-> raw model text, in both directions.

The prompt is rendered by the model's own chat template and the completion is parsed
by that template's declared inverse -- nothing here hand-approximates the model's
grammar. Pure functions; imports neither tinker nor fastapi.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

#: Every one of these is optional on tinker's SamplingParams, so omitting an absent key
#: genuinely means "model default" rather than a hidden server-side choice.
FORWARDED_SAMPLING_PARAMS = ("max_tokens", "temperature", "top_p", "top_k", "stop", "seed")

#: Inverse of the Qwen3.5 chat template (Qwen3.6 shares it; classic Qwen3 declares
#: JSON-style tool calls instead), in transformers' declarative response_template
#: form. Qwen ships no response_template of its own, so this states its grammar:
#: reasoning in <think>, tool calls as <tool_call><function=NAME><parameter=KEY>.
QWEN3_5_RESPONSE_TEMPLATE: dict[str, Any] = {
    "version": 1,
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            # A tool call ends reasoning even without a closing tag; without the
            # lookahead the whole call is swallowed as reasoning and the agent is
            # handed no action at all.
            "close_pattern": r"</think>|(?=<tool_call>)",
            "optional": True,
        },
        "tool_calls": {
            "open_pattern": r"(?:<tool_call>\s*)?<function=(?P<name>[^>]+)>",
            "close_pattern": r"</function>\s*</tool_call>|</function>",
            "content": "xml-inline",
            "content_args": {
                # \n? strips only the template's own wrapping newlines; anything more
                # would eat indentation and trailing blank lines that belong to the
                # argument. The trailing alternation recovers a value whose
                # </parameter> the model omitted.
                "tag_pattern": r"<parameter=(?P<key>[^>]+)>\n?(?P<value>.*?)\n?"
                r"(?:</parameter>|(?=<parameter=)|\Z)"
            },
            "repeats": True,
            "transform": {"name": "{name}", "arguments": "{content}"},
        },
        "content": {"optional": True},
    },
}


def normalize_model_ref(model: str) -> str:
    """litellm-side configs name us ``openai/<ref>``; the wire may carry the prefix."""
    return model.removeprefix("openai/").strip()


def sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    """The sampling knobs the caller actually set; absent stays absent, null is absent."""
    params = {
        key: payload[key] for key in FORWARDED_SAMPLING_PARAMS if payload.get(key) is not None
    }
    # OpenAI's current name for max_tokens; it wins when both are sent.
    if payload.get("max_completion_tokens"):
        params["max_tokens"] = payload["max_completion_tokens"]
    return params


def coerce_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn ``function.arguments`` from a JSON string into a dict.

    The one adaptation the chat template needs: it iterates the arguments as a mapping,
    while the OpenAI wire format (and litellm's ``Function.arguments: str``) carries a
    JSON string. Unparseable values are left as-is so the template's own error surfaces
    instead of a guess made here. Never mutates the caller's messages.
    """
    adapted: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            adapted.append(message)
            continue
        rewritten = []
        for call in tool_calls:
            function = (call or {}).get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    call = {**call, "function": {**function, "arguments": parsed}}
            rewritten.append(call)
        adapted.append({**message, "tool_calls": rewritten})
    return adapted


def parse_completion(prompt_text: str, completion: str) -> dict[str, Any]:
    """Split a raw Qwen completion into content / reasoning_content / tool_calls.

    ``prompt_text`` is the rendered prompt: it ends part-way into ``<think>``, and the
    parser must see that to know the assistant message is already open.

    Assembled from the parser's event stream rather than its summary dict: text can
    appear both before and after a tool call, and the summary keeps only the last
    region. Stray duplicate ``</think>``/``<think>`` tags are dropped the way vLLM's
    parser absorbs them.
    """
    from transformers.utils.chat_parsing import ResponseParser

    parser = ResponseParser(QWEN3_5_RESPONSE_TEMPLATE, prefix=prompt_text)
    events = parser.feed(completion)
    _, final_events = parser.finalize()

    reasoning = ""
    content = ""
    tool_calls: list[dict[str, Any]] = []
    for event in [*events, *final_events]:
        match event.get("type"), event.get("field"):
            case ("region_close", "tool_calls"):
                tool_calls.append(event["value"])
            case ("region_chunk", "reasoning_content"):
                reasoning += event["text"]
            case ("region_chunk", "content"):
                content += event["text"]

    reasoning = reasoning.replace("<think>", "")
    content = content.replace("</think>", "")

    message: dict[str, Any] = {}
    if reasoning.strip():
        message["reasoning_content"] = reasoning
    if content.strip():
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def plain_completion(completion: str) -> dict[str, Any]:
    """For model families we have no response template for: no tool parsing."""
    return {"content": completion}


def build_chat_completion(
    *,
    model: str,
    parsed: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cost_usd: float | None,
    stop_reason: str | None,
) -> dict[str, Any]:
    """Wrap a parsed assistant message in an OpenAI ``chat.completion`` body."""
    message: dict[str, Any] = {"role": "assistant", "content": parsed.get("content") or ""}
    if parsed.get("reasoning_content"):
        message["reasoning_content"] = parsed["reasoning_content"]

    finish_reason = "length" if stop_reason == "length" else "stop"
    calls = parsed.get("tool_calls") or []
    if calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                # Back to a JSON string: that is what OpenAI clients expect on the wire.
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments") or {}),
                },
            }
            for call in calls
        ]
        # A complete tool call outranks hitting the token ceiling; reporting "length"
        # would make the agent drop an action it should have run.
        finish_reason = "tool_calls"

    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "cost_usd": cost_usd,
        },
    }
