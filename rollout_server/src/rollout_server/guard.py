"""Request fields this server blocks."""

from __future__ import annotations

from typing import Any

# Tools the caller runs itself; every other type runs provider-side.
CLIENT_TOOL_TYPES = ("function", "custom")


def _is_image_url(value: str) -> bool:
    return not value.lower().startswith("data:")


def _blocked_content_part(part: Any, path: str) -> str | None:
    if not isinstance(part, dict):
        return None

    if part.get("type") == "image_url":
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else None
        if isinstance(url, str) and _is_image_url(url):
            return f"{path}.image_url.url"

    if part.get("type") == "file":
        file = part.get("file")
        if isinstance(file, dict) and file.get("file_id"):
            return f"{path}.file.file_id"

    return None


def _blocked_message_fields(message: Any, path: str) -> list[str]:
    if not isinstance(message, dict):
        return []
    fields = []
    audio = message.get("audio")
    if isinstance(audio, dict) and audio.get("id"):
        fields.append(f"{path}.audio.id")
    content = message.get("content")
    if isinstance(content, list):
        for index, part in enumerate(content):
            blocked = _blocked_content_part(part, f"{path}.content[{index}]")
            if blocked:
                fields.append(blocked)
    return fields


def blocked_fields(payload: dict[str, Any]) -> list[str]:
    fields = []

    for field in ("web_search_options", "plugins"):
        if payload.get(field) is not None:
            fields.append(field)

    audio = payload.get("audio")
    voice = audio.get("voice") if isinstance(audio, dict) else None
    if isinstance(voice, dict) and voice.get("id"):
        fields.append("audio.voice.id")

    tools = payload.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if isinstance(tool, dict) and tool.get("type", "function") not in CLIENT_TOOL_TYPES:
                fields.append(f"tools[{index}].type")

    messages = payload.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            fields.extend(_blocked_message_fields(message, f"messages[{index}]"))

    return fields
