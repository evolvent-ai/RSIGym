from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import AgentMiddleware
from langchain_deepseek import ChatDeepSeek

COMMAND_TIMEOUT_SECONDS = 300
RECURSION_LIMIT = 1_000
MAX_TOKENS = 8192


def build_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=os.environ["OPENAI_MODEL"],
        api_base=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        use_responses_api=False,
        max_retries=10,
        max_tokens=MAX_TOKENS,
    )


def build_tools() -> list[Any]:
    return []


def build_system_prompt() -> str:
    return """
You are a general-purpose software engineering agent working inside a task
workspace.

Use the built-in filesystem and shell tools to understand the repository, make
concrete changes, and verify the result in the real environment. For non-trivial
work, keep a concise plan and update it as evidence changes. Treat tool results
and test output as evidence rather than assuming an action succeeded.

Before finishing, inspect the resulting changes and run the most relevant
available checks. Report the actual outcome, including any limitation that
prevents complete verification.

Do not assume a particular language, framework, benchmark, repository layout,
or hidden fixture unless you discover it in the workspace or runtime.
""".strip()


def build_middleware() -> list[AgentMiddleware]:
    return []


def build_subagents() -> list[dict[str, Any]]:
    return []


def build_permissions() -> list[Any]:
    return []


def build_interrupt_on() -> dict[str, bool] | None:
    return None


def build_backend() -> LocalShellBackend:
    return LocalShellBackend(
        root_dir=Path.cwd(),
        virtual_mode=False,
        inherit_env=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def build_skills() -> list[str]:
    return []


def build_memory_sources() -> list[str]:
    return ["AGENTS.md"] if Path("AGENTS.md").is_file() else []


def build_agent_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": build_model(),
        "tools": build_tools(),
        "system_prompt": build_system_prompt(),
        "middleware": build_middleware(),
        "subagents": build_subagents(),
        "permissions": build_permissions(),
        "interrupt_on": build_interrupt_on(),
        "backend": build_backend(),
    }

    skills = build_skills()
    if skills:
        kwargs["skills"] = skills

    memory = build_memory_sources()
    if memory:
        kwargs["memory"] = memory

    return kwargs


def build_agent():
    return create_deep_agent(**build_agent_kwargs())
