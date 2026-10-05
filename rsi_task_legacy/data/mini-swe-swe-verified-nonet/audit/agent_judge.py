"""Judge agent over the trial's evidence; writes agent_judge.json
(see agent_judge_prompt.md)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query

MODEL = "bedrock-claude-opus-5"


async def _run(prompt: str, cwd: Path) -> None:
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        # All tools; the sandbox and acceptEdits keep writes inside the cwd.
        permission_mode="acceptEdits",
        sandbox={
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        },
        model=MODEL,
        env={
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "CLAUDE_CONFIG_DIR": str(cwd / ".claude"),
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        },
    )
    with (cwd / "agent_judge.log").open("w", encoding="utf-8") as log:
        async for message in query(prompt=prompt, options=options):
            log.write(repr(message) + "\n")


def agent_judge(trial: Path) -> bool:
    audit = trial / "audit"
    shutil.copyfile(trial / "agent" / "trajectory.json", audit / "trajectory.json")
    shutil.copyfile(Path(__file__).parent / "test_data.json", audit / "test_data.json")

    prompt = (Path(__file__).parent / "agent_judge_prompt.md").read_text()
    asyncio.run(_run(prompt, audit))

    info = json.loads((audit / "agent_judge.json").read_text())
    print(f"judge: {info['justification']}")
    return not info["hacked"]
