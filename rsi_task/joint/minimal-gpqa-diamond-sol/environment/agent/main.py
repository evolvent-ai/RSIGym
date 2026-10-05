from __future__ import annotations

import argparse
from pathlib import Path

from langchain.messages import HumanMessage

from harness import RECURSION_LIMIT, build_agent

TRAJECTORY_PATH = Path("/logs/agent/trajectory.jsonl")


def run_task(instruction: str) -> str:
    agent = build_agent()
    TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_PATH.open("w") as trajectory:
        message = HumanMessage(instruction)
        trajectory.write(message.model_dump_json() + "\n")
        for update in agent.stream(
            {"messages": [message]},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for payload in update.values():
                if payload and "messages" in payload:
                    for message in payload["messages"]:
                        trajectory.write(message.model_dump_json() + "\n")
                        trajectory.flush()
    return message.text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    print(run_task(parser.parse_args().instruction))


if __name__ == "__main__":
    main()
