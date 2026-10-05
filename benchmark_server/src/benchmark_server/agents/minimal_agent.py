"""Fixed adapter for the minimal agent."""

from __future__ import annotations

from pathlib import Path

from benchmark_server.agents.custom_agent import CustomAgent

ARCHIVE_PATH = Path(__file__).parent / "minimal_agent.tar.gz"


class MinimalAgent(CustomAgent):
    def __init__(self, logs_dir, **kwargs) -> None:
        super().__init__(
            logs_dir,
            archive_path=str(ARCHIVE_PATH),
            install_cmd="bash /agent/install.sh",
            run_cmd="bash /agent/run.sh",
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "minimal"
