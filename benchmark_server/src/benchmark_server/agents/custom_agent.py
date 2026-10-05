"""Fixed adapter for user-uploaded agents.

Runs on the server but only uploads the user's code and execs their commands in the
sandbox; the user's own OPENAI_* / model config rides in the agent env, untouched here.
"""

from __future__ import annotations

import uuid
import shlex

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

AGENT_DIR = "/agent"


class CustomAgent(BaseInstalledAgent):
    def __init__(
        self, logs_dir, *, archive_path: str, install_cmd: str, run_cmd: str, **kwargs
    ) -> None:
        self._archive_path = archive_path
        self._install_cmd = install_cmd
        self._run_cmd = run_cmd
        super().__init__(logs_dir, **kwargs)

    @staticmethod
    def name() -> str:
        return "custom"

    async def install(self, environment: BaseEnvironment) -> None:
        remote_archive = f"/tmp/harbor-custom-agent-{uuid.uuid4().hex}.tar.gz"
        await environment.upload_file(self._archive_path, remote_archive)
        chown = ""
        if environment.default_user is not None:
            chown = (
                " && chown -R -- "
                f"{shlex.quote(str(environment.default_user))} {AGENT_DIR}"
            )
        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {AGENT_DIR} && "
                f"tar -xzpf {remote_archive} -C {AGENT_DIR} && "
                f"rm -f {remote_archive}{chown}"
            ),
        )
        await self.exec_as_agent(environment, command=self._install_cmd)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        instr_var = f"HARBOR_CUSTOM_INSTRUCTION_{uuid.uuid4().hex}"
        await self.exec_as_agent(
            environment,
            command=(
                f'printf "%s" "${instr_var}" | {self._run_cmd} '
                f"2>&1 | tee /logs/agent/custom-agent.txt"
            ),
            env={instr_var: instruction},
        )
