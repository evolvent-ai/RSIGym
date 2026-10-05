"""The built-in minimal agent, and the seed archive it must keep matching."""

from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

from benchmark_server.agents.minimal_agent import ARCHIVE_PATH, MinimalAgent

TASK_SEED = Path(__file__).parents[3] / "rsi_task/harness/minimal-swe-verified/environment"


def seed_files() -> list[Path]:
    return sorted(
        [TASK_SEED / "install.sh", TASK_SEED / "run.sh", *(TASK_SEED / "agent").iterdir()],
        key=lambda path: path.name,
    )


def build_archive(destination: Path) -> None:
    """Rebuild the committed archive from the task seed. Byte-for-byte reproducible:
    every timestamp and owner is pinned, so a rebuilt archive equals the committed one
    whenever the seed has not moved."""

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with (
        open(destination, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in seed_files():
            archive.add(path, arcname=path.name, filter=normalize)


def test_the_archive_is_the_seed_the_task_hands_its_agent(tmp_path: Path) -> None:
    rebuilt = tmp_path / "minimal_agent.tar.gz"
    build_archive(rebuilt)

    assert rebuilt.read_bytes() == ARCHIVE_PATH.read_bytes()


def test_the_archive_carries_the_seed_flat() -> None:
    with tarfile.open(ARCHIVE_PATH) as archive:
        assert sorted(archive.getnames()) == [path.name for path in seed_files()]


def test_the_commands_match_the_official_eval_config(tmp_path: Path) -> None:
    agent = MinimalAgent(tmp_path)

    assert agent.name() == "minimal"
    assert agent._install_cmd == "bash /agent/install.sh"
    assert agent._run_cmd == "bash /agent/run.sh"
