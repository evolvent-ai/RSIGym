"""Red-line checks for one trial: run.py <trial-dir>."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_judge import agent_judge

CHECKS = (agent_judge,)


def main() -> None:
    trial = Path(sys.argv[1])
    load_dotenv()
    audit = trial / "audit"
    shutil.rmtree(audit, ignore_errors=True)  # never reuse a previous run's output
    audit.mkdir()
    for check in CHECKS:
        try:
            print(f"check {check.__name__}: {'pass' if check(trial) else 'FAIL'}")
        except Exception as error:
            print(f"check {check.__name__} errored: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
