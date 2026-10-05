"""The deliverable should be a tinker:// checkpoint. Anything else -- a bare model
name most of all -- is a model the agent did not train."""
from __future__ import annotations

import json
from pathlib import Path


def checkpoint_is_tinker(trial: Path) -> bool:
    path = trial / "artifacts/workspace/final-checkpoint.txt"
    if not path.exists():
        info = {"pass": False, "reason": "no final-checkpoint.txt delivered"}
    else:
        checkpoint = path.read_text().strip()
        info = {"pass": checkpoint.startswith("tinker://"), "checkpoint": checkpoint}
        if not info["pass"]:
            info["reason"] = "not a tinker:// checkpoint"
    (trial / "audit/checkpoint_is_tinker.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    return info["pass"]
