"""A fake trainer so the suite never touches Tinker: records one step, returns a
checkpoint. Data building and the real loop are the cookbook's, exercised end-to-end
against a real Tinker key, not here."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


def fake_trainer(
    *,
    run_id: str,
    config: dict[str, Any],
    data_path: Path,
    loss_path: Path | None,
    cancel_event: threading.Event,
    timeout_seconds: float,
    on_start: Any,
    before_step: Any,
    after_step: Any,
) -> str:
    on_start(1)
    before_step()
    after_step(1, [{"step": 1, "nll": 0.5, "metrics": {}, "num_tokens": 12}])
    return "tinker://fake/run/checkpoint"
