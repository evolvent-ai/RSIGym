"""Run submission and execution: governance in front, the training loop behind.

Training runs in a thread pool because Tinker's training client is synchronous; each run
carries a cancel Event checked between accumulation steps.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from train_server.config import Settings
from train_server.datasets import NAME_RE
from train_server.db import Database
from train_server.pricing import estimate_cost_usd
from train_server.training import TrainingCancelled, train
from train_server.validation import CUSTOM, InputError, check_loss_imports, resolve_config

logger = logging.getLogger("train_server")


class AuthError(Exception):
    """The auth server refused the next step or could not take the bill."""


class RunService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        trainer: Callable[..., str] | None = None,
        auth: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._trainer = trainer or train
        self._auth = auth or httpx.Client(timeout=10)
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_runs)
        self._runs: dict[str, tuple[Future[None], threading.Event]] = {}

    def submit(
        self,
        *,
        key: str,
        policy: dict[str, Any],
        config: Any,
        dataset_bytes: bytes | None,
        loss_bytes: bytes | None,
    ) -> str:
        effective = resolve_config(config, policy.get("allow", {}), policy.get("lock", {}))

        if effective["loss_fn"] == CUSTOM:
            if loss_bytes is None:
                raise InputError("loss_fn=custom requires an uploaded loss file")
            check_loss_imports(loss_bytes)
            effective["loss_sha256"] = _sha256(loss_bytes)
        elif loss_bytes is not None:
            raise InputError("a loss file was uploaded but loss_fn is not 'custom'")

        dataset = effective["dataset"]
        if dataset == CUSTOM:
            if dataset_bytes is None:
                raise InputError("dataset=custom requires an uploaded data file")
            effective["data_sha256"] = _sha256(dataset_bytes)
        else:
            if dataset_bytes is not None:
                raise InputError("data file uploaded but dataset is not 'custom'")
            if not NAME_RE.match(dataset):
                raise InputError(f"invalid dataset name {dataset!r}")
            if not self._settings.dataset_file(dataset).exists():
                raise InputError(f"dataset {dataset!r} is not registered")

        run_id = str(uuid.uuid4())
        run_dir = self._settings.run_dir(run_id)
        try:
            run_dir.mkdir(parents=True)
            loss_path = None
            if loss_bytes is not None:
                loss_path = run_dir / "loss.py"
                loss_path.write_bytes(loss_bytes)
            if dataset == CUSTOM:
                data_path = run_dir / "data.jsonl"
                data_path.write_bytes(dataset_bytes)  # type: ignore[arg-type]
            else:
                data_path = self._settings.dataset_file(dataset)

            self._db.create_run(run_id, effective)
            cancel_event = threading.Event()
            future = self._executor.submit(
                self._run, run_id, key, effective, data_path, loss_path, cancel_event
            )
            self._runs[run_id] = (future, cancel_event)
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return run_id

    def cancel(self, run_id: str) -> bool:
        entry = self._runs.get(run_id)
        if entry is None:
            return False
        future, cancel_event = entry
        cancel_event.set()
        if future.cancel():
            self._db.update_run(run_id, status="cancelled", finished=True)
        return True

    def shutdown(self) -> None:
        for run_id in list(self._runs):
            self.cancel(run_id)
        self._executor.shutdown(wait=True)
        self._auth.close()

    def _run(
        self,
        run_id: str,
        key: str,
        config: dict[str, Any],
        data_path: Path,
        loss_path: Path | None,
        cancel_event: threading.Event,
    ) -> None:
        try:
            if cancel_event.is_set():
                self._db.update_run(run_id, status="cancelled", finished=True)
                return
            self._db.update_run(run_id, status="running", started=True)

            # custom loss runs each token through a forward (logprobs) and a backward,
            # so Tinker bills it at 2x the tokens we submit once.
            cost_multiplier = 2 if config.get("loss_fn") == CUSTOM else 1

            def on_start(total_steps: int) -> None:
                self._db.update_run(
                    run_id,
                    num_steps=total_steps,
                    cost_usd=estimate_cost_usd(config["base_model"], 0),
                )

            def before_step() -> None:
                try:
                    response = self._auth.post(
                        f"{self._settings.auth_server_url}/v1/verify",
                        json={"key": key, "service": "train_server"},
                    )
                except httpx.HTTPError as exc:
                    raise AuthError(f"auth server unreachable: {exc}") from exc
                if response.status_code != 200:
                    raise AuthError(response.json()["detail"])

            def after_step(step: int, metrics: list[dict[str, Any]]) -> None:
                trained_tokens = cost_multiplier * sum(m["num_tokens"] for m in metrics)
                self._db.update_run(
                    run_id,
                    step=step,
                    step_metrics=metrics,
                    cost_usd=estimate_cost_usd(config["base_model"], trained_tokens),
                )
                step_tokens = cost_multiplier * metrics[-1]["num_tokens"]
                try:
                    response = self._auth.post(
                        f"{self._settings.auth_server_url}/v1/bill",
                        json={
                            "key": key,
                            "service": "train_server",
                            "cost_usd": estimate_cost_usd(config["base_model"], step_tokens),
                            "ref": f"{run_id} step {step}",
                        },
                    )
                except httpx.HTTPError as exc:
                    raise AuthError(f"auth server unreachable: {exc}") from exc
                if response.status_code != 200:
                    raise AuthError(response.json()["detail"])

            checkpoint = self._trainer(
                run_id=run_id,
                config=config,
                data_path=data_path,
                loss_path=loss_path,
                cancel_event=cancel_event,
                timeout_seconds=self._settings.run_timeout_seconds,
                on_start=on_start,
                before_step=before_step,
                after_step=after_step,
            )
            self._db.update_run(run_id, status="succeeded", checkpoint=checkpoint, finished=True)
            logger.info("run %s succeeded", run_id)
        except TrainingCancelled:
            self._db.update_run(run_id, status="cancelled", finished=True)
        except Exception as exc:
            logger.exception("run %s failed", run_id)
            self._db.update_run(
                run_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished=True
            )
        finally:
            self._runs.pop(run_id, None)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
