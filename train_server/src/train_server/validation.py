"""Field governance (allow/lock) and the custom-loss import gate."""

from __future__ import annotations

import ast
import sys
from typing import Any, get_args

from tinker.types import AdamParams, LoraConfig

CUSTOM = "custom"
DEFAULT_LOSS_FN = "cross_entropy"
DEFAULT_TRAIN_ON_WHAT = "all_assistant_messages"
DEFAULT_LR_SCHEDULE = "linear"

#: The two nested Tinker objects; a key governs their sub-fields individually, and those
#: sub-fields are exactly the Tinker model's fields -- read them off the type, don't mirror.
_NESTED_MODELS = {"lora_config": LoraConfig, "adam_params": AdamParams}

#: This service's own flat fields (each handled explicitly in _finalize / training.py).
_FLAT_PATHS = {
    "base_model", "loss_fn", "loss_fn_config", "dataset",
    "num_epochs", "batch_size", "max_length", "max_steps", "train_on_what", "lr_schedule",
    "renderer",
}
GOVERNABLE_PATHS = _FLAT_PATHS | {
    f"{group}.{field}" for group, model in _NESTED_MODELS.items() for field in model.model_fields
}


class InputError(ValueError):
    """User-visible validation failure (400)."""

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else errors
        super().__init__("; ".join(self.errors))


def _leaf_paths(config: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for key, value in config.items():
        if key in _NESTED_MODELS and isinstance(value, dict):
            for sub, sub_value in value.items():
                paths[f"{key}.{sub}"] = sub_value
        else:
            paths[key] = value
    return paths


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    if "." in path:
        group, sub = path.split(".", 1)
        target.setdefault(group, {})[sub] = value
    else:
        target[path] = value


def _value_allowed(value: Any, constraint: Any) -> bool:
    if constraint is True:
        return True
    if isinstance(constraint, list):
        return value in constraint
    return False


def resolve_config(caller: Any, allow: Any, lock: dict[str, Any]) -> dict[str, Any]:
    """Enforce the key's allow/lock over the caller's config and return the effective
    config Tinker will run. Raises InputError (400) listing every problem."""
    if not isinstance(caller, dict):
        raise InputError("config must be a JSON object")

    provided = _leaf_paths(caller)
    errors: list[str] = []
    for path, value in provided.items():
        if path not in GOVERNABLE_PATHS:
            errors.append(f"unknown config field: {path}")
        elif path in lock:
            if value != lock[path]:
                errors.append(f"{path} is locked to {lock[path]!r} by this key and cannot be changed")
        elif allow == "*":
            continue
        elif isinstance(allow, dict) and path in allow:
            if not _value_allowed(value, allow[path]):
                errors.append(f"{path}={value!r} is out of this key's allowed values {allow[path]}")
        else:
            errors.append(f"{path} is not settable by this key (not in its allow list)")
    if errors:
        raise InputError(errors)

    # Caller's values first, then lock pinned on top: locked fields always win.
    effective: dict[str, Any] = {}
    for path, value in provided.items():
        _set_path(effective, path, value)
    for path, value in lock.items():
        _set_path(effective, path, value)
    return _finalize(effective)


def _finalize(effective: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError
    from tinker.types.loss_fn_type import LossFnType
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.utils.lr_scheduling import LRSchedule

    errors: list[str] = []
    if not effective.get("base_model"):
        errors.append("base_model is required")
    lora = effective.get("lora_config") or {}
    if "rank" not in lora:
        errors.append("lora_config.rank is required")
    if not effective.get("dataset"):
        errors.append("dataset is required")

    loss_fn = effective.get("loss_fn", DEFAULT_LOSS_FN)
    valid_loss = set(get_args(LossFnType)) | {CUSTOM}
    if loss_fn not in valid_loss:
        errors.append(f"loss_fn={loss_fn!r} must be one of {sorted(valid_loss)}")
    if loss_fn == CUSTOM and effective.get("loss_fn_config"):
        errors.append("loss_fn_config is not used when loss_fn is 'custom'")

    train_on_what = effective.get("train_on_what", DEFAULT_TRAIN_ON_WHAT)
    valid_train_on_what = {member.value for member in TrainOnWhat}
    if train_on_what not in valid_train_on_what:
        errors.append(f"train_on_what={train_on_what!r} must be one of {sorted(valid_train_on_what)}")

    lr_schedule = effective.get("lr_schedule", DEFAULT_LR_SCHEDULE)
    if lr_schedule not in set(get_args(LRSchedule)):
        errors.append(f"lr_schedule={lr_schedule!r} must be one of {sorted(get_args(LRSchedule))}")
    if errors:
        raise InputError(errors)

    # Construct the Tinker types so their own validation runs and defaults fill in, then
    # dump the full config back -- names stay Tinker's, no defaults duplicated here.
    try:
        effective["lora_config"] = LoraConfig(**lora).model_dump()
        effective["adam_params"] = AdamParams(**(effective.get("adam_params") or {})).model_dump()
    except (ValidationError, TypeError) as exc:
        raise InputError(f"invalid Tinker config: {exc}") from exc

    effective["loss_fn"] = loss_fn
    effective["train_on_what"] = train_on_what
    effective["lr_schedule"] = lr_schedule
    return effective


def validate_key_policy(allow: Any, lock: Any) -> None:
    """Reject a malformed allow/lock at key creation, so a typo'd path fails loudly here
    instead of silently never matching later."""
    errors: list[str] = []
    if allow != "*":
        if not isinstance(allow, dict):
            errors.append("allow must be '*' or an object")
        else:
            for path, constraint in allow.items():
                if path not in GOVERNABLE_PATHS:
                    errors.append(f"allow: unknown field path {path!r}")
                elif constraint is not True and not isinstance(constraint, list):
                    errors.append(f"allow[{path}] must be true or a list of values")
    if not isinstance(lock, dict):
        errors.append("lock must be an object")
    else:
        for path in lock:
            if path not in GOVERNABLE_PATHS:
                errors.append(f"lock: unknown field path {path!r}")
    if isinstance(allow, dict) and isinstance(lock, dict):
        overlap = sorted(set(allow) & set(lock))
        if overlap:
            errors.append(f"these fields are in both allow and lock: {overlap}")
    if errors:
        raise InputError(errors)


def check_loss_imports(source: bytes) -> None:
    """Static gate before the custom loss ever runs: torch + standard library imports only."""
    try:
        tree = ast.parse(source.decode("utf-8"), filename="loss.py")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise InputError(f"custom loss does not parse: {exc}") from exc
    allowed = set(sys.stdlib_module_names) | {"torch"}
    errors: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append("custom loss: relative imports are not allowed")
                continue
            modules = [node.module or ""]
        for module in modules:
            if module.split(".", 1)[0] not in allowed:
                errors.append(f"custom loss: import {module!r} is not allowed (torch + stdlib only)")
    if errors:
        raise InputError(errors)
