"""Governance (allow/lock) and the custom-loss import gate."""

from __future__ import annotations

import pytest

from train_server.validation import (
    InputError,
    check_loss_imports,
    resolve_config,
    validate_key_policy,
)

BASE = {"base_model": "Qwen/Qwen3-8B", "lora_config": {"rank": 8}, "dataset": "d"}


def test_star_allow_fills_defaults() -> None:
    effective = resolve_config(BASE, "*", {})
    assert effective["base_model"] == "Qwen/Qwen3-8B"
    assert effective["loss_fn"] == "cross_entropy"
    assert effective["train_on_what"] == "all_assistant_messages"
    assert effective["lr_schedule"] == "linear"
    # AdamParams defaults come from Tinker, not hardcoded here.
    assert effective["adam_params"]["learning_rate"] == pytest.approx(1e-4)
    assert effective["lora_config"]["rank"] == 8
    # num_epochs / batch_size / max_length have no default; absent stays absent.
    assert "num_epochs" not in effective
    assert "max_length" not in effective


def test_lock_supplies_locked_fields() -> None:
    effective = resolve_config(
        {"lora_config": {"rank": 8}, "loss_fn": "custom"},
        {"loss_fn": ["cross_entropy", "custom"], "lora_config.rank": True},
        {"base_model": "Qwen/Qwen3-8B", "dataset": "blessed"},
    )
    assert effective["dataset"] == "blessed"
    assert effective["loss_fn"] == "custom"


def test_setting_a_locked_field_is_rejected() -> None:
    with pytest.raises(InputError) as exc:
        resolve_config({**BASE, "loss_fn": "ppo"}, "*", {"loss_fn": "cross_entropy"})
    assert any("locked" in e for e in exc.value.errors)


def test_setting_locked_field_to_same_value_is_ok() -> None:
    effective = resolve_config({**BASE, "loss_fn": "cross_entropy"}, "*", {"loss_fn": "cross_entropy"})
    assert effective["loss_fn"] == "cross_entropy"


def test_field_outside_allow_is_rejected() -> None:
    with pytest.raises(InputError):
        resolve_config({**BASE, "num_epochs": 50}, {"dataset": ["*"]}, {})


def test_value_outside_enum_is_rejected() -> None:
    with pytest.raises(InputError) as exc:
        resolve_config({**BASE, "loss_fn": "ppo"}, {"loss_fn": ["cross_entropy"]}, {})
    assert any("cross_entropy" in e for e in exc.value.errors)  # message names the allowed set


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(InputError) as exc:
        resolve_config({**BASE, "nonsense": 1}, "*", {})
    assert any("unknown" in e for e in exc.value.errors)


def test_missing_required_base_model() -> None:
    with pytest.raises(InputError) as exc:
        resolve_config(
            {"lora_config": {"rank": 8}, "dataset": "d"},
            {"lora_config.rank": True, "dataset": True},
            {},
        )
    assert any("base_model" in e for e in exc.value.errors)


def test_dataset_allow_list_is_literal() -> None:
    lock = {"base_model": "m", "lora_config.rank": 8}
    resolve_config({"dataset": "blessed"}, {"dataset": ["blessed", "custom"]}, lock)
    with pytest.raises(InputError):
        resolve_config({"dataset": "other"}, {"dataset": ["blessed"]}, lock)
    resolve_config({"dataset": "anything"}, {"dataset": True}, lock)


def test_loss_imports_allow_torch_and_stdlib() -> None:
    check_loss_imports(b"import torch\nimport os\nfrom math import log\n")


def test_loss_imports_reject_third_party() -> None:
    with pytest.raises(InputError):
        check_loss_imports(b"import requests\n")


def test_loss_imports_reject_syntax_error() -> None:
    with pytest.raises(InputError):
        check_loss_imports(b"def loss_fn(:\n")


def test_invalid_train_on_what_and_lr_schedule_rejected() -> None:
    with pytest.raises(InputError):
        resolve_config({**BASE, "train_on_what": "nonsense"}, "*", {})
    with pytest.raises(InputError):
        resolve_config({**BASE, "lr_schedule": "exponential"}, "*", {})


def test_custom_loss_rejects_loss_fn_config() -> None:
    with pytest.raises(InputError) as exc:
        resolve_config({**BASE, "loss_fn": "custom", "loss_fn_config": {"clip": 0.2}}, "*", {})
    assert any("loss_fn_config" in e for e in exc.value.errors)


def test_key_policy_accepts_valid_shapes() -> None:
    validate_key_policy("*", {"dataset": "blessed"})
    validate_key_policy({"loss_fn": ["cross_entropy", "custom"], "dataset": True}, {})


def test_key_policy_rejects_unknown_path() -> None:
    with pytest.raises(InputError):
        validate_key_policy({"nonsense": True}, {})
    with pytest.raises(InputError):
        validate_key_policy("*", {"nonsense": 1})


def test_key_policy_rejects_allow_lock_overlap() -> None:
    with pytest.raises(InputError):
        validate_key_policy({"loss_fn": True}, {"loss_fn": "cross_entropy"})
