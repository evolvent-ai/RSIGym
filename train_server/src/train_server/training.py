"""The supervised LoRA training loop, following tinker_cookbook/recipes/sl_loop.py:
conversations become datums via the cookbook renderer, then batched forward_backward +
optim_step."""

from __future__ import annotations

import importlib.util
import json
import random
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat, get_renderer, register_renderer
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier


# qwen3_5 with historical <think> blocks kept, matching the Qwen3.5 template's
# inference behavior; the cookbook has no built-in name for this variant.
register_renderer(
    "qwen3_5_preserve_thinking",
    lambda tokenizer, image_processor: Qwen3_5Renderer(
        tokenizer, image_processor=image_processor, strip_thinking_from_history=False
    ),
)


class TrainingCancelled(Exception):
    pass


class TrainingTimeout(Exception):
    pass


def load_loss_fn(path: Path) -> Callable[..., Any]:
    """Import the user's custom loss (already import-gated) and return its loss_fn."""
    name = f"train_server_loss_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load custom loss")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss_fn = getattr(module, "loss_fn", None)
    if not callable(loss_fn):
        raise TypeError("custom loss must define a callable loss_fn")
    return loss_fn


def train(
    *,
    run_id: str,
    config: dict[str, Any],
    data_path: Path,
    loss_path: Path | None,
    cancel_event: threading.Event,
    timeout_seconds: float,
    on_start: Callable[[int], None],
    before_step: Callable[[], None],
    after_step: Callable[[int, list[dict[str, Any]]], None],
) -> str:
    service = tinker.ServiceClient(user_metadata={"purpose": "rsi-train-server", "run_id": run_id})
    tc = service.create_lora_training_client(base_model=config["base_model"], **config["lora_config"])
    renderer = get_renderer(
        config.get("renderer") or model_info.get_recommended_renderer_name(config["base_model"]),
        tc.get_tokenizer(),
    )

    # build_supervised_examples is renderer-aware: one datum per conversation normally, one
    # per assistant message for thinking renderers (qwen3, correct per-turn prefix).
    datums = [
        datum_from_model_input_weights(model_input, weights, config.get("max_length"), reduction="mean")
        for record in read_jsonl(data_path)
        for model_input, weights in renderer.build_supervised_examples(
            _render_messages(record, renderer), train_on_what=TrainOnWhat(config["train_on_what"])
        )
    ]

    batch_size = config["batch_size"]
    n_batches = len(datums) // batch_size  # drop the last partial batch
    if n_batches == 0:
        raise ValueError(f"dataset has {len(datums)} example(s), fewer than batch_size {batch_size}")
    total_steps = config["num_epochs"] * n_batches
    if config.get("max_steps"):
        total_steps = min(total_steps, config["max_steps"])
    on_start(total_steps)

    peak_lr = config["adam_params"]["learning_rate"]
    loss_fn_config = config.get("loss_fn_config") or None
    custom_loss = None
    if config["loss_fn"] == "custom":
        if loss_path is None:
            raise ValueError("custom loss requires an uploaded loss file")
        custom_loss = load_loss_fn(loss_path)

    rng = random.Random(0)
    order: list[int] = []
    step_metrics: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    for step in range(1, total_steps + 1):
        _raise_if_cancelled(cancel_event)
        if time.monotonic() > deadline:
            raise TrainingTimeout(f"killed on timeout; a run may take at most {timeout_seconds:g}s")
        before_step()

        batch_pos = (step - 1) % n_batches
        if batch_pos == 0:  # new epoch: reshuffle
            order = list(range(len(datums)))
            rng.shuffle(order)
        batch = [datums[i] for i in order[batch_pos * batch_size : (batch_pos + 1) * batch_size]]

        if custom_loss is not None:
            result = tc.forward_backward_custom(batch, custom_loss).result()
        else:
            result = tc.forward_backward(batch, config["loss_fn"], loss_fn_config).result()

        lr = peak_lr * compute_schedule_lr_multiplier(config["lr_schedule"], step - 1, total_steps)
        tc.optim_step(tinker.AdamParams(**{**config["adam_params"], "learning_rate": lr})).result()

        # No .loss on the output; the batch mean NLL is the loss for SFT (a custom loss's
        # own value is in result.metrics).
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        logprobs = [o["logprobs"] for o in result.loss_fn_outputs]
        step_metrics.append(
            {
                "step": step,
                "nll": compute_mean_nll(logprobs, weights),
                "learning_rate": lr,
                "metrics": _json_safe(result.metrics),
                "num_tokens": sum(d.model_input.length for d in batch),
            }
        )
        after_step(step, step_metrics)

    _raise_if_cancelled(cancel_event)
    save = tc.save_weights_for_sampler(run_id).result()
    checkpoint = str(save.path)
    if not checkpoint.startswith("tinker://"):
        raise ValueError(f"unexpected checkpoint path: {checkpoint}")
    return checkpoint


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise TrainingCancelled


def _render_messages(record: dict[str, Any], renderer: Any) -> list[dict[str, Any]]:
    """Coerce the record's tool_calls, and prepend a tool-spec system prefix when it lists
    `tools` (a leading system message folds into that prefix)."""
    messages = _to_messages(record["messages"])
    tools = record.get("tools")
    if not tools:
        return messages
    system_prompt = ""
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages = messages[1:]
    # Tools stay in OpenAI wire form, matching the model's chat template.
    return renderer.create_conversation_prefix_with_tools(tools, system_prompt=system_prompt) + messages


def _to_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce JSON `tool_calls` (plain dicts) into cookbook `ToolCall` objects and fold an
    assistant `reasoning_content` into a leading thinking part, in place (records come
    single-use from read_jsonl)."""
    for message in messages:
        if message.get("role") == "assistant":
            if not message.get("content"):
                message["content"] = []
            if message.get("reasoning_content"):
                thinking = {"type": "thinking", "thinking": message.pop("reasoning_content")}
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                message["content"] = [thinking, *content]
        calls = message.get("tool_calls")
        if not calls:
            message.pop("tool_calls", None)
            continue
        for call in calls:
            args = call.get("function", {}).get("arguments")
            if args is not None and not isinstance(args, str):
                call["function"]["arguments"] = json.dumps(args)  # OpenAI wants a JSON string
        message["tool_calls"] = [ToolCall.model_validate(c) for c in calls]
    return messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001, S110 - best effort before the string fallback
            pass
    return str(value)
