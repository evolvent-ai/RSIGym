"""One live model, and the pool that caches them by ref."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from model_server.pricing import estimate_cost_usd
from model_server.translation import (
    build_chat_completion,
    coerce_tool_call_arguments,
    parse_completion,
    plain_completion,
    sampling_params,
)


class ModelBackend:
    """A SamplingClient plus its tokenizer, bound to one model ref."""

    def __init__(self, model_ref: str) -> None:
        import tinker

        self.model_ref = model_ref
        service = tinker.ServiceClient(user_metadata={"purpose": "model-server"})
        # A tinker:// path is a saved checkpoint; anything else is a base model name.
        self._client = (
            service.create_sampling_client(model_path=model_ref)
            if model_ref.startswith("tinker://")
            else service.create_sampling_client(base_model=model_ref)
        )
        self._tokenizer = self._client.get_tokenizer()
        self.base_model = self._client.get_base_model()
        self.max_context_length = next(
            model.max_context_length
            for model in service.get_server_capabilities().supported_models
            if model.model_name == self.base_model
        )

    def _render_prompt(self, payload: dict[str, Any]) -> tuple[str, list[int]]:
        """The rendered text is kept because the response parser needs it as prefix:
        the prompt ends part-way into <think>."""
        messages = coerce_tool_call_arguments(payload.get("messages") or [])
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=payload.get("tools") or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        # The template already emits the special tokens.
        return text, self._tokenizer.encode(text, add_special_tokens=False)

    def _parse(self, prompt_text: str, completion: str) -> dict[str, Any]:
        if self.base_model.startswith("Qwen/"):
            return parse_completion(prompt_text, completion)
        return plain_completion(completion)

    def _translate_response(
        self, prompt_text: str, token_ids: list[int], response: Any
    ) -> dict[str, Any]:
        sequence = response.sequences[0]
        output_tokens = list(sequence.tokens)
        # skip_special_tokens drops terminators like <|im_end|> but keeps the grammar
        # tags (<think>, <tool_call> are added tokens, not special ones) -- verified
        # against the real Qwen tokenizer.
        completion = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
        prompt_tokens = len(token_ids)
        completion_tokens = len(output_tokens)
        cached_tokens = response.prompt_cache_hit_tokens
        cost_usd = estimate_cost_usd(
            self.base_model, prompt_tokens, cached_tokens, completion_tokens
        )
        return build_chat_completion(
            model=self.model_ref,
            parsed=self._parse(prompt_text, completion),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            stop_reason=str(getattr(sequence, "stop_reason", "") or ""),
        )

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        from tinker import types

        prompt_text, token_ids = await asyncio.to_thread(self._render_prompt, payload)
        params = sampling_params(payload)
        if "stop" not in params and self.base_model.startswith("Qwen/"):
            # Qwen turns end with <|im_end|>, but base-model checkpoints keep
            # <|endoftext|> as EOS -- unstopped sampling runs to max_tokens.
            params["stop"] = ["<|im_end|>"]
        response = await self._client.sample_async(
            prompt=types.ModelInput.from_ints(token_ids),
            sampling_params=types.SamplingParams(**params),
            num_samples=1,
        )
        return await asyncio.to_thread(
            self._translate_response, prompt_text, token_ids, response
        )


class ModelPool:
    """Lazy per-ref backends with LRU eviction. Creation is slow (client setup +
    tokenizer download), so concurrent requests for the same ref share one
    construction; a failed construction is evicted so a transient failure (a ckpt not
    yet visible, a network blip) does not stay poisoned. Every cached backend holds a
    tinker client thread and a tokenizer, and checkpoints keep being minted, so the
    pool is capped: eviction drops the reference and GC releases the client once
    in-flight requests finish."""

    def __init__(self, max_models: int = 16) -> None:
        self._backends: OrderedDict[str, asyncio.Task[ModelBackend]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_models = max_models

    async def get(self, model_ref: str) -> ModelBackend:
        async with self._lock:
            task = self._backends.get(model_ref)
            if task is None:
                task = asyncio.create_task(asyncio.to_thread(ModelBackend, model_ref))
                self._backends[model_ref] = task
                while len(self._backends) > self._max_models:
                    self._backends.popitem(last=False)
            else:
                self._backends.move_to_end(model_ref)
        try:
            return await asyncio.shield(task)
        except BaseException:
            async with self._lock:
                if self._backends.get(model_ref) is task and task.done():
                    del self._backends[model_ref]
            raise
