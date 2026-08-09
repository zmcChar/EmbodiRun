"""Stock-Transformers full-history runtime for the pinned ActiveVLN model."""

from __future__ import annotations

import gc
import importlib
import threading
import time
from pathlib import Path
from typing import Any

from embodied_runtime.policies.navigation.errors import NavigationPolicyError

from .activevln_transformers_support import (
    MAX_HISTORY_IMAGES,
    SYSTEM_PROMPT_R2R,
    USER_SUFFIX,
    validate_checkpoint_model,
)
from .vvla.activevln import (
    ACTIVEVLN_CHECKPOINT,
    ACTIVEVLN_REVISION,
    DEFAULT_VVLA_ROOT,
    VvlaActiveVLNNavigationPolicy,
)
from .vvla.checkpoint import verify_activevln_checkpoint
from .vvla.hardware import require_activevln_cuda_capacity
from .vvla.mapping import ActiveVLNPrediction, parse_canonical_activevln_actions


class _CancellationCriteria:
    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del input_ids, scores, kwargs
        return self._event.is_set()


class TransformersActiveVLNRuntime:
    """Own a stock HF model and rebuild canonical multimodal history each turn."""

    def __init__(
        self,
        *,
        vvla_root: str | Path = DEFAULT_VVLA_ROOT,
        checkpoint: str | Path = ACTIVEVLN_CHECKPOINT,
        revision: str = ACTIVEVLN_REVISION,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention: str = "eager",
        max_new_tokens: int = 64,
        max_context: int = 32768,
        allow_download: bool = False,
        do_sample: bool = False,
    ) -> None:
        if not Path(checkpoint).expanduser().exists() and revision != ACTIVEVLN_REVISION:
            raise ValueError(
                "remote Transformers ActiveVLN requires the pinned immutable checkpoint revision "
                f"{ACTIVEVLN_REVISION}"
            )
        if attention not in {"eager", "sdpa"}:
            raise ValueError(
                f"Transformers ActiveVLN attention must be 'eager' or 'sdpa', got {attention!r}"
            )
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                "Transformers ActiveVLN dtype must be 'auto', 'float16', 'bfloat16', or 'float32'"
            )
        self.vvla_root = Path(vvla_root).expanduser()
        self.checkpoint = str(checkpoint)
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.attention = attention
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context
        self.allow_download = allow_download
        self.do_sample = do_sample
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._active_episode_id: str | None = None
        self._conversation: list[dict[str, Any]] = []
        self._images: list[Any] = []
        self._cancelled = threading.Event()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def engine_dtype(self) -> str:
        if self._model is None:
            raise RuntimeError("Transformers ActiveVLN runtime is not loaded")
        return str(next(self._model.parameters()).dtype).removeprefix("torch.")

    @staticmethod
    def _torch_dtype(torch: Any, name: str) -> Any:
        return {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]

    def load(self) -> None:
        if self._model is not None:
            return
        verify_activevln_checkpoint(self.vvla_root, self.checkpoint)
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise NavigationPolicyError(
                "ActiveVLN through Transformers requires the isolated vvla-activevln environment"
            ) from error

        require_activevln_cuda_capacity(torch, self.device, self.dtype)
        local_files_only = not self.allow_download
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
            torch_dtype=self._torch_dtype(torch, self.dtype),
            attn_implementation=self.attention,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
            use_fast=False,
        )
        validate_checkpoint_model(model)
        model = model.to(self.device).eval()
        self._torch = torch
        self._processor = processor
        self._model = model
        self.reset()

    @staticmethod
    def _initial_conversation() -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R}],
            }
        ]

    @staticmethod
    def _user_turn(instruction: str, *, initial: bool) -> dict[str, Any]:
        label = "[Initial Observation]:" if initial else "After that, the observation is:"
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": label},
                {"type": "image"},
                {"type": "text", "text": USER_SUFFIX.format(instruction)},
            ],
        }

    def reset(self) -> None:
        self._active_episode_id = None
        self._conversation = []
        self._images = []
        self._cancelled.clear()

    def cancel(self) -> None:
        self._cancelled.set()

    def _synchronize(self) -> None:
        if self._torch is not None and str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    def predict(self, rgb: Any, instruction: str, *, episode_id: str) -> ActiveVLNPrediction:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Transformers ActiveVLN runtime is not loaded")
        if self._cancelled.is_set():
            raise NavigationPolicyError("Transformers ActiveVLN generation was cancelled")
        if not instruction.strip():
            raise NavigationPolicyError("ActiveVLN instruction must not be empty")
        if self._active_episode_id is not None and self._active_episode_id != episode_id:
            self.reset()
        initial = self._active_episode_id is None
        conversation = list(self._conversation or self._initial_conversation())
        images = list(self._images)
        if len(images) >= MAX_HISTORY_IMAGES:
            raise NavigationPolicyError(
                f"ActiveVLN episode exceeds {MAX_HISTORY_IMAGES} history images"
            )

        pil = importlib.import_module("PIL.Image")
        image = pil.fromarray(rgb).convert("RGB")
        conversation.append(self._user_turn(instruction, initial=initial))
        images.append(image)
        text = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=False,
            return_tensors="pt",
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if prompt_tokens + self.max_new_tokens > self.max_context:
            raise NavigationPolicyError(
                f"ActiveVLN full history exceeds max_context={self.max_context} tokens"
            )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        generate_options: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.05,
            "use_cache": True,
            "stopping_criteria": transformers_stopping_criteria(self._cancelled),
        }
        if self.do_sample:
            generate_options.update({"temperature": 0.2, "top_p": 0.8})
        else:
            # The checkpoint's generation_config stores a sampling-only
            # temperature. Explicitly clear it for a warning-free greedy path.
            generate_options["temperature"] = None

        self._synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generate_options)
        self._synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if self._cancelled.is_set():
            raise NavigationPolicyError("Transformers ActiveVLN generation was cancelled")
        generated = output[:, prompt_tokens:]
        token_ids = tuple(int(value) for value in generated[0].tolist())
        response = self._processor.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        try:
            actions = parse_canonical_activevln_actions(response)
        except ValueError as error:
            raise NavigationPolicyError(str(error)) from error

        conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        self._active_episode_id = episode_id
        self._conversation = conversation
        self._images = images
        return ActiveVLNPrediction(
            actions=actions,
            text=response,
            latency_ms=latency_ms,
            token_ids=token_ids,
        )

    def close(self) -> None:
        self.cancel()
        self._model = None
        self._processor = None
        self._torch = None
        self._active_episode_id = None
        self._conversation = []
        self._images = []
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transformers_stopping_criteria(cancelled: threading.Event) -> Any:
    transformers = importlib.import_module("transformers")
    return transformers.StoppingCriteriaList([_CancellationCriteria(cancelled)])


class TransformersActiveVLNNavigationPolicy(VvlaActiveVLNNavigationPolicy):
    """Expose stock-HF full-history ActiveVLN through the navigation contract."""

    def __init__(
        self,
        runtime: TransformersActiveVLNRuntime | None = None,
        **runtime_options: Any,
    ) -> None:
        super().__init__(runtime=runtime or TransformersActiveVLNRuntime(**runtime_options))


__all__ = ["TransformersActiveVLNNavigationPolicy", "TransformersActiveVLNRuntime"]
