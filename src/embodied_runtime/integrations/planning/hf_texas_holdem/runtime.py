"""Lazy Hugging Face text-generation runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .config import HfTexasHoldemPlannerConfig
from .errors import PlannerRuntimeError


@runtime_checkable
class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class HuggingFaceTextGenerator:
    """Small synchronous wrapper around a lazily loaded HF model."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch_module: Any,
        max_new_tokens: int,
    ) -> None:
        self._model = model
        self._processor = processor
        self._torch = torch_module
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a deterministic task planner. Follow the requested JSON "
                            "schema exactly."
                        ),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = getattr(self._model, "device", None)
            if input_device is not None:
                if hasattr(inputs, "to"):
                    inputs = inputs.to(input_device)
                else:
                    inputs = {
                        key: value.to(input_device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
            input_ids = inputs["input_ids"]
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            trimmed_ids = [
                output_ids[len(source_ids) :]
                for source_ids, output_ids in zip(input_ids, generated_ids, strict=False)
            ]
            decoded = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as error:
            raise PlannerRuntimeError("Hugging Face planner generation failed") from error
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)) or not decoded:
            raise PlannerRuntimeError("Hugging Face processor returned no decoded response")
        response = decoded[0]
        if not isinstance(response, str) or not response.strip():
            raise PlannerRuntimeError("Hugging Face planner returned an empty response")
        return response


def load_huggingface_generator(
    config: HfTexasHoldemPlannerConfig,
) -> HuggingFaceTextGenerator:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise PlannerRuntimeError(
            "the Hugging Face planner requires torch and transformers"
        ) from error

    model_options: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "dtype": config.dtype if config.dtype == "auto" else getattr(torch, config.dtype),
    }
    shared_options: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.cache_dir is not None:
        model_options["cache_dir"] = config.cache_dir
        shared_options["cache_dir"] = config.cache_dir
    if config.revision is not None:
        model_options["revision"] = config.revision
        shared_options["revision"] = config.revision
    if config.attn_implementation is not None:
        model_options["attn_implementation"] = config.attn_implementation
    if config.device == "auto":
        model_options["device_map"] = "auto"

    model_class_name = (
        "AutoModelForImageTextToText"
        if config.model_kind == "image_text_to_text"
        else "AutoModelForCausalLM"
    )
    model_class = getattr(transformers, model_class_name, None)
    if model_class is None:
        raise PlannerRuntimeError(f"installed transformers does not provide {model_class_name}")
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            config.checkpoint,
            **shared_options,
        )
        model = model_class.from_pretrained(config.checkpoint, **model_options)
        if config.device != "auto":
            model = model.to(config.device)
        model.eval()
    except Exception as error:
        raise PlannerRuntimeError(
            f"failed to load planner checkpoint {config.checkpoint!r}"
        ) from error
    return HuggingFaceTextGenerator(
        model=model,
        processor=processor,
        torch_module=torch,
        max_new_tokens=config.max_new_tokens,
    )
