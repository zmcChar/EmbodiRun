"""Thin evaluator around the official NaVILA tokenizer, model, and image processor."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from .actions import parse_navila_action
from .config import NaVILAConfig
from .errors import NaVILAInferenceError
from .prediction import NaVILAPrediction
from .preprocessing import prepare_navila_images
from .prompt import NAVILA_STOP_STRING, build_navila_prompt


class NaVILAEvaluator:
    def __init__(
        self,
        config: NaVILAConfig,
        *,
        tokenizer: Any,
        model: Any,
        image_processor: Any,
        process_images: Any,
        tokenizer_image_token: Any,
        stopping_criteria_type: Any,
        image_token_index: int,
        torch_module: Any,
        image_module: Any,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.stopping_criteria_type = stopping_criteria_type
        self.image_token_index = image_token_index
        self.torch = torch_module
        self.image_module = image_module
        self.device = torch_module.device(config.device)
        self.torch_dtype = getattr(torch_module, config.dtype)
        self.pad_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        if not isinstance(self.pad_token_id, int) or not isinstance(self.bos_token_id, int):
            raise NaVILAInferenceError("NaVILA tokenizer requires integer EOS and BOS ids")

    def _tokenize(self, prompt: str) -> Any:
        try:
            return (
                self.tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    self.image_token_index,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )
        except Exception as error:
            raise NaVILAInferenceError("cannot tokenize the NaVILA prompt") from error

    @staticmethod
    def _ids(row: Any) -> tuple[int, ...]:
        return tuple(int(value) for value in row.detach().cpu().tolist())

    def _decode(self, generated: Any, input_ids: Any) -> tuple[str, tuple[int, ...]]:
        try:
            generated_ids = self._ids(generated[0])
            prompt_ids = self._ids(input_ids[0])
            if (
                prompt_ids
                and len(generated_ids) >= len(prompt_ids)
                and generated_ids[: len(prompt_ids)] == prompt_ids
            ):
                output = generated[:, len(prompt_ids) :]
                token_ids = generated_ids[len(prompt_ids) :]
            elif generated_ids[:1] == (self.bos_token_id,):
                output = generated[:, 1:]
                token_ids = generated_ids[1:]
            else:
                output = generated
                token_ids = generated_ids
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        except Exception as error:
            raise NaVILAInferenceError("cannot decode NaVILA generation output") from error
        if decoded.endswith(NAVILA_STOP_STRING):
            decoded = decoded[: -len(NAVILA_STOP_STRING)].rstrip()
        return decoded, token_ids

    def predict(self, rgb_frames: Sequence[Any], instruction: str) -> NaVILAPrediction:
        prompt = build_navila_prompt(instruction)
        images = prepare_navila_images(
            rgb_frames,
            image_processor=self.image_processor,
            model_config=self.model.config,
            process_images=self.process_images,
            image_module=self.image_module,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        input_ids = self._tokenize(prompt)
        try:
            stopping = self.stopping_criteria_type([NAVILA_STOP_STRING], self.tokenizer, input_ids)
            attention_mask = self.torch.ones_like(input_ids)
            started = time.monotonic()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=self.config.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping],
                    pad_token_id=self.pad_token_id,
                )
            elapsed = time.monotonic() - started
        except Exception as error:
            raise NaVILAInferenceError("NaVILA generation failed") from error
        decoded, token_ids = self._decode(generated, input_ids)
        return NaVILAPrediction(
            action=parse_navila_action(decoded),
            raw_output=decoded,
            token_ids=token_ids,
            generation_time_s=elapsed,
        )


__all__ = ["NaVILAEvaluator"]
