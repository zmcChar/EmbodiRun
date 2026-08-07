"""Lazy materialization of the official NaVILA source and checkpoint."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import NaVILAConfig
from .errors import NaVILALoadError
from .evaluator import NaVILAEvaluator
from .repository import activate_repository, validate_module_origins


@dataclass(frozen=True, slots=True)
class MaterializedNaVILA:
    model: Any
    evaluator: Any


EvaluatorFactory = Callable[..., Any]


def materialize_navila(
    config: NaVILAConfig,
    evaluator_factory: EvaluatorFactory = NaVILAEvaluator,
) -> MaterializedNaVILA:
    """Load official objects only when the owning runtime is prepared."""

    try:
        activate_repository(config.navila_root)
        validate_module_origins(config.navila_root, require_loaded=False)
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        constants = importlib.import_module("llava.constants")
        mm_utils = importlib.import_module("llava.mm_utils")
        builder = importlib.import_module("llava.model.builder")
        validate_module_origins(config.navila_root, require_loaded=True)

        device = torch.device(config.device)
        if config.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(
                config.cuda_memory_fraction,
                device=device,
            )
        model_name = mm_utils.get_model_name_from_path(str(config.model_path))
        tokenizer, model, image_processor, _context_length = builder.load_pretrained_model(
            str(config.model_path),
            model_name,
            model_base=None,
            device=config.device,
            local_files_only=config.local_files_only,
            attn_implementation=config.attention_backend,
        )
        if callable(getattr(model, "requires_grad_", None)):
            model.requires_grad_(False)
        if callable(getattr(model, "eval", None)):
            model.eval()
        evaluator = evaluator_factory(
            config,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            process_images=mm_utils.process_images,
            tokenizer_image_token=mm_utils.tokenizer_image_token,
            stopping_criteria_type=mm_utils.KeywordsStoppingCriteria,
            image_token_index=constants.IMAGE_TOKEN_INDEX,
            torch_module=torch,
            image_module=image_module,
        )
        return MaterializedNaVILA(model=model, evaluator=evaluator)
    except NaVILALoadError:
        raise
    except Exception as error:
        raise NaVILALoadError(f"failed to load NaVILA: {error}") from error


__all__ = ["EvaluatorFactory", "MaterializedNaVILA", "materialize_navila"]
