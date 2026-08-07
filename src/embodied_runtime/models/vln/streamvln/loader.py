"""Checkpoint materialization for the lazy StreamVLN runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_SENSOR_CONFIG, StreamVLNConfig
from .repository import activate_repository_imports, validate_repository_module_origins
from .vision import embedded_siglip_vision_initialization


@dataclass(frozen=True, slots=True)
class MaterializedStreamVLN:
    """Objects jointly owned by one loaded runtime."""

    model: Any
    evaluator: Any


def _pretrained_options(config: StreamVLNConfig) -> dict[str, object]:
    options: dict[str, object] = {"local_files_only": config.local_files_only}
    if config.revision is not None:
        options["revision"] = config.revision
    return options


def materialize_streamvln(
    config: StreamVLNConfig,
    evaluator_factory: Callable[..., Any],
) -> MaterializedStreamVLN:
    """Load the official class, checkpoint, tokenizer, and recurrent evaluator."""

    activate_repository_imports(config.streamvln_root)
    validate_repository_module_origins(config.streamvln_root, require_loaded=False)

    torch = importlib.import_module("torch")
    torch_device = torch.device(config.device)
    if config.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(
            config.cuda_memory_fraction,
            device=torch_device,
        )

    transformers = importlib.import_module("transformers")
    pretrained = _pretrained_options(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side="right",
        **pretrained,
    )
    hf_config = transformers.AutoConfig.from_pretrained(config.model_path, **pretrained)
    model_options: dict[str, object] = {
        "torch_dtype": getattr(torch, config.dtype),
        "config": hf_config,
        "low_cpu_mem_usage": False,
        **pretrained,
    }
    if config.attn_implementation is not None:
        model_options["attn_implementation"] = config.attn_implementation

    with embedded_siglip_vision_initialization(config.use_embedded_vision_weights):
        # Import after installing the patch: importing the native class loads
        # llava's builder and captures this tower type.
        native_module = importlib.import_module("model.stream_video_vln")
        validate_repository_module_origins(config.streamvln_root, require_loaded=True)
        model = native_module.StreamVLNForCausalLM.from_pretrained(
            config.model_path,
            **model_options,
        )

    model.model.num_history = config.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(torch_device)
    model.eval()

    evaluator = evaluator_factory(
        DEFAULT_SENSOR_CONFIG,
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        num_frames=config.num_frames,
        num_future_steps=config.num_future_steps,
        num_history=config.num_history,
        max_new_tokens=config.max_new_tokens,
        environment_id=0,
    )
    if config.warmup:
        numpy = importlib.import_module("numpy")
        evaluator.reset_memory()
        evaluator.step(
            0,
            numpy.zeros((480, 640, 3), dtype=numpy.uint8),
            "Move forward 25 centimeters.",
            run_model=True,
        )
        evaluator.reset_memory()

    return MaterializedStreamVLN(model=model, evaluator=evaluator)


__all__ = ["MaterializedStreamVLN", "materialize_streamvln"]
