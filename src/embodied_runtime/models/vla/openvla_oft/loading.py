"""Dependency-aware construction of a verified OpenVLA-OFT runtime."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...errors import ModelPackageError
from .checkpoint import load_checkpoint_components, resolve_openvla_oft_checkpoint
from .configuration import load_model_config
from .constants import VVLA_SOURCE_COMMIT
from .dependencies import require_torch


def load_openvla_oft(
    checkpoint: str | Path,
    *,
    load_device: str = "cpu",
    load_dtype: Any = "bfloat16",
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
    action_dim: int | None = None,
    action_horizon: int | None = None,
    max_prompt_length: int = 50,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load an RLinf-compatible discrete OpenVLA-OFT checkpoint."""

    root = resolve_openvla_oft_checkpoint(
        checkpoint,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    torch = require_torch()
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as error:
        raise ModelPackageError(
            "OpenVLA-OFT requires transformers; install the 'openvla_oft' extra"
        ) from error

    model_config = load_model_config(
        root,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    try:
        from .backbone import PrismaticVisionBackbone
        from .processing_openvla_oft import OpenVLAOFTProcessor
        from .projector import PrismaticProjector
    except ImportError as error:
        raise ModelPackageError(
            "OpenVLA-OFT vision loading requires timm and torchvision; "
            "install the 'openvla_oft' extra"
        ) from error
    from .head import CategoricalActionHead
    from .modeling_openvla_oft import OpenVLAOFTReferenceModule

    dtype = resolve_torch_dtype(load_dtype)
    try:
        # timm 0.9 calls Tensor.item() while constructing stochastic-depth
        # schedules, so only the 7B Llama is initialized on meta.
        with default_dtype(dtype):
            vision_backbone = PrismaticVisionBackbone(
                use_fused_vision_backbone=model_config.use_fused_vision_backbone,
                image_sizes=model_config.image_sizes,
                timm_model_ids=model_config.timm_model_ids,
                timm_override_act_layers=model_config.activation_layers,
            )
            vision_backbone.set_num_images_in_input(model_config.num_images_in_input)
            language_config = LlamaConfig(**model_config.text_config)
            projector = PrismaticProjector(
                use_fused_vision_backbone=model_config.use_fused_vision_backbone,
                vision_dim=vision_backbone.embed_dim,
                llm_dim=int(language_config.hidden_size),
            )
        with torch.device("meta"), default_dtype(dtype):
            language_model = LlamaForCausalLM(language_config)
    except Exception as error:
        raise ModelPackageError(f"could not construct OpenVLA-OFT modules: {error}") from error
    materialize_language_buffers(
        language_model,
        language_config,
        load_device=load_device,
    )
    load_checkpoint_components(
        root,
        vision_backbone=vision_backbone,
        projector=projector,
        language_model=language_model,
        load_device=load_device,
        load_dtype=dtype,
        strict=strict,
    )
    head = CategoricalActionHead(
        model_config.action_vocab_size,
        model_config.n_action_bins,
        model_config.action_dim,
        model_config.action_horizon,
        model_config.q01,
        model_config.q99,
        model_config.action_mask,
    ).to(device=torch.device(load_device))
    processor = OpenVLAOFTProcessor.from_checkpoint(
        root,
        max_length=max_prompt_length,
        local_files_only=True,
    )
    module = OpenVLAOFTReferenceModule(
        vision_backbone,
        projector,
        language_model,
        head,
    ).eval()
    return (
        module,
        processor,
        {
            "checkpoint_root": str(root),
            "statistics_key": model_config.statistics_key,
            "action_vocab_size": model_config.action_vocab_size,
            "padded_vocab_size": model_config.padded_vocab_size,
            "n_action_bins": model_config.n_action_bins,
            "num_images_in_input": vision_backbone.get_num_images_in_input(),
            "image_sizes": model_config.image_sizes,
            "timm_model_ids": model_config.timm_model_ids,
            "load_dtype": str(dtype),
            "vvla_commit": VVLA_SOURCE_COMMIT,
        },
    )


def resolve_torch_dtype(value: Any) -> Any:
    torch = require_torch()
    if isinstance(value, torch.dtype):
        dtype = value
    else:
        normalized = value.lower().replace("torch.", "").replace("_", "")
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = aliases[normalized]
        except KeyError as error:
            raise ModelPackageError(f"unsupported OpenVLA-OFT load dtype {value!r}") from error
    if not dtype.is_floating_point:
        raise ModelPackageError("OpenVLA-OFT load dtype must be floating point")
    return dtype


@contextmanager
def default_dtype(dtype: Any) -> Iterator[None]:
    torch = require_torch()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def materialize_language_buffers(
    language_model: Any,
    language_config: Any,
    *,
    load_device: str,
) -> None:
    """Recreate non-persistent Llama buffers omitted from safetensors."""

    meta_buffers = [name for name, value in language_model.named_buffers() if value.is_meta]
    if not meta_buffers:
        return
    model = getattr(language_model, "model", None)
    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        raise ModelPackageError(
            "OpenVLA-OFT language model retained meta buffers but exposes no rotary_emb"
        )
    torch = require_torch()
    try:
        model.rotary_emb = type(rotary)(
            language_config,
            device=torch.device(load_device),
        )
    except Exception as error:
        raise ModelPackageError(
            f"could not materialize OpenVLA-OFT Llama rotary buffers: {error}"
        ) from error
    remaining = [name for name, value in language_model.named_buffers() if value.is_meta]
    if remaining:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta buffers: " + ", ".join(remaining[:8])
        )


__all__ = ["load_openvla_oft"]
