"""Reference single-forward graph and checkpoint loader for OpenVLA-OFT.

The implementation is adapted from vvla's OpenVLA-OFT integration at commit
``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT,
Copyright 2026 Longxmas) and checked against RLinf's Apache-2.0
``OpenVLAOFTForRLActionPrediction`` execution semantics.

This first integration intentionally uses the stock Hugging Face Llama forward:
vision, projection, multimodal assembly, categorical decoding, and checkpoint
routing are model-owned; scheduling, device placement, compilation, and future
operator replacement are runtime/backend concerns.  Prefix/decode splitting and
RL log-probability recomputation are later optimizations, not hidden in this
correctness path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import torch

from embodied_runtime.contracts import ModelPackageError

from .head import CategoricalActionHead

_VVLA_SOURCE_COMMIT = "80b5cf48c8710c69ed97200903562e9787efe105"
_REQUIRED_METADATA_FILES = (
    "config.json",
    "dataset_statistics.json",
    "preprocessor_config.json",
)
_WEIGHT_INDEX = "model.safetensors.index.json"
_SINGLE_WEIGHTS = "model.safetensors"
_COMPONENT_PREFIXES = {
    "vision_backbone.": "vision_backbone",
    "projector.": "projector",
    "language_model.": "language_model",
}


def _output_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    raise TypeError("OpenVLA-OFT language model output must expose 'logits'")


def _floating_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for value in (*module.parameters(), *module.buffers()):
        if value.is_floating_point():
            return value.dtype
    return None


class OpenVLAOFTReferenceModule(torch.nn.Module):
    """Full causal forward from processed observation to one action chunk."""

    def __init__(
        self,
        vision_backbone: torch.nn.Module,
        projector: torch.nn.Module,
        language_model: torch.nn.Module,
        head: CategoricalActionHead,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.projector = projector
        self.language_model = language_model
        self.head = head
        self.action_dim = head.action_dim
        self.action_horizon = head.num_action_chunks
        self.n_tokens = head.n_tokens

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        getter = getattr(self.language_model, "get_input_embeddings", None)
        if callable(getter):
            embeddings = getter()
        else:
            model = getattr(self.language_model, "model", None)
            embeddings = getattr(model, "embed_tokens", None)
        if embeddings is None:
            raise TypeError(
                "OpenVLA-OFT language model must expose get_input_embeddings() "
                "or model.embed_tokens"
            )
        return embeddings(input_ids)

    def forward(self, inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Run one deterministic OpenVLA-OFT action prediction.

        ``inputs`` contains ``input_ids [B,L]``, ``attention_mask [B,L]`` and
        fused ``pixel_values [B,C,H,W]``.  The Llama output positions beginning
        at the final prompt-space token causally predict every action token.
        """

        try:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
        except KeyError as exc:
            raise ValueError(f"OpenVLA-OFT input is missing {exc.args[0]!r}") from exc
        if not all(
            isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, pixel_values)
        ):
            raise TypeError("OpenVLA-OFT inputs must be torch.Tensor leaves")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length]")
        if pixel_values.ndim != 4 or pixel_values.shape[0] != input_ids.shape[0]:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        if input_ids.shape[1] < 1:
            raise ValueError("OpenVLA-OFT prompt must contain at least one token")
        if torch.any(attention_mask[:, -1] == 0):
            raise ValueError("the final OpenVLA-OFT prompt token must be active")

        batch_size = input_ids.shape[0]
        placeholders = torch.ones(
            (batch_size, self.n_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        complete_ids = torch.cat((input_ids, placeholders), dim=1)
        action_mask = torch.ones(
            (batch_size, self.n_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        complete_mask = torch.cat((attention_mask, action_mask), dim=1)

        token_embeddings = self._input_embeddings(complete_ids)
        # Native OpenVLA-OFT uses placeholder ids only to allocate causal
        # positions; action queries intentionally carry no token content.
        token_embeddings = token_embeddings.clone()
        token_embeddings[:, -self.n_tokens :] = 0

        vision_dtype = _floating_dtype(self.vision_backbone)
        if vision_dtype is not None and pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        vision_features = self.vision_backbone(pixel_values)
        projector_dtype = _floating_dtype(self.projector)
        if projector_dtype is not None and vision_features.dtype != projector_dtype:
            vision_features = vision_features.to(dtype=projector_dtype)
        patch_embeddings = self.projector(vision_features)
        if patch_embeddings.dtype != token_embeddings.dtype:
            patch_embeddings = patch_embeddings.to(dtype=token_embeddings.dtype)
        if patch_embeddings.ndim != 3:
            raise ValueError("OpenVLA-OFT projected vision features must have shape [B,P,D]")
        if (
            patch_embeddings.shape[0] != batch_size
            or patch_embeddings.shape[-1] != token_embeddings.shape[-1]
        ):
            raise ValueError(
                "OpenVLA-OFT vision patches do not match language batch/hidden dimensions"
            )

        multimodal_embeddings = torch.cat(
            (
                token_embeddings[:, :1],
                patch_embeddings,
                token_embeddings[:, 1:],
            ),
            dim=1,
        )
        patch_mask = torch.ones(
            patch_embeddings.shape[:2],
            dtype=complete_mask.dtype,
            device=complete_mask.device,
        )
        multimodal_mask = torch.cat(
            (
                complete_mask[:, :1],
                patch_mask,
                complete_mask[:, 1:],
            ),
            dim=1,
        )
        position_ids = multimodal_mask.long().cumsum(dim=1) - 1
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = _output_logits(outputs)
        expected_sequence = multimodal_embeddings.shape[1]
        if logits.ndim != 3 or logits.shape[:2] != (
            batch_size,
            expected_sequence,
        ):
            raise ValueError(
                "OpenVLA-OFT language logits must have shape "
                f"[{batch_size}, {expected_sequence}, vocabulary]"
            )

        # Causal read: prompt space predicts action 0; action-query i predicts
        # action i+1.  The final query would predict STOP and is not consumed.
        action_logits = logits[:, -self.n_tokens - 1 : -1]
        action_tokens, _ = self.head.greedy(action_logits)
        actions = self.head.tokens_to_actions(action_tokens)
        return {
            "actions": actions,
            "action_tokens": action_tokens,
        }


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelPackageError(f"{description} {path} must contain a JSON object")
    return value


def resolve_openvla_oft_checkpoint(
    checkpoint: str | Path,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or one complete Hugging Face snapshot."""

    source = str(checkpoint)
    local = Path(source).expanduser()
    if local.is_dir():
        root = local.resolve()
    elif local.is_absolute() or source.startswith(("./", "../", "~")):
        raise ModelPackageError(f"OpenVLA-OFT checkpoint directory does not exist: {local}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelPackageError(
                "resolving a Hugging Face OpenVLA-OFT checkpoint requires "
                "huggingface-hub; install the 'openvla_oft' extra"
            ) from exc
        try:
            root = Path(
                snapshot_download(
                    repo_id=source,
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=(
                        "*.json",
                        "*.model",
                        "tokenizer.*",
                        "added_tokens.json",
                        "special_tokens_map.json",
                        "*.safetensors",
                    ),
                )
            )
        except Exception as exc:
            mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub"
            raise ModelPackageError(
                f"could not resolve OpenVLA-OFT checkpoint {source!r} from {mode}: {exc}"
            ) from exc

    missing = [name for name in _REQUIRED_METADATA_FILES if not (root / name).is_file()]
    if not (root / _WEIGHT_INDEX).is_file() and not (root / _SINGLE_WEIGHTS).is_file():
        missing.append(f"{_WEIGHT_INDEX} or {_SINGLE_WEIGHTS}")
    if missing:
        raise ModelPackageError(f"OpenVLA-OFT checkpoint {root} is missing: {', '.join(missing)}")
    return root


def _weight_shards(root: Path) -> tuple[Path, ...]:
    index_path = root / _WEIGHT_INDEX
    if index_path.is_file():
        index = _read_json(index_path, description="OpenVLA-OFT weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelPackageError(f"OpenVLA-OFT weight index {index_path} has no weight_map")
        shard_names = tuple(sorted({str(name) for name in weight_map.values()}))
        shards = tuple(root / name for name in shard_names)
    else:
        shards = (root / _SINGLE_WEIGHTS,)
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint is missing weight shard(s): " + ", ".join(missing)
        )
    return shards


def _component_for_key(key: str) -> tuple[str, str] | None:
    for prefix, component in _COMPONENT_PREFIXES.items():
        if key.startswith(prefix):
            return component, key[len(prefix) :]
    return None


def _load_checkpoint_components(
    root: Path,
    *,
    vision_backbone: torch.nn.Module,
    projector: torch.nn.Module,
    language_model: torch.nn.Module,
    load_device: str,
    load_dtype: torch.dtype,
    strict: bool,
) -> None:
    """Assign sharded safetensors into meta-initialized component modules."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint loading requires safetensors; install the 'openvla_oft' extra"
        ) from exc

    components = {
        "vision_backbone": vision_backbone,
        "projector": projector,
        "language_model": language_model,
    }
    expected_parameters = {
        name: set(dict(module.named_parameters(remove_duplicate=False)))
        for name, module in components.items()
    }
    loaded: dict[str, set[str]] = {name: set() for name in components}
    ignored_prefixes = ("value_head.",)
    unknown_source_keys: list[str] = []

    for shard in _weight_shards(root):
        shard_state: dict[str, dict[str, torch.Tensor]] = {name: {} for name in components}
        try:
            with safe_open(str(shard), framework="pt", device=load_device) as handle:
                for source_key in handle.keys():
                    target = _component_for_key(source_key)
                    if target is None:
                        if not source_key.startswith(ignored_prefixes):
                            unknown_source_keys.append(source_key)
                        continue
                    component_name, destination_key = target
                    tensor = handle.get_tensor(source_key)
                    if tensor.is_floating_point() and tensor.dtype != load_dtype:
                        tensor = tensor.to(dtype=load_dtype)
                    shard_state[component_name][destination_key] = tensor
        except Exception as exc:
            raise ModelPackageError(
                f"could not read OpenVLA-OFT weight shard {shard}: {exc}"
            ) from exc

        for component_name, partial_state in shard_state.items():
            if not partial_state:
                continue
            try:
                incompatible = components[component_name].load_state_dict(
                    partial_state,
                    strict=False,
                    assign=True,
                )
            except Exception as exc:
                raise ModelPackageError(
                    f"could not route {shard.name} into OpenVLA-OFT {component_name}: {exc}"
                ) from exc
            if incompatible.unexpected_keys:
                names = ", ".join(incompatible.unexpected_keys[:8])
                raise ModelPackageError(
                    f"OpenVLA-OFT {component_name} has incompatible checkpoint keys: {names}"
                )
            loaded[component_name].update(partial_state)

    missing_parameters = {
        name: sorted(parameters.difference(loaded[name]))
        for name, parameters in expected_parameters.items()
    }
    missing_parameters = {name: names for name, names in missing_parameters.items() if names}
    if missing_parameters:
        details = "; ".join(
            f"{name}: {', '.join(keys[:8])}" for name, keys in missing_parameters.items()
        )
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint left model parameters uninitialized: " + details
        )
    if strict and unknown_source_keys:
        names = ", ".join(sorted(unknown_source_keys)[:8])
        raise ModelPackageError(f"OpenVLA-OFT checkpoint has unknown component keys: {names}")

    tie_weights = getattr(language_model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    meta_parameters = [
        f"{component_name}.{parameter_name}"
        for component_name, module in components.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.is_meta
    ]
    if meta_parameters:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta parameters: " + ", ".join(meta_parameters[:8])
        )
    meta_buffers = [
        f"{component_name}.{buffer_name}"
        for component_name, module in components.items()
        for buffer_name, buffer in module.named_buffers()
        if buffer.is_meta
    ]
    if meta_buffers:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta buffers: " + ", ".join(meta_buffers[:8])
        )


def _materialize_language_buffers(
    language_model: torch.nn.Module,
    language_config: Any,
    *,
    load_device: str,
) -> None:
    """Recreate non-persistent Llama buffers omitted from safetensors.

    Transformers registers rotary frequencies with ``persistent=False``.
    Building the Llama on meta therefore leaves them meta-backed even after
    every checkpoint parameter is assigned.
    """

    meta_buffers = [name for name, value in language_model.named_buffers() if value.is_meta]
    if not meta_buffers:
        return
    model = getattr(language_model, "model", None)
    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        raise ModelPackageError(
            "OpenVLA-OFT language model retained meta buffers but exposes no rotary_emb"
        )
    try:
        model.rotary_emb = type(rotary)(
            language_config,
            device=torch.device(load_device),
        )
    except Exception as exc:
        raise ModelPackageError(
            f"could not materialize OpenVLA-OFT Llama rotary buffers: {exc}"
        ) from exc
    remaining = [name for name, value in language_model.named_buffers() if value.is_meta]
    if remaining:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta buffers: " + ", ".join(remaining[:8])
        )


def _action_statistics(
    root: Path,
    config: Mapping[str, Any],
    *,
    action_dim: int | None,
) -> tuple[list[float], list[float], list[bool], str]:
    dataset_statistics = _read_json(
        root / "dataset_statistics.json",
        description="OpenVLA-OFT dataset statistics",
    )
    requested_key = config.get("unnorm_key")
    candidate_keys = (
        (
            str(requested_key),
            f"{requested_key}_no_noops",
        )
        if requested_key
        else ()
    )
    selected_key = next(
        (key for key in candidate_keys if key in dataset_statistics),
        next(iter(dataset_statistics), ""),
    )
    if not selected_key:
        raise ModelPackageError("OpenVLA-OFT dataset_statistics.json is empty")
    entry = dataset_statistics[selected_key]
    try:
        stats = entry["action"]
        q01 = list(stats["q01"])
        q99 = list(stats["q99"])
    except (KeyError, TypeError) as exc:
        raise ModelPackageError(
            f"OpenVLA-OFT statistics {selected_key!r} have no action q01/q99"
        ) from exc
    inferred_dim = len(q01)
    if len(q99) != inferred_dim:
        raise ModelPackageError("OpenVLA-OFT action q01/q99 lengths differ")
    if action_dim is not None and action_dim != inferred_dim:
        raise ModelPackageError(
            f"requested action_dim={action_dim}, but checkpoint statistics have {inferred_dim}"
        )
    mask = list(stats.get("mask", [True] * inferred_dim))
    if len(mask) != inferred_dim:
        raise ModelPackageError("OpenVLA-OFT action mask length differs from q01/q99")
    return q01, q99, mask, selected_key


def _resolve_torch_dtype(value: str | torch.dtype) -> torch.dtype:
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
        except KeyError as exc:
            raise ModelPackageError(f"unsupported OpenVLA-OFT load dtype {value!r}") from exc
    if not dtype.is_floating_point:
        raise ModelPackageError("OpenVLA-OFT load dtype must be floating point")
    return dtype


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_openvla_oft(
    checkpoint: str | Path,
    *,
    load_device: str = "cpu",
    load_dtype: str | torch.dtype = "bfloat16",
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
    action_dim: int | None = None,
    action_horizon: int | None = None,
    max_prompt_length: int = 50,
) -> tuple[OpenVLAOFTReferenceModule, Any, dict[str, Any]]:
    """Load an RLinf-compatible discrete OpenVLA-OFT checkpoint.

    The 7B Llama is created on the meta device and shards are assigned directly,
    avoiding a second full-size randomly initialized language-model allocation.
    timm 0.9 cannot initialize wholly on meta, so its smaller vision/projector
    leaf is constructed on CPU directly in ``load_dtype``.
    """

    root = resolve_openvla_oft_checkpoint(
        checkpoint,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        raise ModelPackageError(
            "OpenVLA-OFT requires transformers; install the 'openvla_oft' extra"
        ) from exc
    config = _read_json(root / "config.json", description="OpenVLA-OFT config")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ModelPackageError("OpenVLA-OFT config.json has no text_config object")
    fused = bool(config.get("use_fused_vision_backbone", True))
    num_images_in_input = int(config.get("num_images_in_input") or 1)
    if num_images_in_input != 1:
        raise ModelPackageError(
            "the first OpenVLA-OFT adapter slice supports exactly one input image; "
            f"checkpoint config requests {num_images_in_input}"
        )
    try:
        timm_model_ids = tuple(config["timm_model_ids"])
        image_sizes = tuple(config.get("image_sizes", (224, 224)))
    except (KeyError, TypeError) as exc:
        raise ModelPackageError("OpenVLA-OFT config lacks timm_model_ids/image_sizes") from exc
    activation_layers = tuple(config.get("timm_override_act_layers", (None,) * len(timm_model_ids)))
    q01, q99, mask, statistics_key = _action_statistics(
        root,
        config,
        action_dim=action_dim,
    )
    resolved_action_dim = len(q01)
    resolved_horizon = int(
        action_horizon if action_horizon is not None else config.get("num_action_chunks", 8)
    )
    if resolved_horizon <= 0:
        raise ModelPackageError("OpenVLA-OFT action_horizon must be greater than zero")
    pad_multiple = int(config.get("pad_to_multiple_of", 64))
    try:
        padded_vocab_size = int(text_config["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelPackageError("OpenVLA-OFT text_config has no valid vocab_size") from exc
    action_vocab_size = padded_vocab_size - pad_multiple
    n_action_bins = int(config.get("n_action_bins", 256))
    if action_vocab_size <= 0:
        raise ModelPackageError("OpenVLA-OFT unpadded vocabulary size is invalid")

    try:
        from .processing_openvla_oft import OpenVLAOFTProcessor
        from .vision_prismatic import PrismaticProjector, PrismaticVisionBackbone
    except ImportError as exc:
        raise ModelPackageError(
            "OpenVLA-OFT vision loading requires timm and torchvision; "
            "install the 'openvla_oft' extra"
        ) from exc

    dtype = _resolve_torch_dtype(load_dtype)
    try:
        # timm 0.9 calls Tensor.item() while constructing stochastic-depth
        # schedules, which is unsupported for meta tensors. Building this leaf
        # directly in the checkpoint dtype keeps the temporary allocation
        # bounded; the 7B Llama below remains meta-initialized.
        with _default_dtype(dtype):
            vision_backbone = PrismaticVisionBackbone(
                use_fused_vision_backbone=fused,
                image_sizes=image_sizes,
                timm_model_ids=timm_model_ids,
                timm_override_act_layers=activation_layers,
            )
            vision_backbone.set_num_images_in_input(num_images_in_input)
            language_config = LlamaConfig(**text_config)
            projector = PrismaticProjector(
                use_fused_vision_backbone=fused,
                vision_dim=vision_backbone.embed_dim,
                llm_dim=int(language_config.hidden_size),
            )
        with torch.device("meta"), _default_dtype(dtype):
            language_model = LlamaForCausalLM(language_config)
    except Exception as exc:
        raise ModelPackageError(f"could not construct OpenVLA-OFT modules: {exc}") from exc
    _materialize_language_buffers(
        language_model,
        language_config,
        load_device=load_device,
    )

    _load_checkpoint_components(
        root,
        vision_backbone=vision_backbone,
        projector=projector,
        language_model=language_model,
        load_device=load_device,
        load_dtype=dtype,
        strict=strict,
    )
    head = CategoricalActionHead(
        action_vocab_size,
        n_action_bins,
        resolved_action_dim,
        resolved_horizon,
        q01,
        q99,
        mask,
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
            "statistics_key": statistics_key,
            "action_vocab_size": action_vocab_size,
            "padded_vocab_size": padded_vocab_size,
            "n_action_bins": n_action_bins,
            "num_images_in_input": vision_backbone.get_num_images_in_input(),
            "image_sizes": image_sizes,
            "timm_model_ids": timm_model_ids,
            "load_dtype": str(dtype),
            "vvla_commit": _VVLA_SOURCE_COMMIT,
        },
    )


__all__ = [
    "OpenVLAOFTReferenceModule",
    "load_openvla_oft",
    "resolve_openvla_oft_checkpoint",
]
