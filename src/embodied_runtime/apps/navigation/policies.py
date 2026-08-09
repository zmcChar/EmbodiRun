"""Construct the selected model-coupled navigation policy."""

from __future__ import annotations

from embodied_runtime.policies.navigation import (
    InternVLANavigationPolicy,
    NaVILANavigationPolicy,
    QwenNavigationPolicy,
    StreamVLNNavigationPolicy,
)

from .interfaces import ManagedNavigationPolicy
from .settings import PolicySettings

SUPPORTED_RUNTIMES_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen": ("transformers",),
    "streamvln": ("transformers", "vllm-omni"),
    "internvla": ("transformers",),
    "navila": ("transformers", "vllm-omni"),
    "activevln": ("transformers", "vvla"),
}


def supported_runtimes(model: str) -> tuple[str, ...]:
    """Return selectable runtimes for one navigation model."""

    try:
        return SUPPORTED_RUNTIMES_BY_MODEL[model]
    except KeyError as error:
        raise ValueError(f"unsupported navigation model: {model}") from error


def _require_supported_runtime(settings: PolicySettings) -> None:
    supported = supported_runtimes(settings.model)
    if settings.runtime not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"runtime {settings.runtime!r} is not supported for navigation model "
            f"{settings.model!r}; supported runtimes: {choices}"
        )


def _build_vllm_omni_policy(settings: PolicySettings) -> ManagedNavigationPolicy:
    # Keep this optional integration out of every Transformers-only import and
    # command.  Capability validation runs before the import as a fail-fast
    # guard for unsupported model/runtime pairs.
    from embodied_runtime.integrations.navigation.vllm_omni import (
        VllmOmniNaVILANavigationPolicy,
        VllmOmniStreamVLNNavigationPolicy,
    )

    options = {
        "url": settings.vllm_omni_url,
        "timeout_s": settings.vllm_omni_timeout_s,
        "session_id": settings.vllm_omni_session_id,
    }
    if settings.model == "streamvln":
        return VllmOmniStreamVLNNavigationPolicy(**options)
    if settings.model == "navila":
        return VllmOmniNaVILANavigationPolicy(**options)
    # Kept as a defensive boundary if the capability table is changed without
    # adding the matching factory branch.
    raise ValueError(f"runtime 'vllm-omni' has no navigation policy factory for {settings.model!r}")


def _build_vvla_policy(settings: PolicySettings) -> ManagedNavigationPolicy:
    from embodied_runtime.integrations.navigation.vvla import VvlaActiveVLNNavigationPolicy

    if settings.model != "activevln":
        raise ValueError(f"runtime 'vvla' has no navigation policy factory for {settings.model!r}")
    return VvlaActiveVLNNavigationPolicy(
        vvla_root=settings.vvla_root,
        checkpoint=settings.activevln_checkpoint,
        revision=settings.activevln_revision,
        device=settings.activevln_device,
        dtype=settings.activevln_dtype,
        attention=settings.activevln_attention,
        max_new_tokens=settings.activevln_max_new_tokens,
        max_context=settings.activevln_max_context,
        allow_download=settings.activevln_allow_download,
        do_sample=settings.activevln_do_sample,
    )


def _build_transformers_activevln_policy(settings: PolicySettings) -> ManagedNavigationPolicy:
    from embodied_runtime.integrations.navigation.activevln_transformers import (
        TransformersActiveVLNNavigationPolicy,
    )

    if settings.activevln_attention == "eager_bc":
        raise ValueError("attention 'eager_bc' is VVLA-only; use eager or sdpa with Transformers")
    return TransformersActiveVLNNavigationPolicy(
        vvla_root=settings.vvla_root,
        checkpoint=settings.activevln_checkpoint,
        revision=settings.activevln_revision,
        device=settings.activevln_device,
        dtype=settings.activevln_dtype,
        attention=settings.activevln_attention,
        max_new_tokens=settings.activevln_max_new_tokens,
        max_context=settings.activevln_max_context,
        allow_download=settings.activevln_allow_download,
        do_sample=settings.activevln_do_sample,
    )


def build_navigation_policy(settings: PolicySettings) -> ManagedNavigationPolicy:
    """Construct one policy; existing external model servers are never managed."""

    _require_supported_runtime(settings)
    if settings.runtime == "vllm-omni":
        return _build_vllm_omni_policy(settings)
    if settings.runtime == "vvla":
        return _build_vvla_policy(settings)
    if settings.model == "activevln":
        return _build_transformers_activevln_policy(settings)

    if settings.model == "qwen":
        return QwenNavigationPolicy(
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            timeout_s=settings.qwen_timeout_s,
        )
    if settings.model == "streamvln":
        if settings.streamvln_root is None or settings.streamvln_model_path is None:
            raise ValueError("StreamVLN requires repository and checkpoint paths")
        return StreamVLNNavigationPolicy(
            streamvln_root=settings.streamvln_root,
            model_path=settings.streamvln_model_path,
            device=settings.streamvln_device,
            cuda_memory_fraction=settings.cuda_memory_fraction,
            max_new_tokens=settings.max_new_tokens,
            local_files_only=settings.local_files_only,
            warmup=settings.warmup,
        )
    if settings.model == "internvla":
        return InternVLANavigationPolicy(
            variant=settings.internvla_variant,
            model_path=settings.internvla_model_path,
            device=settings.internvla_device,
            internnav_root=settings.internvla_root,
        )
    if settings.model == "navila":
        if settings.navila_root is None or settings.navila_model_path is None:
            raise ValueError("NaVILA requires repository and checkpoint paths")
        return NaVILANavigationPolicy(
            navila_root=settings.navila_root,
            model_path=settings.navila_model_path,
            device=settings.navila_device,
            cuda_memory_fraction=settings.navila_cuda_memory_fraction,
            max_new_tokens=settings.navila_max_new_tokens,
            local_files_only=settings.navila_local_files_only,
        )
    raise ValueError(f"unsupported navigation model: {settings.model}")


__all__ = [
    "SUPPORTED_RUNTIMES_BY_MODEL",
    "build_navigation_policy",
    "supported_runtimes",
]
