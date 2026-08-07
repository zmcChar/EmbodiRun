"""Construct the selected model-coupled navigation policy."""

from embodied_runtime.policies.navigation import (
    InternVLANavigationPolicy,
    QwenNavigationPolicy,
    StreamVLNNavigationPolicy,
)
from embodied_runtime.tasks.navigation import NavigationPolicy

from .settings import PolicySettings


def build_navigation_policy(settings: PolicySettings) -> NavigationPolicy:
    """Construct one policy; existing external model servers are never managed."""

    if settings.backend == "qwen":
        return QwenNavigationPolicy(
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            timeout_s=settings.qwen_timeout_s,
        )
    if settings.backend == "streamvln":
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
    return InternVLANavigationPolicy(
        variant=settings.internvla_variant,
        model_path=settings.internvla_model_path,
        device=settings.internvla_device,
        internnav_root=settings.internvla_root,
    )


__all__ = ["build_navigation_policy"]
