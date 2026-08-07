"""Scoped SigLIP initialization used by the official StreamVLN loader."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_VISION_INITIALIZATION_LOCK = threading.Lock()


@contextmanager
def embedded_siglip_vision_initialization(
    enabled: bool = True,
) -> Iterator[None]:
    """Build the vision tower from the main checkpoint without a Hub lookup.

    The published StreamVLN shard index contains all 421 vision-tower tensors
    (26 encoder layers), but upstream still calls ``from_pretrained`` on
    ``google/siglip-so400m-patch14-384`` while constructing the parent model.
    We temporarily replace only that constructor hook with an equivalent empty
    27-layer tower, remove the final layer exactly as upstream does, and let
    Transformers populate its remaining weights from the main checkpoint.

    The class method is restored on context exit. A process-wide lock keeps
    multiple runtimes created through this module from observing one another's
    temporary patch.
    """

    if not enabled:
        yield
        return

    with _VISION_INITIALIZATION_LOCK:
        siglip = importlib.import_module("llava.model.multimodal_encoder.siglip_encoder")
        tower_type = siglip.SigLipVisionTower
        original_load_model = tower_type.load_model

        def load_from_embedded_checkpoint(tower: Any, device_map: Any = None) -> None:
            # ``device_map`` belongs to the redundant standalone download. The
            # parent StreamVLN model owns placement after loading its shards.
            del device_map
            if tower.is_loaded:
                return
            tower.vision_tower = siglip.SigLipVisionModel(tower.config)
            del tower.vision_tower.vision_model.encoder.layers[-1:]
            tower.vision_tower.vision_model.head = siglip.nn.Identity()
            tower.vision_tower.requires_grad_(False)
            tower.is_loaded = True

        tower_type.load_model = load_from_embedded_checkpoint
        try:
            yield
        finally:
            # Do not overwrite an unrelated third-party patch installed while
            # model construction was in progress.
            if tower_type.load_model is load_from_embedded_checkpoint:
                tower_type.load_model = original_load_model


__all__ = ["embedded_siglip_vision_initialization"]
