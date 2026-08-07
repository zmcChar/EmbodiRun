from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from embodied_runtime.models.vla.navila.errors import NaVILAInferenceError
from embodied_runtime.models.vla.navila.preprocessing import prepare_navila_images


class _Tensor:
    shape = (8, 3, 384, 384)

    def __init__(self) -> None:
        self.moves = []

    def to(self, device, *, dtype):
        self.moves.append((device, dtype))
        return self


def test_preprocessing_left_pads_black_and_keeps_current_last() -> None:
    captured = []
    tensor = _Tensor()

    def process_images(images, processor, model_config):
        assert processor.size == {"height": 384, "width": 384}
        assert model_config == "config"
        captured.extend(image.getpixel((0, 0)) for image in images)
        return tensor

    frames = (
        np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8),
        np.full((3, 4, 3), (40, 50, 60), dtype=np.uint8),
    )
    result = prepare_navila_images(
        frames,
        image_processor=SimpleNamespace(size={"height": 384, "width": 384}),
        model_config="config",
        process_images=process_images,
        image_module=Image,
        device="cuda:0",
        torch_dtype="float16",
    )
    assert result is tensor
    assert captured == [(0, 0, 0)] * 6 + [(10, 20, 30), (40, 50, 60)]
    assert tensor.moves == [("cuda:0", "float16")]


def test_preprocessing_rejects_wrong_processor_and_tensor_shapes() -> None:
    frame = (np.zeros((2, 2, 3), dtype=np.uint8),)
    with pytest.raises(NaVILAInferenceError, match="384x384"):
        prepare_navila_images(
            frame,
            image_processor=SimpleNamespace(size={"height": 224, "width": 224}),
            model_config=object(),
            process_images=lambda *_: _Tensor(),
            image_module=Image,
            device="cpu",
            torch_dtype="float16",
        )

    wrong = SimpleNamespace(shape=(1, 3, 384, 384), to=lambda *args, **kwargs: None)
    with pytest.raises(NaVILAInferenceError, match="shape"):
        prepare_navila_images(
            frame,
            image_processor=SimpleNamespace(size={"height": 384, "width": 384}),
            model_config=object(),
            process_images=lambda *_: wrong,
            image_module=Image,
            device="cpu",
            torch_dtype="float16",
        )
