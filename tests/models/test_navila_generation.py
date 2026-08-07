from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_runtime.models.vla.navila import evaluator as evaluator_module
from embodied_runtime.models.vla.navila.actions import NaVILAPrimitive
from embodied_runtime.models.vla.navila.config import NaVILAConfig
from embodied_runtime.models.vla.navila.evaluator import NaVILAEvaluator


class _Row:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values

    def detach(self) -> _Row:
        return self

    def cpu(self) -> _Row:
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class _Batch:
    def __init__(self, rows: list[list[int]] | tuple[tuple[int, ...], ...]) -> None:
        self.rows = tuple(tuple(row) for row in rows)

    def unsqueeze(self, dimension: int) -> _Batch:
        assert dimension == 0
        return self

    def to(self, device: str) -> _Batch:
        assert device == "cuda:0"
        return self

    def __getitem__(self, key):
        if isinstance(key, int):
            return _Row(self.rows[key])
        row_slice, column_slice = key
        rows = self.rows[row_slice]
        return _Batch(tuple(row[column_slice] for row in rows))


class _Tokenizer:
    eos_token_id = 128001
    bos_token_id = 128000

    @staticmethod
    def batch_decode(output: _Batch, *, skip_special_tokens: bool) -> list[str]:
        assert output.rows == ((17, 18),)
        assert skip_special_tokens is True
        return ["The action is move forward 25 cm.<|eot_id|>"]


class _Model:
    config = object()

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return _Batch([[101, 102, 17, 18]])


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "NaVILA"
    for relative in ("llava/constants.py", "llava/mm_utils.py", "llava/model/builder.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return root


def test_generation_disables_checkpoint_sampling_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    torch = SimpleNamespace(
        device=lambda value: value,
        float16=object(),
        ones_like=lambda _value: "attention-mask",
        inference_mode=nullcontext,
    )
    monkeypatch.setattr(evaluator_module, "prepare_navila_images", lambda *_args, **_kwargs: "images")
    evaluator = NaVILAEvaluator(
        NaVILAConfig(_source_tree(tmp_path)),
        tokenizer=_Tokenizer(),
        model=model,
        image_processor=object(),
        process_images=object(),
        tokenizer_image_token=lambda *_args, **_kwargs: _Batch([[101, 102]]),
        stopping_criteria_type=lambda *_args: "stop-criteria",
        image_token_index=-200,
        torch_module=torch,
        image_module=object(),
    )

    prediction = evaluator.predict((object(),), "go to the target")

    assert prediction.action.primitive is NaVILAPrimitive.MOVE_FORWARD
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["temperature"] is None
    assert model.generate_kwargs["top_p"] is None
    assert model.generate_kwargs["top_k"] is None
    assert model.generate_kwargs["attention_mask"] == "attention-mask"
