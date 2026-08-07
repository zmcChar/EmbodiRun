from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodied_runtime.models.vla.navila.config import NaVILAConfig
from embodied_runtime.models.vla.navila.errors import NaVILALoadError
from embodied_runtime.models.vla.navila.evaluator import NaVILAEvaluator
from embodied_runtime.models.vla.navila.loader import materialize_navila
from embodied_runtime.models.vla.navila.repository import validate_module_origins

OFFICIAL_MODULES = (
    "llava.constants",
    "llava.mm_utils",
    "llava.model.builder",
)


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "NaVILA"
    for relative in ("llava/constants.py", "llava/mm_utils.py", "llava/model/builder.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return root


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

    def __getitem__(self, key):
        if isinstance(key, int):
            return _Row(self.rows[key])
        row_slice, column_slice = key
        selected = self.rows[row_slice]
        return _Batch(tuple(row[column_slice] for row in selected))


class _Tokenizer:
    eos_token_id = 128001
    bos_token_id = 128000

    def __init__(self) -> None:
        self.decoded_batches: list[tuple[tuple[int, ...], ...]] = []

    def batch_decode(self, batch: _Batch, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        self.decoded_batches.append(batch.rows)
        return ["The action is stop.<|eot_id|>" if batch.rows[0] else ""]


def _evaluator(tmp_path: Path) -> tuple[NaVILAEvaluator, _Tokenizer]:
    tokenizer = _Tokenizer()
    torch = SimpleNamespace(
        device=lambda value: f"device:{value}",
        float16=object(),
    )
    evaluator = NaVILAEvaluator(
        NaVILAConfig(_source_tree(tmp_path)),
        tokenizer=tokenizer,
        model=SimpleNamespace(config=object()),
        image_processor=object(),
        process_images=lambda *_: None,
        tokenizer_image_token=lambda *_args, **_kwargs: None,
        stopping_criteria_type=object,
        image_token_index=-200,
        torch_module=torch,
        image_module=object(),
    )
    return evaluator, tokenizer


@pytest.mark.parametrize(
    ("generated", "expected_decoded_ids"),
    [
        pytest.param([[17, 18]], ((17, 18),), id="completion-only"),
        pytest.param(
            [[101, 102, 17, 18]],
            ((17, 18),),
            id="prompt-plus-completion",
        ),
        pytest.param([[128000, 17, 18]], ((17, 18),), id="bos-prefixed-completion"),
    ],
)
def test_evaluator_decodes_supported_generate_return_shapes(
    tmp_path: Path,
    generated: list[list[int]],
    expected_decoded_ids: tuple[tuple[int, ...], ...],
) -> None:
    evaluator, tokenizer = _evaluator(tmp_path)

    decoded, token_ids = evaluator._decode(
        _Batch(generated),
        _Batch([[101, 102]]),
    )

    assert decoded == "The action is stop."
    assert token_ids == (17, 18)
    assert tokenizer.decoded_batches == [expected_decoded_ids]


def test_evaluator_does_not_decode_an_echoed_prompt_as_a_stop_action(tmp_path: Path) -> None:
    evaluator, tokenizer = _evaluator(tmp_path)

    decoded, token_ids = evaluator._decode(
        _Batch([[101, 102]]),
        _Batch([[101, 102]]),
    )

    assert decoded == ""
    assert token_ids == ()
    assert tokenizer.decoded_batches == [((),)]


class _Model:
    def __init__(self) -> None:
        self.config = "official-model-config"
        self.requires_grad_calls: list[bool] = []
        self.eval_calls = 0

    def requires_grad_(self, enabled: bool) -> _Model:
        self.requires_grad_calls.append(enabled)
        return self

    def eval(self) -> _Model:
        self.eval_calls += 1
        return self


def test_loader_forwards_official_local_options_and_constructs_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodied_runtime.models.vla.navila import loader as loader_module

    root = _source_tree(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = NaVILAConfig(
        root,
        checkpoint,
        device="cuda:7",
        cuda_memory_fraction=0.40,
        local_files_only=True,
    )
    events: list[tuple[object, ...]] = []
    tokenizer = object()
    model = _Model()
    image_processor = object()
    evaluator = object()
    process_images = object()
    tokenizer_image_token = object()
    stopping_criteria_type = object()

    class Builder:
        @staticmethod
        def load_pretrained_model(*args, **kwargs):
            events.append(("load_pretrained_model", args, kwargs))
            return tokenizer, model, image_processor, 4096

    mm_utils = SimpleNamespace(
        get_model_name_from_path=lambda path: (
            events.append(("model_name", path)) or "navila-checkpoint"
        ),
        process_images=process_images,
        tokenizer_image_token=tokenizer_image_token,
        KeywordsStoppingCriteria=stopping_criteria_type,
    )
    torch = SimpleNamespace(
        device=lambda value: events.append(("device", value)) or f"device:{value}",
        cuda=SimpleNamespace(
            set_per_process_memory_fraction=lambda fraction, *, device: events.append(
                ("memory_fraction", fraction, device)
            )
        ),
        float16=object(),
    )
    modules = {
        "torch": torch,
        "PIL.Image": "pil-image-module",
        "llava.constants": SimpleNamespace(IMAGE_TOKEN_INDEX=-200),
        "llava.mm_utils": mm_utils,
        "llava.model.builder": Builder,
    }
    monkeypatch.setattr(
        loader_module,
        "activate_repository",
        lambda selected: events.append(("activate", selected)),
    )
    monkeypatch.setattr(
        loader_module,
        "validate_module_origins",
        lambda selected, *, require_loaded: events.append(("validate", selected, require_loaded)),
    )
    monkeypatch.setattr(loader_module.importlib, "import_module", modules.__getitem__)

    def evaluator_factory(selected_config, **kwargs):
        events.append(("evaluator", selected_config, kwargs))
        return evaluator

    loaded = materialize_navila(config, evaluator_factory=evaluator_factory)

    assert loaded.model is model
    assert loaded.evaluator is evaluator
    assert events[0:2] == [
        ("activate", root.resolve()),
        ("validate", root.resolve(), False),
    ]
    assert ("validate", root.resolve(), True) in events
    load_event = next(event for event in events if event[0] == "load_pretrained_model")
    assert load_event[1] == (str(checkpoint.resolve()), "navila-checkpoint")
    assert load_event[2] == {
        "model_base": None,
        "device": "cuda:7",
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    assert ("memory_fraction", 0.40, "device:cuda:7") in events
    assert model.requires_grad_calls == [False]
    assert model.eval_calls == 1
    evaluator_event = next(event for event in events if event[0] == "evaluator")
    assert evaluator_event[1] is config
    assert evaluator_event[2] == {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "stopping_criteria_type": stopping_criteria_type,
        "image_token_index": -200,
        "torch_module": torch,
        "image_module": "pil-image-module",
    }


def test_repository_origin_validation_accepts_only_the_selected_local_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _source_tree(tmp_path)
    relative_paths = {
        "llava.constants": "llava/constants.py",
        "llava.mm_utils": "llava/mm_utils.py",
        "llava.model.builder": "llava/model/builder.py",
    }
    for name, relative in relative_paths.items():
        module = ModuleType(name)
        module.__file__ = str(root / relative)
        monkeypatch.setitem(sys.modules, name, module)

    validate_module_origins(root, require_loaded=True)

    foreign = tmp_path / "foreign" / "constants.py"
    foreign.parent.mkdir()
    foreign.write_text("# foreign fixture\n", encoding="utf-8")
    foreign_module = ModuleType("llava.constants")
    foreign_module.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "llava.constants", foreign_module)

    with pytest.raises(NaVILALoadError, match="outside"):
        validate_module_origins(root, require_loaded=False)


def test_repository_origin_validation_requires_all_official_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _source_tree(tmp_path)
    for name in OFFICIAL_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(NaVILALoadError, match="did not load"):
        validate_module_origins(root, require_loaded=True)
