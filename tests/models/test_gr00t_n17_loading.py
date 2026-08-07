from __future__ import annotations

import pytest

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_EMBODIMENT_TAG,
    load_gr00t_n17,
)
from embodied_runtime.models.vla.gr00t_n17 import loading as loading_module


def test_loader_accepts_local_directory_without_network(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeOfficialPolicy:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        loading_module,
        "require_gr00t_policy",
        lambda: FakeOfficialPolicy,
    )
    policy = load_gr00t_n17(tmp_path, strict=False)

    assert isinstance(policy, FakeOfficialPolicy)
    assert calls == [
        {
            "embodiment_tag": DEFAULT_EMBODIMENT_TAG,
            "model_path": str(tmp_path.resolve()),
            "device": "cpu",
            "strict": False,
        }
    ]


def test_loader_resolves_hugging_face_repository_without_real_network(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    class FakeOfficialPolicy:
        def __init__(self, **kwargs) -> None:
            calls.append(("policy", kwargs))

    def fake_download(checkpoint, **kwargs):
        calls.append(("download", checkpoint, kwargs))
        return tmp_path

    monkeypatch.setattr(
        loading_module,
        "require_gr00t_policy",
        lambda: FakeOfficialPolicy,
    )
    monkeypatch.setattr(loading_module, "download_hf_snapshot", fake_download)

    load_gr00t_n17(
        "nvidia/GR00T-N1.7-3B",
        cache_dir="/cache",
        revision="revision-a",
        local_files_only=True,
    )

    assert calls[0] == (
        "download",
        "nvidia/GR00T-N1.7-3B",
        {
            "cache_dir": "/cache",
            "revision": "revision-a",
            "local_files_only": True,
        },
    )
    assert calls[1][0] == "policy"
    assert calls[1][1]["model_path"] == str(tmp_path)


def test_loader_rejects_direct_accelerator_placement(tmp_path) -> None:
    with pytest.raises(ModelPackageError, match="backend owns device placement"):
        load_gr00t_n17(tmp_path, load_device="cuda:0")
