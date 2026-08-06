"""Lazy, stateful runtime for the official StreamVLN real-world checkpoint."""

from __future__ import annotations

import importlib
import math
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import (
    MAX_FUTURE_ACTIONS,
    StreamVLNNativeOutputError,
    StreamVLNWaypoint,
    StreamVLNWaypointPlan,
    actions_to_cumulative_waypoints,
    normalize_native_actions,
)
from .evaluator import StreamVLNEvaluator

DEFAULT_MODEL = "mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
DEFAULT_SENSOR_CONFIG: dict[str, object] = {
    "rgb_height": 1.25,
    "camera_intrinsic": (
        (192.0, 0.0, 191.42857143, 0.0),
        (0.0, 192.0, 191.42857143, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
}

_REPOSITORY_PATH_LOCK = threading.Lock()
_VISION_INITIALIZATION_LOCK = threading.Lock()


class StreamVLNLoadError(RuntimeError):
    """The official repository, dependencies, or checkpoint could not load."""


class StreamVLNInferenceError(RuntimeError):
    """The recurrent evaluator failed after it had been loaded."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cuda_fraction(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004
            "cuda_memory_fraction must be a number in (0, 1]"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("cuda_memory_fraction must be a finite number in (0, 1]")
    return result


def _checkpoint_name(value: str | Path) -> str:
    if isinstance(value, Path):
        text = str(value.expanduser())
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise TypeError("model_path must be a string or pathlib.Path")
    if not text:
        raise ValueError("model_path must not be empty")
    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return text


@dataclass(frozen=True, slots=True)
class StreamVLNConfig:
    """Validated settings for one checkpoint-owning runtime process."""

    streamvln_root: str | Path
    model_path: str | Path = DEFAULT_MODEL
    device: str = "cuda:0"
    cuda_memory_fraction: float | None = None
    num_future_steps: int = MAX_FUTURE_ACTIONS
    num_frames: int = 32
    num_history: int = 8
    model_max_length: int = 4096
    max_new_tokens: int = 10_000
    dtype: str = "bfloat16"
    attn_implementation: str | None = "sdpa"
    revision: str | None = None
    local_files_only: bool = False
    use_embedded_vision_weights: bool = True
    warmup: bool = True

    def __post_init__(self) -> None:
        root = Path(self.streamvln_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"StreamVLN root does not exist: {root}")
        object.__setattr__(self, "streamvln_root", root)
        object.__setattr__(self, "model_path", _checkpoint_name(self.model_path))
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        object.__setattr__(self, "device", self.device.strip())
        object.__setattr__(
            self,
            "cuda_memory_fraction",
            _cuda_fraction(self.cuda_memory_fraction),
        )
        for name in (
            "num_future_steps",
            "num_frames",
            "num_history",
            "model_max_length",
            "max_new_tokens",
        ):
            _positive_int(getattr(self, name), name)
        if self.num_future_steps > MAX_FUTURE_ACTIONS:
            raise ValueError(f"num_future_steps must not exceed {MAX_FUTURE_ACTIONS}")
        if self.dtype != "bfloat16":
            raise ValueError("StreamVLN currently supports only dtype='bfloat16'")
        if self.attn_implementation is not None and (
            not isinstance(self.attn_implementation, str) or not self.attn_implementation.strip()
        ):
            raise ValueError("attn_implementation must be None or a non-empty string")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be None or a non-empty string")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")  # noqa: TRY004
        if not isinstance(self.use_embedded_vision_weights, bool):
            raise ValueError(  # noqa: TRY004
                "use_embedded_vision_weights must be a boolean"
            )
        if not isinstance(self.warmup, bool):
            raise ValueError("warmup must be a boolean")  # noqa: TRY004


@dataclass(frozen=True, slots=True)
class StreamVLNPrediction:
    """Native checkpoint output plus a model-local cumulative Go2 path."""

    actions: tuple[int, ...]
    raw_output: str
    generation_time_s: float
    plan: StreamVLNWaypointPlan

    @property
    def waypoints(self) -> tuple[StreamVLNWaypoint, ...]:
        return self.plan.waypoints

    @property
    def terminal(self) -> bool:
        return self.plan.terminal

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "raw_output": self.raw_output,
            "generation_time_s": self.generation_time_s,
            **self.plan.as_dict(),
        }


def repository_import_paths(streamvln_root: str | Path) -> tuple[Path, Path]:
    """Return the two import roots required by the upstream source layout.

    The repository root exposes ``llava`` and the ``streamvln`` namespace.  Its
    nested ``streamvln`` directory exposes the upstream top-level ``model`` and
    ``utils`` namespaces.  The official loader needs both.
    """

    root = Path(streamvln_root).expanduser().resolve()
    return root, root / "streamvln"


def activate_repository_imports(streamvln_root: str | Path) -> tuple[Path, Path]:
    """Put the official source roots at deterministic import precedence."""

    root, nested = repository_import_paths(streamvln_root)
    with _REPOSITORY_PATH_LOCK:
        # Inserting root first leaves nested first, which is required for the
        # official ``from model...`` and ``from utils...`` imports.
        for path in (root, nested):
            text = str(path)
            while text in sys.path:
                sys.path.remove(text)
            sys.path.insert(0, text)
    return root, nested


_OFFICIAL_MODULES = (
    "llava",
    "llava.model.multimodal_encoder.siglip_encoder",
    "model.stream_video_vln",
    "utils.utils",
)


def validate_repository_module_origins(
    streamvln_root: str | Path,
    *,
    require_loaded: bool,
) -> None:
    """Fail fast if generic upstream module names resolve to another project.

    StreamVLN uses the process-global names ``llava``, ``model``, and ``utils``.
    Changing ``sys.path`` cannot replace an object already cached in
    ``sys.modules``, so silently accepting a foreign module would mix model
    implementations in one process.  A dedicated model worker remains the
    safest deployment; this check makes an accidental shared-process collision
    explicit.
    """

    root = Path(streamvln_root).expanduser().resolve()
    for name in _OFFICIAL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            if require_loaded:
                raise StreamVLNLoadError(
                    f"official StreamVLN import did not load required module {name!r}"
                )
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not source:
            raise StreamVLNLoadError(f"cannot verify cached module {name!r}: it has no __file__")
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise StreamVLNLoadError(
                f"cached module {name!r} comes from {resolved}, outside {root}"
            )


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

    The class method is restored on context exit.  A process-wide lock keeps
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
            # ``device_map`` belongs to the redundant standalone download.  The
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


class StreamVLNRuntime:
    """Lazy owner of one model and one recurrent evaluator.

    Construction performs validation only.  PyTorch, Transformers, the
    official repository, and checkpoint weights are first touched by
    :meth:`load`, :meth:`reset_memory`, or :meth:`predict`.
    """

    def __init__(
        self,
        *,
        streamvln_root: str | Path,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cuda:0",
        cuda_memory_fraction: float | None = None,
        num_future_steps: int = MAX_FUTURE_ACTIONS,
        num_frames: int = 32,
        num_history: int = 8,
        model_max_length: int = 4096,
        max_new_tokens: int = 10_000,
        dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        revision: str | None = None,
        local_files_only: bool = False,
        use_embedded_vision_weights: bool = True,
        warmup: bool = True,
        evaluator_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = StreamVLNConfig(
            streamvln_root=streamvln_root,
            model_path=model_path,
            device=device,
            cuda_memory_fraction=cuda_memory_fraction,
            num_future_steps=num_future_steps,
            num_frames=num_frames,
            num_history=num_history,
            model_max_length=model_max_length,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            attn_implementation=attn_implementation,
            revision=revision,
            local_files_only=local_files_only,
            use_embedded_vision_weights=use_embedded_vision_weights,
            warmup=warmup,
        )
        self._evaluator_factory = evaluator_factory or StreamVLNEvaluator
        self._load_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._evaluator: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None
        self._state_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._evaluator is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def evaluator(self) -> Any | None:
        return self._evaluator

    @property
    def requires_reset(self) -> bool:
        """Whether a failed partial cadence blocks further inference."""

        with self._state_lock:
            return self._state_error is not None

    def _pretrained_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "local_files_only": self.config.local_files_only,
        }
        if self.config.revision is not None:
            options["revision"] = self.config.revision
        return options

    def _load(self) -> Any:
        if self._evaluator is not None:
            return self._evaluator
        with self._load_lock:
            if self._evaluator is not None:
                return self._evaluator
            try:
                activate_repository_imports(self.config.streamvln_root)
                validate_repository_module_origins(
                    self.config.streamvln_root,
                    require_loaded=False,
                )
                torch = importlib.import_module("torch")
                torch_device = torch.device(self.config.device)
                if self.config.cuda_memory_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.cuda_memory_fraction,
                        device=torch_device,
                    )

                transformers = importlib.import_module("transformers")
                pretrained = self._pretrained_options()
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    model_max_length=self.config.model_max_length,
                    padding_side="right",
                    **pretrained,
                )
                hf_config = transformers.AutoConfig.from_pretrained(
                    self.config.model_path,
                    **pretrained,
                )
                model_dtype = getattr(torch, self.config.dtype)
                model_options: dict[str, object] = {
                    "torch_dtype": model_dtype,
                    "config": hf_config,
                    "low_cpu_mem_usage": False,
                    **pretrained,
                }
                if self.config.attn_implementation is not None:
                    model_options["attn_implementation"] = self.config.attn_implementation
                with embedded_siglip_vision_initialization(self.config.use_embedded_vision_weights):
                    # Import after installing the patch: importing the native
                    # class loads llava's builder and captures this tower type.
                    native_module = importlib.import_module("model.stream_video_vln")
                    validate_repository_module_origins(
                        self.config.streamvln_root,
                        require_loaded=True,
                    )
                    model_type = native_module.StreamVLNForCausalLM
                    model = model_type.from_pretrained(
                        self.config.model_path,
                        **model_options,
                    )
                model.model.num_history = self.config.num_history
                model.reset(1)
                model.requires_grad_(False)
                model.to(torch_device)
                model.eval()

                evaluator = self._evaluator_factory(
                    DEFAULT_SENSOR_CONFIG,
                    model=model,
                    tokenizer=tokenizer,
                    device=torch_device,
                    num_frames=self.config.num_frames,
                    num_future_steps=self.config.num_future_steps,
                    num_history=self.config.num_history,
                    max_new_tokens=self.config.max_new_tokens,
                    environment_id=0,
                )
                if self.config.warmup:
                    numpy = importlib.import_module("numpy")
                    evaluator.reset_memory()
                    evaluator.step(
                        0,
                        numpy.zeros((480, 640, 3), dtype=numpy.uint8),
                        "Move forward 25 centimeters.",
                        run_model=True,
                    )
                    evaluator.reset_memory()

                self._model = model
                self._evaluator = evaluator
                self._load_error = None
            except Exception as error:
                self._load_error = f"{type(error).__name__}: {error}"
                raise StreamVLNLoadError(f"failed to load StreamVLN: {self._load_error}") from error
        return self._evaluator

    def load(self) -> StreamVLNRuntime:
        """Explicitly materialize the lazy runtime and return ``self``."""

        self._load()
        return self

    def reset_memory(self) -> None:
        with self._state_lock:
            evaluator = self._load()
            try:
                evaluator.reset_memory()
            except Exception as error:
                self._state_error = f"reset failed: {type(error).__name__}: {error}"
                raise StreamVLNInferenceError(self._state_error) from error
            self._state_error = None

    def reset(self) -> None:
        """Short spelling used by episode-oriented providers."""

        self.reset_memory()

    def predict(self, rgb_image: Any, instruction: str) -> StreamVLNPrediction:
        """Run one official four-step real-world service cadence."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        with self._state_lock:
            if self._state_error is not None:
                raise StreamVLNInferenceError(
                    "StreamVLN state was discarded after a partial inference; "
                    "call reset() before retrying"
                )
            return self._predict_locked(rgb_image, instruction)

    def _discard_failed_state(self, evaluator: Any, error: Exception) -> str:
        detail = f"{type(error).__name__}: {error}"
        try:
            evaluator.reset_memory()
        except Exception as reset_error:  # noqa: BLE001 - preserve the primary failure
            detail += f"; evaluator reset also failed: {type(reset_error).__name__}: {reset_error}"
        # Even when the defensive reset succeeds, callers must explicitly
        # acknowledge that the episode's recurrent context was lost.
        self._state_error = detail
        return detail

    def _predict_locked(self, rgb_image: Any, instruction: str) -> StreamVLNPrediction:
        evaluator = self._load()
        latest_actions: Sequence[object] | None = None
        latest_output = ""
        generation_time_s = 0.0
        try:
            for _ in range(self.config.num_future_steps):
                run_model = evaluator.step_id % self.config.num_future_steps == 0
                actions, generate_time, raw_output = evaluator.step(
                    0,
                    rgb_image,
                    instruction,
                    run_model=run_model,
                )
                if actions is not None:
                    latest_actions = actions
                if raw_output is not None:
                    latest_output = str(raw_output)
                if generate_time is not None:
                    generation_time_s = max(generation_time_s, float(generate_time))
                evaluator.step_id += 1
            if latest_actions is None:
                raise StreamVLNNativeOutputError("StreamVLN evaluator returned no action sequence")

            normalized = normalize_native_actions(
                latest_actions,
                max_actions=self.config.num_future_steps,
            )
            plan = actions_to_cumulative_waypoints(
                normalized,
                max_actions=self.config.num_future_steps,
            )
            return StreamVLNPrediction(
                actions=normalized,
                raw_output=latest_output,
                generation_time_s=generation_time_s,
                plan=plan,
            )
        except StreamVLNNativeOutputError as error:
            self._discard_failed_state(evaluator, error)
            raise
        except Exception as error:
            detail = self._discard_failed_state(evaluator, error)
            raise StreamVLNInferenceError(
                f"StreamVLN evaluator failed and requires reset: {detail}"
            ) from error

    infer = predict


# The migration source called this object an engine.  Keep a source-compatible
# spelling while new providers use the more precise runtime name.
LazyStreamVLNEngine = StreamVLNRuntime


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SENSOR_CONFIG",
    "LazyStreamVLNEngine",
    "StreamVLNConfig",
    "StreamVLNInferenceError",
    "StreamVLNLoadError",
    "StreamVLNPrediction",
    "StreamVLNRuntime",
    "activate_repository_imports",
    "embedded_siglip_vision_initialization",
    "repository_import_paths",
    "validate_repository_module_origins",
]
