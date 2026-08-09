"""Episode-aware ActiveVLN policy backed by the vendored VVLA engine."""

from __future__ import annotations

import asyncio
import gc
import importlib
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from embodied_runtime.policies.navigation.episode import EpisodeCursor
from embodied_runtime.policies.navigation.errors import NavigationPolicyError
from embodied_runtime.policies.navigation.validation import navigation_request
from embodied_runtime.policies.navigation.vision import decode_rgb
from embodied_runtime.tasks.navigation import NavigationRequest, WaypointPlan

from .backend import ValidatedActiveVLNBackend
from .checkpoint import verify_activevln_checkpoint
from .hardware import require_activevln_cuda_capacity
from .mapping import ActiveVLNPrediction, activevln_actions_to_waypoint_plan

ACTIVEVLN_CHECKPOINT = "Arvil/Qwen2.5-VL-3B_rl_r2r_4000"
ACTIVEVLN_REVISION = "160987313e3e869705f42400d1b8f28177044518"
DEFAULT_VVLA_ROOT = Path(__file__).resolve().parents[5] / "third_party" / "vvla"


class VvlaActiveVLNRuntime:
    """Lazy, single-GPU owner for VVLA's recurrent ActiveVLN engine."""

    def __init__(
        self,
        *,
        vvla_root: str | Path = DEFAULT_VVLA_ROOT,
        checkpoint: str | Path = ACTIVEVLN_CHECKPOINT,
        revision: str = ACTIVEVLN_REVISION,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention: str = "eager",
        max_new_tokens: int = 64,
        max_context: int = 32768,
        allow_download: bool = False,
        do_sample: bool = False,
        env_id: str = "rlinf-deploy",
    ) -> None:
        if not Path(checkpoint).expanduser().exists() and revision != ACTIVEVLN_REVISION:
            raise ValueError(
                "remote VVLA ActiveVLN requires the pinned immutable checkpoint revision "
                f"{ACTIVEVLN_REVISION}"
            )
        self.vvla_root = Path(vvla_root).expanduser()
        self.checkpoint = str(checkpoint)
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.attention = attention
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context
        self.allow_download = allow_download
        self.do_sample = do_sample
        self.env_id = env_id
        self._engine: Any | None = None
        self._observation_type: Any | None = None
        self._session_key_type: Any | None = None
        self._active_key: Any | None = None
        self._cancelled = threading.Event()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def backend(self) -> Any:
        if self._engine is None:
            raise RuntimeError("VVLA runtime is not loaded")
        return ValidatedActiveVLNBackend(self._engine.backend)

    @property
    def engine_dtype(self) -> str:
        if self._engine is None:
            raise RuntimeError("VVLA runtime is not loaded")
        return str(self._engine.core.dtype).removeprefix("torch.")

    def _make_importable(self) -> None:
        root = self.vvla_root.resolve()
        if not (root / "pyproject.toml").is_file() or not (root / "vvla" / "__init__.py").is_file():
            raise NavigationPolicyError(
                f"VVLA source is missing at {root}; initialize third_party/vvla or install vvla"
            )
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    @staticmethod
    def _torch_dtype(torch: Any, name: str) -> Any:
        choices = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            return choices[name]
        except KeyError as error:
            raise ValueError(
                "VVLA dtype must be 'auto', 'float16', 'bfloat16', or 'float32'"
            ) from error

    def load(self) -> None:
        if self._engine is not None:
            return
        self._make_importable()
        verify_activevln_checkpoint(self.vvla_root, self.checkpoint)
        try:
            torch = importlib.import_module("torch")
            vvla = importlib.import_module("vvla")
            policies = importlib.import_module("vvla.policies")
            types = importlib.import_module("vvla.types")
        except ImportError as error:
            raise NavigationPolicyError(
                "ActiveVLN through VVLA requires the isolated vvla-activevln environment"
            ) from error

        require_activevln_cuda_capacity(torch, self.device, self.dtype)

        policy = policies.make_policy(
            "activevln",
            checkpoint=self.checkpoint,
            revision=self.revision,
            allow_download=self.allow_download,
            attention=self.attention,
            max_new_tokens=self.max_new_tokens,
            max_context=self.max_context,
            do_sample=self.do_sample,
        )
        if self.dtype != "auto":
            # Cast before EngineCore transfers weights to avoid a transient fp32 GPU copy.
            policy = policy.to(self._torch_dtype(torch, self.dtype))
        engine_config = vvla.EngineConfig(
            device=self.device,
            dtype=self.dtype,
            max_batch_size=1,
            use_cuda_graph=False,
            capture_full_loop=False,
        )
        self._engine = vvla.Vvla(policy, engine_config=engine_config)
        self._observation_type = types.Observation
        self._session_key_type = types.SessionKey

    def _key(self, episode_id: str) -> Any:
        if self._session_key_type is None:
            raise RuntimeError("VVLA runtime is not loaded")
        return self._session_key_type(env_id=self.env_id, episode_id=episode_id)

    def reset(self) -> None:
        if self._engine is not None and self._active_key is not None:
            self._engine.backend.reset_sessions([self._active_key])
        self._active_key = None
        self._cancelled.clear()

    def cancel(self) -> None:
        self._cancelled.set()
        if self._engine is not None and self._active_key is not None:
            self._engine.backend.cancel_sessions([self._active_key])

    def predict(self, rgb: Any, instruction: str, *, episode_id: str) -> ActiveVLNPrediction:
        if self._engine is None or self._observation_type is None:
            raise RuntimeError("VVLA runtime is not loaded")
        if self._cancelled.is_set():
            raise NavigationPolicyError("VVLA ActiveVLN generation was cancelled")
        torch = importlib.import_module("torch")
        pixels = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        observation = self._observation_type(
            images=pixels,
            state=torch.empty(0),
            instruction_tokens=torch.empty(0, dtype=torch.long),
            instruction=instruction,
        )
        key = self._key(episode_id)
        if self._active_key is not None and self._active_key != key:
            self.reset()
        self._active_key = key
        if self._cancelled.is_set():
            self.cancel()
            raise NavigationPolicyError("VVLA ActiveVLN generation was cancelled")
        chunk = self.backend.generate([observation], session_ids=[key])[0]
        trace = chunk.trace
        parsed = trace.parsed_actions
        token_ids = tuple(int(value) for value in trace.token_ids.tolist())
        return ActiveVLNPrediction(
            actions=tuple(parsed.actions),
            text=parsed.raw_text,
            latency_ms=float(chunk.latency_ms),
            token_ids=token_ids,
        )

    def close(self) -> None:
        self.reset()
        self._engine = None
        self._observation_type = None
        self._session_key_type = None
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VvlaActiveVLNNavigationPolicy:
    """Adapt an ActiveVLN runtime to the navigation task contract.

    The historical public name is retained for compatibility; the lifecycle is
    backend-neutral and is also shared by the stock-Transformers runtime.
    """

    def __init__(
        self,
        runtime: Any | None = None,
        **runtime_options: Any,
    ) -> None:
        self.runtime = runtime or VvlaActiveVLNRuntime(**runtime_options)
        self._cursor = EpisodeCursor()
        self._lock = asyncio.Lock()
        self._closed = False

    async def prepare(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("ActiveVLN policy is closed")
            worker = asyncio.create_task(asyncio.to_thread(self.runtime.load))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Keep the lock until the non-cancellable load thread exits, then release its model.
                with suppress(Exception):
                    await worker
                await asyncio.to_thread(self.runtime.close)
                raise

    async def plan(self, request: NavigationRequest) -> WaypointPlan:
        navigation = navigation_request(request)
        observation = navigation.observation
        async with self._lock:
            if self._closed:
                raise RuntimeError("ActiveVLN policy is closed")
            should_reset = self._cursor.requires_reset(observation)

            def infer() -> WaypointPlan:
                if should_reset:
                    self.runtime.reset()
                prediction = self.runtime.predict(
                    decode_rgb(observation.latest_rgb),
                    navigation.instruction,
                    episode_id=observation.episode_id,
                )
                try:
                    return activevln_actions_to_waypoint_plan(
                        prediction.actions,
                        observation_sequence=observation.sequence,
                    )
                except ValueError as error:
                    raise NavigationPolicyError(str(error)) from error

            worker = asyncio.create_task(asyncio.to_thread(infer))
            try:
                plan = await asyncio.shield(worker)
            except asyncio.CancelledError:
                await asyncio.to_thread(self.runtime.cancel)
                with suppress(Exception):
                    await worker
                await asyncio.to_thread(self.runtime.reset)
                self._cursor.clear()
                raise
            except Exception:
                await asyncio.to_thread(self.runtime.reset)
                self._cursor.clear()
                raise
            self._cursor.commit(observation)
            return plan

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cursor.clear()
            worker = asyncio.create_task(asyncio.to_thread(self.runtime.close))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Closing owns the final model/session release.  Drain the
                # non-cancellable thread while keeping the policy closed.
                with suppress(Exception):
                    await worker
                raise
