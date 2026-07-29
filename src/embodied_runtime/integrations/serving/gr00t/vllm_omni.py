"""Native vLLM-Omni GR00T provider over its OpenPI WebSocket endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    RawRequest,
)
from embodied_runtime.distributed.communication import OpenPiWebSocketEndpoint
from embodied_runtime.integrations.serving.base import ProviderCapabilities
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_ACTION_HORIZON,
    DEFAULT_ACTION_KEYS,
    DEFAULT_EMBODIMENT_TAG,
    Gr00tN17Adapter,
)


def _prepared_request(
    adapter: Gr00tN17Adapter,
    request: InferenceRequest,
) -> InferenceRequest:
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be an InferenceRequest")
    payload = request.payload
    if isinstance(payload, RawRequest):
        payload = adapter.preprocess_one(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("GR00T provider payload must be RawRequest or a mapping")
    return replace(request, payload=payload)


class _RemoteOpenPiGr00tProvider:
    """Model-aware normalization around a model-neutral OpenPI endpoint."""

    provider_name = "openpi"
    provider_runtime = "openpi"

    def __init__(
        self,
        adapter: Gr00tN17Adapter,
        endpoint: OpenPiWebSocketEndpoint,
    ) -> None:
        self.adapter = adapter
        self.endpoint = endpoint
        self._closed = False
        self._capabilities = ProviderCapabilities(
            name=type(self).provider_name,
            runtime=type(self).provider_runtime,
            model=adapter.describe(),
            is_remote=True,
            transport="openpi_websocket",
            features=frozenset(
                {
                    "async_infer",
                    "connection_epoch",
                    "policy_handshake",
                    "session_reset",
                }
            ),
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        session_id: str | None = None,
        timeout_s: float = 120.0,
        adapter: Gr00tN17Adapter | None = None,
        **endpoint_options: Any,
    ):
        adapter = adapter or Gr00tN17Adapter()
        spec = adapter.describe()
        endpoint = OpenPiWebSocketEndpoint(
            url,
            session_id=session_id,
            timeout_s=timeout_s,
            expected_batch_size=1,
            expected_action_horizon=spec.action_horizon,
            expected_action_dim=spec.action_dim,
            **endpoint_options,
        )
        return cls(adapter, endpoint)

    @property
    def connected(self) -> bool:
        return not self._closed and self.endpoint.connected

    @property
    def connection_epoch(self) -> int:
        return self.endpoint.connection_epoch

    @property
    def server_metadata(self) -> Mapping[str, Any] | None:
        return self.endpoint.server_metadata

    async def connect(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        metadata = await self.endpoint.connect()
        try:
            embodiment = metadata.get("embodiment_tag")
            if embodiment is not None and embodiment != DEFAULT_EMBODIMENT_TAG:
                raise RuntimeError(
                    "GR00T server embodiment does not match the initial DROID-only "
                    f"client contract: {embodiment!r}"
                )
            horizon = metadata.get("action_horizon")
            if horizon is not None and int(horizon) != DEFAULT_ACTION_HORIZON:
                raise RuntimeError(
                    f"GR00T server action_horizon does not match the DROID contract: {horizon!r}"
                )
            keys = metadata.get("action_keys")
            if keys is not None and set(keys) != set(DEFAULT_ACTION_KEYS):
                raise RuntimeError(
                    f"GR00T server action_keys do not match the DROID contract: {keys!r}"
                )
        except Exception:
            await self.aclose()
            raise
        return metadata

    async def reset(
        self,
        reset_info: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> str:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        return await self.endpoint.reset(reset_info, session_id=session_id)

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        prepared = _prepared_request(self.adapter, request)
        remote_result = await self.endpoint.infer_async(prepared)
        raw_output = remote_result.output
        if not isinstance(raw_output, Mapping):
            raise TypeError("OpenPI endpoint output must be a mapping")
        actions = raw_output.get("actions")
        if not isinstance(actions, Mapping) or not actions:
            raise TypeError("GR00T OpenPI output must contain a named action mapping")

        # vLLM-Omni returns B=1. Accept an equivalent unbatched response from a
        # conforming OpenPI endpoint, then restore the adapter's explicit batch
        # boundary before unbatching exactly once.
        batched_actions: dict[str, Any] = {}
        for name, value in actions.items():
            ndim = getattr(value, "ndim", None)
            if ndim == 2:
                value = value[None, ...]
            batched_actions[str(name)] = value
        output = self.adapter.unbatch(
            {
                "actions": batched_actions,
                "info": {
                    "server_metadata": dict(self.endpoint.server_metadata or {}),
                    "response_metadata": remote_result.metadata.get(
                        "response_metadata",
                        {},
                    ),
                },
            },
            batch_size=1,
        )[0]
        return InferenceResult(
            request_id=remote_result.request_id,
            output=output,
            status=remote_result.status,
            queue_time_s=remote_result.queue_time_s,
            execution_time_s=remote_result.execution_time_s,
            metadata={
                **remote_result.metadata,
                "provider": self.provider_name,
                "provider_runtime": self.provider_runtime,
            },
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.endpoint.aclose()
        self._closed = True


class VllmOmniGr00tProvider(_RemoteOpenPiGr00tProvider):
    """The native vLLM-Omni GR00T N1.7 OpenPI provider."""

    provider_name = "vllm-omni"
    provider_runtime = "vllm_omni_native"


__all__ = ["VllmOmniGr00tProvider"]
