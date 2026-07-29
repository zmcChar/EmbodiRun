"""A local Hugging Face cloud planner for VLABench Texas Hold'em.

The provider deliberately consumes only the public planning contract.  A
simulator, robot, or dataset adapter must publish the visible cards through
``PlanRequest.goal.metadata`` or observation metadata before calling it.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from embodied_runtime.contracts import PlanEnvelope, PlanRequest, PlanStep

_METADATA_NAMESPACE = "texas_holdem"
_SUPPORTED_HAND_TYPES = frozenset(
    {
        "royal_flush",
        "straight_flush",
        "four_of_a_kind",
        "full_house",
        "flush",
        "straight",
        "three_of_a_kind",
        "two_pair",
        "one_pair",
        "high_card",
    }
)
_TARGET_COUNT_BY_HAND_TYPE = {
    "royal_flush": 5,
    "straight_flush": 5,
    "four_of_a_kind": 4,
    "full_house": 5,
    "flush": 5,
    "straight": 5,
    "three_of_a_kind": 3,
    "two_pair": 4,
    "one_pair": 2,
    "high_card": 1,
}
_THINK_THEN_JSON = re.compile(r"\A<think>.*?</think>\s*(\{.*\})\Z", re.DOTALL)
_SINGLE_JSON_FENCE = re.compile(r"\A```json\s*(\{.*\})\s*```\Z", re.DOTALL)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class PlannerInputError(ValueError):
    """The structured planner request is absent or violates its schema."""


class PlannerOutputError(ValueError):
    """The model response is not a valid, grounded planning result."""


class PlannerRuntimeError(RuntimeError):
    """The Hugging Face runtime could not load or generate a response."""


@dataclass(frozen=True, slots=True)
class TexasHoldemCard:
    """One visible card supplied to the planner without simulator references."""

    name: str
    value: str
    suit: str

    def __post_init__(self) -> None:
        for field_name in ("name", "value", "suit"):
            raw = getattr(self, field_name)
            if not isinstance(raw, str):
                raise PlannerInputError(f"card {field_name} must be a string")
            normalized = raw.strip()
            if not normalized:
                raise PlannerInputError(f"card {field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

    @property
    def primitive_instruction(self) -> str:
        return f"primitive: Please pick the poker {self.value} of {self.suit}"


@dataclass(frozen=True, slots=True)
class HfTexasHoldemPlannerConfig:
    """Placement, loading, and generation settings for one planner model."""

    checkpoint: str
    device: str = "cuda"
    dtype: str = "float16"
    model_kind: str = "image_text_to_text"
    max_new_tokens: int = 512
    plan_ttl_s: float = 300.0
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"
    skill: str = "pick_and_place_poker"
    allow_single_json_fence: bool = False

    def __post_init__(self) -> None:
        for field_name in ("checkpoint", "device", "dtype", "model_kind", "skill"):
            raw = getattr(self, field_name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, raw.strip())
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")
        if self.model_kind not in {"image_text_to_text", "causal_lm"}:
            raise ValueError("model_kind must be 'image_text_to_text' or 'causal_lm'")
        if not _IDENTIFIER.fullmatch(self.skill):
            raise ValueError("skill must be a valid planning identifier")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if isinstance(self.plan_ttl_s, bool) or not isinstance(self.plan_ttl_s, (int, float)):
            raise TypeError("plan_ttl_s must be a real number")
        normalized_ttl = float(self.plan_ttl_s)
        if not math.isfinite(normalized_ttl) or normalized_ttl <= 0.0:
            raise ValueError("plan_ttl_s must be finite and greater than zero")
        object.__setattr__(self, "plan_ttl_s", normalized_ttl)
        for field_name in (
            "local_files_only",
            "trust_remote_code",
            "allow_single_json_fence",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in ("cache_dir", "revision", "attn_implementation"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or a non-empty string")
            if isinstance(value, str):
                object.__setattr__(self, field_name, value.strip())


@runtime_checkable
class _TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class _TexasHoldemSelection:
    hand_type: str
    target_card_names: tuple[str, ...]


class _HuggingFaceTextGenerator:
    """Small synchronous wrapper around a lazily loaded HF model."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch_module: Any,
        max_new_tokens: int,
    ) -> None:
        self._model = model
        self._processor = processor
        self._torch = torch_module
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a deterministic task planner. Follow the requested JSON "
                            "schema exactly."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = getattr(self._model, "device", None)
            if input_device is not None:
                if hasattr(inputs, "to"):
                    inputs = inputs.to(input_device)
                else:
                    inputs = {
                        key: value.to(input_device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
            input_ids = inputs["input_ids"]
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            trimmed_ids = [
                output_ids[len(source_ids) :]
                for source_ids, output_ids in zip(input_ids, generated_ids, strict=False)
            ]
            decoded = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as error:
            raise PlannerRuntimeError("Hugging Face planner generation failed") from error
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)) or not decoded:
            raise PlannerRuntimeError("Hugging Face processor returned no decoded response")
        response = decoded[0]
        if not isinstance(response, str) or not response.strip():
            raise PlannerRuntimeError("Hugging Face planner returned an empty response")
        return response


class HfTexasHoldemPlanner:
    """Generate grounded task plans from structured card metadata.

    Construction never imports Transformers or loads weights.  The first
    ``plan`` or ``plan_async`` call creates one runtime and subsequent calls
    reuse it.  Generation is serialized because a single local model instance
    is not assumed to be re-entrant.
    """

    def __init__(
        self,
        config: HfTexasHoldemPlannerConfig,
        *,
        clock: Callable[[], float] = time.time,
        generator_factory: Callable[[], _TextGenerator] | None = None,
    ) -> None:
        if not isinstance(config, HfTexasHoldemPlannerConfig):
            raise TypeError("config must be an HfTexasHoldemPlannerConfig")
        self.config = config
        self._clock = clock
        self._generator_factory = generator_factory
        self._generator: _TextGenerator | None = None
        self._runtime_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._generator is not None

    def plan(self, request: PlanRequest) -> PlanEnvelope:
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")
        cards = _cards_from_request(request)
        if request.goal.allowed_skills and self.config.skill not in request.goal.allowed_skills:
            raise PlannerInputError(
                f"planner skill {self.config.skill!r} is not allowed by the task goal"
            )
        prompt = build_texas_holdem_prompt(
            instruction=request.goal.instruction,
            cards=cards,
        )
        with self._runtime_lock:
            generator = self._get_or_load_generator()
            try:
                response = generator.generate(prompt)
            except (PlannerInputError, PlannerOutputError, PlannerRuntimeError):
                raise
            except Exception as error:
                raise PlannerRuntimeError("planner text generation failed") from error
        selection = parse_texas_holdem_selection(
            response,
            cards=cards,
            allow_single_json_fence=self.config.allow_single_json_fence,
        )
        card_by_name = {card.name: card for card in cards}
        steps = tuple(
            PlanStep(
                step_id=f"select_card_{index:02d}",
                instruction=card_by_name[name].primitive_instruction,
                skill=self.config.skill,
                success_criteria=f"{name} is contained by the target placemat.",
                metadata={
                    "poker_name": name,
                    "target_index": index - 1,
                    "hand_type": selection.hand_type,
                },
            )
            for index, name in enumerate(selection.target_card_names, start=1)
        )
        created_at_s = _valid_clock_value(self._clock())
        revision = 1 if request.active_revision is None else request.active_revision + 1
        return PlanEnvelope(
            request_id=request.request_id,
            task_id=request.goal.task_id,
            session_id=request.goal.session_id,
            revision=revision,
            steps=steps,
            created_at_s=created_at_s,
            expires_at_s=created_at_s + self.config.plan_ttl_s,
            based_on_observation_id=request.observation_id,
            metadata={
                "planner": "huggingface_texas_holdem",
                "checkpoint": self.config.checkpoint,
                "hand_type": selection.hand_type,
                "target_card_names": selection.target_card_names,
            },
        )

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope:
        return await asyncio.to_thread(self.plan, request)

    def _get_or_load_generator(self) -> _TextGenerator:
        if self._generator is None:
            factory = self._generator_factory
            try:
                generator = (
                    factory() if factory is not None else _load_huggingface_generator(self.config)
                )
            except PlannerRuntimeError:
                raise
            except Exception as error:
                raise PlannerRuntimeError(
                    f"failed to load planner checkpoint {self.config.checkpoint!r}"
                ) from error
            if not isinstance(generator, _TextGenerator):
                raise PlannerRuntimeError("generator factory returned an incompatible object")
            self._generator = generator
        return self._generator


def build_texas_holdem_prompt(
    *,
    instruction: str,
    cards: Sequence[TexasHoldemCard],
) -> str:
    """Build an oracle-comparable prompt using VLABench's target-card rules."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise PlannerInputError("instruction must be a non-empty string")
    normalized_cards = _validate_cards(cards)
    card_payload = [
        {"name": card.name, "value": card.value, "suit": card.suit} for card in normalized_cards
    ]
    hand_types = ", ".join(sorted(_SUPPORTED_HAND_TYPES))
    return (
        "Solve this Texas Hold'em card-selection task from the structured card list.\n"
        f"Task instruction: {instruction.strip()}\n"
        f"Cards in input order: {json.dumps(card_payload, ensure_ascii=True)}\n\n"
        "Use standard five-card poker category strength. Evaluate every five-card "
        "combination in input order and use the first combination that reaches the "
        "strongest category. Return only the category-defining cards: one card for "
        "high_card, two for one_pair, four for two_pair, three for three_of_a_kind, "
        "four for four_of_a_kind, and all five for straight, flush, full_house, "
        "straight_flush, or royal_flush. Every target must copy a card name exactly "
        "from the input.\n\n"
        "Return exactly one JSON object and no Markdown or explanation. The object "
        "must contain exactly these keys:\n"
        '{"hand_type":"<category>","target_card_names":["<input card name>",...]}\n'
        f"hand_type must be one of: {hand_types}."
    )


def parse_texas_holdem_selection(
    response: str,
    *,
    cards: Sequence[TexasHoldemCard],
    allow_single_json_fence: bool = False,
) -> _TexasHoldemSelection:
    """Parse a strict JSON model response and enforce grounding to input cards."""

    if not isinstance(response, str) or not response.strip():
        raise PlannerOutputError("planner response must be a non-empty string")
    if not isinstance(allow_single_json_fence, bool):
        raise TypeError("allow_single_json_fence must be a bool")
    normalized_cards = _validate_cards(cards)
    payload_text = response.strip()
    think_match = _THINK_THEN_JSON.fullmatch(payload_text)
    if think_match is not None:
        payload_text = think_match.group(1)
    elif allow_single_json_fence:
        fence_match = _SINGLE_JSON_FENCE.fullmatch(payload_text)
        if fence_match is not None:
            payload_text = fence_match.group(1)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise PlannerOutputError("planner response is not strict JSON") from error
    if not isinstance(payload, Mapping):
        raise PlannerOutputError("planner response must be a JSON object")
    expected_keys = {"hand_type", "target_card_names"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys.difference(payload))
        extra = sorted(set(payload).difference(expected_keys))
        raise PlannerOutputError(
            f"planner response schema mismatch; missing={missing}, extra={extra}"
        )
    hand_type = payload["hand_type"]
    if not isinstance(hand_type, str) or hand_type not in _SUPPORTED_HAND_TYPES:
        raise PlannerOutputError(f"unsupported hand_type: {hand_type!r}")
    raw_names = payload["target_card_names"]
    if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
        raise PlannerOutputError("target_card_names must be a JSON array")
    names = tuple(raw_names)
    if not names:
        raise PlannerOutputError("target_card_names must not be empty")
    if len(names) > 5:
        raise PlannerOutputError("target_card_names must contain at most five cards")
    if any(not isinstance(name, str) or not name for name in names):
        raise PlannerOutputError("every target_card_names item must be a non-empty string")
    if len(set(names)) != len(names):
        raise PlannerOutputError("target_card_names must not contain duplicates")
    expected_count = _TARGET_COUNT_BY_HAND_TYPE[hand_type]
    if len(names) != expected_count:
        raise PlannerOutputError(
            f"hand_type {hand_type!r} requires exactly {expected_count} target cards"
        )
    allowed_names = {card.name for card in normalized_cards}
    unknown = sorted(set(names).difference(allowed_names))
    if unknown:
        raise PlannerOutputError(f"planner selected cards absent from the input: {unknown}")
    return _TexasHoldemSelection(
        hand_type=hand_type,
        target_card_names=names,
    )


def _cards_from_request(request: PlanRequest) -> tuple[TexasHoldemCard, ...]:
    sources: list[tuple[str, Any]] = []
    goal_section = request.goal.metadata.get(_METADATA_NAMESPACE)
    if goal_section is not None:
        sources.append(("goal.metadata", goal_section))
    observation = request.observation
    if isinstance(observation, Mapping):
        observation_metadata = observation.get("metadata")
        if isinstance(observation_metadata, Mapping):
            observation_section = observation_metadata.get(_METADATA_NAMESPACE)
            if observation_section is not None:
                sources.append(("observation metadata", observation_section))
    if not sources:
        raise PlannerInputError(
            "missing texas_holdem metadata in goal.metadata or observation metadata"
        )
    parsed = tuple((name, _cards_from_metadata(value, source=name)) for name, value in sources)
    reference = parsed[0][1]
    for source_name, cards in parsed[1:]:
        if cards != reference:
            raise PlannerInputError(f"{source_name} cards conflict with {parsed[0][0]} cards")
    return reference


def _cards_from_metadata(value: Any, *, source: str) -> tuple[TexasHoldemCard, ...]:
    if not isinstance(value, Mapping):
        raise PlannerInputError(f"{source}.{_METADATA_NAMESPACE} must be a mapping")
    if set(value) != {"cards"}:
        raise PlannerInputError(
            f"{source}.{_METADATA_NAMESPACE} must contain exactly the 'cards' key"
        )
    raw_cards = value["cards"]
    if isinstance(raw_cards, (str, bytes)) or not isinstance(raw_cards, Sequence):
        raise PlannerInputError(f"{source} cards must be a sequence")
    cards: list[TexasHoldemCard] = []
    for index, raw_card in enumerate(raw_cards):
        if not isinstance(raw_card, Mapping):
            raise PlannerInputError(f"{source} card {index} must be a mapping")
        if set(raw_card) != {"name", "value", "suit"}:
            raise PlannerInputError(
                f"{source} card {index} must contain exactly name, value, and suit"
            )
        cards.append(
            TexasHoldemCard(
                name=raw_card["name"],
                value=raw_card["value"],
                suit=raw_card["suit"],
            )
        )
    return _validate_cards(cards)


def _validate_cards(cards: Sequence[TexasHoldemCard]) -> tuple[TexasHoldemCard, ...]:
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise PlannerInputError("cards must be a sequence of TexasHoldemCard values")
    normalized = tuple(cards)
    if len(normalized) < 5:
        raise PlannerInputError("Texas Hold'em planning requires at least five cards")
    if any(not isinstance(card, TexasHoldemCard) for card in normalized):
        raise PlannerInputError("cards must contain only TexasHoldemCard values")
    names = tuple(card.name for card in normalized)
    if len(set(names)) != len(names):
        raise PlannerInputError("input card names must be unique")
    physical_cards = tuple((card.value, card.suit) for card in normalized)
    if len(set(physical_cards)) != len(physical_cards):
        raise PlannerInputError("input value/suit card pairs must be unique")
    return normalized


def _valid_clock_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerRuntimeError("planner clock must return a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise PlannerRuntimeError("planner clock must return a finite non-negative value")
    return normalized


def _load_huggingface_generator(
    config: HfTexasHoldemPlannerConfig,
) -> _HuggingFaceTextGenerator:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise PlannerRuntimeError(
            "the Hugging Face planner requires torch and transformers"
        ) from error

    model_options: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "dtype": config.dtype if config.dtype == "auto" else getattr(torch, config.dtype),
    }
    shared_options: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.cache_dir is not None:
        model_options["cache_dir"] = config.cache_dir
        shared_options["cache_dir"] = config.cache_dir
    if config.revision is not None:
        model_options["revision"] = config.revision
        shared_options["revision"] = config.revision
    if config.attn_implementation is not None:
        model_options["attn_implementation"] = config.attn_implementation
    if config.device == "auto":
        model_options["device_map"] = "auto"

    model_class_name = (
        "AutoModelForImageTextToText"
        if config.model_kind == "image_text_to_text"
        else "AutoModelForCausalLM"
    )
    model_class = getattr(transformers, model_class_name, None)
    if model_class is None:
        raise PlannerRuntimeError(f"installed transformers does not provide {model_class_name}")
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            config.checkpoint,
            **shared_options,
        )
        model = model_class.from_pretrained(config.checkpoint, **model_options)
        if config.device != "auto":
            model = model.to(config.device)
        model.eval()
    except Exception as error:
        raise PlannerRuntimeError(
            f"failed to load planner checkpoint {config.checkpoint!r}"
        ) from error
    return _HuggingFaceTextGenerator(
        model=model,
        processor=processor,
        torch_module=torch,
        max_new_tokens=config.max_new_tokens,
    )


__all__ = [
    "HfTexasHoldemPlanner",
    "HfTexasHoldemPlannerConfig",
    "PlannerInputError",
    "PlannerOutputError",
    "PlannerRuntimeError",
    "TexasHoldemCard",
    "build_texas_holdem_prompt",
    "parse_texas_holdem_selection",
]
