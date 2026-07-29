from __future__ import annotations

import asyncio
import json

import pytest

from embodied_runtime.contracts import PlanRequest, TaskGoal
from embodied_runtime.distributed import AsyncPlannerEndpoint
from embodied_runtime.integrations.planning import (
    HfTexasHoldemPlanner,
    HfTexasHoldemPlannerConfig,
    PlannerInputError,
    PlannerOutputError,
    PlannerRuntimeError,
    TexasHoldemCard,
    build_texas_holdem_prompt,
    parse_texas_holdem_selection,
)

_CARDS = (
    TexasHoldemCard("ace_of_hearts", "ace", "hearts"),
    TexasHoldemCard("ace_of_spades", "ace", "spades"),
    TexasHoldemCard("king_of_clubs", "king", "clubs"),
    TexasHoldemCard("seven_of_diamonds", "7", "diamonds"),
    TexasHoldemCard("four_of_hearts", "4", "hearts"),
    TexasHoldemCard("three_of_clubs", "3", "clubs"),
    TexasHoldemCard("two_of_spades", "2", "spades"),
)


def _card_payload() -> list[dict[str, str]]:
    return [{"name": card.name, "value": card.value, "suit": card.suit} for card in _CARDS]


def _request(
    *,
    goal_cards: bool = True,
    observation_cards: bool = False,
    active_revision: int | None = None,
) -> PlanRequest:
    goal_metadata = {"texas_holdem": {"cards": _card_payload()}} if goal_cards else {}
    observation = (
        {"metadata": {"texas_holdem": {"cards": _card_payload()}}}
        if observation_cards
        else {"camera": "frame"}
    )
    return PlanRequest(
        goal=TaskGoal(
            task_id="texas_holdem",
            session_id="robot-1",
            instruction="Show me the largest cards on the placemat.",
            allowed_skills=("pick_and_place_poker",),
            metadata=goal_metadata,
        ),
        observation=observation,
        observation_id="frame-1",
        observation_timestamp_s=10.0,
        active_plan_id="old-plan" if active_revision is not None else None,
        active_revision=active_revision,
        requested_at_s=11.0,
        request_id="request-1",
    )


class _FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class _BrokenGenerator:
    def generate(self, prompt: str) -> str:
        del prompt
        raise OSError("synthetic model failure")


def _pair_response() -> str:
    return json.dumps(
        {
            "hand_type": "one_pair",
            "target_card_names": ["ace_of_hearts", "ace_of_spades"],
        }
    )


def test_provider_is_lazy_and_builds_bound_plan_envelope() -> None:
    generator = _FakeGenerator(_pair_response())
    factory_calls = 0

    def factory() -> _FakeGenerator:
        nonlocal factory_calls
        factory_calls += 1
        return generator

    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(
            checkpoint="/models/cosmos-reason2-2b",
            device="cuda:0",
            dtype="bfloat16",
            max_new_tokens=128,
            plan_ttl_s=60.0,
        ),
        clock=lambda: 100.0,
        generator_factory=factory,
    )

    assert isinstance(planner, AsyncPlannerEndpoint)
    assert not planner.loaded
    assert factory_calls == 0

    plan = planner.plan(_request(active_revision=4))

    assert planner.loaded
    assert factory_calls == 1
    assert plan.request_id == "request-1"
    assert plan.task_id == "texas_holdem"
    assert plan.session_id == "robot-1"
    assert plan.revision == 5
    assert plan.created_at_s == 100.0
    assert plan.expires_at_s == 160.0
    assert plan.based_on_observation_id == "frame-1"
    assert plan.metadata["hand_type"] == "one_pair"
    assert plan.metadata["target_card_names"] == (
        "ace_of_hearts",
        "ace_of_spades",
    )
    assert tuple(step.metadata["poker_name"] for step in plan.steps) == (
        "ace_of_hearts",
        "ace_of_spades",
    )
    assert tuple(step.instruction for step in plan.steps) == (
        "primitive: Please pick the poker ace of hearts",
        "primitive: Please pick the poker ace of spades",
    )
    assert all(step.skill == "pick_and_place_poker" for step in plan.steps)
    assert "Cards in input order" in generator.prompts[0]
    assert '"name": "ace_of_hearts"' in generator.prompts[0]

    planner.plan(_request())
    assert factory_calls == 1


def test_provider_reads_observation_metadata_without_simulator_objects() -> None:
    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="unused"),
        clock=lambda: 20.0,
        generator_factory=lambda: _FakeGenerator(_pair_response()),
    )

    plan = planner.plan(_request(goal_cards=False, observation_cards=True))

    assert tuple(step.metadata["poker_name"] for step in plan.steps) == (
        "ace_of_hearts",
        "ace_of_spades",
    )


def test_provider_rejects_conflicting_goal_and_observation_cards() -> None:
    request = _request(goal_cards=True, observation_cards=True)
    conflicting_observation = {
        "metadata": {
            "texas_holdem": {
                "cards": [
                    *_card_payload()[:-1],
                    {"name": "nine_of_spades", "value": "9", "suit": "spades"},
                ]
            }
        }
    }
    request = PlanRequest(
        goal=request.goal,
        observation=conflicting_observation,
        observation_id=request.observation_id,
        observation_timestamp_s=request.observation_timestamp_s,
        requested_at_s=request.requested_at_s,
        request_id=request.request_id,
    )
    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="unused"),
        generator_factory=lambda: _FakeGenerator(_pair_response()),
    )

    with pytest.raises(PlannerInputError, match="conflict"):
        planner.plan(request)

    assert not planner.loaded


def test_provider_requires_structured_card_metadata_and_allowed_skill() -> None:
    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="unused"),
        generator_factory=lambda: _FakeGenerator(_pair_response()),
    )
    with pytest.raises(PlannerInputError, match="missing texas_holdem metadata"):
        planner.plan(_request(goal_cards=False))

    request = _request()
    disallowed_goal = TaskGoal(
        task_id=request.goal.task_id,
        session_id=request.goal.session_id,
        instruction=request.goal.instruction,
        allowed_skills=("navigate",),
        metadata=request.goal.metadata,
    )
    disallowed_request = PlanRequest(
        goal=disallowed_goal,
        observation=request.observation,
        observation_id=request.observation_id,
        observation_timestamp_s=request.observation_timestamp_s,
    )
    with pytest.raises(PlannerInputError, match="not allowed"):
        planner.plan(disallowed_request)

    assert not planner.loaded


def test_async_planning_uses_the_same_contract() -> None:
    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="unused"),
        clock=lambda: 40.0,
        generator_factory=lambda: _FakeGenerator(_pair_response()),
    )

    plan = asyncio.run(planner.plan_async(_request()))

    assert plan.revision == 1
    assert plan.created_at_s == 40.0


@pytest.mark.parametrize(
    ("response", "match"),
    (
        ("not-json", "strict JSON"),
        (
            ('{"hand_type":"one_pair","target_card_names":["ace_of_hearts","ace_of_hearts"]}'),
            "duplicates",
        ),
        (
            ('{"hand_type":"one_pair","target_card_names":["ace_of_hearts","queen_of_moons"]}'),
            "absent from the input",
        ),
        (
            '{"hand_type":"one_pair","target_card_names":["ace_of_hearts"]}',
            "requires exactly 2",
        ),
        (
            (
                '{"hand_type":"one_pair","target_card_names":'
                '["ace_of_hearts","ace_of_spades"],"reason":"pair"}'
            ),
            "schema mismatch",
        ),
        (
            (
                '```json\n{"hand_type":"one_pair","target_card_names":'
                '["ace_of_hearts","ace_of_spades"]}\n```'
            ),
            "strict JSON",
        ),
    ),
)
def test_parser_rejects_malformed_or_ungrounded_output(
    response: str,
    match: str,
) -> None:
    with pytest.raises(PlannerOutputError, match=match):
        parse_texas_holdem_selection(response, cards=_CARDS)


def test_parser_accepts_cosmos_think_block_followed_by_strict_json() -> None:
    response = "<think>I compare all five-card combinations.</think>\n" + _pair_response()

    selection = parse_texas_holdem_selection(response, cards=_CARDS)

    assert selection.hand_type == "one_pair"
    assert selection.target_card_names == ("ace_of_hearts", "ace_of_spades")


def test_parser_optionally_accepts_one_bare_json_code_fence() -> None:
    response = f"```json\n{_pair_response()}\n```"

    selection = parse_texas_holdem_selection(
        response,
        cards=_CARDS,
        allow_single_json_fence=True,
    )

    assert selection.hand_type == "one_pair"
    assert selection.target_card_names == ("ace_of_hearts", "ace_of_spades")


def test_prompt_documents_vlabench_target_card_cardinality() -> None:
    prompt = build_texas_holdem_prompt(
        instruction="select the strongest cards",
        cards=_CARDS,
    )

    assert "one card for high_card" in prompt
    assert "two for one_pair" in prompt
    assert "four for two_pair" in prompt
    assert "all five for straight" in prompt
    assert "Return exactly one JSON object" in prompt


def test_runtime_failures_are_explicit_and_keep_the_cause() -> None:
    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="unused"),
        generator_factory=lambda: _BrokenGenerator(),
    )

    with pytest.raises(PlannerRuntimeError, match="text generation") as captured:
        planner.plan(_request())

    assert isinstance(captured.value.__cause__, OSError)


def test_generator_factory_failure_is_wrapped_with_checkpoint() -> None:
    def fail() -> _FakeGenerator:
        raise OSError("weights unavailable")

    planner = HfTexasHoldemPlanner(
        HfTexasHoldemPlannerConfig(checkpoint="/missing/checkpoint"),
        generator_factory=fail,
    )

    with pytest.raises(PlannerRuntimeError, match="/missing/checkpoint") as captured:
        planner.plan(_request())

    assert isinstance(captured.value.__cause__, OSError)
