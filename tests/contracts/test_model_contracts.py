from __future__ import annotations

import pytest

from embodied_runtime.contracts import (
    IterativeFlowPlan,
    ModelPackage,
    ModelSpec,
    SingleForwardPlan,
)


def test_flow_schedule_integrates_from_one_to_zero() -> None:
    schedule = IterativeFlowPlan(default_num_steps=4).schedule()
    assert schedule == ((1.0, -0.25), (0.75, -0.25), (0.5, -0.25), (0.25, -0.25))
    assert schedule[-1][0] + schedule[-1][1] == 0.0


def test_model_package_rejects_missing_plan_entrypoints() -> None:
    with pytest.raises(ValueError, match="denoise_step"):
        ModelPackage(
            spec=ModelSpec(model_id="test", family="flow"),
            entrypoints={
                "encode_prefix": lambda value: value,
                "init_state": lambda value: value,
                "finalize": lambda value: value,
            },
            plan=IterativeFlowPlan(),
        )


def test_model_package_has_nonempty_per_instance_identity() -> None:
    plan = IterativeFlowPlan()
    entrypoints = {name: (lambda value: value) for name in plan.required_entrypoints()}
    first = ModelPackage(
        spec=ModelSpec(model_id="test", family="flow"),
        entrypoints=entrypoints,
        plan=plan,
    )
    second = ModelPackage(
        spec=ModelSpec(model_id="test", family="flow"),
        entrypoints=entrypoints,
        plan=plan,
    )
    assert first.package_id
    assert first.package_id != second.package_id


def test_single_forward_plan_requires_only_forward_entrypoint() -> None:
    plan = SingleForwardPlan(forward="predict")
    package = ModelPackage(
        spec=ModelSpec(model_id="test", family="single"),
        entrypoints={"predict": lambda value: value},
        plan=plan,
    )

    assert plan.kind == "single_forward"
    assert plan.required_entrypoints() == ("predict",)
    with pytest.raises(AttributeError, match="iterative-flow"):
        _ = package.recipe


def test_iterative_flow_recipe_property_is_read_only_compatibility_view() -> None:
    plan = IterativeFlowPlan()
    package = ModelPackage(
        spec=ModelSpec(model_id="test", family="flow"),
        entrypoints={name: (lambda value: value) for name in plan.required_entrypoints()},
        plan=plan,
    )

    assert package.recipe is plan
    with pytest.raises(AttributeError):
        package.recipe = IterativeFlowPlan(default_num_steps=2)


def test_plan_validation_rejects_invalid_flow_controls() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        IterativeFlowPlan(default_num_steps=0)
    with pytest.raises(ValueError, match="distinct"):
        IterativeFlowPlan(encode="same", initialize="same")
    with pytest.raises(ValueError, match="must not be empty"):
        SingleForwardPlan(forward="")
