"""Canonical application API for the paired VLABench get-coffee pilot."""

from .composition import run_vlabench_pilot
from .settings import (
    GET_COFFEE_COMPOSITE_PROMPT,
    GET_COFFEE_ORACLE_STEPS,
    GET_COFFEE_TASK,
    VLABenchPilotConfig,
)

__all__ = [
    "GET_COFFEE_COMPOSITE_PROMPT",
    "GET_COFFEE_ORACLE_STEPS",
    "GET_COFFEE_TASK",
    "VLABenchPilotConfig",
    "run_vlabench_pilot",
]
