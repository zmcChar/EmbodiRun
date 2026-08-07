"""Report orchestration and deterministic output renderers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ._serialization import compact_json, csv_value, render_csv, trial_dict
from .comparisons import PairedComparison, paired_binary_comparison
from .oracle import OracleGateConfig, OracleGateResult, evaluate_oracle_gate
from .records import ExperimentCondition, TrialRecord, normalize_condition, record_sort_key
from .summaries import ConditionSummary, summarize_condition


class EvaluationReport:
    """Validated collection of paired trials with deterministic renderers."""

    def __init__(self, records: Iterable[TrialRecord]) -> None:
        try:
            normalized = tuple(records)
        except TypeError as error:
            raise TypeError("records must be an iterable of TrialRecord values") from error
        if not normalized:
            raise ValueError("an evaluation report requires at least one trial")
        if any(not isinstance(record, TrialRecord) for record in normalized):
            raise TypeError("records must contain only TrialRecord values")

        identities: dict[str, tuple[str, int]] = {}
        fingerprints: dict[str, str] = {}
        seen: set[tuple[ExperimentCondition, str]] = set()
        for record in normalized:
            key = (record.condition, record.pair_id)
            if key in seen:
                raise ValueError("each condition may contain at most one trial for a given pair_id")
            seen.add(key)
            identity = (record.task_id, record.seed)
            previous = identities.setdefault(record.pair_id, identity)
            if previous != identity:
                raise ValueError(
                    "a pair_id must refer to the same task_id and seed in every condition"
                )
            if record.initial_fingerprint is not None:
                previous_fingerprint = fingerprints.setdefault(
                    record.pair_id,
                    record.initial_fingerprint,
                )
                if previous_fingerprint != record.initial_fingerprint:
                    raise ValueError(
                        "paired conditions must start from the same observation fingerprint"
                    )

        self._records = tuple(sorted(normalized, key=record_sort_key))

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return self._records

    def summaries(self) -> tuple[ConditionSummary, ...]:
        present = {record.condition for record in self._records}
        return tuple(
            summarize_condition(record for record in self._records if record.condition is condition)
            for condition in ExperimentCondition
            if condition in present
        )

    def comparisons(
        self,
        *,
        baseline: ExperimentCondition = ExperimentCondition.EDGE_ONLY,
    ) -> tuple[PairedComparison, ...]:
        baseline = normalize_condition(baseline)
        present = {record.condition for record in self._records}
        if baseline not in present:
            return ()
        return tuple(
            paired_binary_comparison(
                self._records,
                baseline=baseline,
                comparator=condition,
            )
            for condition in ExperimentCondition
            if condition is not baseline and condition in present
        )

    def oracle_gate(
        self,
        config: OracleGateConfig | None = None,
    ) -> OracleGateResult:
        return evaluate_oracle_gate(self._records, config=config)

    def to_dict(
        self,
        *,
        oracle_gate_config: OracleGateConfig | None = None,
    ) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": 1,
            "trials": [trial_dict(record) for record in self._records],
            "condition_summaries": [summary.to_dict() for summary in self.summaries()],
            "paired_comparisons": [comparison.to_dict() for comparison in self.comparisons()],
        }
        if oracle_gate_config is not None:
            document["oracle_gate"] = self.oracle_gate(oracle_gate_config).to_dict()
        return document

    def to_json(
        self,
        *,
        oracle_gate_config: OracleGateConfig | None = None,
        indent: int | None = 2,
    ) -> str:
        if indent is not None and (
            isinstance(indent, bool) or not isinstance(indent, int) or indent < 0
        ):
            raise ValueError("indent must be a non-negative integer or None")
        return (
            json.dumps(
                self.to_dict(oracle_gate_config=oracle_gate_config),
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    def trials_csv(self) -> str:
        header = (
            "pair_id",
            "task_id",
            "seed",
            "condition",
            "success",
            "steps",
            "edge_latencies_ms",
            "planner_latencies_ms",
            "chunk_generation_latencies_ms",
            "queue_hit_latencies_ms",
            "deadline_misses",
            "subgoals_completed",
            "subgoals_total",
            "initial_fingerprint",
        )
        rows = (
            (
                record.pair_id,
                record.task_id,
                record.seed,
                record.condition.value,
                int(record.success),
                record.steps,
                compact_json(record.edge_latencies_ms),
                compact_json(record.planner_latencies_ms),
                compact_json(record.chunk_generation_latencies_ms),
                compact_json(record.queue_hit_latencies_ms),
                record.deadline_misses,
                record.subgoals_completed,
                record.subgoals_total,
                record.initial_fingerprint or "",
            )
            for record in self._records
        )
        return render_csv(header, rows)

    def summaries_csv(self) -> str:
        dictionaries = tuple(summary.to_dict() for summary in self.summaries())
        header = tuple(ConditionSummary.__dataclass_fields__)
        return render_csv(
            header,
            (tuple(csv_value(document[name]) for name in header) for document in dictionaries),
        )

    def comparisons_csv(
        self,
        *,
        baseline: ExperimentCondition = ExperimentCondition.EDGE_ONLY,
    ) -> str:
        header = (
            "baseline",
            "comparator",
            "matched_pairs",
            "baseline_unmatched_trials",
            "comparator_unmatched_trials",
            "both_succeeded",
            "both_failed",
            "baseline_only_succeeded",
            "comparator_only_succeeded",
            "discordant_pairs",
            "effect_percentage_points",
            "exact_two_sided_p_value",
        )
        dictionaries = tuple(
            comparison.to_dict() for comparison in self.comparisons(baseline=baseline)
        )
        return render_csv(
            header,
            (tuple(csv_value(document[name]) for name in header) for document in dictionaries),
        )
