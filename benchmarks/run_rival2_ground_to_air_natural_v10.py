"""Train the causal prompt airborne follow before higher aerial continuation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_runner  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v7 as balanced  # noqa: E402
from rivalsim.rival2_ground_to_air_prompt_follow_v10 import (  # noqa: E402
    GROUND_TO_AIR_PROMPT_FOLLOW_V10_VERSION,
    PromptAerialFollowTrainingTracker,
)

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V10"
AUTHORITY = balanced.ROOT / "results/rival2/ground_to_air_natural_v10/authority.json"
AUTHORITY_SHA256 = "55CEC4D1AD920C4233BDA5E79FFB03EC34F4EFE904699F9DF1D3F4A290C718BD"
RESULTS = balanced.ROOT / "results/rival2/ground_to_air_natural_v10"
CHECKPOINTS = balanced.ROOT / "checkpoints/rival2/ground_to_air_natural_v10"
CHECKPOINT_NAME = "rival2_ground_to_air_natural_v10.pt"
STAGE_NAME = "ground_to_air_natural_v10"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v10")
POP_CONTROL_IDENTITY = (
    "RIVAL2_GROUND_TO_AIR_POP_CONTROL_V7+V8_HOLD24_RELEASE4+"
    + GROUND_TO_AIR_PROMPT_FOLLOW_V10_VERSION
)
OPTIMIZER_FORMAT = "RIVAL2_GROUND_TO_AIR_NATURAL_V10_FRESH_BALANCED_ADAMW"

_BASE_COLLECT = balanced.collect_rollout
_BASE_SELECTION_KEY = balanced.natural_v6.selection_key
_BASE_PASSES_GATE = balanced.natural_v4.passes_gate

_OVERRIDES: dict[str, Any] = {
    "VERSION": VERSION,
    "AUTHORITY": AUTHORITY,
    "AUTHORITY_SHA256": AUTHORITY_SHA256,
    "RESULTS": RESULTS,
    "CHECKPOINTS": CHECKPOINTS,
    "CHECKPOINT_NAME": CHECKPOINT_NAME,
    "STAGE_NAME": STAGE_NAME,
    "DEFAULT_RUN_DIR": DEFAULT_RUN_DIR,
    "POP_CONTROL_IDENTITY": POP_CONTROL_IDENTITY,
    "OPTIMIZER_FORMAT": OPTIMIZER_FORMAT,
    "VALIDATION_PHYSICAL_PROBE": True,
}


def collect_rollout(*args: Any, **kwargs: Any) -> Any:
    """Use the unchanged rollout path with the V10 training-only tracker."""

    original_tracker = goal_runner.GoalDirectedTrainingTracker
    try:
        goal_runner.GoalDirectedTrainingTracker = PromptAerialFollowTrainingTracker
        rollout, metrics = _BASE_COLLECT(*args, **kwargs)
    finally:
        goal_runner.GoalDirectedTrainingTracker = original_tracker
    metrics["fractions"]["prompt_airborne_follow"] = (
        metrics["telemetry"]["prompt_airborne_follow_touches"]
        / metrics["worlds"]
    )
    return rollout, metrics


def _row_gate(row: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    return authority["acceptance"]["per_setup_and_defender"][row["setup"]][
        row["defender_mode"]
    ]


def selection_key(
    rows: list[dict[str, Any]], authority: dict[str, Any]
) -> tuple[float, float]:
    """First satisfy causal prompt coverage, then optimize complete outcomes."""

    prompt_ratios = [
        min(
            float(row["fractions"]["prompt_airborne_follow"])
            / float(_row_gate(row, authority)["prompt_airborne_follow_fraction_min"]),
            1.0,
        )
        for row in rows
    ]
    prompt_score = 0.0
    for row in rows:
        weight = 1.5 if row["defender_mode"] == "live" else 1.0
        prompt_score += weight * float(
            row["fractions"]["prompt_airborne_follow"]
        )
    prompt_score /= len(rows)
    downstream = balanced.natural_v4.evaluation_score(rows)
    return float(min(prompt_ratios)), float(downstream + 0.5 * prompt_score)


def passes_gate(rows: list[dict[str, Any]], authority: dict[str, Any]) -> bool:
    acceptance = authority["acceptance"]
    for row in rows:
        fractions = row["fractions"]
        gate = _row_gate(row, authority)
        for name in (
            "pop_touch",
            "prompt_airborne_follow",
            "elevated_follow_touch",
            "high_follow_touch",
            "productive_continuation",
            "goal_within_contact_budget",
        ):
            if fractions[name] < float(gate[f"{name}_fraction_min"]):
                return False
        if fractions["contact_budget_exceeded"] > float(
            acceptance["contact_budget_exceeded_fraction_max"]
        ):
            return False
        if fractions["unassisted_or_ground_goal"] > float(
            acceptance["unassisted_or_ground_goal_fraction_max"]
        ):
            return False
        if not row["finite"] or max(row["analog_saturation_fraction"]) >= float(
            acceptance["maximum_analog_saturation_fraction"]
        ):
            return False
    return True


@contextmanager
def _configured_runner() -> Iterator[None]:
    """Bind the generic balanced runner to the prospective V10 authority."""

    original = {name: getattr(balanced, name) for name in _OVERRIDES}
    original_collect = balanced.collect_rollout
    original_selection = balanced.natural_v6.selection_key
    original_gate = balanced.natural_v4.passes_gate
    try:
        for name, value in _OVERRIDES.items():
            setattr(balanced, name, value)
        balanced.collect_rollout = collect_rollout
        balanced.natural_v6.selection_key = selection_key
        balanced.natural_v4.passes_gate = passes_gate
        yield
    finally:
        balanced.collect_rollout = original_collect
        balanced.natural_v6.selection_key = original_selection
        balanced.natural_v4.passes_gate = original_gate
        for name, value in original.items():
            setattr(balanced, name, value)


def load_authority() -> dict[str, Any]:
    with _configured_runner():
        authority = balanced.load_authority()
    if float(authority["reward"]["prompt_airborne_follow_event"]) <= 0.0:
        raise RuntimeError("V10 prompt airborne-follow event must be positive")
    if authority["integrity"]["named_mechanic_classifier_used"]:
        raise RuntimeError("V10 cannot use a named-mechanic classifier")
    return authority


def run(args: argparse.Namespace) -> int:
    with _configured_runner():
        return balanced.run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=balanced.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-stratum", type=int, default=256)
    parser.add_argument("--evaluation-worlds-per-row", type=int, default=256)
    parser.add_argument("--test-worlds-per-row", type=int, default=512)
    parser.add_argument("--maximum-blocks", type=int, default=96)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
