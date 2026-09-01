"""Deterministic, evaluation-only Nexto check for the masked Fresh Human Seed.

The selected Stage-1 checkpoint is loaded directly.  Rival acts once per 120 Hz
physics tick with deterministic hybrid actions; no reward or optimizer is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_fresh_human_seed_no_previous_action_v1 import (  # noqa: E402
    CHECKPOINT,
    CHECKPOINT_FORMAT,
    RESULTS,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.nexto_short_eval import (  # noqa: E402
    PHYSICS_HZ,
    TERMINATION_GOAL,
    TERMINATION_HARD_TIME,
    TERMINATION_NO_TOUCH,
    NextoShortEpisodeRunner,
)
from rivalsim.open_play import TOUCH_FORWARD  # noqa: E402
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402

COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
OUTPUT = RESULTS / "deterministic_nexto_closed_loop.json"
EVALUATION_SEED = 2026090108
WORLDS_PER_SIDE = 128


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def assignments(worlds_per_side: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(worlds_per_side, dtype=np.int32)
    side = np.concatenate(
        (
            np.zeros(worlds_per_side, dtype=np.int32),
            np.ones(worlds_per_side, dtype=np.int32),
        )
    )
    layout = np.concatenate((local % 5, local % 5)).astype(np.int32)
    return side, layout


def ratio(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(count / denominator)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_sha = file_sha256(args.checkpoint)
    side, layout = assignments(args.worlds_per_side)
    runner = NextoShortEpisodeRunner(
        side.size,
        str(args.collision_root),
        args.checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        starting_layout=layout,
        rival_side=side,
        stochastic_rival=False,
        evaluation_seed=args.seed,
        device=args.device,
        rival_policy_hz=120,
        accepted_stage1_checkpoint_format=CHECKPOINT_FORMAT,
        collect_open_play_telemetry=True,
    )
    if not runner.rival_policy.config.zero_previous_action_inputs:
        raise RuntimeError("closed-loop policy did not load the permanent input mask")
    timing = runner.run()
    export = runner.export()
    raw = export["raw"]
    open_raw = export["open_play_raw"]
    rows = np.arange(side.size, dtype=np.int64)
    opponent_side = 1 - side
    rival_touches_by_episode = raw["touch_count"][rows, side]
    nexto_touches_by_episode = raw["touch_count"][rows, opponent_side]
    rival_touches = int(rival_touches_by_episode.sum())
    nexto_touches = int(nexto_touches_by_episode.sum())
    episodes_with_rival_touch = int((rival_touches_by_episode > 0).sum())
    termination = raw["termination_kind"]
    winner = raw["winner"]
    decisive = termination == TERMINATION_GOAL
    rival_goals = int((decisive & (winner == side)).sum())
    nexto_goals = int((decisive & (winner == opponent_side)).sum())
    no_touch = int((termination == TERMINATION_NO_TOUCH).sum())
    hard_time = int((termination == TERMINATION_HARD_TIME).sum())
    rival_first = int(((raw["first_toucher"] >= 0) & (raw["first_toucher"] == side)).sum())
    # Open-play telemetry does not own the short evaluator's no-touch stop bit.
    # Exclude those worlds so state after their lifecycle reset cannot enter these
    # first-episode ball-direction and possession statistics.
    behavior_rows = np.flatnonzero(termination != TERMINATION_NO_TOUCH)
    behavior_side = side[behavior_rows]
    rival_forward_impulse_touches = int(
        open_raw["direction_count"][
            behavior_rows, behavior_side, TOUCH_FORWARD
        ].sum()
    )
    rival_forward_possessions = int(
        open_raw["displacement_count"][
            behavior_rows, behavior_side, TOUCH_FORWARD
        ].sum()
    )
    rival_challenge_exchanges = int(
        open_raw["possession_opponent"][behavior_rows, behavior_side].sum()
    )
    rival_ticks = int(raw["simulated_ticks"][rows, side].sum())
    rival_minutes = rival_ticks / (PHYSICS_HZ * 60.0)
    mean_speed = (
        None
        if rival_ticks == 0
        else float(raw["speed_sum"][rows, side].sum(dtype=np.float64) / rival_ticks)
    )
    episodes = int(side.size)

    # This is a direct behavioral description, not a model-selection metric.
    # It asks only whether the seed visibly participates in the game often enough
    # to be a credible future PPO starting point.
    functional = bool(
        episodes_with_rival_touch >= episodes // 2
        and no_touch < episodes // 2
        and rival_touches > 0
        and rival_first > 0
        and rival_forward_impulse_touches > 0
        and rival_challenge_exchanges > 0
    )
    result = {
        "format": "RIVAL2_FRESH_HUMAN_SEED_NO_PREVIOUS_ACTION_V1_DETERMINISTIC_NEXTO",
        "created_utc": utc_now(),
        "checkpoint": {
            "path": args.checkpoint.relative_to(ROOT).as_posix(),
            "sha256": checkpoint_sha,
            "format": CHECKPOINT_FORMAT,
            "policy_config_hash": Rival2PolicyConfig(
                zero_previous_action_inputs=True
            ).content_hash,
        },
        "execution": {
            "rival_policy_hz": 120,
            "physics_hz": PHYSICS_HZ,
            "rival_action": "deterministic_tanh_mean_and_button_threshold",
            "gaussian_sampling": False,
            "bernoulli_sampling": False,
            "previous_action_fields_forced_zero_inside_policy": True,
            "reward_used_for_optimization_or_selection": False,
            "optimizer_steps": 0,
            "episodes": episodes,
            "seed": args.seed,
            "worlds_per_side": args.worlds_per_side,
            "paired_kickoff_layouts": True,
        },
        "gameplay": {
            "rival_touches": rival_touches,
            "nexto_touches": nexto_touches,
            "rival_touches_per_minute": (
                None if rival_minutes == 0 else rival_touches / rival_minutes
            ),
            "episodes_with_rival_touch": episodes_with_rival_touch,
            "episodes_with_rival_touch_fraction": ratio(
                episodes_with_rival_touch, episodes
            ),
            "no_touch_resets": no_touch,
            "no_touch_fraction": ratio(no_touch, episodes),
            "hard_timeouts": hard_time,
            "rival_first_touches": rival_first,
            "rival_challenge_possession_exchanges": rival_challenge_exchanges,
            "rival_forward_ball_velocity_contacts": rival_forward_impulse_touches,
            "rival_forward_ball_displacement_possessions": rival_forward_possessions,
            "rival_goals": rival_goals,
            "nexto_goals": nexto_goals,
            "mean_rival_speed_uu_per_s": mean_speed,
        },
        "behavioral_interpretation": {
            "approaches_ball": episodes_with_rival_touch >= episodes // 2,
            "produces_legitimate_physics_touches": rival_touches > 0,
            "avoids_persistent_no_touch_resets": no_touch < episodes // 2,
            "challenges": rival_challenge_exchanges > 0 and rival_first > 0,
            "moves_ball_toward_opponent_goal": (
                rival_forward_impulse_touches > 0 and rival_forward_possessions > 0
            ),
            "scores_or_concedes_observed": rival_goals + nexto_goals > 0,
            "functional_gameplay_demonstrated": functional,
        },
        "performance": {
            "wall_seconds": timing.seconds,
            "world_ticks_per_second": timing.world_ticks_per_second,
            "peak_cuda_bytes": export["peak_cuda_bytes"],
        },
    }
    write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worlds-per-side", type=int, default=WORLDS_PER_SIDE)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.collision_root.is_dir():
        raise FileNotFoundError(args.collision_root)
    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
