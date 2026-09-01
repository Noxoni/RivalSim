"""Deterministic Nexto evaluation for Human Sequence recurrent PPO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
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

from rivalsim.nexto_short_eval import (  # noqa: E402
    PHYSICS_HZ,
    TERMINATION_GOAL,
    TERMINATION_HARD_TIME,
    TERMINATION_NO_TOUCH,
)
from rivalsim.open_play import TOUCH_FORWARD  # noqa: E402
from rivalsim.recurrent_nexto_eval import RecurrentNextoEpisodeRunner  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES  # noqa: E402
from rivalsim.rival2_recurrent_training import CHECKPOINT_FORMAT  # noqa: E402

FORMAT = "RIVAL2_HUMAN_SEQUENCE_RECURRENT_PPO_DETERMINISTIC_NEXTO"
COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
DEFAULT_SEED = 2026090305
DEFAULT_WORLDS_PER_SIDE = 128
LAYOUT_NAMES = (
    "diagonal_left",
    "diagonal_right",
    "off_center_left",
    "off_center_right",
    "center",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def evaluate_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    collision_root: Path = COLLISION_ROOT,
    expected_format: str = CHECKPOINT_FORMAT,
    worlds_per_side: int = DEFAULT_WORLDS_PER_SIDE,
    seed: int = DEFAULT_SEED,
    device: str = "cuda:0",
) -> dict[str, Any]:
    side, layout = assignments(worlds_per_side)
    runner = RecurrentNextoEpisodeRunner(
        side.size,
        str(collision_root),
        checkpoint,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        expected_checkpoint_format=expected_format,
        starting_layout=layout,
        rival_side=side,
        evaluation_seed=seed,
        device=device,
    )
    initial = runner.initial_action_probe()
    initial_vectors = []
    for layout_index in range(5):
        row = next(
            index
            for index in range(side.size)
            if side[index] == 0 and layout[index] == layout_index
        )
        initial_vectors.append(
            {
                "layout": layout_index,
                "layout_name": LAYOUT_NAMES[layout_index],
                "hidden_state": "zero",
                "controller_output": {
                    name: float(initial[row, channel])
                    for channel, name in enumerate(ACTION_NAMES)
                },
            }
        )
    timing = runner.run()
    export = runner.export()
    raw = export["raw"]
    open_raw = export["open_play_raw"]
    rows = np.arange(side.size, dtype=np.int64)
    opponent_side = 1 - side
    rival_by_episode = raw["touch_count"][rows, side]
    nexto_by_episode = raw["touch_count"][rows, opponent_side]
    rival_touches = int(rival_by_episode.sum())
    nexto_touches = int(nexto_by_episode.sum())
    episodes_with_touch = int((rival_by_episode > 0).sum())
    termination = raw["termination_kind"]
    winner = raw["winner"]
    decisive = termination == TERMINATION_GOAL
    rival_goals = int((decisive & (winner == side)).sum())
    nexto_goals = int((decisive & (winner == opponent_side)).sum())
    no_touch = int((termination == TERMINATION_NO_TOUCH).sum())
    hard_time = int((termination == TERMINATION_HARD_TIME).sum())
    rival_first = int(((raw["first_toucher"] >= 0) & (raw["first_toucher"] == side)).sum())
    behavior_rows = np.flatnonzero(termination != TERMINATION_NO_TOUCH)
    behavior_side = side[behavior_rows]
    forward_contacts = int(
        open_raw["direction_count"][behavior_rows, behavior_side, TOUCH_FORWARD].sum()
    )
    forward_possessions = int(
        open_raw["displacement_count"][behavior_rows, behavior_side, TOUCH_FORWARD].sum()
    )
    challenge_exchanges = int(
        open_raw["possession_opponent"][behavior_rows, behavior_side].sum()
    )
    rival_ticks = int(raw["simulated_ticks"][rows, side].sum())
    mean_speed = (
        None
        if rival_ticks == 0
        else float(raw["speed_sum"][rows, side].sum(dtype=np.float64) / rival_ticks)
    )
    episodes = int(side.size)
    touch_fraction = float(episodes_with_touch / episodes)
    no_touch_fraction = float(no_touch / episodes)
    routine_acquisition = bool(
        touch_fraction >= 0.90
        and no_touch_fraction <= 0.01
        and rival_touches >= episodes_with_touch
        and forward_contacts > 0
    )
    result = {
        "format": FORMAT,
        "created_utc": utc_now(),
        "checkpoint": export["checkpoint_identity"],
        "execution": {
            "episodes": episodes,
            "worlds_per_side": worlds_per_side,
            "paired_balanced_standard_kickoffs": True,
            "seed": seed,
            "physics_hz": PHYSICS_HZ,
            "rival_policy_hz": PHYSICS_HZ,
            "deterministic_actor_means_and_button_threshold": True,
            "sampling": False,
            "initial_hidden_zero_at_playable_kickoff": True,
            "hidden_continuous_within_episode": True,
            "hidden_reset_only_at_native_episode_reset": True,
            "optimizer_steps": 0,
            "reward_optimization": False,
        },
        "five_initial_kickoff_action_vectors": initial_vectors,
        "gameplay": {
            "rival_touches": rival_touches,
            "nexto_touches": nexto_touches,
            "episodes_with_rival_touch": episodes_with_touch,
            "episodes_with_rival_touch_fraction": touch_fraction,
            "rival_first_touches": rival_first,
            "rival_challenge_possession_exchanges": challenge_exchanges,
            "rival_forward_ball_velocity_contacts": forward_contacts,
            "rival_forward_ball_displacement_possessions": forward_possessions,
            "rival_goals": rival_goals,
            "nexto_goals": nexto_goals,
            "no_touch_truncations": no_touch,
            "no_touch_fraction": no_touch_fraction,
            "hard_timeouts": hard_time,
            "mean_rival_speed_uu_per_s": mean_speed,
        },
        "phase_a_transition_evidence": {
            "routine_acquisition": routine_acquisition,
            "touch_fraction_reference": 0.90,
            "no_touch_fraction_reference": 0.01,
            "references_are_transition_guidance_not_checkpoint_selection": True,
        },
        "performance": {
            "wall_seconds": timing.seconds,
            "world_ticks_per_second": timing.world_ticks_per_second,
            "peak_cuda_bytes": export["peak_cuda_bytes"],
        },
    }
    write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--expected-format", default=CHECKPOINT_FORMAT)
    parser.add_argument("--worlds-per-side", type=int, default=DEFAULT_WORLDS_PER_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.output,
        collision_root=args.collision_root,
        expected_format=args.expected_format,
        worlds_per_side=args.worlds_per_side,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
