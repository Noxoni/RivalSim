"""Evaluate deterministic Rival-vs-Nexto play over a continuous match window."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.full_match import PHYSICS_HZ, FullMatchRunner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2_026_090_206)
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.window_seconds >= 300:
        raise ValueError("match window must be between 1 and 299 seconds")

    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256.upper():
        raise RuntimeError("checkpoint SHA-256 mismatch")
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = FullMatchRunner(
        10,
        str(args.collision_root.resolve()),
        checkpoint,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        evaluation_seed=int(args.seed),
    )
    timing = runner.run_ticks(int(args.window_seconds) * PHYSICS_HZ)
    exported = runner.export()
    raw = exported.pop("raw")
    rows = np.arange(10, dtype=np.int64)
    opponent_side = 1 - rival_side
    blue_score = raw["match.blue_score"].astype(np.int64)
    orange_score = raw["match.orange_score"].astype(np.int64)
    rival_goals = np.where(rival_side == 0, blue_score, orange_score)
    opponent_goals = np.where(rival_side == 0, orange_score, blue_score)
    rival_touches = raw["touch_count"][rows, rival_side].astype(np.int64)
    opponent_touches = raw["touch_count"][rows, opponent_side].astype(np.int64)
    rival_first = raw["kickoff_first_touch_count"][rows, rival_side].astype(np.int64)
    opponent_first = raw["kickoff_first_touch_count"][rows, opponent_side].astype(
        np.int64
    )

    def side_summary(side: int) -> dict[str, int | float]:
        selected = rival_side == side
        touch_denominator = int(
            rival_touches[selected].sum() + opponent_touches[selected].sum()
        )
        return {
            "worlds": int(selected.sum()),
            "goals_for": int(rival_goals[selected].sum()),
            "goals_against": int(opponent_goals[selected].sum()),
            "goal_differential": int(
                rival_goals[selected].sum() - opponent_goals[selected].sum()
            ),
            "rival_touches": int(rival_touches[selected].sum()),
            "opponent_touches": int(opponent_touches[selected].sum()),
            "rival_touch_share": float(
                rival_touches[selected].sum() / max(touch_denominator, 1)
            ),
            "rival_kickoff_first_touches": int(rival_first[selected].sum()),
            "opponent_kickoff_first_touches": int(opponent_first[selected].sum()),
        }

    total_touches = int(rival_touches.sum() + opponent_touches.sum())
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_MATCH_WINDOW_EVALUATION",
        "checkpoint": exported["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "opponent": "Nexto",
        "action_mode": "deterministic_deployment",
        "window_seconds": int(args.window_seconds),
        "physics_ticks": int(args.window_seconds) * PHYSICS_HZ,
        "matrix": "five standard kickoff layouts x both Rival sides",
        "seed": int(args.seed),
        "overall": {
            "goals_for": int(rival_goals.sum()),
            "goals_against": int(opponent_goals.sum()),
            "goal_differential": int(rival_goals.sum() - opponent_goals.sum()),
            "rival_touches": int(rival_touches.sum()),
            "opponent_touches": int(opponent_touches.sum()),
            "touch_differential": int(rival_touches.sum() - opponent_touches.sum()),
            "rival_touch_share": float(rival_touches.sum() / max(total_touches, 1)),
            "rival_kickoff_first_touches": int(rival_first.sum()),
            "opponent_kickoff_first_touches": int(opponent_first.sum()),
        },
        "by_rival_side": {
            "blue": side_summary(0),
            "orange": side_summary(1),
        },
        "per_layout_side": [
            {
                "layout": int(layout[index]),
                "rival_side": "Blue" if int(rival_side[index]) == 0 else "Orange",
                "goals_for": int(rival_goals[index]),
                "goals_against": int(opponent_goals[index]),
                "rival_touches": int(rival_touches[index]),
                "opponent_touches": int(opponent_touches[index]),
            }
            for index in range(10)
        ],
        "performance": {
            "seconds": timing.seconds,
            "world_ticks_per_second": timing.world_ticks_per_second,
            "peak_cuda_bytes": exported["peak_cuda_bytes"],
            "world_h2d_bytes": exported[
                "world_host_to_device_bytes_after_initialization"
            ],
            "world_d2h_bytes_before_export": exported[
                "world_device_to_host_bytes_before_export"
            ],
            "nexto_timed_h2d_bytes": exported["nexto_timed_h2d_bytes"],
            "nexto_timed_d2h_bytes": exported["nexto_timed_d2h_bytes"],
        },
        "policy_mutation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
