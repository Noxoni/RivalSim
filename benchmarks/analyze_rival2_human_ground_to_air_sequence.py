"""Read-only timing summary of accepted human ground-to-air demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_codex_autonomous_v1 import load_human_data  # noqa: E402
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)

REVIEW_ATTEMPTS = (
    ROOT
    / "results/rival2/human_demo_review_v2/attempts/"
    / "24A888CE-9F8A-4FBB-8FC9-F26730314ACF.jsonl"
)
DATASET_MANIFEST = ROOT / "results/rival2/human_demo_dataset_v1/dataset_manifest.json"


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(array.mean()),
    }


def analyze_split(data: Any) -> dict[str, Any]:
    labels = np.asarray(data.mechanic_label)
    selected = np.flatnonzero(labels == "groundtoairdribble")
    groups: dict[str, list[int]] = defaultdict(list)
    for index in selected.tolist():
        groups[data.mechanic_attempt[index]].append(index)
    attempts: list[dict[str, Any]] = []
    for identity in sorted(groups):
        index = torch.as_tensor(groups[identity], dtype=torch.int64)
        observation = data.mechanic_observation.index_select(0, index)
        action = data.mechanic_action.index_select(0, index)
        ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        car_height = observation[:, FIELD["self.position.z"]] * POSITION_SCALE[2]
        touch = observation[:, FIELD["lifecycle.self_touch_event"]] >= 0.5
        low_touch_rows = torch.nonzero(touch & (ball_height <= 170.0), as_tuple=False).flatten()
        if low_touch_rows.numel() == 0:
            continue
        touch_row = int(low_touch_rows[0])
        jump = action[:, 5] >= 0.5
        jump_onset = jump & ~torch.cat((torch.zeros(1, dtype=torch.bool), jump[:-1]))
        eligible_jump_rows = torch.nonzero(
            jump_onset & (torch.arange(jump.shape[0]) <= touch_row),
            as_tuple=False,
        ).flatten()
        jump_row = int(eligible_jump_rows[-1]) if eligible_jump_rows.numel() else -1
        post_limit = min(observation.shape[0], touch_row + 121)
        post_jump_rows = torch.nonzero(
            jump_onset
            & (torch.arange(jump.shape[0]) >= touch_row)
            & (torch.arange(jump.shape[0]) < post_limit),
            as_tuple=False,
        ).flatten()
        post_jump_row = int(post_jump_rows[0]) if post_jump_rows.numel() else -1
        boost = action[:, 6] >= 0.5
        boost_onset = boost & ~torch.cat((torch.zeros(1, dtype=torch.bool), boost[:-1]))
        post_boost_rows = torch.nonzero(
            boost_onset
            & (torch.arange(boost.shape[0]) >= touch_row)
            & (torch.arange(boost.shape[0]) < post_limit),
            as_tuple=False,
        ).flatten()
        post_boost_row = int(post_boost_rows[0]) if post_boost_rows.numel() else -1
        on_ground = observation[:, FIELD["self.on_ground"]] >= 0.5
        left_ground = ~on_ground & torch.cat((torch.zeros(1, dtype=torch.bool), on_ground[:-1]))
        airborne_rows = torch.nonzero(
            left_ground
            & (torch.arange(on_ground.shape[0]) >= touch_row)
            & (torch.arange(on_ground.shape[0]) < post_limit),
            as_tuple=False,
        ).flatten()
        airborne_row = int(airborne_rows[0]) if airborne_rows.numel() else -1
        follow_touch_rows = torch.nonzero(
            touch & (torch.arange(touch.shape[0]) >= touch_row + 4) & (car_height >= 100.0),
            as_tuple=False,
        ).flatten()
        follow_touch_row = int(follow_touch_rows[0]) if follow_touch_rows.numel() else -1
        relative = torch.stack(
            [
                observation[:, FIELD[f"relative.ball_position.{axis}"]] * POSITION_SCALE[axis_index]
                for axis_index, axis in enumerate("xyz")
            ],
            dim=-1,
        )
        ball_vertical = observation[:, FIELD["ball.linear_velocity.z"]] * BALL_LINEAR_SPEED_SCALE
        post_stop = min(observation.shape[0], touch_row + 121)
        pre_start = max(0, touch_row - 30)
        row = {
            "attempt": identity,
            "frames": int(observation.shape[0]),
            "low_pop_touch_row": touch_row,
            "jump_onset_row": jump_row,
            "jump_lead_ticks": touch_row - jump_row if jump_row >= 0 else None,
            "post_pop_jump_onset_row": post_jump_row,
            "post_pop_jump_delay_ticks": (
                post_jump_row - touch_row if post_jump_row >= 0 else None
            ),
            "post_pop_boost_onset_row": post_boost_row,
            "post_pop_boost_delay_ticks": (
                post_boost_row - touch_row if post_boost_row >= 0 else None
            ),
            "first_airborne_row": airborne_row,
            "post_pop_airborne_delay_ticks": (
                airborne_row - touch_row if airborne_row >= 0 else None
            ),
            "elevated_follow_touch_row": follow_touch_row,
            "pop_to_elevated_follow_touch_ticks": (
                follow_touch_row - touch_row if follow_touch_row >= 0 else None
            ),
            "planar_distance_at_jump_uu": (
                float(torch.linalg.vector_norm(relative[jump_row, :2])) if jump_row >= 0 else None
            ),
            "planar_distance_at_touch_uu": float(torch.linalg.vector_norm(relative[touch_row, :2])),
            "car_height_at_touch_uu": float(car_height[touch_row]),
            "ball_height_at_touch_uu": float(ball_height[touch_row]),
            "ball_vertical_velocity_at_touch_uu_per_second": float(ball_vertical[touch_row]),
            "maximum_ball_height_next_120_ticks_uu": float(ball_height[touch_row:post_stop].max()),
            "action_mean_30_before_through_touch": action[pre_start : touch_row + 1]
            .mean(dim=0)
            .tolist(),
            "action_mean_first_30_after_touch": action[
                touch_row : min(observation.shape[0], touch_row + 30)
            ]
            .mean(dim=0)
            .tolist(),
            "action_windows_after_touch": {
                f"{start:02d}_{stop:02d}": action[
                    touch_row + start : min(observation.shape[0], touch_row + stop)
                ]
                .mean(dim=0)
                .tolist()
                for start, stop in ((0, 6), (6, 16), (16, 30), (30, 60), (60, 90))
                if touch_row + start < observation.shape[0]
            },
        }
        attempts.append(row)
    numeric = {
        key: quantiles([float(row[key]) for row in attempts if row.get(key) is not None])
        for key in (
            "jump_lead_ticks",
            "post_pop_jump_delay_ticks",
            "post_pop_boost_delay_ticks",
            "post_pop_airborne_delay_ticks",
            "pop_to_elevated_follow_touch_ticks",
            "planar_distance_at_jump_uu",
            "planar_distance_at_touch_uu",
            "car_height_at_touch_uu",
            "ball_height_at_touch_uu",
            "ball_vertical_velocity_at_touch_uu_per_second",
            "maximum_ball_height_next_120_ticks_uu",
        )
    }
    return {
        "split": data.split,
        "attempts_with_low_pop_touch": len(attempts),
        "summaries": numeric,
        "attempts": attempts,
    }


def analyze_source_bound_review() -> dict[str, Any]:
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    split_by_attempt = {
        str(row["attempt_id"]): str(row["split"]) for row in manifest["mechanic_positive_attempts"]
    }
    rows: list[dict[str, Any]] = []
    pair_specs: list[tuple[int, int]] = []
    excluded_test_pairs = 0
    with REVIEW_ATTEMPTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            attempt = json.loads(line)
            attempt_id = str(attempt["attempt_id"])
            split = split_by_attempt.get(attempt_id)
            assessment = attempt.get("mechanic_assessment", {})
            if assessment.get("verdict") != "success":
                continue
            evidence = assessment.get("evidence", {})
            if split not in {"train", "validation"}:
                if split == "test":
                    excluded_test_pairs += len(evidence.get("ground_to_air_pop_followup_pairs", []))
                continue
            contacts = {
                int(contact.get("physics_frame", contact.get("contact_physics_frame"))): contact
                for contact in attempt.get("ball_outcome", {}).get("contacts", [])
            }
            for pair in evidence.get("ground_to_air_pop_followup_pairs", []):
                pop_frame = int(pair["ground_pop_physics_frame"])
                follow_frame = int(pair["first_later_air_touch_physics_frame"])
                contact = contacts.get(pop_frame)
                if contact is None:
                    continue
                car = contact["car_at_contact"]
                ball = contact["ball_at_contact"]
                geometry = contact["contact_geometry"]
                rows.append(
                    {
                        "attempt_id": attempt_id,
                        "split": split,
                        "ground_pop_physics_frame": pop_frame,
                        "first_later_air_touch_physics_frame": follow_frame,
                        "car_to_ball_distance_uu": float(geometry["car_to_ball_distance_uu"]),
                        "car_to_ball_forward_alignment": float(
                            geometry["car_to_ball_forward_alignment"]
                        ),
                        "car_to_ball_up_alignment": float(geometry["car_to_ball_up_alignment"]),
                        "car_planar_speed_uu_per_second": float(car["planar_speed_uu_per_s"]),
                        "ball_speed_uu_per_second": float(ball["speed_uu_per_s"]),
                        "ball_height_uu": float(ball["position"][2]),
                        "ball_vertical_velocity_uu_per_second": float(ball["linear_velocity"][2]),
                        "pop_vertical_delta_12_ticks_uu_per_second": float(
                            pair["ground_pop_vertical_delta_uu_per_s"]
                        ),
                        "ticks_from_pop_to_air_touch": float(pair["ticks_from_pop_to_air_touch"]),
                    }
                )
                pair_specs.append((pop_frame, follow_frame))

    if pair_specs:
        needed = {
            frame
            for pop_frame, follow_frame in pair_specs
            for frame in range(pop_frame - 30, follow_frame + 2)
        }
        session_uuid = REVIEW_ATTEMPTS.stem
        source_root = Path(os.environ["APPDATA"]) / "bakkesmod/bakkesmod/data/rival2/human_demos"
        frames = {
            int(frame["physics_frame"]): frame
            for frame in SessionReader(source_root / session_uuid).iter_frames()
            if int(frame["physics_frame"]) in needed
        }
        action_names = (
            "throttle",
            "steer",
            "pitch",
            "yaw",
            "roll",
            "jump",
            "boost",
            "handbrake",
        )

        def local_car(frame: dict[str, Any]) -> dict[str, Any]:
            matches = [
                car for car in frame["cars"] if bool(car.get("flags", {}).get("is_local_human"))
            ]
            if len(matches) != 1:
                raise ValueError("native frame does not contain one local human car")
            return matches[0]

        for row, (pop_frame, follow_frame) in zip(rows, pair_specs, strict=True):
            ordered = [
                frames[physics_frame]
                for physics_frame in range(pop_frame - 30, follow_frame + 1)
                if physics_frame in frames
            ]
            action_rows: list[tuple[int, list[float]]] = []
            events: dict[str, int | None] = {
                "jump_onset": None,
                "boost_onset": None,
                "left_ground": None,
                "double_jump_onset": None,
            }
            previous_jump = False
            previous_boost = False
            previous_ground = bool(local_car(ordered[0])["flags"]["on_ground"])
            previous_double = bool(local_car(ordered[0])["flags"]["double_jumped"])
            for frame in ordered:
                physics_frame = int(frame["physics_frame"])
                action = frame["rival_action"]
                flags = local_car(frame)["flags"]
                jump = bool(action["jump"])
                boost = bool(action["boost"])
                on_ground = bool(flags["on_ground"])
                double_jumped = bool(flags["double_jumped"])
                in_search_window = physics_frame >= pop_frame - 10
                if in_search_window and events["jump_onset"] is None and jump and not previous_jump:
                    events["jump_onset"] = physics_frame
                if (
                    in_search_window
                    and events["boost_onset"] is None
                    and boost
                    and not previous_boost
                ):
                    events["boost_onset"] = physics_frame
                if (
                    in_search_window
                    and events["left_ground"] is None
                    and not on_ground
                    and previous_ground
                ):
                    events["left_ground"] = physics_frame
                if (
                    in_search_window
                    and events["double_jump_onset"] is None
                    and double_jumped
                    and not previous_double
                ):
                    events["double_jump_onset"] = physics_frame
                action_rows.append((physics_frame, [float(action[name]) for name in action_names]))
                previous_jump = jump
                previous_boost = boost
                previous_ground = on_ground
                previous_double = double_jumped

            for name, physics_frame in events.items():
                row[f"native_{name}_delay_ticks"] = (
                    int(physics_frame - pop_frame) if physics_frame is not None else None
                )
            row["native_action_windows_after_pop"] = {}
            for start, stop in ((0, 6), (6, 16), (16, 30), (30, 46), (46, 70)):
                values = [
                    action
                    for physics_frame, action in action_rows
                    if pop_frame + start <= physics_frame < pop_frame + stop
                ]
                if values:
                    row["native_action_windows_after_pop"][f"{start:02d}_{stop:02d}"] = (
                        np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
                    )
    keys = sorted(rows[0]) if rows else []
    return {
        "source": REVIEW_ATTEMPTS.relative_to(ROOT).as_posix(),
        "dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
        "splits_used": ["train", "validation"],
        "test_split_loaded": False,
        "excluded_test_pairs": excluded_test_pairs,
        "successful_pop_followup_pairs": len(rows),
        "summaries": {
            key: quantiles(
                [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            )
            for key in keys
            if key
            not in {
                "attempt_id",
                "split",
                "native_action_windows_after_pop",
            }
        },
        "rows": rows,
    }


def run(args: argparse.Namespace) -> int:
    if args.review_only:
        result = {
            "format": "RIVAL2_HUMAN_GROUND_TO_AIR_SEQUENCE_DIAGNOSTIC_V1",
            "test_split_loaded": False,
            "human_identity": None,
            "splits": [],
            "source_bound_review": analyze_source_bound_review(),
        }
    else:
        train, validation, _teacher, identity = load_human_data(device=args.device)
        result = {
            "format": "RIVAL2_HUMAN_GROUND_TO_AIR_SEQUENCE_DIAGNOSTIC_V1",
            "test_split_loaded": False,
            "human_identity": identity,
            "splits": [analyze_split(train), analyze_split(validation)],
            "source_bound_review": analyze_source_bound_review(),
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
