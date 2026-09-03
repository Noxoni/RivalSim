"""Extract source-bound 120 Hz ground-to-air transition evidence.

This read-only analysis uses only the already accepted Human Demo Review V2
attempts.  It does not re-adjudicate demonstrations, define a production
mechanic detector, change rewards, or train a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSION_UUID = "24A888CE-9F8A-4FBB-8FC9-F26730314ACF"
SOURCE = (
    ROOT
    / "results/rival2/human_demo_review_v2/attempts"
    / f"{SESSION_UUID}.jsonl"
)
CANDIDATES = (
    ROOT / "results/rival2/human_demo_review_v2/behavior_cloning_candidates.json"
)
OUTPUT = (
    ROOT
    / "results/rival2/ground_to_air_human_physics_v1/human_transition.json"
)
FORMAT = "RIVAL2_HUMAN_GROUND_TO_AIR_PHYSICS_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def load_attempts() -> list[dict[str, Any]]:
    accepted = {
        row["attempt_id"]
        for row in json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]
        if row["declared_label"] == "groundtoairdribble"
        and row["session_uuid"] == SESSION_UUID
    }
    attempts: list[dict[str, Any]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["attempt_id"] not in accepted:
                continue
            assessment = row["mechanic_assessment"]
            if assessment["behavior_cloning_eligible"] is not True:
                raise RuntimeError("candidate manifest includes an ineligible attempt")
            if row["data_quality"]["noncontiguous_adjacent_frame_count"] != 0:
                raise RuntimeError("accepted attempt contains noncontiguous paired frames")
            attempts.append(row)
    if {row["attempt_id"] for row in attempts} != accepted:
        raise RuntimeError("accepted ground-to-air attempt identity is incomplete")
    return attempts


def contact_at_frame(row: dict[str, Any], frame: int) -> dict[str, Any]:
    matching = [
        contact
        for contact in row["ball_outcome"]["contacts"]
        if contact["contact_physics_frame"] == frame
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one contact at frame {frame}")
    return matching[0]


def analyze_attempt(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row["mechanic_assessment"]["evidence"]
    pairs = evidence["ground_to_air_pop_followup_pairs"]
    if not pairs:
        raise RuntimeError("accepted attempt has no ground-to-air pair")
    pair = pairs[0]
    pop_frame = int(pair["ground_pop_physics_frame"])
    first_air_frame = int(pair["first_later_air_touch_physics_frame"])
    pop = contact_at_frame(row, pop_frame)
    first_air = contact_at_frame(row, first_air_frame)
    all_contacts = row["ball_outcome"]["contacts"]
    airborne = [
        contact
        for contact in all_contacts
        if contact["car_at_contact"]["wheel_world_contact_count"] == 0
    ]
    airborne_frames = [int(contact["contact_physics_frame"]) for contact in airborne]
    if first_air_frame not in airborne_frames:
        raise RuntimeError("accepted first air contact is not wheel-contact-free")
    jumps = sorted(
        int(event["physics_frame"])
        for event in row["jump_onset_events"]
        if pop_frame <= int(event["physics_frame"]) <= first_air_frame
    )
    gaps = [right - left for left, right in itertools.pairwise(airborne_frames)]
    geometry = first_air["contact_geometry"]
    car = first_air["car_at_contact"]
    ball = first_air["ball_at_contact"]
    return {
        "attempt_id": row["attempt_id"],
        "attempt_number": int(row["attempt_number"]),
        "pop_frame": pop_frame,
        "first_air_contact_frame": first_air_frame,
        "pop_to_first_air_contact_ticks": first_air_frame - pop_frame,
        "jump_onset_frames": jumps,
        "jump_onset_offsets_from_pop_ticks": [frame - pop_frame for frame in jumps],
        "last_jump_to_first_air_contact_ticks": (
            None if not jumps else first_air_frame - jumps[-1]
        ),
        "jump_onset_count_before_first_air_contact": len(jumps),
        "first_air_contact": {
            "ball_height_uu": float(ball["position"][2]),
            "car_height_uu": float(car["position"][2]),
            "car_speed_uu_per_second": float(car["speed_uu_per_s"]),
            "car_vertical_speed_uu_per_second": float(car["linear_velocity"][2]),
            "car_angular_speed_rad_per_second": float(car["angular_speed_rad_per_s"]),
            "double_jumped": bool(car["double_jumped"]),
            "has_flip": bool(car["has_flip"]),
            "car_to_ball_distance_uu": float(geometry["car_to_ball_distance_uu"]),
            "car_to_ball_forward_alignment": float(
                geometry["car_to_ball_forward_alignment"]
            ),
            "car_to_ball_up_alignment": float(geometry["car_to_ball_up_alignment"]),
            "ball_velocity_before_uu_per_second": [
                float(value) for value in first_air["ball_before"]["linear_velocity"]
            ],
            "ball_velocity_after_12_ticks_uu_per_second": [
                float(value)
                for value in first_air["ball_after_12_ticks"]["linear_velocity"]
            ],
            "ball_delta_velocity_12_ticks_uu_per_second": [
                float(value) for value in first_air["ball_delta_velocity_12_ticks"]
            ],
        },
        "ground_pop_vertical_delta_12_ticks_uu_per_second": float(
            pop["ball_delta_velocity_12_ticks"][2]
        ),
        "airborne_contact_count": len(airborne),
        "airborne_contact_frames": airborne_frames,
        "airborne_contact_gaps_ticks": gaps,
        "first_to_last_airborne_contact_ticks": (
            0 if not airborne_frames else airborne_frames[-1] - airborne_frames[0]
        ),
        "within_six_airborne_contacts": len(airborne) <= 6,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = [row["first_air_contact"] for row in rows]
    all_gaps = [
        gap for row in rows for gap in row["airborne_contact_gaps_ticks"]
    ]
    two_or_more = [row for row in rows if row["airborne_contact_count"] >= 2]
    return {
        "attempts": len(rows),
        "attempts_with_one_jump_onset_before_first_air_contact": sum(
            row["jump_onset_count_before_first_air_contact"] == 1 for row in rows
        ),
        "attempts_with_two_jump_onsets_before_first_air_contact": sum(
            row["jump_onset_count_before_first_air_contact"] == 2 for row in rows
        ),
        "attempts_with_at_least_two_airborne_contacts": len(two_or_more),
        "attempts_with_one_to_six_airborne_contacts": sum(
            1 <= row["airborne_contact_count"] <= 6 for row in rows
        ),
        "attempt_fraction_with_one_to_six_airborne_contacts": sum(
            1 <= row["airborne_contact_count"] <= 6 for row in rows
        )
        / len(rows),
        "pop_to_first_air_contact_ticks": distribution(
            [row["pop_to_first_air_contact_ticks"] for row in rows]
        ),
        "last_jump_to_first_air_contact_ticks": distribution(
            [
                row["last_jump_to_first_air_contact_ticks"]
                for row in rows
                if row["last_jump_to_first_air_contact_ticks"] is not None
            ]
        ),
        "first_to_second_airborne_contact_ticks": distribution(
            [row["airborne_contact_gaps_ticks"][0] for row in two_or_more]
        ),
        "all_consecutive_airborne_contact_gaps_ticks": distribution(all_gaps),
        "airborne_contact_count": distribution(
            [row["airborne_contact_count"] for row in rows]
        ),
        "first_to_last_airborne_contact_ticks": distribution(
            [row["first_to_last_airborne_contact_ticks"] for row in rows]
        ),
        "first_air_contact_ball_height_uu": distribution(
            [row["ball_height_uu"] for row in first]
        ),
        "first_air_contact_car_height_uu": distribution(
            [row["car_height_uu"] for row in first]
        ),
        "first_air_contact_car_vertical_speed_uu_per_second": distribution(
            [row["car_vertical_speed_uu_per_second"] for row in first]
        ),
        "first_air_contact_car_angular_speed_rad_per_second": distribution(
            [row["car_angular_speed_rad_per_second"] for row in first]
        ),
        "first_air_contact_distance_uu": distribution(
            [row["car_to_ball_distance_uu"] for row in first]
        ),
        "first_air_contact_forward_alignment": distribution(
            [row["car_to_ball_forward_alignment"] for row in first]
        ),
        "first_air_contact_up_alignment": distribution(
            [row["car_to_ball_up_alignment"] for row in first]
        ),
        "first_air_contact_vertical_ball_transfer_12_ticks_uu_per_second": (
            distribution(
                [
                    row["ball_delta_velocity_12_ticks_uu_per_second"][2]
                    for row in first
                ]
            )
        ),
        "first_air_contact_double_jumped_fraction": sum(
            row["double_jumped"] for row in first
        )
        / len(first),
        "first_air_contact_has_flip_fraction": sum(row["has_flip"] for row in first)
        / len(first),
    }


def run(args: argparse.Namespace) -> int:
    attempts = load_attempts()
    rows = [analyze_attempt(row) for row in attempts]
    if not all(math.isfinite(float(row["pop_to_first_air_contact_ticks"])) for row in rows):
        raise RuntimeError("non-finite human transition evidence")
    payload = {
        "format": FORMAT,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": {
            "attempts_path": SOURCE.relative_to(ROOT).as_posix(),
            "attempts_sha256": sha256_file(SOURCE),
            "candidate_manifest_path": CANDIDATES.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": sha256_file(CANDIDATES),
            "session_uuid": SESSION_UUID,
            "declared_label": "groundtoairdribble",
            "review_version": "RIVALRL_HUMAN_DEMO_REVIEW_V2",
        },
        "semantics": {
            "read_only": True,
            "accepted_attempts_only": True,
            "new_adjudication": False,
            "production_detector": False,
            "reward_definition": False,
            "training_performed": False,
            "physics_hz": 120,
            "contact_count": (
                "Native human ball-touch episodes with zero world-contact wheels "
                "inside the accepted reset-bounded attempt."
            ),
        },
        "summary": summarize(rows),
        "attempts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
