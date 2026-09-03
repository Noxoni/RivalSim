"""Extract literal 120 Hz dash timing from accepted human demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.human_demo.reader import SessionReader  # noqa: E402

VERSION = "RIVAL2_HUMAN_DASH_TIMING_CALIBRATION_V1"
CANDIDATES = (
    ROOT
    / "results/rival2/human_demo_review_v2/behavior_cloning_candidates.json"
)
INVENTORY = ROOT / "results/rival2/human_demo_review_v2/source_inventory.json"
ATTEMPTS = ROOT / "results/rival2/human_demo_review_v2/attempts"
DEFAULT_OUTPUT = (
    ROOT / "results/rival2/dash_physical_calibration_v1/human_timing.json"
)
LABELS = ("wavedash", "walldash")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _human_car(frame: dict[str, Any]) -> dict[str, Any]:
    cars = [car for car in frame["cars"] if car["flags"]["is_local_human"]]
    if len(cars) != 1:
        raise RuntimeError(
            f"frame {frame['physics_frame']} has {len(cars)} local human cars"
        )
    return cars[0]


def _frame_evidence(frame: dict[str, Any]) -> dict[str, Any]:
    car = _human_car(frame)
    velocity = [float(value) for value in car["linear_velocity"]]
    return {
        "physics_frame": int(frame["physics_frame"]),
        "sequence": int(frame["sequence"]),
        "engine_physics_time": float(frame["engine_physics_time"]),
        "native_input": frame["native_input"],
        "rival_action": frame["rival_action"],
        "position": [float(value) for value in car["position"]],
        "rotation_unreal_units": [int(value) for value in car["rotation"]],
        "linear_velocity": velocity,
        "speed_uu_per_second": float(math.sqrt(sum(v * v for v in velocity))),
        "on_ground": bool(car["flags"]["on_ground"]),
        "world_contact_wheels": int(
            sum(bool(wheel["has_world_contact"]) for wheel in car["wheels"])
        ),
        "world_contact_normals": [
            [float(value) for value in wheel["contact_normal"]]
            for wheel in car["wheels"]
            if wheel["has_world_contact"]
        ],
        "jump_component_active": bool(car["jump_component"]["active"]),
        "dodge_component_active": bool(car["dodge_component"]["active"]),
        "dodge_direction": [
            float(value) for value in car["dodge_component"]["direction"]
        ],
        "has_flip": bool(car["flags"]["has_flip"]),
    }


def _first_release_frame(
    frames: dict[int, dict[str, Any]],
    press_frame: int,
    upper_bound: int,
) -> int | None:
    observed_press = False
    for physics_frame in range(press_frame, upper_bound + 1):
        frame = frames.get(physics_frame)
        if frame is None:
            continue
        pressed = bool(frame["native_input"]["jump"])
        observed_press |= pressed
        if observed_press and not pressed:
            return physics_frame
    return None


def _load_attempt_records(session_uuid: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    path = ATTEMPTS / f"{session_uuid}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[str(record["attempt_id"])] = record
    return records


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LABELS:
        grouped = [row for row in rows if row["declared_label"] == label]

        def values(
            name: str,
            grouped_rows: list[dict[str, Any]] = grouped,
        ) -> list[float]:
            return [
                float(row[name])
                for row in grouped_rows
                if row[name] is not None
            ]

        def summary(name: str) -> dict[str, float | int | None]:
            observed = values(name)
            return {
                "count": len(observed),
                "minimum": min(observed) if observed else None,
                "median": statistics.median(observed) if observed else None,
                "maximum": max(observed) if observed else None,
            }

        result[label] = {
            "attempt_count": len(grouped),
            "jump_to_dodge_ticks": summary("jump_to_dodge_ticks"),
            "jump_release_to_dodge_ticks": summary(
                "jump_release_to_dodge_ticks"
            ),
            "dodge_to_landing_ticks": summary("dodge_to_landing_ticks"),
            "surface_tangent_speed_gain_uu_per_second": summary(
                "surface_tangent_speed_gain_uu_per_second"
            ),
        }
    return result


def run(args: argparse.Namespace) -> int:
    candidates_payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    selected = [
        row
        for row in candidates_payload["candidates"]
        if row["declared_label"] in LABELS
    ]
    inventory_payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    inventory = {
        row["session_uuid"]: row for row in inventory_payload["sessions"]
    }
    rows: list[dict[str, Any]] = []
    session_evidence: dict[str, Any] = {}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for candidate in selected:
        by_session.setdefault(candidate["session_uuid"], []).append(candidate)

    for session_uuid, session_candidates in sorted(by_session.items()):
        source = inventory[session_uuid]
        reader = SessionReader(source["source_directory"])
        validation = reader.validate()
        if not validation.container_valid or not validation.manifest_hashes_valid:
            raise RuntimeError(f"invalid human dash source session: {session_uuid}")
        frames = {
            int(frame["physics_frame"]): frame for frame in reader.iter_frames()
        }
        attempts = _load_attempt_records(session_uuid)
        session_evidence[session_uuid] = {
            "source_directory": source["source_directory"],
            "source_file_set_sha256": source["source_file_set_sha256"],
            "frame_count": validation.frame_count,
            "container_valid": validation.container_valid,
            "manifest_hashes_valid": validation.manifest_hashes_valid,
        }
        for candidate in session_candidates:
            record = attempts[candidate["attempt_id"]]
            if record["mechanic_assessment"]["verdict"] != "success":
                raise RuntimeError("candidate no longer has a successful verdict")
            evidence = record["mechanic_assessment"]["evidence"]
            dodge_onsets = evidence["native_dodge_onsets"]
            landings = evidence["dash_landing_outcomes"]
            if not dodge_onsets or not landings:
                raise RuntimeError("accepted dash attempt lacks physical evidence")
            dodge_frame = min(
                int(row["physics_frame"]) for row in dodge_onsets
            )
            jump_events = sorted(
                int(row["physics_frame"])
                for row in record["jump_onset_events"]
                if int(row["physics_frame"]) <= dodge_frame
            )
            if not jump_events:
                raise RuntimeError("accepted dash attempt lacks jump onset")
            jump_frame = jump_events[0]
            release_frame = _first_release_frame(
                frames,
                jump_frame,
                dodge_frame,
            )
            landing = min(
                landings,
                key=lambda row: (
                    int(row["landing_physics_frame"]),
                    -float(row["surface_tangent_speed_change_uu_per_s"]),
                ),
            )
            landing_frame = int(landing["landing_physics_frame"])
            salient = {
                jump_frame - 1,
                jump_frame,
                jump_frame + 1,
                dodge_frame - 1,
                dodge_frame,
                dodge_frame + 1,
                landing_frame - 1,
                landing_frame,
                landing_frame + 1,
            }
            if release_frame is not None:
                salient.update(
                    (release_frame - 1, release_frame, release_frame + 1)
                )
            timeline = [
                _frame_evidence(frames[physics_frame])
                for physics_frame in sorted(salient)
                if physics_frame in frames
            ]
            rows.append(
                {
                    "attempt_id": candidate["attempt_id"],
                    "declared_label": candidate["declared_label"],
                    "session_uuid": session_uuid,
                    "source_file_set_sha256": candidate[
                        "source_file_set_sha256"
                    ],
                    "jump_onset_physics_frame": jump_frame,
                    "jump_release_physics_frame": release_frame,
                    "dodge_onset_physics_frame": dodge_frame,
                    "landing_physics_frame": landing_frame,
                    "jump_to_dodge_ticks": dodge_frame - jump_frame,
                    "jump_release_to_dodge_ticks": (
                        None
                        if release_frame is None
                        else dodge_frame - release_frame
                    ),
                    "dodge_to_landing_ticks": landing_frame - dodge_frame,
                    "surface_contact_normal": landing[
                        "surface_contact_normal"
                    ],
                    "surface_tangent_speed_gain_uu_per_second": float(
                        landing["surface_tangent_speed_change_uu_per_s"]
                    ),
                    "timeline": timeline,
                }
            )

    payload = {
        "format": VERSION,
        "created_utc": _utc_now(),
        "source_authority": {
            "candidate_path": CANDIDATES.relative_to(ROOT).as_posix(),
            "candidate_sha256": _sha256(CANDIDATES),
            "inventory_path": INVENTORY.relative_to(ROOT).as_posix(),
            "inventory_sha256": _sha256(INVENTORY),
        },
        "sessions": session_evidence,
        "attempts": rows,
        "summary": _summarize(rows),
        "interpretation": (
            "Source-bound physical timing only. This does not create a "
            "production detector or reward and does not modify any policy."
        ),
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
        "verdict": "PASS",
    }
    _write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
