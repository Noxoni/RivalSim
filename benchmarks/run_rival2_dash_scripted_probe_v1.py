"""Run the source-bound dash timing sweep in native RivalSim physics.

This is a deterministic, no-policy, no-reward, no-learning physical probe.
It applies control sequences bounded by accepted 120 Hz human timing evidence
from the recorder and measures literal flip, contact, and tangent-speed results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.controls import ControlBatch  # noqa: E402
from rivalsim.rival2_capability_curriculum_v2 import (  # noqa: E402
    _quat_from_basis,
)
from rivalsim.rival2_dash_scripted_probe import (  # noqa: E402
    DASH_SCRIPTED_PROBE_VERSION,
    action_matrix,
    analyze_dash_trace,
    build_dash_probe_cases,
    summarize_dash_results,
)
from rivalsim.state import StateSnapshot  # noqa: E402
from rivalsim.static_world import CompleteWorldSim  # noqa: E402

HUMAN_TIMING = (
    ROOT / "results/rival2/dash_physical_calibration_v1/human_timing.json"
)
OUTPUT = (
    ROOT
    / "results/rival2/dash_physical_calibration_v1/native_scripted_probe.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


def initial_state(cases: list[Any]) -> StateSnapshot:
    worlds = len(cases)
    state = StateSnapshot.empty(worlds)
    state.boost.fill(100.0)
    state.ball_pos[:] = np.asarray((0.0, 3600.0, 1100.0), dtype=np.float32)
    state.car_pos[:, 1] = np.asarray((0.0, -3600.0, 17.0), dtype=np.float32)
    state.car_quat[:, 1] = yaw_quat(-math.pi / 2.0)
    state.on_ground[:, 1] = 1
    floor_quat = yaw_quat(math.pi / 2.0)
    for world, case in enumerate(cases):
        if case.family == "floor_wavedash":
            state.car_pos[world, 0] = (0.0, 0.0, 17.0)
            state.car_quat[world, 0] = floor_quat
        else:
            forward = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
            up = np.asarray((-float(case.wall_sign), 0.0, 0.0), dtype=np.float64)
            state.car_pos[world, 0] = (
                float(case.wall_sign) * 4060.0,
                0.0,
                450.0,
            )
            state.car_quat[world, 0] = _quat_from_basis(forward, up)
        state.car_vel[world, 0] = (
            0.0,
            float(case.initial_speed_uu_per_second),
            0.0,
        )
        state.on_ground[world, 0] = 1
    state.validate()
    return state


def control_batch(actions: np.ndarray) -> ControlBatch:
    worlds = actions.shape[0]
    if actions.shape != (worlds, 8):
        raise ValueError("action matrix must have shape [worlds, 8]")
    controls = ControlBatch.zeros(worlds)
    for channel, name in enumerate(
        ("throttle", "steer", "pitch", "yaw", "roll")
    ):
        getattr(controls, name)[:, 0] = actions[:, channel]
    for channel, name in enumerate(("jump", "boost", "handbrake"), start=5):
        getattr(controls, name)[:, 0] = actions[:, channel] != 0.0
    controls.validate()
    return controls


def capture_state(sim: CompleteWorldSim) -> dict[str, np.ndarray]:
    state = sim.snapshot()
    vehicle = sim.vehicle_snapshot()
    return {
        "car_position": state.car_pos[:, 0].copy(),
        "car_velocity": state.car_vel[:, 0].copy(),
        "on_ground": state.on_ground[:, 0].copy(),
        "has_flipped": state.has_flipped[:, 0].copy(),
        "is_flipping": state.is_flipping[:, 0].copy(),
        "wheel_contact_count": vehicle.wheels_with_contact[::2].copy(),
        "world_contact_normal": vehicle.world_contact_normal[::2].copy(),
    }


def run(args: argparse.Namespace) -> int:
    human_timing = json.loads(HUMAN_TIMING.read_text(encoding="utf-8"))
    cases = build_dash_probe_cases(human_timing)
    if args.horizon <= max(case.second_jump_tick for case in cases) + 8:
        raise ValueError("horizon does not leave enough post-landing observation time")
    state = initial_state(cases)
    geometry = ArenaGeometry.load_soccar(args.collision_root)
    meshes = WarpArenaMeshes(geometry, args.device)
    sim = CompleteWorldSim(
        len(cases),
        str(args.collision_root),
        device=args.device,
        seed=args.seed,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        auto_kickoff=False,
    )
    captured = [capture_state(sim)]
    applied_actions: list[np.ndarray] = []
    for tick in range(args.horizon):
        actions = action_matrix(cases, tick)
        sim.set_controls(control_batch(actions))
        sim.step()
        applied_actions.append(actions)
        captured.append(capture_state(sim))

    trace = {
        name: np.stack([row[name] for row in captured])
        for name in captured[0]
    }
    trace["action"] = np.stack(applied_actions)
    rows = analyze_dash_trace(cases, trace)
    result: dict[str, Any] = {
        "format": DASH_SCRIPTED_PROBE_VERSION,
        "human_timing_path": HUMAN_TIMING.relative_to(ROOT).as_posix(),
        "human_timing_sha256": sha256_file(HUMAN_TIMING),
        "physics_hz": 120,
        "device": args.device,
        "seed": args.seed,
        "horizon_ticks": args.horizon,
        "case_count": len(cases),
        "summary": summarize_dash_results(rows),
        "rows": rows,
        "policy_loaded": False,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "named_mechanic_detector_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026090301)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
