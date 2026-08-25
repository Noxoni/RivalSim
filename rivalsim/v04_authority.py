"""Content-addressed native RocketSim authority for RivalSim v0.4."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.arena import ArenaGeometry
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_PRIMARY_COMMIT,
)

SCHEMA = "rivalsim-v0.4-native-authority-v1"
GENERATOR_SEED = 20260825
KICKOFF_SEEDS = (7, 3, 2, 1, 0)
RESPAWN_SEEDS = (0, 1, 3, 5)
GOAL_THRESHOLD = np.float32(5124.25 + 91.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _asset_identity(collision_root: Path) -> dict[str, object]:
    geometry = ArenaGeometry.load_soccar(collision_root)
    return {
        "format": "RocketSim CMF",
        "combined_content_sha256": geometry.content_sha256,
        "files": [
            {
                "name": mesh.path.name,
                "bytes": mesh.path.stat().st_size,
                "sha256": mesh.sha256,
            }
            for mesh in geometry.meshes
        ],
    }


def authority_identity(collision_root: str | Path) -> tuple[str, dict[str, object]]:
    import RocketSim as rs

    root = Path(collision_root).resolve()
    extension = Path(rs.__file__).resolve()
    generator = Path(__file__).resolve()
    inputs: dict[str, object] = {
        "schema": SCHEMA,
        "rocket_sim_primary_commit": ROCKETSIM_PRIMARY_COMMIT,
        "rocket_sim_binding_commit": ROCKETSIM_BINDING_COMMIT,
        "rocket_sim_package": f"rocketsim=={version('rocketsim')}",
        "rocket_sim_extension": {
            "name": extension.name,
            "bytes": extension.stat().st_size,
            "sha256": _sha256(extension),
        },
        "collision_assets": _asset_identity(root),
        "generator": {
            "name": generator.name,
            "sha256": _sha256(generator),
            "seed": GENERATOR_SEED,
        },
        "corpus": {
            "pads": {"count": 34, "cars": [0, 1]},
            "kickoff_layouts": list(range(5)),
            "kickoff_seeds": list(KICKOFF_SEEDS),
            "respawn_locations": list(range(4)),
            "respawn_seeds": list(RESPAWN_SEEDS),
            "goal_y": [
                float(GOAL_THRESHOLD - np.float32(0.25)),
                float(GOAL_THRESHOLD),
                float(GOAL_THRESHOLD + np.float32(0.25)),
            ],
        },
        "authority_settings": {
            "game_mode": "SOCCAR",
            "tick_rate": 120,
            "car_config": "OCTANE",
            "teams": ["BLUE", "ORANGE"],
            "auto_reset_composition": "goal callback then reset_kickoff(explicit seed)",
            "randomness": "explicit layout selectors; no ambient result selection",
        },
        "bounded_rivalsim_contract": {
            "kickoff_selector_advance": "(layout + 1) mod 5",
            "respawn_selector_advance": "(location + 1) mod 4",
            "terminal": "raw events only; terminated=truncated=0",
        },
    }
    return hashlib.sha256(_canonical(inputs)).hexdigest().upper(), inputs


def cache_directory(cache_root: str | Path, identity: str) -> Path:
    return Path(cache_root).resolve() / identity


def _vec(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _matrix(value: Any) -> list[list[float]]:
    return [
        [float(value.forward.x), float(value.right.x), float(value.up.x)],
        [float(value.forward.y), float(value.right.y), float(value.up.y)],
        [float(value.forward.z), float(value.right.z), float(value.up.z)],
    ]


def _car_record(car: Any) -> dict[str, object]:
    state = car.get_state()
    return {
        "pos": _vec(state.pos),
        "vel": _vec(state.vel),
        "rot_mat": _matrix(state.rot_mat),
        "ang_vel": _vec(state.ang_vel),
        "boost": float(state.boost),
        "is_on_ground": bool(state.is_on_ground),
        "is_demoed": bool(state.is_demoed),
        "demo_respawn_timer": float(state.demo_respawn_timer),
        "car_contact_id": int(state.car_contact_id),
        "car_contact_cooldown_timer": float(state.car_contact_cooldown_timer),
        "tick_count_since_update": int(state.tick_count_since_update),
    }


def _ball_record(ball: Any) -> dict[str, object]:
    state = ball.get_state()
    return {
        "pos": _vec(state.pos),
        "vel": _vec(state.vel),
        "rot_mat": _matrix(state.rot_mat),
        "ang_vel": _vec(state.ang_vel),
        "tick_count_since_update": int(state.tick_count_since_update),
    }


def _pad_record(pad: Any) -> dict[str, object]:
    state = pad.get_state()
    return {
        "is_active": bool(state.is_active),
        "cooldown": float(state.cooldown),
        "previous_locked_car_id": int(state.prev_locked_car_id),
    }


def _new_arena(rs: Any) -> tuple[Any, list[Any]]:
    arena = rs.Arena(rs.GameMode.SOCCAR, tick_rate=120.0)
    cars = [arena.add_car(rs.Team.BLUE), arena.add_car(rs.Team.ORANGE)]
    return arena, cars


def _canonical_pads(arena: Any) -> list[Any]:
    """Map binding/RLGym pad order back to Arena's six-big-then-small order."""

    unordered = arena.get_boost_pads()
    result: list[Any] = []
    for expected in SOCCAR_PAD_POSITIONS:
        matches = [
            pad
            for pad in unordered
            if np.array_equal(np.asarray(_vec(pad.get_pos()), dtype=np.float32), expected)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"failed to map native boost pad at {expected.tolist()}")
        result.append(matches[0])
    return result


def _collect_pads(rs: Any) -> dict[str, object]:
    arena, cars = _new_arena(rs)
    arena.set_car_car_collision(False)
    arena.ball.set_state(rs.BallState(pos=rs.Vec(0.0, 0.0, 1500.0)))
    pads = _canonical_pads(arena)
    cases: list[dict[str, object]] = []
    for pad_index, expected_pos in enumerate(SOCCAR_PAD_POSITIONS):
        for local_car in range(2):
            arena.reset_kickoff(0)
            target = cars[local_car]
            other = cars[1 - local_car]
            target.set_state(
                rs.CarState(
                    pos=rs.Vec(*(float(item) for item in expected_pos)),
                    boost=0.0,
                    is_on_ground=False,
                )
            )
            other.set_state(rs.CarState(pos=rs.Vec(0.0, 0.0, 1500.0), is_on_ground=False))
            arena.step(1)
            cases.append(
                {
                    "pad": pad_index,
                    "car": local_car,
                    "pad_position": _vec(pads[pad_index].get_pos()),
                    "pad_state": _pad_record(pads[pad_index]),
                    "car_boost": float(target.get_state().boost),
                }
            )

    cooldowns: list[dict[str, object]] = []
    for pad_index in (0, 6):
        arena.reset_kickoff(0)
        pad = pads[pad_index]
        position = pad.get_pos()
        cars[0].set_state(rs.CarState(pos=position, boost=0.0, is_on_ground=False))
        cars[1].set_state(rs.CarState(pos=rs.Vec(0.0, 0.0, 1500.0), is_on_ground=False))
        arena.step(1)
        initial = _pad_record(pad)
        cars[0].set_state(rs.CarState(pos=rs.Vec(0.0, 0.0, 1500.0), is_on_ground=False))
        reactivation_tick = -1
        before: dict[str, object] | None = None
        at: dict[str, object] | None = None
        after: dict[str, object] | None = None
        limit = 1300 if pad_index == 0 else 600
        for tick in range(1, limit + 1):
            arena.step(1)
            record = _pad_record(pad)
            if bool(record["is_active"]):
                reactivation_tick = tick
                at = record
                break
            before = record
        arena.step(1)
        after = _pad_record(pad)
        cars[0].set_state(rs.CarState(pos=position, boost=0.0, is_on_ground=False))
        arena.step(1)
        cooldowns.append(
            {
                "pad": pad_index,
                "initial": initial,
                "reactivation_tick": reactivation_tick,
                "before": before,
                "at": at,
                "after": after,
                "second_pickup": _pad_record(pad),
                "second_boost": float(cars[0].get_state().boost),
            }
        )
    return {"pickup_cases": cases, "cooldown_cases": cooldowns}


def _collect_goals_kickoff(rs: Any) -> dict[str, object]:
    boundary: list[dict[str, object]] = []
    for sign in (-1.0, 1.0):
        for magnitude in (
            GOAL_THRESHOLD - np.float32(0.25),
            GOAL_THRESHOLD,
            GOAL_THRESHOLD + np.float32(0.25),
        ):
            arena, _cars = _new_arena(rs)
            events: list[int] = []

            def callback(*, team: int, _events: list[int] = events, **_kwargs: Any) -> None:
                _events.append(int(team))

            arena.set_goal_score_callback(callback)
            requested_y = float(np.float32(sign) * magnitude)
            arena.ball.set_state(rs.BallState(pos=rs.Vec(0.0, requested_y, 93.15)))
            arena.step(1)
            boundary.append(
                {
                    "requested_y": requested_y,
                    "readback_y": float(arena.ball.get_state().pos.y),
                    "goal": bool(events),
                    "team": -1 if not events else events[0],
                    "blue_score": int(arena.blue_score),
                    "orange_score": int(arena.orange_score),
                }
            )

    kickoff: list[dict[str, object]] = []
    for layout, seed in enumerate(KICKOFF_SEEDS):
        arena, cars = _new_arena(rs)
        pads = _canonical_pads(arena)
        cars[0].demolish()
        pad_state = pads[0].get_state()
        pad_state.is_active = False
        pad_state.cooldown = 7.0
        pads[0].set_state(pad_state)
        events: list[int] = []

        def reset_callback(
            *,
            arena: Any,
            team: int,
            _events: list[int] = events,
            _seed: int = seed,
            **_kwargs: Any,
        ) -> None:
            _events.append(int(team))
            arena.reset_kickoff(_seed)

        arena.set_goal_score_callback(reset_callback)
        arena.ball.set_state(rs.BallState(pos=rs.Vec(0.0, 5300.0, 93.15)))
        arena.step(1)
        kickoff.append(
            {
                "layout": layout,
                "seed": seed,
                "events": events,
                "blue_score": int(arena.blue_score),
                "orange_score": int(arena.orange_score),
                "cars": [_car_record(car) for car in cars],
                "ball": _ball_record(arena.ball),
                "pads": [_pad_record(pad) for pad in pads],
            }
        )
    return {"boundary_cases": boundary, "kickoff_cases": kickoff}


def _collect_demolition_respawn(rs: Any) -> dict[str, object]:
    poses: list[dict[str, object]] = []
    arena, cars = _new_arena(rs)
    for local_car, car in enumerate(cars):
        for location, seed in enumerate(RESPAWN_SEEDS):
            car.respawn(seed=seed)
            poses.append(
                {
                    "car": local_car,
                    "team": int(car.team),
                    "location": location,
                    "seed": seed,
                    "state": _car_record(car),
                }
            )

    arena, cars = _new_arena(rs)
    victim = cars[0]
    victim.set_state(
        rs.CarState(
            pos=rs.Vec(123.0, 456.0, 1000.0),
            vel=rs.Vec(10.0, 20.0, 30.0),
            boost=41.0,
            is_on_ground=False,
        )
    )
    victim.demolish()
    timer_records = [{"relative_tick": 0, "state": _car_record(victim)}]
    for tick in range(1, 362):
        arena.step(1)
        if tick in (1, 358, 359, 360, 361):
            timer_records.append({"relative_tick": tick, "state": _car_record(victim)})
    return {"respawn_poses": poses, "timer_trace": timer_records}


def collect_native_authority(collision_root: str | Path) -> dict[str, object]:
    import RocketSim as rs

    rs.init(str(Path(collision_root).resolve()))
    return {
        "schema": SCHEMA,
        "boost_pads": _collect_pads(rs),
        "goals_kickoff": _collect_goals_kickoff(rs),
        "demolition_respawn": _collect_demolition_respawn(rs),
    }


def build_cache(
    collision_root: str | Path,
    cache_root: str | Path,
) -> tuple[Path, dict[str, object]]:
    identity, inputs = authority_identity(collision_root)
    directory = cache_directory(cache_root, identity)
    frozen_path = directory / "frozen.json"
    if frozen_path.exists():
        return directory, load_cache(collision_root, cache_root)
    directory.mkdir(parents=True, exist_ok=False)
    authority = collect_native_authority(collision_root)
    authority_bytes = _canonical(authority)
    (directory / "identity.json").write_text(
        json.dumps({"identity": identity, "inputs": inputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "authority.json").write_bytes(authority_bytes + b"\n")
    frozen = {
        "schema": SCHEMA,
        "identity": identity,
        "authority_sha256": hashlib.sha256(authority_bytes + b"\n").hexdigest(),
        "complete": True,
        "live_fallback": False,
    }
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return directory, authority


def load_cache(
    collision_root: str | Path,
    cache_root: str | Path,
) -> dict[str, object]:
    identity, _inputs = authority_identity(collision_root)
    directory = cache_directory(cache_root, identity)
    frozen_path = directory / "frozen.json"
    authority_path = directory / "authority.json"
    if not frozen_path.is_file() or not authority_path.is_file():
        raise FileNotFoundError(
            f"complete v0.4 authority cache {identity} is missing; no live fallback"
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload = authority_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != frozen["authority_sha256"]:
        raise RuntimeError("v0.4 authority cache hash mismatch")
    if frozen.get("identity") != identity or not frozen.get("complete"):
        raise RuntimeError("v0.4 authority cache identity/integrity mismatch")
    return json.loads(payload)


__all__ = [
    "GENERATOR_SEED",
    "GOAL_THRESHOLD",
    "KICKOFF_SEEDS",
    "RESPAWN_SEEDS",
    "SCHEMA",
    "authority_identity",
    "build_cache",
    "cache_directory",
    "load_cache",
]
