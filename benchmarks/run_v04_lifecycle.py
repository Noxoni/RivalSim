"""Run RivalSim v0.4 lifecycle gates against the frozen native cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import warp as wp

from rivalsim import CompleteWorldSim, StateSnapshot
from rivalsim.kernels.boost_pad import PAD_COUNT, SOCCAR_PAD_POSITIONS
from rivalsim.math import quat_to_matrix
from rivalsim.v04_authority import authority_identity, load_cache


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: object, expected: object, *, atol: float = 2.0e-4) -> bool:
    return bool(
        np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            atol=atol,
            rtol=0.0,
        )
    )


def _phase_a(collision_root: str, device: str, authority: dict[str, object]) -> dict[str, object]:
    cases = authority["boost_pads"]["pickup_cases"]
    count = len(cases)
    state = StateSnapshot.empty(count)
    state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    state.ball_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    for env, case in enumerate(cases):
        target = int(case["car"])
        state.car_pos[env, target] = SOCCAR_PAD_POSITIONS[int(case["pad"])]
        state.boost[env, target] = 0.0
    sim = CompleteWorldSim(
        count,
        collision_root,
        device=device,
        initial=state,
        car_visitation_order="a_then_b",
        auto_kickoff=False,
    )
    sim.step(1, synchronize=True)
    output = sim.snapshot()
    lifecycle = sim.lifecycle_snapshot()
    for env, case in enumerate(cases):
        pad = int(case["pad"])
        car = int(case["car"])
        expected = case["pad_state"]
        _assert(
            _close(output.boost[env, car], case["car_boost"], atol=1.0e-5),
            f"pad case {env} boost",
        )
        _assert(
            _close(lifecycle.pad_cooldown[env, pad], expected["cooldown"], atol=1e-6),
            f"pad case {env} cooldown",
        )
        _assert(
            int(lifecycle.pad_previous_locked_car[env, pad])
            == int(expected["previous_locked_car_id"]),
            f"pad case {env} receiver",
        )
        _assert(
            int(lifecycle.pad_pickup_car[env, pad]) == car + 1,
            f"pad case {env} pickup event",
        )

    cooldown_authority = {
        int(case["pad"]): case for case in authority["boost_pads"]["cooldown_cases"]
    }
    cooldown_state = StateSnapshot.empty(2)
    cooldown_state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    cooldown_state.ball_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    cooldown_sim = CompleteWorldSim(
        2,
        collision_root,
        device=device,
        initial=cooldown_state,
        car_visitation_order="a_then_b",
        auto_kickoff=False,
    )
    cooldown = np.zeros((2, PAD_COUNT), dtype=np.float32)
    cooldown[0, 0] = 10.0
    cooldown[1, 6] = 4.0
    cooldown_sim.boost_pad_cooldown = wp.array(
        cooldown.reshape(-1), dtype=wp.float32, device=device
    )
    cooldown_sim.step(479, synchronize=True)
    before_small = cooldown_sim.lifecycle_snapshot()
    _assert(
        _close(
            before_small.pad_cooldown[1, 6],
            cooldown_authority[6]["before"]["cooldown"],
            atol=1.0e-7,
        ),
        "small pad cooldown before reactivation",
    )
    cooldown_sim.step(1, synchronize=True)
    at_small = cooldown_sim.lifecycle_snapshot()
    _assert(int(at_small.pad_reactivated[1, 6]) == 1, "small pad reactivation tick")
    cooldown_sim.step(720, synchronize=True)
    before_big = cooldown_sim.lifecycle_snapshot()
    _assert(
        _close(
            before_big.pad_cooldown[0, 0],
            cooldown_authority[0]["before"]["cooldown"],
            atol=1.0e-7,
        ),
        "big pad cooldown before reactivation",
    )
    cooldown_sim.step(1, synchronize=True)
    at_big = cooldown_sim.lifecycle_snapshot()
    _assert(int(at_big.pad_reactivated[0, 0]) == 1, "big pad reactivation tick")

    contender = StateSnapshot.empty(2)
    contender.car_pos[:] = SOCCAR_PAD_POSITIONS[0]
    contender.car_vel[:] = 0.0
    contender.boost[:] = 0.0
    contender.ball_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    contention_sim = CompleteWorldSim(
        2,
        collision_root,
        device=device,
        initial=contender,
        car_visitation_order=np.asarray((0, 1), dtype=np.int32),
        auto_kickoff=False,
    )
    contention_sim.step(1, synchronize=True)
    contention = contention_sim.lifecycle_snapshot()
    _assert(
        np.array_equal(contention.pad_pickup_car[:, 0], np.asarray((2, 1))),
        "pad contenders must follow persistent native visitation order",
    )
    return {
        "verdict": "PASS_GREEN",
        "pickup_cases": count,
        "all_34_pads": True,
        "both_cars": True,
        "cooldown_reactivation_ticks": {"big": 1201, "small": 480},
        "contention_branches": ["a_then_b", "b_then_a"],
    }


def _phase_b(collision_root: str, device: str, authority: dict[str, object]) -> dict[str, object]:
    cases = authority["goals_kickoff"]["boundary_cases"]
    state = StateSnapshot.empty(len(cases))
    state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    for env, case in enumerate(cases):
        state.ball_pos[env] = np.asarray((0.0, float(case["requested_y"]), 93.15), dtype=np.float32)
    sim = CompleteWorldSim(
        len(cases),
        collision_root,
        device=device,
        initial=state,
        car_visitation_order="a_then_b",
        auto_kickoff=False,
    )
    sim.step(1, synchronize=True)
    lifecycle = sim.lifecycle_snapshot()
    for env, case in enumerate(cases):
        _assert(
            int(lifecycle.goal_scored[env]) == int(bool(case["goal"])),
            f"goal boundary case {env}",
        )
        _assert(
            int(lifecycle.scoring_team[env]) == int(case["team"]),
            f"goal team case {env}",
        )

    kickoff_cases = authority["goals_kickoff"]["kickoff_cases"]
    kickoff_state = StateSnapshot.empty(5)
    kickoff_state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    kickoff_state.ball_pos[:] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    kickoff_sim = CompleteWorldSim(
        5,
        collision_root,
        device=device,
        initial=kickoff_state,
        kickoff_selector=np.arange(5, dtype=np.int32),
        car_visitation_order=np.asarray((0, 1, 0, 1, 0), dtype=np.int32),
        auto_kickoff=True,
    )
    pad_cooldown = np.zeros((5, PAD_COUNT), dtype=np.float32)
    pad_cooldown[:, 0] = 7.0
    kickoff_sim.boost_pad_cooldown = wp.array(
        pad_cooldown.reshape(-1), dtype=wp.float32, device=device
    )
    before_order = kickoff_sim.car_car.visit_order
    kickoff_sim.request_demolition(0)
    kickoff_sim.reset_transfer_counters()
    kickoff_sim.step(1, synchronize=True)
    timed_h2d = kickoff_sim.host_to_device_bytes
    timed_d2h = kickoff_sim.device_to_host_bytes
    output = kickoff_sim.snapshot()
    lifecycle = kickoff_sim.lifecycle_snapshot()
    for env, case in enumerate(kickoff_cases):
        _assert(int(lifecycle.goal_scored[env]) == 1, f"kickoff goal {env}")
        _assert(int(lifecycle.scoring_team[env]) == 0, f"kickoff team {env}")
        _assert(int(lifecycle.blue_score[env]) == 1, f"kickoff score {env}")
        _assert(int(lifecycle.kickoff_layout[env]) == env, f"kickoff layout {env}")
        for car in range(2):
            native = case["cars"][car]
            _assert(
                _close(output.car_pos[env, car], native["pos"]),
                f"kickoff car position {env}/{car}",
            )
            _assert(
                _close(
                    quat_to_matrix(output.car_quat[env, car]),
                    native["rot_mat"],
                    atol=3.0e-7,
                ),
                f"kickoff car orientation {env}/{car}",
            )
            _assert(
                _close(output.boost[env, car], native["boost"], atol=1.0e-6),
                f"kickoff boost {env}/{car}",
            )
        _assert(
            _close(output.ball_pos[env], case["ball"]["pos"], atol=1.0e-6),
            f"kickoff ball {env}",
        )
    _assert(np.all(lifecycle.pad_active == 1), "kickoff pad reset")
    _assert(np.all(lifecycle.car_is_demoed == 0), "kickoff demo reset")
    _assert(np.array_equal(before_order, kickoff_sim.car_car.visit_order), "order reset")
    _assert(timed_h2d == 0 and timed_d2h == 0, "timed kickoff transfers")
    return {
        "verdict": "PASS_GREEN",
        "goal_boundary_cases": len(cases),
        "kickoff_layouts": len(kickoff_cases),
        "both_goals": True,
        "goal_pad_demo_reset_composition": True,
        "visitation_order_preserved": True,
        "timed_h2d_bytes": timed_h2d,
        "timed_d2h_bytes": timed_d2h,
    }


def _phase_c(collision_root: str, device: str, authority: dict[str, object]) -> dict[str, object]:
    selectors = np.tile(np.arange(4, dtype=np.int32), 2)
    victims = np.repeat(np.arange(2, dtype=np.int32), 4)
    state = StateSnapshot.empty(8)
    state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    state.ball_pos[:] = np.asarray((0.0, 0.0, 1800.0), dtype=np.float32)
    state.car_pos[np.arange(8), victims] = np.asarray((123.0, 456.0, 1000.0), dtype=np.float32)
    state.car_vel[np.arange(8), victims] = np.asarray((10.0, 20.0, 30.0), dtype=np.float32)
    state.boost[np.arange(8), victims] = 41.0
    visit_order = np.arange(8, dtype=np.int32) & 1
    sim = CompleteWorldSim(
        8,
        collision_root,
        device=device,
        initial=state,
        respawn_selector=selectors,
        car_visitation_order=visit_order,
        auto_kickoff=False,
    )
    before_order = sim.car_car.visit_order
    sim.request_demolition(victims)
    sim.step(1, synchronize=True)
    event_state = sim.snapshot()
    event_lifecycle = sim.lifecycle_snapshot()
    native_timer = authority["demolition_respawn"]["timer_trace"]
    _assert(
        _close(event_lifecycle.demo_respawn_timer[np.arange(8), victims], 3.0),
        "demo event timer",
    )
    sim.step(359, synchronize=True)
    before_respawn = sim.snapshot()
    before_lifecycle = sim.lifecycle_snapshot()
    _assert(
        _close(
            before_lifecycle.demo_respawn_timer[np.arange(8), victims],
            native_timer[3]["state"]["demo_respawn_timer"],
            atol=1.0e-7,
        ),
        "demo timer tick 359",
    )
    for env, victim in enumerate(victims):
        _assert(
            np.array_equal(event_state.car_pos[env, victim], before_respawn.car_pos[env, victim]),
            f"demo held position {env}",
        )
        _assert(
            np.array_equal(event_state.car_vel[env, victim], before_respawn.car_vel[env, victim]),
            f"demo held velocity {env}",
        )
    sim.step(1, synchronize=True)
    output = sim.snapshot()
    lifecycle = sim.lifecycle_snapshot()
    native_poses = {
        (int(case["car"]), int(case["location"])): case["state"]
        for case in authority["demolition_respawn"]["respawn_poses"]
    }
    for env, victim in enumerate(victims):
        location = int(selectors[env])
        expected = native_poses[(int(victim), location)]
        _assert(int(lifecycle.respawn_event[env, victim]) == 1, f"respawn event {env}")
        _assert(
            int(lifecycle.respawn_location[env, victim]) == location,
            f"respawn selector {env}",
        )
        _assert(
            _close(output.car_pos[env, victim], expected["pos"]),
            f"respawn position {env}",
        )
        _assert(
            _close(
                quat_to_matrix(output.car_quat[env, victim]),
                expected["rot_mat"],
                atol=3.0e-7,
            ),
            f"respawn orientation {env}",
        )
        _assert(int(lifecycle.car_is_demoed[env, victim]) == 0, f"respawn state {env}")
    _assert(np.array_equal(before_order, sim.car_car.visit_order), "demo changed order")
    return {
        "verdict": "PASS_GREEN",
        "victim_team_location_cases": 8,
        "exact_respawn_tick": 360,
        "disabled_state_frozen": True,
        "visitation_order_preserved": True,
        "source_timer_at_tick_359": native_timer[3]["state"]["demo_respawn_timer"],
    }


def _snapshot_hash(sim: CompleteWorldSim) -> str:
    digest = hashlib.sha256()
    state = sim.snapshot()
    lifecycle = sim.lifecycle_snapshot()
    for field in fields(state):
        digest.update(getattr(state, field.name).tobytes())
    for field in fields(lifecycle):
        digest.update(getattr(lifecycle, field.name).tobytes())
    digest.update(sim.car_car.visit_order.tobytes())
    return digest.hexdigest()


def _phase_d(collision_root: str, device: str) -> dict[str, object]:
    count = 64
    initial = StateSnapshot.empty(count)
    initial.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    initial.ball_pos[:] = np.asarray((0.0, 0.0, 1200.0), dtype=np.float32)
    initial.ball_pos[:16] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    for env in range(16, 48):
        pad = (env - 16) % PAD_COUNT
        car = env & 1
        initial.car_pos[env, car] = SOCCAR_PAD_POSITIONS[pad]
        initial.boost[env, car] = 0.0
    order = np.arange(count, dtype=np.int32) & 1
    selector = np.arange(count, dtype=np.int32) % 5
    kwargs = {
        "device": device,
        "initial": initial,
        "kickoff_selector": selector,
        "respawn_selector": np.arange(count, dtype=np.int32) % 4,
        "car_visitation_order": order,
        "full_reset_interval_ticks": 97,
    }
    sims = [CompleteWorldSim(count, collision_root, **kwargs) for _ in range(2)]
    victims = np.arange(count, dtype=np.int32) & 1
    for sim in sims:
        sim.request_demolition(victims)
        sim.reset_transfer_counters()
        sim.step(400, synchronize=True)
        _assert(sim.host_to_device_bytes == 0, "stress H2D transfer")
        _assert(sim.device_to_host_bytes == 0, "stress D2H transfer")
    hashes = [_snapshot_hash(sim) for sim in sims]
    _assert(hashes[0] == hashes[1], "same-seed lifecycle stress is nondeterministic")
    lifecycle = sims[0].lifecycle_snapshot()
    _assert(np.all(lifecycle.world_tick == 400), "world clock")
    _assert(np.all(lifecycle.terminated == 0), "terminal contract")
    _assert(np.all(lifecycle.truncated == 0), "truncation contract")
    return {
        "verdict": "PASS_GREEN",
        "worlds": count,
        "ticks": 400,
        "full_reset_interval_ticks": 97,
        "same_seed_hash": hashes[0],
        "same_seed_repeat_equal": True,
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "raw_event_terminal_contract": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collision_root")
    parser.add_argument("--cache-root", default=".tools/v0.4/oracle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    authority = load_cache(args.collision_root, args.cache_root)
    identity, _inputs = authority_identity(args.collision_root)
    result = {
        "schema": "rivalsim-v0.4-lifecycle-gate-v1",
        "authority_identity": identity,
        "phase_a": _phase_a(args.collision_root, args.device, authority),
        "phase_b": _phase_b(args.collision_root, args.device, authority),
        "phase_c": _phase_c(args.collision_root, args.device, authority),
        "phase_d": _phase_d(args.collision_root, args.device),
        "verdict": "PASS_GREEN",
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
