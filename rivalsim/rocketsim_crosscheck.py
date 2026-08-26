"""Evaluation-only RocketSim runtimes for reciprocal Rival/Nexto validation.

The module deliberately keeps RocketSim authoritative.  It does not alter
RivalSim physics or either policy contract; it owns only match lifecycle,
adapter memory, compact telemetry, and public-state continuation capture.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from rivalsim.rocketsim_adapter import (
    PHYSICS_HZ,
    RIVAL_CADENCE_TICKS,
    FrozenRivalPolicy,
    RocketSimBatchState,
    RocketSimRivalMemory,
    SourceNextoPolicy,
    actions_to_controls,
    build_rival2_observation,
    canonical_boost_pads,
    car_hit_ticks,
    read_rocketsim_batch,
    update_rival_memory_after_tick,
)
from rivalsim.rival2_contracts import ORANGE_PAD_REMAP

REGULATION_TICKS = 5 * 60 * PHYSICS_HZ
DUEL_LIMIT_TICKS = 60 * PHYSICS_HZ
KICKOFF_SEEDS = np.asarray((7, 3, 2, 1, 0), dtype=np.int64)
GOAL_PLANE_Y = np.float32(5120.0)
GOAL_HALF_WIDTH = np.float32(893.0)
GOAL_HEIGHT = np.float32(642.775)
_ROCKETSIM_INITIALIZED_ROOT: Path | None = None


def initialize_rocketsim(collision_root: str | Path) -> Any:
    import RocketSim as rs

    global _ROCKETSIM_INITIALIZED_ROOT
    resolved = Path(collision_root).resolve()
    if _ROCKETSIM_INITIALIZED_ROOT is None:
        try:
            rs.init(str(resolved))
        except RuntimeError as error:
            if "Already inited" not in str(error):
                raise
        _ROCKETSIM_INITIALIZED_ROOT = resolved
    elif _ROCKETSIM_INITIALIZED_ROOT != resolved:
        raise RuntimeError(
            f"RocketSim already initialized from {_ROCKETSIM_INITIALIZED_ROOT}, not {resolved}"
        )
    return rs


def _new_world(rs: Any, kickoff_layout: int = 0) -> tuple[Any, list[Any], list[Any]]:
    arena = rs.Arena(rs.GameMode.SOCCAR, tick_rate=float(PHYSICS_HZ))
    cars = [arena.add_car(rs.Team.BLUE), arena.add_car(rs.Team.ORANGE)]
    pads = canonical_boost_pads(arena)
    arena.reset_kickoff(int(KICKOFF_SEEDS[int(kickoff_layout) % 5]))
    return arena, cars, pads


def _scatter_batch_state(
    base: RocketSimBatchState,
    subset: RocketSimBatchState,
    rows: np.ndarray,
) -> RocketSimBatchState:
    values: dict[str, np.ndarray] = {}
    for field in fields(RocketSimBatchState):
        value = getattr(base, field.name).copy()
        value[rows] = getattr(subset, field.name)
        values[field.name] = value
    return RocketSimBatchState(**values)


def _team_index(car: Any) -> int:
    return 0 if int(car.team) == 0 else 1


@dataclass(frozen=True, slots=True)
class RuntimeTiming:
    worlds: int
    physics_ticks: int
    seconds: float

    @property
    def world_ticks_per_second(self) -> float:
        return self.worlds * self.physics_ticks / self.seconds


class RivalMechanicsTelemetry:
    """Read-only controller and mechanic transition telemetry for Rival only.

    Candidate labels are deliberately conservative operational classifications,
    not claims that a named mechanic was executed intentionally.
    """

    WAVEDASH_LANDING_WINDOW_TICKS = 24
    RAPID_LANDING_TO_JUMP_TICKS = 12
    RAPID_JUMP_TO_FLIP_TICKS = 30
    DOUBLE_DASH_WINDOW_TICKS = 90
    LOW_AIR_TIME_SECONDS = 0.35

    def __init__(self, worlds: int, rival_side: np.ndarray):
        self.worlds = int(worlds)
        self.rival_side = np.asarray(rival_side, dtype=np.int32)
        self.decision_count = np.zeros(worlds, dtype=np.int64)
        self.decision_channel_active = np.zeros((worlds, 8), dtype=np.int64)
        self.physics_control_ticks = np.zeros(worlds, dtype=np.int64)
        self.physics_channel_active = np.zeros((worlds, 8), dtype=np.int64)
        self.jump_rising_edges = np.zeros(worlds, dtype=np.int64)
        self.jump_held_ticks_total = np.zeros(worlds, dtype=np.int64)
        self.jump_held_durations: list[list[int]] = [[] for _ in range(worlds)]
        self.first_jump_onsets = np.zeros(worlds, dtype=np.int64)
        self.double_jump_onsets = np.zeros(worlds, dtype=np.int64)
        self.flip_onsets = np.zeros(worlds, dtype=np.int64)
        self.unavailable_jump_presses = np.zeros(worlds, dtype=np.int64)
        self.previous_jump_control = np.zeros(worlds, dtype=bool)
        self.current_jump_hold = np.zeros(worlds, dtype=np.int32)
        self.last_wheel_contact_tick = np.full(worlds, -1, dtype=np.int64)
        self.last_landing_tick = np.full(worlds, -1, dtype=np.int64)
        self.last_first_jump_tick = np.full(worlds, -1, dtype=np.int64)
        self.last_flip_tick = np.full(worlds, -1, dtype=np.int64)
        self.last_flip_event_index = np.full(worlds, -1, dtype=np.int64)
        self.event_ledger: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]

    @staticmethod
    def _active_channels(action: np.ndarray) -> np.ndarray:
        analog = np.abs(action[..., :5]) > np.float32(1e-6)
        binary = action[..., 5:] >= np.float32(0.5)
        return np.concatenate((analog, binary), axis=-1)

    def record_decision(self, action: np.ndarray, active: np.ndarray) -> None:
        rows = np.flatnonzero(active)
        if not rows.size:
            return
        self.decision_count[rows] += 1
        self.decision_channel_active[rows] += self._active_channels(action[rows])

    @staticmethod
    def _flip_direction(action: np.ndarray, torque: np.ndarray) -> str:
        pitch = float(action[2])
        yaw = float(action[3])
        roll = float(action[4])
        if abs(pitch) + abs(yaw) + abs(roll) < 0.1:
            return "neutral_double_jump"
        if abs(pitch) >= abs(yaw) and abs(pitch) >= abs(roll):
            return "forward" if pitch < 0 else "backward"
        if abs(yaw) >= abs(roll):
            return "right" if yaw > 0 else "left"
        return "right_roll" if roll > 0 else "left_roll"

    def _append_candidate(self, world: int, event_index: int, label: str, evidence: dict[str, Any]) -> None:
        if event_index < 0:
            return
        event = self.event_ledger[world][event_index]
        if label not in event["candidate_labels"]:
            event["candidate_labels"].append(label)
            event["classification_evidence"][label] = evidence

    def after_reset(self, rows: np.ndarray | Sequence[int]) -> None:
        index = np.asarray(rows, dtype=np.int64)
        for world in index:
            if self.current_jump_hold[world] > 0:
                self.jump_held_durations[world].append(int(self.current_jump_hold[world]))
        self.previous_jump_control[index] = False
        self.current_jump_hold[index] = 0
        self.last_wheel_contact_tick[index] = -1
        self.last_landing_tick[index] = -1
        self.last_first_jump_tick[index] = -1
        self.last_flip_tick[index] = -1
        self.last_flip_event_index[index] = -1

    def update(
        self,
        tick: int,
        before: RocketSimBatchState,
        after: RocketSimBatchState,
        actions: np.ndarray,
        active: np.ndarray,
    ) -> None:
        rows = np.arange(self.worlds)
        side = self.rival_side
        selected_action = actions[rows, side]
        active_channels = self._active_channels(selected_action)
        active_rows = np.flatnonzero(active)
        self.physics_control_ticks[active_rows] += 1
        self.physics_channel_active[active_rows] += active_channels[active_rows]
        jump = selected_action[:, 5] >= 0.5
        rising = active & jump & ~self.previous_jump_control
        falling = active & ~jump & self.previous_jump_control
        for world in np.flatnonzero(falling):
            self.jump_held_durations[world].append(int(self.current_jump_hold[world]))
            self.current_jump_hold[world] = 0
        self.current_jump_hold[active & jump] += 1
        self.jump_held_ticks_total[active & jump] += 1
        self.jump_rising_edges[rising] += 1

        b_ground = before.on_ground[rows, side] != 0
        a_ground = after.on_ground[rows, side] != 0
        b_wheels = np.sum(before.wheels[rows, side] != 0, axis=1)
        a_wheels = np.sum(after.wheels[rows, side] != 0, axis=1)
        contact_before = b_wheels > 0
        contact_after = a_wheels > 0
        self.last_wheel_contact_tick[active & contact_before] = tick
        landing = active & ~contact_before & contact_after
        for world in np.flatnonzero(landing):
            self.last_landing_tick[world] = tick + 1
            event_index = int(self.last_flip_event_index[world])
            if event_index >= 0:
                event = self.event_ledger[world][event_index]
                since_flip = tick + 1 - int(event["tick"])
                event["ticks_flip_to_next_wheel_contact"] = since_flip
                if (
                    since_flip <= self.WAVEDASH_LANDING_WINDOW_TICKS
                    and float(event["air_time_before_seconds"]) <= self.LOW_AIR_TIME_SECONDS
                ):
                    self._append_candidate(
                        world,
                        event_index,
                        "wavedash_candidate",
                        {
                            "rule": "actual flip onset while airborne followed by wheel contact within 24 ticks and pre-flip air time <=0.35s",
                            "ticks_flip_to_wheel_contact": since_flip,
                            "air_time_before_seconds": float(event["air_time_before_seconds"]),
                        },
                    )

        first_jump = active & (before.has_jumped[rows, side] == 0) & (after.has_jumped[rows, side] != 0)
        double_jump = active & (before.has_double_jumped[rows, side] == 0) & (after.has_double_jumped[rows, side] != 0)
        flip = active & (before.has_flipped[rows, side] == 0) & (after.has_flipped[rows, side] != 0)
        self.first_jump_onsets[first_jump] += 1
        self.double_jump_onsets[double_jump] += 1
        self.flip_onsets[flip] += 1

        available = (
            b_ground
            | (before.has_jumped[rows, side] == 0)
            | (
                (before.has_double_jumped[rows, side] == 0)
                & (before.has_flipped[rows, side] == 0)
                & (before.air_time_since_jump[rows, side] < 1.25)
            )
        )
        self.unavailable_jump_presses[rising & ~available] += 1
        for world in np.flatnonzero(first_jump):
            self.last_first_jump_tick[world] = tick + 1

        speed_before = np.linalg.norm(before.car_vel[rows, side], axis=1)
        speed_after = np.linalg.norm(after.car_vel[rows, side], axis=1)
        for world in np.flatnonzero(flip):
            previous_flip = int(self.last_flip_tick[world])
            previous_event = int(self.last_flip_event_index[world])
            last_contact = int(self.last_wheel_contact_tick[world])
            last_jump = int(self.last_first_jump_tick[world])
            last_landing = int(self.last_landing_tick[world])
            torque = after.flip_rel_torque[world, side[world]].copy()
            event = {
                "tick": tick + 1,
                "side": int(side[world]),
                "direction": self._flip_direction(selected_action[world], torque),
                "controller": selected_action[world].astype(float).tolist(),
                "flip_rel_torque": torque.astype(float).tolist(),
                "on_ground_before": bool(b_ground[world]),
                "on_ground_after": bool(a_ground[world]),
                "wheel_contacts_before": int(b_wheels[world]),
                "wheel_contacts_after": int(a_wheels[world]),
                "ticks_last_wheel_contact_to_flip": None if last_contact < 0 else tick + 1 - last_contact,
                "ticks_last_wheel_contact_to_jump": None if last_contact < 0 or last_jump < 0 else last_jump - last_contact,
                "ticks_jump_to_flip": None if last_jump < 0 else tick + 1 - last_jump,
                "air_time_before_seconds": float(before.air_time[world, side[world]]),
                "air_time_since_jump_before_seconds": float(before.air_time_since_jump[world, side[world]]),
                "ticks_landing_to_flip": None if last_landing < 0 else tick + 1 - last_landing,
                "speed_before_uu_per_s": float(speed_before[world]),
                "speed_after_uu_per_s": float(speed_after[world]),
                "speed_delta_uu_per_s": float(speed_after[world] - speed_before[world]),
                "ticks_flip_to_next_wheel_contact": 0 if contact_after[world] else None,
                "candidate_labels": [],
                "classification_evidence": {},
            }
            self.event_ledger[world].append(event)
            event_index = len(self.event_ledger[world]) - 1
            if contact_before[world] or contact_after[world]:
                self._append_candidate(
                    world,
                    event_index,
                    "ground_contact_dodge_candidate",
                    {
                        "rule": "actual flip onset with wheel contact immediately before or after the transition",
                        "wheel_contacts_before": int(b_wheels[world]),
                        "wheel_contacts_after": int(a_wheels[world]),
                    },
                )
            landing_to_jump = None if last_landing < 0 or last_jump < last_landing else last_jump - last_landing
            jump_to_flip = None if last_jump < 0 else tick + 1 - last_jump
            if (
                landing_to_jump is not None
                and landing_to_jump <= self.RAPID_LANDING_TO_JUMP_TICKS
                and jump_to_flip is not None
                and jump_to_flip <= self.RAPID_JUMP_TO_FLIP_TICKS
                and float(event["air_time_before_seconds"]) <= self.LOW_AIR_TIME_SECONDS
            ):
                self._append_candidate(
                    world,
                    event_index,
                    "zapdash_candidate",
                    {
                        "rule": "landing-to-first-jump <=12 ticks, jump-to-actual-flip <=30 ticks, and pre-flip air time <=0.35s",
                        "ticks_landing_to_jump": landing_to_jump,
                        "ticks_jump_to_flip": jump_to_flip,
                    },
                )
            if previous_flip >= 0 and tick + 1 - previous_flip <= self.DOUBLE_DASH_WINDOW_TICKS:
                previous = self.event_ledger[world][previous_event]
                landing_between = last_landing >= previous_flip
                low_air_pair = (
                    float(previous["air_time_before_seconds"]) <= self.LOW_AIR_TIME_SECONDS
                    and float(event["air_time_before_seconds"]) <= self.LOW_AIR_TIME_SECONDS
                )
                if landing_between and low_air_pair:
                    evidence = {
                        "rule": "two actual flip onsets within 90 ticks with intervening landing and both pre-flip air times <=0.35s",
                        "ticks_between_flips": tick + 1 - previous_flip,
                        "intervening_landing_tick": last_landing,
                    }
                    self._append_candidate(world, previous_event, "double_dash_candidate", evidence)
                    self._append_candidate(world, event_index, "double_dash_candidate", evidence)
            self.last_flip_tick[world] = tick + 1
            self.last_flip_event_index[world] = event_index

        self.previous_jump_control[active] = jump[active]
        self.previous_jump_control[~active] = False

    def finalize(self) -> None:
        for world in range(self.worlds):
            if self.current_jump_hold[world] > 0:
                self.jump_held_durations[world].append(int(self.current_jump_hold[world]))
                self.current_jump_hold[world] = 0

    def world_rows(self) -> list[dict[str, Any]]:
        self.finalize()
        result: list[dict[str, Any]] = []
        for world in range(self.worlds):
            result.append(
                {
                    "decision_count": int(self.decision_count[world]),
                    "decision_channel_active": self.decision_channel_active[world].astype(int).tolist(),
                    "physics_control_ticks": int(self.physics_control_ticks[world]),
                    "physics_channel_active": self.physics_channel_active[world].astype(int).tolist(),
                    "jump_rising_edges": int(self.jump_rising_edges[world]),
                    "jump_held_ticks_total": int(self.jump_held_ticks_total[world]),
                    "jump_held_durations_ticks": self.jump_held_durations[world],
                    "first_jump_onsets": int(self.first_jump_onsets[world]),
                    "double_jump_onsets": int(self.double_jump_onsets[world]),
                    "flip_onsets": int(self.flip_onsets[world]),
                    "unavailable_jump_presses": int(self.unavailable_jump_presses[world]),
                    "flip_events": self.event_ledger[world],
                }
            )
        return result


class ComprehensiveBehaviorTelemetry:
    """Low-overhead physical-team accumulators plus authoritative event rows."""

    ACTION_MAG_EDGES = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    BOOST_EDGES = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    SPEED_EDGES = np.linspace(0.0, 2400.0, 49, dtype=np.float32)
    HEIGHT_EDGES = np.linspace(0.0, 2100.0, 43, dtype=np.float32)
    DISTANCE_EDGES = np.linspace(0.0, 10_000.0, 41, dtype=np.float32)
    BOOST_ADVANTAGE_EDGES = np.linspace(-100.0, 100.0, 41, dtype=np.float32)

    def __init__(self, worlds: int, rival_side: np.ndarray):
        self.worlds = int(worlds)
        self.rival_side = np.asarray(rival_side, dtype=np.int32)
        shape = (worlds, 2)
        self.tick_count = np.zeros(shape, dtype=np.int64)
        self.decision_count = np.zeros(shape, dtype=np.int64)
        self.decision_action_sum = np.zeros(shape + (8,), dtype=np.float64)
        self.decision_action_abs_sum = np.zeros(shape + (8,), dtype=np.float64)
        self.decision_action_active = np.zeros(shape + (8,), dtype=np.int64)
        self.action_sum = np.zeros(shape + (8,), dtype=np.float64)
        self.action_abs_sum = np.zeros(shape + (8,), dtype=np.float64)
        self.action_active = np.zeros(shape + (8,), dtype=np.int64)
        self.action_abs_hist = np.zeros(shape + (8, len(self.ACTION_MAG_EDGES) - 1), dtype=np.int64)
        self.boost_sum = np.zeros(shape, dtype=np.float64)
        self.boost_hist = np.zeros(shape + (len(self.BOOST_EDGES) - 1,), dtype=np.int64)
        self.boost_starved_ticks = np.zeros(shape, dtype=np.int64)
        self.boost_consumed_no_pickup_ticks = np.zeros(shape, dtype=np.float64)
        self.boost_advantage_sum = np.zeros(shape, dtype=np.float64)
        self.boost_advantage_ticks = np.zeros(shape, dtype=np.int64)
        self.boost_disadvantage_ticks = np.zeros(shape, dtype=np.int64)
        self.boost_advantage_hist = np.zeros(shape + (len(self.BOOST_ADVANTAGE_EDGES) - 1,), dtype=np.int64)
        self.speed_sum = np.zeros(shape, dtype=np.float64)
        self.speed_hist = np.zeros(shape + (len(self.SPEED_EDGES) - 1,), dtype=np.int64)
        self.supersonic_ticks = np.zeros(shape, dtype=np.int64)
        self.distance_traveled = np.zeros(shape, dtype=np.float64)
        self.grounded_ticks = np.zeros(shape, dtype=np.int64)
        self.airborne_ticks = np.zeros(shape, dtype=np.int64)
        self.airborne_height_sum = np.zeros(shape, dtype=np.float64)
        self.airborne_height_count = np.zeros(shape, dtype=np.int64)
        self.airborne_height_hist = np.zeros(shape + (len(self.HEIGHT_EDGES) - 1,), dtype=np.int64)
        self.maximum_height = np.zeros(shape, dtype=np.float32)
        self.demoed_ticks = np.zeros(shape, dtype=np.int64)
        self.car_ball_distance_sum = np.zeros(shape, dtype=np.float64)
        self.car_ball_distance_hist = np.zeros(shape + (len(self.DISTANCE_EDGES) - 1,), dtype=np.int64)
        self.car_opponent_distance_sum = np.zeros(shape, dtype=np.float64)
        self.car_opponent_distance_hist = np.zeros(shape + (len(self.DISTANCE_EDGES) - 1,), dtype=np.int64)
        self.field_occupancy = np.zeros(shape + (3,), dtype=np.int64)
        self.pad_pickups_small = np.zeros(shape, dtype=np.int64)
        self.pad_pickups_big = np.zeros(shape, dtype=np.int64)
        self.demos_inflicted = np.zeros(shape, dtype=np.int64)
        self.demos_suffered = np.zeros(shape, dtype=np.int64)
        self.shots = np.zeros(shape, dtype=np.int64)
        self.saves = np.zeros(shape, dtype=np.int64)
        self.pad_pickup_this_tick = np.zeros(shape, dtype=bool)
        self.previous_ball_pos = np.zeros((worlds, 3), dtype=np.float32)
        self.previous_ball_vel = np.zeros((worlds, 3), dtype=np.float32)
        self.touch_events: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.goal_events: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.pad_events: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.demo_events: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.shot_save_events: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.possession_chains: list[list[dict[str, Any]]] = [[] for _ in range(worlds)]
        self.chain_toucher = np.full(worlds, -1, dtype=np.int32)
        self.chain_start_tick = np.full(worlds, -1, dtype=np.int64)
        self.chain_last_touch_tick = np.full(worlds, -1, dtype=np.int64)
        self.chain_length = np.zeros(worlds, dtype=np.int32)
        self.last_touch_event_index = np.full(worlds, -1, dtype=np.int64)

    @staticmethod
    def _hist_add(target: np.ndarray, values: np.ndarray, edges: np.ndarray, active: np.ndarray) -> None:
        bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        rows = np.flatnonzero(active)
        world_index = np.repeat(rows, values.shape[1])
        side_index = np.tile(np.arange(values.shape[1]), rows.size)
        np.add.at(target, (world_index, side_index, bins[rows].reshape(-1)), 1)

    def begin_tick(self, state: RocketSimBatchState) -> None:
        self.previous_ball_pos[:] = state.ball_pos
        self.previous_ball_vel[:] = state.ball_vel
        self.pad_pickup_this_tick.fill(False)

    def record_decisions(self, sides: np.ndarray, actions: np.ndarray, active: np.ndarray) -> None:
        rows = np.flatnonzero(active)
        for world in rows:
            side = int(sides[world])
            action = actions[world]
            self.decision_count[world, side] += 1
            self.decision_action_sum[world, side] += action
            self.decision_action_abs_sum[world, side] += np.abs(action)
            self.decision_action_active[world, side] += np.concatenate(
                (np.abs(action[:5]) > 1e-6, action[5:] >= 0.5)
            )

    def update(
        self,
        before: RocketSimBatchState,
        after: RocketSimBatchState,
        actions: np.ndarray,
        active: np.ndarray,
    ) -> None:
        rows = np.flatnonzero(active)
        if not rows.size:
            return
        self.tick_count[rows] += 1
        self.action_sum[rows] += actions[rows]
        self.action_abs_sum[rows] += np.abs(actions[rows])
        self.action_active[rows] += np.concatenate(
            (np.abs(actions[rows, :, :5]) > 1e-6, actions[rows, :, 5:] >= 0.5), axis=2
        )
        action_bins = np.clip(
            np.searchsorted(self.ACTION_MAG_EDGES, np.abs(actions), side="right") - 1,
            0,
            len(self.ACTION_MAG_EDGES) - 2,
        )
        world_index = np.repeat(rows, 2 * 8)
        side_index = np.tile(np.repeat(np.arange(2), 8), rows.size)
        channel_index = np.tile(np.arange(8), rows.size * 2)
        np.add.at(
            self.action_abs_hist,
            (world_index, side_index, channel_index, action_bins[rows].reshape(-1)),
            1,
        )
        boost = after.boost
        self.boost_sum[rows] += boost[rows]
        self._hist_add(self.boost_hist, boost, self.BOOST_EDGES, active)
        self.boost_starved_ticks[rows] += boost[rows] < 1.0
        no_pickup = ~self.pad_pickup_this_tick
        decrease = np.maximum(before.boost - after.boost, 0.0)
        self.boost_consumed_no_pickup_ticks[rows] += decrease[rows] * no_pickup[rows]
        advantage = boost - boost[:, ::-1]
        self.boost_advantage_sum[rows] += advantage[rows]
        self.boost_advantage_ticks[rows] += advantage[rows] > 0
        self.boost_disadvantage_ticks[rows] += advantage[rows] < 0
        self._hist_add(self.boost_advantage_hist, advantage, self.BOOST_ADVANTAGE_EDGES, active)
        speed = np.linalg.norm(after.car_vel, axis=2)
        self.speed_sum[rows] += speed[rows]
        self._hist_add(self.speed_hist, speed, self.SPEED_EDGES, active)
        self.supersonic_ticks[rows] += after.is_supersonic[rows] != 0
        self.distance_traveled[rows] += np.linalg.norm(after.car_pos[rows] - before.car_pos[rows], axis=2)
        grounded = after.on_ground != 0
        self.grounded_ticks[rows] += grounded[rows]
        self.airborne_ticks[rows] += ~grounded[rows]
        height = after.car_pos[:, :, 2]
        airborne = ~grounded
        self.airborne_height_sum[rows] += height[rows] * airborne[rows]
        self.airborne_height_count[rows] += airborne[rows]
        height_bins = np.clip(
            np.searchsorted(self.HEIGHT_EDGES, height, side="right") - 1,
            0,
            len(self.HEIGHT_EDGES) - 2,
        )
        height_active = active[:, None] & airborne
        height_world, height_side = np.nonzero(height_active)
        np.add.at(
            self.airborne_height_hist,
            (height_world, height_side, height_bins[height_world, height_side]),
            1,
        )
        self.maximum_height[rows] = np.maximum(self.maximum_height[rows], height[rows])
        self.demoed_ticks[rows] += after.is_demoed[rows] != 0
        ball_distance = np.linalg.norm(after.car_pos - after.ball_pos[:, None, :], axis=2)
        opponent_distance = np.linalg.norm(after.car_pos[:, 0] - after.car_pos[:, 1], axis=1)
        opponent_pair = np.stack((opponent_distance, opponent_distance), axis=1)
        self.car_ball_distance_sum[rows] += ball_distance[rows]
        self.car_opponent_distance_sum[rows] += opponent_pair[rows]
        self._hist_add(self.car_ball_distance_hist, ball_distance, self.DISTANCE_EDGES, active)
        self._hist_add(self.car_opponent_distance_hist, opponent_pair, self.DISTANCE_EDGES, active)
        for side in range(2):
            canonical_y = after.car_pos[:, side, 1] * (1.0 if side == 0 else -1.0)
            region = np.where(canonical_y > 1706.7, 2, np.where(canonical_y < -1706.7, 0, 1))
            np.add.at(self.field_occupancy, (rows, side, region[rows]), 1)

    def _finalize_touch_displacement(self, world: int, ball_pos: np.ndarray, tick: int) -> None:
        index = int(self.last_touch_event_index[world])
        if index < 0:
            return
        event = self.touch_events[world][index]
        if event.get("result_finalized"):
            return
        start = np.asarray(event["ball_position_after_touch"], dtype=np.float32)
        displacement = ball_pos - start
        side = int(event["side"])
        forward = float(displacement[1] * (1.0 if side == 0 else -1.0))
        event.update(
            {
                "result_finalized": True,
                "result_duration_ticks": int(tick - event["tick"]),
                "result_ball_displacement": displacement.astype(float).tolist(),
                "result_forward_displacement_uu": forward,
                "result_direction": "offensive" if forward > 100 else ("defensive" if forward < -100 else "neutral"),
            }
        )

    def _finalize_chain(self, world: int, end_tick: int, reason: str) -> None:
        if self.chain_toucher[world] < 0:
            return
        self.possession_chains[world].append(
            {
                "side": int(self.chain_toucher[world]),
                "start_tick": int(self.chain_start_tick[world]),
                "end_tick": int(end_tick),
                "duration_ticks": int(end_tick - self.chain_start_tick[world]),
                "touches": int(self.chain_length[world]),
                "end_reason": reason,
            }
        )
        self.chain_toucher[world] = -1
        self.chain_length[world] = 0

    def on_touch(self, world: int, side: int, arena: Any, kickoff_active: bool) -> None:
        tick = int(arena.tick_count)
        ball = arena.ball.get_state()
        pos = _vector(ball.pos)
        vel = _vector(ball.vel)
        self._finalize_touch_displacement(world, pos, tick)
        previous_toucher = int(self.chain_toucher[world])
        if previous_toucher != side:
            self._finalize_chain(world, tick, "opponent_handoff" if previous_toucher >= 0 else "start")
            self.chain_toucher[world] = side
            self.chain_start_tick[world] = tick
            self.chain_length[world] = 0
        self.chain_length[world] += 1
        self.chain_last_touch_tick[world] = tick
        canonical_y = float(pos[1] * (1.0 if side == 0 else -1.0))
        event = {
            "tick": tick,
            "side": side,
            "kickoff_phase": bool(kickoff_active),
            "field_region": "offensive" if canonical_y > 1706.7 else ("defensive" if canonical_y < -1706.7 else "midfield"),
            "ball_position_after_touch": pos.astype(float).tolist(),
            "ball_velocity_before_tick": self.previous_ball_vel[world].astype(float).tolist(),
            "ball_velocity_after_touch": vel.astype(float).tolist(),
            "ball_speed_before_tick": float(np.linalg.norm(self.previous_ball_vel[world])),
            "ball_speed_after_touch": float(np.linalg.norm(vel)),
            "ball_speed_delta": float(np.linalg.norm(vel) - np.linalg.norm(self.previous_ball_vel[world])),
            "result_finalized": False,
        }
        self.touch_events[world].append(event)
        self.last_touch_event_index[world] = len(self.touch_events[world]) - 1

    def on_pad(self, world: int, side: int, pad: Any, car: Any, arena: Any) -> None:
        big = bool(pad.is_big)
        if big:
            self.pad_pickups_big[world, side] += 1
        else:
            self.pad_pickups_small[world, side] += 1
        self.pad_pickup_this_tick[world, side] = True
        state = car.get_state()
        self.pad_events[world].append(
            {"tick": int(arena.tick_count), "side": side, "big": big, "boost_after_pickup": float(state.boost)}
        )

    def on_demo(self, world: int, bumper_side: int, victim_side: int, arena: Any) -> None:
        self.demos_inflicted[world, bumper_side] += 1
        self.demos_suffered[world, victim_side] += 1
        self.demo_events[world].append(
            {"tick": int(arena.tick_count), "bumper_side": bumper_side, "victim_side": victim_side}
        )

    def on_shot(self, world: int, side: int, arena: Any) -> None:
        self.shots[world, side] += 1
        self.shot_save_events[world].append({"tick": int(arena.tick_count), "kind": "shot", "side": side})

    def on_save(self, world: int, side: int, arena: Any) -> None:
        self.saves[world, side] += 1
        self.shot_save_events[world].append({"tick": int(arena.tick_count), "kind": "save", "side": side})

    def on_goal(
        self,
        world: int,
        scorer: int,
        arena: Any,
        kickoff_touch_count: int,
        *,
        entry_valid: bool,
        entry_x: float,
        entry_z: float,
        entry_speed: float,
        entry_angle: float | None,
    ) -> None:
        tick = int(arena.tick_count)
        ball = arena.ball.get_state()
        pos = _vector(ball.pos)
        vel = _vector(ball.vel)
        self._finalize_touch_displacement(world, pos, tick)
        last_toucher = int(self.chain_toucher[world])
        last_touch_tick = int(self.chain_last_touch_tick[world])
        self._finalize_chain(world, tick, "goal")
        scoring_forward = float(vel[1] * (1.0 if scorer == 0 else -1.0))
        speed = float(np.linalg.norm(vel))
        angle = None if speed <= 1e-6 else float(np.degrees(np.arccos(np.clip(scoring_forward / speed, -1.0, 1.0))))
        self.goal_events[world].append(
            {
                "tick": tick,
                "scorer": scorer,
                "kickoff_touch_count": int(kickoff_touch_count),
                "phase": "kickoff" if kickoff_touch_count <= 1 else "established_open_play",
                "last_toucher": last_toucher,
                "scorer_matches_last_toucher": bool(last_toucher == scorer),
                "final_touch_to_goal_ticks": None if last_touch_tick < 0 else tick - last_touch_tick,
                "ball_position": pos.astype(float).tolist(),
                "ball_velocity": vel.astype(float).tolist(),
                "goal_entry_speed_uu_per_s": speed,
                "goal_entry_angle_degrees_from_goal_normal": angle,
                "scoring_plane_entry_valid": entry_valid,
                "scoring_plane_entry_x_uu": entry_x,
                "scoring_plane_entry_z_uu": entry_z,
                "scoring_plane_entry_speed_uu_per_s": entry_speed,
                "scoring_plane_entry_angle_degrees_from_goal_normal": entry_angle,
            }
        )

    def after_reset(self, rows: np.ndarray | Sequence[int], tick: int) -> None:
        for world in np.asarray(rows, dtype=np.int64):
            self._finalize_chain(int(world), tick, "reset")
            self.last_touch_event_index[world] = -1
            self.chain_last_touch_tick[world] = -1

    def world_rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for world in range(self.worlds):
            result.append(
                {
                    "tick_count": self.tick_count[world].astype(int).tolist(),
                    "decision_count": self.decision_count[world].astype(int).tolist(),
                    "decision_action_sum": self.decision_action_sum[world].tolist(),
                    "decision_action_abs_sum": self.decision_action_abs_sum[world].tolist(),
                    "decision_action_active": self.decision_action_active[world].astype(int).tolist(),
                    "action_sum": self.action_sum[world].tolist(),
                    "action_abs_sum": self.action_abs_sum[world].tolist(),
                    "action_active": self.action_active[world].astype(int).tolist(),
                    "action_abs_hist": self.action_abs_hist[world].astype(int).tolist(),
                    "boost_sum": self.boost_sum[world].tolist(),
                    "boost_hist": self.boost_hist[world].astype(int).tolist(),
                    "boost_starved_ticks": self.boost_starved_ticks[world].astype(int).tolist(),
                    "boost_consumed_no_pickup_ticks": self.boost_consumed_no_pickup_ticks[world].tolist(),
                    "boost_advantage_sum": self.boost_advantage_sum[world].tolist(),
                    "boost_advantage_ticks": self.boost_advantage_ticks[world].astype(int).tolist(),
                    "boost_disadvantage_ticks": self.boost_disadvantage_ticks[world].astype(int).tolist(),
                    "boost_advantage_hist": self.boost_advantage_hist[world].astype(int).tolist(),
                    "speed_sum": self.speed_sum[world].tolist(),
                    "speed_hist": self.speed_hist[world].astype(int).tolist(),
                    "supersonic_ticks": self.supersonic_ticks[world].astype(int).tolist(),
                    "distance_traveled": self.distance_traveled[world].tolist(),
                    "grounded_ticks": self.grounded_ticks[world].astype(int).tolist(),
                    "airborne_ticks": self.airborne_ticks[world].astype(int).tolist(),
                    "airborne_height_sum": self.airborne_height_sum[world].tolist(),
                    "airborne_height_count": self.airborne_height_count[world].astype(int).tolist(),
                    "airborne_height_hist": self.airborne_height_hist[world].astype(int).tolist(),
                    "maximum_height": self.maximum_height[world].tolist(),
                    "demoed_ticks": self.demoed_ticks[world].astype(int).tolist(),
                    "car_ball_distance_sum": self.car_ball_distance_sum[world].tolist(),
                    "car_ball_distance_hist": self.car_ball_distance_hist[world].astype(int).tolist(),
                    "car_opponent_distance_sum": self.car_opponent_distance_sum[world].tolist(),
                    "car_opponent_distance_hist": self.car_opponent_distance_hist[world].astype(int).tolist(),
                    "field_occupancy": self.field_occupancy[world].astype(int).tolist(),
                    "pad_pickups_small": self.pad_pickups_small[world].astype(int).tolist(),
                    "pad_pickups_big": self.pad_pickups_big[world].astype(int).tolist(),
                    "demos_inflicted": self.demos_inflicted[world].astype(int).tolist(),
                    "demos_suffered": self.demos_suffered[world].astype(int).tolist(),
                    "shots": self.shots[world].astype(int).tolist(),
                    "saves": self.saves[world].astype(int).tolist(),
                    "touch_events": self.touch_events[world],
                    "goal_events": self.goal_events[world],
                    "pad_events": self.pad_events[world],
                    "demo_events": self.demo_events[world],
                    "shot_save_events": self.shot_save_events[world],
                    "possession_chains": self.possession_chains[world],
                    "histogram_edges": {
                        "action_magnitude": self.ACTION_MAG_EDGES.tolist(),
                        "boost": self.BOOST_EDGES.tolist(),
                        "speed": self.SPEED_EDGES.tolist(),
                        "airborne_height": self.HEIGHT_EDGES.tolist(),
                        "distance": self.DISTANCE_EDGES.tolist(),
                        "boost_advantage": self.BOOST_ADVANTAGE_EDGES.tolist(),
                    },
                }
            )
        return result


class RocketSimEventTelemetry:
    """Callback-backed event ledger shared by matches, harvest, and duels."""

    def __init__(self, worlds: int, behavior: ComprehensiveBehaviorTelemetry | None = None):
        self.worlds = int(worlds)
        self.goal_team = np.full(worlds, -1, dtype=np.int32)
        self.goal_tick = np.full(worlds, -1, dtype=np.int64)
        self.goal_entry_x = np.zeros(worlds, dtype=np.float32)
        self.goal_entry_z = np.zeros(worlds, dtype=np.float32)
        self.goal_entry_valid = np.zeros(worlds, dtype=bool)
        self.goal_entry_speed = np.zeros(worlds, dtype=np.float32)
        self.goal_entry_angle = np.zeros(worlds, dtype=np.float32)
        self.pending_entry_valid = np.zeros(worlds, dtype=bool)
        self.pending_entry_team = np.full(worlds, -1, dtype=np.int32)
        self.pending_entry_x = np.zeros(worlds, dtype=np.float32)
        self.pending_entry_z = np.zeros(worlds, dtype=np.float32)
        self.pending_entry_speed = np.zeros(worlds, dtype=np.float32)
        self.pending_entry_angle = np.zeros(worlds, dtype=np.float32)
        self.touch_count = np.zeros((worlds, 2), dtype=np.int64)
        self.kickoff_first_touch_count = np.zeros((worlds, 2), dtype=np.int64)
        self.kickoff_goal_count = np.zeros((worlds, 2), dtype=np.int64)
        self.demo_count = np.zeros((worlds, 2), dtype=np.int64)
        self.possession_total = np.zeros((worlds, 2), dtype=np.int64)
        self.possession_same = np.zeros((worlds, 2), dtype=np.int64)
        self.possession_opponent = np.zeros((worlds, 2), dtype=np.int64)
        self.kickoff_touch_count = np.zeros(worlds, dtype=np.int32)
        self.kickoff_active = np.ones(worlds, dtype=bool)
        self.last_toucher = np.full(worlds, -1, dtype=np.int32)
        self.last_touch_callback_tick = np.full((worlds, 2), -1, dtype=np.int64)
        self.previous_ball_position = np.zeros((worlds, 3), dtype=np.float32)
        self.goal_callback: Callable[..., None] | None = None
        self.behavior = behavior
        self.enabled = np.ones(worlds, dtype=bool)

    def attach(
        self,
        arenas: Sequence[Any],
        *,
        goal_callback: Callable[[int, int, Any], None] | None = None,
    ) -> None:
        self.goal_callback = goal_callback

        def touch(*, arena: Any, car: Any, data: Any, **_kwargs: Any) -> None:
            world = int(data)
            if not self.enabled[world]:
                return
            side = _team_index(car)
            tick = int(arena.tick_count)
            if self.last_touch_callback_tick[world, side] == tick:
                return
            self.last_touch_callback_tick[world, side] = tick
            previous = int(self.last_toucher[world])
            kickoff_phase = bool(self.kickoff_active[world])
            if previous >= 0:
                self.possession_total[world, previous] += 1
                if previous == side:
                    self.possession_same[world, previous] += 1
                else:
                    self.possession_opponent[world, previous] += 1
            self.touch_count[world, side] += 1
            if self.kickoff_touch_count[world] == 0:
                self.kickoff_first_touch_count[world, side] += 1
            self.kickoff_touch_count[world] += 1
            self.kickoff_active[world] = False
            self.last_toucher[world] = side
            if self.behavior is not None:
                self.behavior.on_touch(world, side, arena, kickoff_phase)

        def demo(*, arena: Any, bumper: Any, victim: Any, data: Any, **_kwargs: Any) -> None:
            world = int(data)
            if not self.enabled[world]:
                return
            if bumper is not None:
                self.demo_count[world, _team_index(bumper)] += 1
            if self.behavior is not None and bumper is not None and victim is not None:
                self.behavior.on_demo(world, _team_index(bumper), _team_index(victim), arena)

        def boost_pickup(*, arena: Any, car: Any, boost_pad: Any, data: Any, **_kwargs: Any) -> None:
            if not self.enabled[int(data)]:
                return
            if self.behavior is not None and car is not None and boost_pad is not None:
                self.behavior.on_pad(int(data), _team_index(car), boost_pad, car, arena)

        def shot(*, arena: Any, shooter: Any, data: Any, **_kwargs: Any) -> None:
            if not self.enabled[int(data)]:
                return
            if self.behavior is not None and shooter is not None:
                self.behavior.on_shot(int(data), _team_index(shooter), arena)

        def save(*, arena: Any, saver: Any, data: Any, **_kwargs: Any) -> None:
            if not self.enabled[int(data)]:
                return
            if self.behavior is not None and saver is not None:
                self.behavior.on_save(int(data), _team_index(saver), arena)

        def goal(*, arena: Any, team: int, data: Any, **_kwargs: Any) -> None:
            world = int(data)
            if not self.enabled[world]:
                return
            side = int(team)
            if self.goal_team[world] >= 0:
                return
            ball = arena.ball.get_state()
            after = np.asarray((ball.pos.x, ball.pos.y, ball.pos.z), dtype=np.float32)
            before = self.previous_ball_position[world]
            scoring_plane = GOAL_PLANE_Y if side == 0 else -GOAL_PLANE_Y
            delta = float(after[1] - before[1])
            if abs(delta) > 1e-6:
                fraction = float((scoring_plane - before[1]) / delta)
                if 0.0 <= fraction <= 1.0:
                    crossing = before + np.float32(fraction) * (after - before)
                    self.goal_entry_x[world] = (1.0 if side == 0 else -1.0) * crossing[0]
                    self.goal_entry_z[world] = crossing[2]
                    self.goal_entry_valid[world] = True
                    velocity = np.asarray((ball.vel.x, ball.vel.y, ball.vel.z), dtype=np.float32)
                    speed = float(np.linalg.norm(velocity))
                    forward = float(velocity[1] * (1.0 if side == 0 else -1.0))
                    self.goal_entry_speed[world] = speed
                    self.goal_entry_angle[world] = 0.0 if speed <= 1e-6 else float(np.degrees(np.arccos(np.clip(forward / speed, -1.0, 1.0))))
            if not self.goal_entry_valid[world] and self.pending_entry_valid[world] and self.pending_entry_team[world] == side:
                self.goal_entry_valid[world] = True
                self.goal_entry_x[world] = self.pending_entry_x[world]
                self.goal_entry_z[world] = self.pending_entry_z[world]
                self.goal_entry_speed[world] = self.pending_entry_speed[world]
                self.goal_entry_angle[world] = self.pending_entry_angle[world]
            self.goal_team[world] = side
            self.goal_tick[world] = int(arena.tick_count)
            if self.kickoff_touch_count[world] <= 1:
                self.kickoff_goal_count[world, side] += 1
            if self.behavior is not None:
                self.behavior.on_goal(
                    world,
                    side,
                    arena,
                    int(self.kickoff_touch_count[world]),
                    entry_valid=bool(self.goal_entry_valid[world]),
                    entry_x=float(self.goal_entry_x[world]),
                    entry_z=float(self.goal_entry_z[world]),
                    entry_speed=float(self.goal_entry_speed[world]),
                    entry_angle=float(self.goal_entry_angle[world]) if self.goal_entry_valid[world] else None,
                )
            self.pending_entry_valid[world] = False
            if self.goal_callback is not None:
                self.goal_callback(world, side, arena)

        for world, arena in enumerate(arenas):
            arena.set_ball_touch_callback(touch, world)
            arena.set_car_demo_callback(demo, world)
            arena.set_boost_pickup_callback(boost_pickup, world)
            arena.set_shot_event_callback(shot, world)
            arena.set_save_event_callback(save, world)
            arena.set_goal_score_callback(goal, world)

    def begin_tick(self, state: RocketSimBatchState) -> None:
        self.goal_team.fill(-1)
        self.goal_tick.fill(-1)
        self.goal_entry_valid.fill(False)
        self.previous_ball_position[:] = state.ball_pos
        if self.behavior is not None:
            self.behavior.begin_tick(state)

    def after_step(self, before: RocketSimBatchState, after: RocketSimBatchState) -> None:
        for side, sign in ((0, 1.0), (1, -1.0)):
            before_y = sign * before.ball_pos[:, 1]
            after_y = sign * after.ball_pos[:, 1]
            crossing = (before_y < GOAL_PLANE_Y) & (after_y >= GOAL_PLANE_Y)
            for world in np.flatnonzero(crossing):
                delta = float(after.ball_pos[world, 1] - before.ball_pos[world, 1])
                if abs(delta) <= 1e-6:
                    continue
                plane = float(sign * GOAL_PLANE_Y)
                fraction = float((plane - before.ball_pos[world, 1]) / delta)
                position = before.ball_pos[world] + np.float32(fraction) * (after.ball_pos[world] - before.ball_pos[world])
                velocity = before.ball_vel[world] + np.float32(fraction) * (after.ball_vel[world] - before.ball_vel[world])
                speed = float(np.linalg.norm(velocity))
                forward = float(sign * velocity[1])
                self.pending_entry_valid[world] = True
                self.pending_entry_team[world] = side
                self.pending_entry_x[world] = sign * position[0]
                self.pending_entry_z[world] = position[2]
                self.pending_entry_speed[world] = speed
                self.pending_entry_angle[world] = 0.0 if speed <= 1e-6 else float(np.degrees(np.arccos(np.clip(forward / speed, -1.0, 1.0))))
        returned_inside = np.abs(after.ball_pos[:, 1]) < GOAL_PLANE_Y
        no_goal = self.goal_team < 0
        self.pending_entry_valid[returned_inside & no_goal] = False

    def after_reset(self, rows: np.ndarray | Sequence[int]) -> None:
        index = np.asarray(rows, dtype=np.int64)
        self.kickoff_touch_count[index] = 0
        self.kickoff_active[index] = True
        self.last_toucher[index] = -1
        self.last_touch_callback_tick[index] = -1
        self.enabled[index] = True


class RocketSimPolicyRuntime:
    """Mixed-cadence RocketSim batch with callback-authoritative memory."""

    def __init__(
        self,
        collision_root: str | Path,
        checkpoint: str | Path,
        nexto_model: str | Path,
        layouts: np.ndarray,
        rival_side: np.ndarray,
        *,
        stochastic_rival: bool,
        seed: int,
        reset_goals: bool,
    ):
        self.rs = initialize_rocketsim(collision_root)
        self.layouts = np.asarray(layouts, dtype=np.int32)
        self.rival_side = np.asarray(rival_side, dtype=np.int32)
        if self.layouts.shape != self.rival_side.shape:
            raise ValueError("layout and side arrays differ")
        self.worlds = int(self.layouts.size)
        built = [_new_world(self.rs, int(layout)) for layout in self.layouts]
        self.arenas = [item[0] for item in built]
        self.cars = [item[1] for item in built]
        self.pads = [item[2] for item in built]
        self.memory = RocketSimRivalMemory.create(self.worlds)
        self.rival = FrozenRivalPolicy(
            checkpoint, stochastic=stochastic_rival, seed=seed
        )
        self.nexto = SourceNextoPolicy(self.worlds, nexto_model)
        self.nexto.reset_policy_memory(np.arange(self.worlds))
        self.actions = np.zeros((self.worlds, 2, 8), dtype=np.float32)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)
        self.host_tick = 0
        self.reset_goals = bool(reset_goals)
        self.reset_counter = np.zeros(self.worlds, dtype=np.int32)
        self.goal_total = np.zeros((self.worlds, 2), dtype=np.int32)
        self.goal_events: list[list[dict[str, Any]]] = [[] for _ in range(self.worlds)]
        self.behavior = ComprehensiveBehaviorTelemetry(self.worlds, self.rival_side)
        self.event = RocketSimEventTelemetry(self.worlds, self.behavior)
        self.event.attach(self.arenas, goal_callback=self._on_goal)
        self.event.previous_ball_position[:] = self.state.ball_pos
        self.done = np.zeros(self.worlds, dtype=bool)
        self.completion_tick = np.full(self.worlds, -1, dtype=np.int64)
        self.overtime = np.zeros(self.worlds, dtype=bool)
        self.total_kickoffs = np.ones(self.worlds, dtype=np.int32)
        self.mechanics = RivalMechanicsTelemetry(self.worlds, self.rival_side)
        initial_contact = np.any(self.state.wheels != 0, axis=2)
        rows = np.arange(self.worlds)
        self.mechanics.last_wheel_contact_tick[
            initial_contact[rows, self.rival_side]
        ] = 0

    def _on_goal(self, world: int, side: int, arena: Any) -> None:
        if self.done[world]:
            return
        self.goal_total[world, side] += 1
        self.goal_events[world].append(
            {
                "team": side,
                "host_tick": int(self.host_tick),
                "overtime": bool(self.overtime[world]),
                "kickoff_goal": bool(self.event.kickoff_touch_count[world] <= 1),
                "entry_valid": bool(self.event.goal_entry_valid[world]),
                "entry_x": float(self.event.goal_entry_x[world]),
                "entry_z": float(self.event.goal_entry_z[world]),
            }
        )
        if self.overtime[world] or not self.reset_goals:
            self.done[world] = True
            self.completion_tick[world] = self.host_tick + 1
            self.event.enabled[world] = False
        elif self.reset_goals:
            self.reset_counter[world] += 1
            layout = int((self.layouts[world] + self.reset_counter[world]) % 5)
            arena.reset_kickoff(int(KICKOFF_SEEDS[layout]))
            self.total_kickoffs[world] += 1

    def force_fresh_kickoff(self, rows: np.ndarray | Sequence[int]) -> None:
        index = np.asarray(rows, dtype=np.int64)
        for world in index:
            self.reset_counter[world] += 1
            layout = int((self.layouts[world] + self.reset_counter[world]) % 5)
            self.arenas[world].reset_kickoff(int(KICKOFF_SEEDS[layout]))
            self.total_kickoffs[world] += 1
        self.memory.reset_rows(index)
        self.nexto.reset_policy_memory(index)
        self.event.after_reset(index)
        self.mechanics.after_reset(index)
        self.behavior.after_reset(index, self.host_tick)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)

    def tick(self, active: np.ndarray | None = None) -> None:
        if active is None:
            active = np.ones(self.worlds, dtype=bool)
        active = np.asarray(active, dtype=bool)
        rows = np.arange(self.worlds)
        active_before_tick = active & ~self.done
        active_rows = np.flatnonzero(active_before_tick)
        if not active_rows.size:
            return
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            observation = build_rival2_observation(self.state, self.memory)
            rival_observation = observation[active_rows, self.rival_side[active_rows]]
            action = self.rival.act(rival_observation)
            self.actions[active_rows, self.rival_side[active_rows]] = action
            self.memory.previous_action[active_rows, self.rival_side[active_rows]] = action
            rival_actions = self.actions[rows, self.rival_side]
            self.mechanics.record_decision(rival_actions, active_before_tick)
            self.behavior.record_decisions(self.rival_side, rival_actions, active_before_tick)
            self.memory.clear_interval_events()
        nexto_side = 1 - self.rival_side
        nexto_decision = self.nexto.cadence_tick == 0
        nexto_action = self.nexto.tick(
            self.state, nexto_side, self.event.kickoff_active, active_before_tick
        )
        self.actions[active_rows, nexto_side[active_rows]] = nexto_action[active_rows]
        if nexto_decision:
            self.behavior.record_decisions(nexto_side, nexto_action, active_before_tick)
        controls = actions_to_controls(self.rs, self.actions[active_rows])
        for local, world in enumerate(active_rows):
            self.cars[world][0].set_controls(controls[local][0])
            self.cars[world][1].set_controls(controls[local][1])
        before = self.state
        self.event.begin_tick(before)
        self.rs.Arena.multi_step([self.arenas[index] for index in active_rows], 1)
        subset = read_rocketsim_batch(
            [self.arenas[index] for index in active_rows],
            [self.cars[index] for index in active_rows],
            [self.pads[index] for index in active_rows],
        )
        self.state = _scatter_batch_state(before, subset, active_rows)
        self.event.after_step(before, self.state)
        hit_ticks = np.full((self.worlds, 2), -1, dtype=np.int64)
        hit_ticks[active_rows] = car_hit_ticks([self.cars[index] for index in active_rows])
        update_rival_memory_after_tick(
            self.memory,
            before,
            self.state,
            self.actions,
            hit_ticks,
            active=active_before_tick,
        )
        self.mechanics.update(
            self.host_tick, before, self.state, self.actions, active_before_tick
        )
        self.behavior.update(before, self.state, self.actions, active_before_tick)
        goal_rows = np.flatnonzero(self.event.goal_team >= 0)
        reset_rows = goal_rows[~self.done[goal_rows]] if goal_rows.size else goal_rows
        if self.reset_goals and reset_rows.size:
            self.memory.reset_rows(reset_rows)
            self.nexto.reset_policy_memory(reset_rows)
            self.event.after_reset(reset_rows)
            self.mechanics.after_reset(reset_rows)
            self.behavior.after_reset(reset_rows, self.host_tick + 1)
        moved = self.state.ball_pos[:, 1] != 0.0
        self.event.kickoff_active[moved] = False
        self.host_tick += 1

    def run_ticks(self, ticks: int) -> RuntimeTiming:
        started = time.perf_counter()
        for _ in range(int(ticks)):
            self.tick()
        return RuntimeTiming(self.worlds, int(ticks), time.perf_counter() - started)

    def run_full_matches(self, *, overtime_block_ticks: int = 3600) -> list[RuntimeTiming]:
        timings = [self.run_ticks(REGULATION_TICKS)]
        tied = np.flatnonzero(self.goal_total[:, 0] == self.goal_total[:, 1])
        untied = np.setdiff1d(np.arange(self.worlds), tied, assume_unique=True)
        self.done[untied] = True
        self.event.enabled[untied] = False
        self.completion_tick[untied] = REGULATION_TICKS
        if tied.size:
            self.overtime[tied] = True
            self.force_fresh_kickoff(tied)
        while np.any(~self.done):
            timings.append(self.run_ticks(overtime_block_ticks))
        return timings

    def export_match_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        mechanics = self.mechanics.world_rows()
        behavior = self.behavior.world_rows()
        for world in range(self.worlds):
            side = int(self.rival_side[world])
            blue, orange = (int(item) for item in self.goal_total[world])
            winner = 0 if blue > orange else 1
            rows.append(
                {
                    "world": world,
                    "starting_layout": int(self.layouts[world]),
                    "rival_side": side,
                    "blue_score": blue,
                    "orange_score": orange,
                    "winner": winner,
                    "entered_overtime": bool(self.overtime[world]),
                    "total_physics_ticks": int(self.completion_tick[world]),
                    "total_kickoffs": int(self.total_kickoffs[world]),
                    "touch_blue": int(self.event.touch_count[world, 0]),
                    "touch_orange": int(self.event.touch_count[world, 1]),
                    "kickoff_first_touch_blue": int(self.event.kickoff_first_touch_count[world, 0]),
                    "kickoff_first_touch_orange": int(self.event.kickoff_first_touch_count[world, 1]),
                    "kickoff_goal_blue": int(self.event.kickoff_goal_count[world, 0]),
                    "kickoff_goal_orange": int(self.event.kickoff_goal_count[world, 1]),
                    "demo_blue": int(self.event.demo_count[world, 0]),
                    "demo_orange": int(self.event.demo_count[world, 1]),
                    "possession_total_blue": int(self.event.possession_total[world, 0]),
                    "possession_total_orange": int(self.event.possession_total[world, 1]),
                    "possession_same_blue": int(self.event.possession_same[world, 0]),
                    "possession_same_orange": int(self.event.possession_same[world, 1]),
                    "possession_opponent_blue": int(self.event.possession_opponent[world, 0]),
                    "possession_opponent_orange": int(self.event.possession_opponent[world, 1]),
                    "goal_events": self.goal_events[world],
                    "rival_mechanics": mechanics[world],
                    "behavior": behavior[world],
                    "rival_won": winner == side,
                }
            )
        return rows


class SelfPlayHarvestRuntime:
    """RocketSim source-trajectory generator for prospective open-play capture."""

    def __init__(
        self,
        collision_root: str | Path,
        checkpoint: str | Path,
        nexto_model: str | Path,
        layouts: np.ndarray,
        *,
        source: str,
        seed: int,
    ):
        if source not in {"rival_stochastic", "nexto_deterministic"}:
            raise ValueError(source)
        self.source = source
        self.rs = initialize_rocketsim(collision_root)
        self.layouts = np.asarray(layouts, dtype=np.int32)
        self.worlds = int(self.layouts.size)
        built = [_new_world(self.rs, int(layout)) for layout in self.layouts]
        self.arenas = [item[0] for item in built]
        self.cars = [item[1] for item in built]
        self.pads = [item[2] for item in built]
        self.memory = RocketSimRivalMemory.create(self.worlds)
        self.actions = np.zeros((self.worlds, 2, 8), dtype=np.float32)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)
        self.event = RocketSimEventTelemetry(self.worlds)
        self.event.attach(self.arenas, goal_callback=self._on_goal)
        self.event.previous_ball_position[:] = self.state.ball_pos
        self.host_tick = 0
        self.ticks_since_reset = np.zeros(self.worlds, dtype=np.int32)
        self.touched_since_reset = np.zeros(self.worlds, dtype=bool)
        self.reset_counter = np.zeros(self.worlds, dtype=np.int32)
        if source == "rival_stochastic":
            self.rival = FrozenRivalPolicy(checkpoint, stochastic=True, seed=seed)
            self.nexto_blue = None
            self.nexto_orange = None
        else:
            self.rival = None
            self.nexto_blue = SourceNextoPolicy(self.worlds, nexto_model)
            self.nexto_orange = SourceNextoPolicy(self.worlds, nexto_model)
            rows = np.arange(self.worlds)
            self.nexto_blue.reset_policy_memory(rows)
            self.nexto_orange.reset_policy_memory(rows)

    def _on_goal(self, world: int, _side: int, arena: Any) -> None:
        self.reset_counter[world] += 1
        layout = int((self.layouts[world] + self.reset_counter[world]) % 5)
        arena.reset_kickoff(int(KICKOFF_SEEDS[layout]))

    def tick(self, active: np.ndarray | None = None) -> None:
        if active is None:
            active = np.ones(self.worlds, dtype=bool)
        active = np.asarray(active, dtype=bool)
        if self.source == "rival_stochastic":
            if self.host_tick % RIVAL_CADENCE_TICKS == 0:
                observation = build_rival2_observation(self.state, self.memory)
                flat = observation.reshape(self.worlds * 2, -1)
                action = self.rival.act(flat).reshape(self.worlds, 2, 8)
                self.actions[:] = action
                self.memory.previous_action[:] = action
                self.memory.clear_interval_events()
        else:
            assert self.nexto_blue is not None and self.nexto_orange is not None
            kickoff_active = self.event.kickoff_active
            self.actions[:, 0] = self.nexto_blue.tick(
                self.state, np.zeros(self.worlds, dtype=np.int32), kickoff_active, active
            )
            self.actions[:, 1] = self.nexto_orange.tick(
                self.state, np.ones(self.worlds, dtype=np.int32), kickoff_active, active
            )
        controls = actions_to_controls(self.rs, self.actions)
        for world, pair in enumerate(self.cars):
            pair[0].set_controls(controls[world][0])
            pair[1].set_controls(controls[world][1])
        before = self.state
        previous_touches = self.event.touch_count.copy()
        self.event.begin_tick(before)
        self.rs.Arena.multi_step([self.arenas[index] for index in np.flatnonzero(active)], 1)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)
        self.event.after_step(before, self.state)
        update_rival_memory_after_tick(
            self.memory, before, self.state, self.actions, car_hit_ticks(self.cars)
        )
        touched = np.any(self.event.touch_count > previous_touches, axis=1)
        self.touched_since_reset |= touched
        self.ticks_since_reset[active] += 1
        goal_rows = np.flatnonzero(self.event.goal_team >= 0)
        if goal_rows.size:
            self.ticks_since_reset[goal_rows] = 0
            self.touched_since_reset[goal_rows] = False
            self.memory.reset_rows(goal_rows)
            self.event.after_reset(goal_rows)
            if self.nexto_blue is not None:
                self.nexto_blue.reset_policy_memory(goal_rows)
                self.nexto_orange.reset_policy_memory(goal_rows)
        self.event.kickoff_active[self.state.ball_pos[:, 1] != 0.0] = False
        self.host_tick += 1

    def reassign_rows(self, rows: np.ndarray, global_index: np.ndarray) -> None:
        selected = np.asarray(rows, dtype=np.int64)
        assigned = np.asarray(global_index, dtype=np.int64)
        if selected.size != assigned.size:
            raise ValueError("harvest reassignment size mismatch")
        for row, index in zip(selected, assigned, strict=True):
            self.layouts[row] = int(index % 5)
            self.arenas[row].reset_kickoff(int(KICKOFF_SEEDS[int(index % 5)]))
            self.reset_counter[row] = 0
        self.memory.reset_rows(selected)
        self.actions[selected] = 0
        self.event.after_reset(selected)
        self.ticks_since_reset[selected] = 0
        self.touched_since_reset[selected] = False
        if self.nexto_blue is not None:
            self.nexto_blue.reset_policy_memory(selected)
            self.nexto_orange.reset_policy_memory(selected)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)

    def eligible(self, target_age: np.ndarray) -> np.ndarray:
        inside = np.abs(self.state.ball_pos[:, 1]) < GOAL_PLANE_Y
        active = ~np.any(self.state.is_demoed != 0, axis=1)
        no_goal = self.event.goal_team < 0
        return (
            (self.ticks_since_reset >= target_age)
            & (self.ticks_since_reset >= 600)
            & self.touched_since_reset
            & inside
            & active
            & no_goal
        )


def _vector(value: Any) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float32)


def _rot(value: Any) -> np.ndarray:
    return np.asarray(value.as_numpy(), dtype=np.float32).reshape(3, 3)


def capture_public_state(
    runtime: SelfPlayHarvestRuntime, rows: np.ndarray, global_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Serialize every public RocketSim continuation field used by SOCCAR."""

    selected = np.asarray(rows, dtype=np.int64)
    count = int(selected.size)
    result: dict[str, np.ndarray] = {
        "base_index": np.asarray(global_index, dtype=np.int32),
        "source": np.full(count, 0 if runtime.source == "rival_stochastic" else 1, dtype=np.int8),
        "capture_tick": runtime.ticks_since_reset[selected].astype(np.int32),
        "ball_pos": runtime.state.ball_pos[selected].copy(),
        "ball_vel": runtime.state.ball_vel[selected].copy(),
        "ball_ang_vel": runtime.state.ball_ang_vel[selected].copy(),
        "car_pos": runtime.state.car_pos[selected].copy(),
        "car_vel": runtime.state.car_vel[selected].copy(),
        "car_ang_vel": runtime.state.car_ang_vel[selected].copy(),
        "car_boost": runtime.state.boost[selected].copy(),
        "pad_cooldown": runtime.state.pad_cooldown[selected].copy(),
        "pad_active": runtime.state.pad_active[selected].astype(np.int8),
    }
    result["ball_rot"] = np.stack([_rot(runtime.arenas[row].ball.get_state().rot_mat) for row in selected])
    car_states = [[runtime.cars[row][side].get_state() for side in range(2)] for row in selected]
    result["car_rot"] = np.asarray(
        [[_rot(states[side].rot_mat) for side in range(2)] for states in car_states], dtype=np.float32
    )
    vector_fields = ("flip_rel_torque", "world_contact_normal")
    float_fields = (
        "air_time", "air_time_since_jump", "auto_flip_timer", "auto_flip_torque_scale",
        "boosting_time", "car_contact_cooldown_timer", "demo_respawn_timer", "flip_time",
        "handbrake_val", "jump_time", "supersonic_time", "time_spent_boosting",
    )
    int_fields = (
        "car_contact_id", "has_double_jumped", "has_flipped", "has_jumped",
        "has_world_contact", "is_auto_flipping", "is_demoed", "is_flipping",
        "is_jumping", "is_on_ground", "is_supersonic",
    )
    for field in vector_fields:
        result[f"car_{field}"] = np.asarray(
            [[_vector(getattr(states[side], field)) for side in range(2)] for states in car_states],
            dtype=np.float32,
        )
    for field in float_fields:
        result[f"car_{field}"] = np.asarray(
            [[float(getattr(states[side], field)) for side in range(2)] for states in car_states],
            dtype=np.float32,
        )
    for field in int_fields:
        result[f"car_{field}"] = np.asarray(
            [[int(getattr(states[side], field)) for side in range(2)] for states in car_states],
            dtype=np.int32,
        )
    result["car_wheels_with_contact"] = np.asarray(
        [[states[side].wheels_with_contact for side in range(2)] for states in car_states],
        dtype=np.int8,
    )
    controls = np.empty((count, 2, 8), dtype=np.float32)
    for local, states in enumerate(car_states):
        for side, state in enumerate(states):
            value = state.last_controls
            controls[local, side] = (
                value.throttle, value.steer, value.pitch, value.yaw, value.roll,
                value.jump, value.boost, value.handbrake,
            )
    result["car_last_controls"] = controls
    for name in (
        "episode_ticks", "no_touch_ticks", "kickoff_indicator", "touch_event",
        "demoed_event", "time_since_boosted", "sticky_ticks", "last_hit_tick",
        "previous_demoed",
    ):
        result[f"memory_{name}"] = getattr(runtime.memory, name)[selected].copy()
    result["memory_previous_action"] = np.zeros((count, 2, 8), dtype=np.float32)
    return result


def concatenate_banks(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("empty state-bank parts")
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise ValueError("state-bank schemas differ")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in sorted(keys)}


def mirror_bank(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in bank.items()}
    vec_keys = ["ball_pos", "ball_vel", "ball_ang_vel"]
    for key in vec_keys:
        result[key][..., :2] *= -1
    result["ball_rot"][..., :, :2] *= -1
    for key, value in list(result.items()):
        if key.startswith("car_") and value.ndim >= 2 and value.shape[1] == 2:
            result[key] = value[:, ::-1].copy()
    for key in ("car_pos", "car_vel", "car_ang_vel", "car_world_contact_normal"):
        result[key][..., :2] *= -1
    result["car_rot"][..., :, :2] *= -1
    for key, value in list(result.items()):
        if key.startswith("memory_") and value.ndim >= 2 and value.shape[1] == 2:
            result[key] = value[:, ::-1].copy()
    result["pad_cooldown"] = result["pad_cooldown"][:, ORANGE_PAD_REMAP]
    result["pad_active"] = result["pad_active"][:, ORANGE_PAD_REMAP]
    return result


def _make_vec(rs: Any, value: np.ndarray) -> Any:
    return rs.Vec(float(value[0]), float(value[1]), float(value[2]))


def _make_rot(rs: Any, value: np.ndarray) -> Any:
    return rs.RotMat(
        _make_vec(rs, value[0]),
        _make_vec(rs, value[1]),
        _make_vec(rs, value[2]),
    )


def restore_public_state(
    rs: Any,
    arenas: Sequence[Any],
    cars: Sequence[Sequence[Any]],
    pads: Sequence[Sequence[Any]],
    bank: dict[str, np.ndarray],
) -> RocketSimRivalMemory:
    count = len(arenas)
    if bank["ball_pos"].shape[0] != count:
        raise ValueError("restore bank size mismatch")
    vector_fields = ("flip_rel_torque", "world_contact_normal")
    float_fields = (
        "air_time", "air_time_since_jump", "auto_flip_timer", "auto_flip_torque_scale",
        "boosting_time", "car_contact_cooldown_timer", "demo_respawn_timer", "flip_time",
        "handbrake_val", "jump_time", "supersonic_time", "time_spent_boosting",
    )
    bool_fields = (
        "has_double_jumped", "has_flipped", "has_jumped", "has_world_contact",
        "is_auto_flipping", "is_demoed", "is_flipping", "is_jumping",
        "is_on_ground", "is_supersonic",
    )
    for world, arena in enumerate(arenas):
        ball = rs.BallState()
        ball.pos = _make_vec(rs, bank["ball_pos"][world])
        ball.vel = _make_vec(rs, bank["ball_vel"][world])
        ball.ang_vel = _make_vec(rs, bank["ball_ang_vel"][world])
        ball.rot_mat = _make_rot(rs, bank["ball_rot"][world])
        arena.ball.set_state(ball)
        for side in range(2):
            state = rs.CarState()
            state.pos = _make_vec(rs, bank["car_pos"][world, side])
            state.vel = _make_vec(rs, bank["car_vel"][world, side])
            state.ang_vel = _make_vec(rs, bank["car_ang_vel"][world, side])
            state.rot_mat = _make_rot(rs, bank["car_rot"][world, side])
            state.boost = float(bank["car_boost"][world, side])
            for field in vector_fields:
                setattr(state, field, _make_vec(rs, bank[f"car_{field}"][world, side]))
            for field in float_fields:
                setattr(state, field, float(bank[f"car_{field}"][world, side]))
            for field in bool_fields:
                setattr(state, field, bool(bank[f"car_{field}"][world, side]))
            state.wheels_with_contact = tuple(
                bool(value) for value in bank["car_wheels_with_contact"][world, side]
            )
            row = bank["car_last_controls"][world, side]
            state.last_controls = rs.CarControls(
                throttle=float(row[0]), steer=float(row[1]), pitch=float(row[2]),
                yaw=float(row[3]), roll=float(row[4]), jump=bool(row[5]),
                boost=bool(row[6]), handbrake=bool(row[7]),
            )
            cars[world][side].set_state(state)
        for pad_index, pad in enumerate(pads[world]):
            state = pad.get_state()
            state.is_active = bool(bank["pad_active"][world, pad_index])
            state.cooldown = float(bank["pad_cooldown"][world, pad_index])
            state.prev_locked_car_id = 0
            pad.set_state(state)
    memory = RocketSimRivalMemory.create(count)
    for name in (
        "episode_ticks", "no_touch_ticks", "kickoff_indicator", "touch_event",
        "demoed_event", "previous_action", "time_since_boosted", "sticky_ticks",
        "last_hit_tick", "previous_demoed",
    ):
        getattr(memory, name)[:] = bank[f"memory_{name}"]
    memory.previous_action.fill(0)
    return memory


class OpenPlayDuelRuntime(RocketSimPolicyRuntime):
    """Four-way restored first-goal duels; no kickoff and no goal reset."""

    def __init__(
        self,
        collision_root: str | Path,
        checkpoint: str | Path,
        nexto_model: str | Path,
        bank: dict[str, np.ndarray],
        rival_side: np.ndarray,
        *,
        seed: int,
    ):
        layouts = np.zeros(len(rival_side), dtype=np.int32)
        super().__init__(
            collision_root, checkpoint, nexto_model, layouts, rival_side,
            stochastic_rival=False, seed=seed, reset_goals=False,
        )
        self.memory = restore_public_state(self.rs, self.arenas, self.cars, self.pads, bank)
        self.state = read_rocketsim_batch(self.arenas, self.cars, self.pads)
        self.actions.fill(0)
        self.nexto.previous_action.fill(0)
        self.nexto.kickoff_index.fill(-1)
        self.event.kickoff_active.fill(False)
        self.event.kickoff_touch_count.fill(0)
        self.event.previous_ball_position[:] = self.state.ball_pos
        self.total_kickoffs.fill(0)
        self.mechanics.after_reset(np.arange(self.worlds))
        self.behavior.after_reset(np.arange(self.worlds), 0)
        initial_contact = np.any(self.state.wheels != 0, axis=2)
        rows = np.arange(self.worlds)
        self.mechanics.last_wheel_contact_tick[
            initial_contact[rows, self.rival_side]
        ] = 0
        self.initial_ball = self.state.ball_pos.copy()
        self.initial_car = self.state.car_pos.copy()
        self.initial_boost = self.state.boost.copy()

    def run_duels(self, limit_ticks: int = DUEL_LIMIT_TICKS) -> RuntimeTiming:
        started = time.perf_counter()
        ticks = 0
        while ticks < limit_ticks and np.any(~self.done):
            self.tick()
            ticks += 1
        return RuntimeTiming(self.worlds, ticks, time.perf_counter() - started)

    def export_duel_rows(
        self,
        base_index: np.ndarray,
        variant: np.ndarray,
        source: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        mechanics = self.mechanics.world_rows()
        behavior = self.behavior.world_rows()
        for world in range(self.worlds):
            side = int(self.rival_side[world])
            winner = int(np.argmax(self.goal_total[world])) if self.done[world] else -1
            distance = np.linalg.norm(
                self.initial_car[world] - self.initial_ball[world, None, :], axis=1
            )
            inherited = 0 if int(variant[world]) < 2 else 1
            inherited = side if inherited == 0 else 1 - side
            rows.append(
                {
                    "base_index": int(base_index[world]),
                    "variant": int(variant[world]),
                    "source": "rival_stochastic" if int(source[world]) == 0 else "nexto_deterministic",
                    "mirrored": bool(int(variant[world]) >= 2),
                    "rival_side": side,
                    "winner": winner,
                    "outcome": "draw" if winner < 0 else ("rival" if winner == side else "nexto"),
                    "elapsed_ticks": int(self.completion_tick[world] if self.completion_tick[world] >= 0 else DUEL_LIMIT_TICKS),
                    "elapsed_seconds": float((self.completion_tick[world] if self.completion_tick[world] >= 0 else DUEL_LIMIT_TICKS) / PHYSICS_HZ),
                    "rival_inherited_original_physical_car": "Blue" if inherited == 0 else "Orange",
                    "rival_closest_to_ball": bool(distance[side] <= distance[1 - side]),
                    "ball_field_third": "blue" if self.initial_ball[world, 1] < -1706.7 else ("orange" if self.initial_ball[world, 1] > 1706.7 else "middle"),
                    "ball_height_band": "ground" if self.initial_ball[world, 2] < 250 else ("mid" if self.initial_ball[world, 2] < 900 else "high"),
                    "rival_boost_advantage": float(self.initial_boost[world, side] - self.initial_boost[world, 1 - side]),
                    "touch_rival": int(self.event.touch_count[world, side]),
                    "touch_nexto": int(self.event.touch_count[world, 1 - side]),
                    "demo_rival": int(self.event.demo_count[world, side]),
                    "demo_nexto": int(self.event.demo_count[world, 1 - side]),
                    "possession_total_rival": int(self.event.possession_total[world, side]),
                    "possession_same_rival": int(self.event.possession_same[world, side]),
                    "possession_opponent_rival": int(self.event.possession_opponent[world, side]),
                    "goal_entry_valid": bool(self.event.goal_entry_valid[world]),
                    "goal_entry_x": float(self.event.goal_entry_x[world]),
                    "goal_entry_z": float(self.event.goal_entry_z[world]),
                    "rival_mechanics": mechanics[world],
                    "behavior": behavior[world],
                }
            )
        return rows


__all__ = [
    "DUEL_LIMIT_TICKS", "GOAL_HALF_WIDTH", "GOAL_HEIGHT", "GOAL_PLANE_Y",
    "KICKOFF_SEEDS", "OpenPlayDuelRuntime", "REGULATION_TICKS",
    "RocketSimEventTelemetry", "RocketSimPolicyRuntime", "RuntimeTiming",
    "SelfPlayHarvestRuntime", "capture_public_state", "concatenate_banks",
    "initialize_rocketsim", "mirror_bank", "restore_public_state",
]
