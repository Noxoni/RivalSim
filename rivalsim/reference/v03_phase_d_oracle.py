"""Pinned RocketSim authority adapter for v0.3 Phase D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    _import_rocketsim,
    _matrix,
    _previous_controls_at,
    _to_controls_at,
    _to_rot_mat,
    _to_vec,
    _vec,
)
from rivalsim.reference.v03_phase_c_oracle import (
    MAX_CAR_BUMP_EVENTS_PER_TICK,
    PHASE_C_NATIVE_BRANCHES,
)
from rivalsim.state import StateSnapshot

PHASE_D_NATIVE_BRANCHES = PHASE_C_NATIVE_BRANCHES
_PHASE_D_INIT_ROOT: str | None = None


@dataclass(slots=True)
class IntegratedBatchOracleFrame:
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_matrix: np.ndarray
    car_ang_vel: np.ndarray
    car_boost: np.ndarray
    car_handbrake: np.ndarray
    car_on_ground: np.ndarray
    car_wheel_contacts: np.ndarray
    car_world_contact: np.ndarray
    car_world_contact_normal: np.ndarray
    car_is_supersonic: np.ndarray
    car_supersonic_time: np.ndarray
    car_contact_id: np.ndarray
    car_contact_cooldown: np.ndarray
    car_is_demoed: np.ndarray
    car_demo_respawn_timer: np.ndarray
    car_ball_hit_valid: np.ndarray
    car_ball_hit_tick: np.ndarray
    car_ball_extra_hit_vel: np.ndarray
    car_ball_relative_pos: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_matrix: np.ndarray
    ball_ang_vel: np.ndarray
    ball_last_hit_car_id: np.ndarray
    bump_event_count: np.ndarray
    bump_event_bumper: np.ndarray
    bump_event_victim: np.ndarray
    bump_event_is_demo: np.ndarray


def _record_bump(
    *, arena: Any, bumper: Any, victim: Any, is_demo: bool, data: Any
) -> None:
    del arena
    events, car_ids = data
    events.append((car_ids[int(bumper.id)], car_ids[int(victim.id)], bool(is_demo)))


class RocketSimIntegratedBatchOracle:
    """One fresh two-Octane/one-ball Soccar arena per Phase D case."""

    def __init__(
        self,
        state: StateSnapshot,
        collision_root: str,
        *,
        pre_tick_visit_order: str | np.ndarray | None = None,
        max_construction_attempts: int = 1536,
    ):
        global _PHASE_D_INIT_ROOT

        if state.num_envs <= 0:
            raise ValueError("the integrated batch oracle requires at least one world")
        self.rs = _import_rocketsim()
        if _PHASE_D_INIT_ROOT is None:
            self.rs.init(collision_root)
            _PHASE_D_INIT_ROOT = collision_root
        elif collision_root != _PHASE_D_INIT_ROOT:
            raise RuntimeError(
                "RocketSim is process-global and was initialized with a different collision root"
            )
        if not hasattr(self.rs.Arena, "_get_pre_tick_visit_order"):
            raise RuntimeError(
                "Phase D relational authority requires the read-only logical-order diagnostic build"
            )

        requested_orders = _normalize_authority_orders(pre_tick_visit_order, state.num_envs)
        self.arenas: list[Any] = []
        self.cars: list[tuple[Any, Any]] = []
        self._events: list[list[tuple[int, int, bool]]] = []
        self.pre_tick_first_car = np.empty(state.num_envs, dtype=np.int32)
        self.construction_attempts = np.empty(state.num_envs, dtype=np.int32)

        for env_index in range(state.num_envs):
            requested = requested_orders[env_index]
            for _attempt in range(1, max_construction_attempts + 1):
                arena, car_a, car_b = self._new_arena()
                ids = tuple(int(value) for value in arena._get_pre_tick_visit_order())
                logical_ids = (int(car_a.id), int(car_b.id))
                if ids == logical_ids:
                    observed = 0
                elif ids == logical_ids[::-1]:
                    observed = 1
                else:
                    raise RuntimeError("native logical-order diagnostic returned an unknown car")
                if requested < 0 or observed == requested:
                    break
            else:
                raise RuntimeError(
                    "native source did not produce the requested Phase D visitation branch"
                )
            self.pre_tick_first_car[env_index] = observed
            self.construction_attempts[env_index] = _attempt
            self._set_car_state(car_a, state, env_index, 0)
            self._set_car_state(car_b, state, env_index, 1)
            ball_state = self.rs.BallState()
            ball_state.pos = _to_vec(self.rs, state.ball_pos[env_index])
            ball_state.vel = _to_vec(self.rs, state.ball_vel[env_index])
            ball_state.ang_vel = _to_vec(self.rs, state.ball_ang_vel[env_index])
            ball_state.rot_mat = _to_rot_mat(self.rs, state.ball_quat[env_index])
            arena.ball.set_state(ball_state)
            events: list[tuple[int, int, bool]] = []
            car_ids = {int(car_a.id): 0, int(car_b.id): 1}
            arena.set_car_bump_callback(_record_bump, (events, car_ids))
            self.arenas.append(arena)
            self.cars.append((car_a, car_b))
            self._events.append(events)
        self.arena = self.arenas[0]

    @property
    def num_envs(self) -> int:
        return len(self.arenas)

    def set_controls(self, controls: ControlBatch) -> None:
        if controls.num_envs != self.num_envs:
            raise ValueError("control batch size differs from the authority batch")
        for env_index, cars in enumerate(self.cars):
            for car_index, car in enumerate(cars):
                car.set_controls(_to_controls_at(self.rs, controls, env_index, car_index))

    def step(self) -> None:
        for events in self._events:
            events.clear()
        for arena in self.arenas:
            arena.step(1)

    def frame(self) -> IntegratedBatchOracleFrame:
        count = self.num_envs
        car_pos = np.empty((count, 2, 3), dtype=np.float32)
        car_vel = np.empty((count, 2, 3), dtype=np.float32)
        car_matrix = np.empty((count, 2, 3, 3), dtype=np.float32)
        car_ang_vel = np.empty((count, 2, 3), dtype=np.float32)
        car_boost = np.empty((count, 2), dtype=np.float32)
        car_handbrake = np.empty((count, 2), dtype=np.float32)
        car_on_ground = np.empty((count, 2), dtype=np.bool_)
        car_wheel_contacts = np.empty((count, 2, 4), dtype=np.bool_)
        car_world_contact = np.empty((count, 2), dtype=np.bool_)
        car_world_contact_normal = np.empty((count, 2, 3), dtype=np.float32)
        car_is_supersonic = np.empty((count, 2), dtype=np.bool_)
        car_supersonic_time = np.empty((count, 2), dtype=np.float32)
        car_contact_id = np.empty((count, 2), dtype=np.uint32)
        car_contact_cooldown = np.empty((count, 2), dtype=np.float32)
        car_is_demoed = np.empty((count, 2), dtype=np.bool_)
        car_demo_respawn_timer = np.empty((count, 2), dtype=np.float32)
        car_ball_hit_valid = np.empty((count, 2), dtype=np.bool_)
        car_ball_hit_tick = np.empty((count, 2), dtype=np.uint64)
        car_ball_extra_hit_vel = np.empty((count, 2, 3), dtype=np.float32)
        car_ball_relative_pos = np.empty((count, 2, 3), dtype=np.float32)
        ball_pos = np.empty((count, 3), dtype=np.float32)
        ball_vel = np.empty((count, 3), dtype=np.float32)
        ball_matrix = np.empty((count, 3, 3), dtype=np.float32)
        ball_ang_vel = np.empty((count, 3), dtype=np.float32)
        ball_last_hit_car_id = np.empty(count, dtype=np.uint32)
        bump_event_count = np.zeros(count, dtype=np.int32)
        bump_event_bumper = np.full(
            (count, MAX_CAR_BUMP_EVENTS_PER_TICK), -1, dtype=np.int32
        )
        bump_event_victim = np.full(
            (count, MAX_CAR_BUMP_EVENTS_PER_TICK), -1, dtype=np.int32
        )
        bump_event_is_demo = np.zeros(
            (count, MAX_CAR_BUMP_EVENTS_PER_TICK), dtype=np.bool_
        )

        for env_index, (arena, cars) in enumerate(zip(self.arenas, self.cars, strict=True)):
            id_map = {int(cars[0].id): 0, int(cars[1].id): 1}
            for car_index, car in enumerate(cars):
                source = car.get_state()
                hit = source.ball_hit_info
                car_pos[env_index, car_index] = _vec(source.pos)
                car_vel[env_index, car_index] = _vec(source.vel)
                car_matrix[env_index, car_index] = _matrix(source.rot_mat)
                car_ang_vel[env_index, car_index] = _vec(source.ang_vel)
                car_boost[env_index, car_index] = float(source.boost)
                car_handbrake[env_index, car_index] = float(source.handbrake_val)
                car_on_ground[env_index, car_index] = bool(source.is_on_ground)
                car_wheel_contacts[env_index, car_index] = tuple(
                    bool(value) for value in source.wheels_with_contact
                )
                car_world_contact[env_index, car_index] = bool(source.has_world_contact)
                car_world_contact_normal[env_index, car_index] = _vec(
                    source.world_contact_normal
                )
                car_is_supersonic[env_index, car_index] = bool(source.is_supersonic)
                car_supersonic_time[env_index, car_index] = float(source.supersonic_time)
                contact_id = int(source.car_contact_id)
                car_contact_id[env_index, car_index] = np.uint32(
                    id_map.get(contact_id, contact_id)
                )
                car_contact_cooldown[env_index, car_index] = float(
                    source.car_contact_cooldown_timer
                )
                car_is_demoed[env_index, car_index] = bool(source.is_demoed)
                car_demo_respawn_timer[env_index, car_index] = float(
                    source.demo_respawn_timer
                )
                car_ball_hit_valid[env_index, car_index] = bool(hit.is_valid)
                car_ball_hit_tick[env_index, car_index] = int(hit.tick_count_when_hit)
                car_ball_extra_hit_vel[env_index, car_index] = _vec(hit.extra_hit_vel)
                car_ball_relative_pos[env_index, car_index] = _vec(
                    hit.relative_pos_on_ball
                )
            ball = arena.ball.get_state()
            ball_pos[env_index] = _vec(ball.pos)
            ball_vel[env_index] = _vec(ball.vel)
            ball_matrix[env_index] = _matrix(ball.rot_mat)
            ball_ang_vel[env_index] = _vec(ball.ang_vel)
            last_hit = int(ball.last_hit_car_id)
            ball_last_hit_car_id[env_index] = np.uint32(id_map.get(last_hit, last_hit))
            events = self._events[env_index]
            if len(events) > MAX_CAR_BUMP_EVENTS_PER_TICK:
                raise RuntimeError("Phase D bump event capacity exceeded")
            bump_event_count[env_index] = len(events)
            for event_index, (bumper, victim, is_demo) in enumerate(events):
                bump_event_bumper[env_index, event_index] = bumper
                bump_event_victim[env_index, event_index] = victim
                bump_event_is_demo[env_index, event_index] = is_demo

        return IntegratedBatchOracleFrame(
            car_pos=car_pos,
            car_vel=car_vel,
            car_matrix=car_matrix,
            car_ang_vel=car_ang_vel,
            car_boost=car_boost,
            car_handbrake=car_handbrake,
            car_on_ground=car_on_ground,
            car_wheel_contacts=car_wheel_contacts,
            car_world_contact=car_world_contact,
            car_world_contact_normal=car_world_contact_normal,
            car_is_supersonic=car_is_supersonic,
            car_supersonic_time=car_supersonic_time,
            car_contact_id=car_contact_id,
            car_contact_cooldown=car_contact_cooldown,
            car_is_demoed=car_is_demoed,
            car_demo_respawn_timer=car_demo_respawn_timer,
            car_ball_hit_valid=car_ball_hit_valid,
            car_ball_hit_tick=car_ball_hit_tick,
            car_ball_extra_hit_vel=car_ball_extra_hit_vel,
            car_ball_relative_pos=car_ball_relative_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            ball_matrix=ball_matrix,
            ball_ang_vel=ball_ang_vel,
            ball_last_hit_car_id=ball_last_hit_car_id,
            bump_event_count=bump_event_count,
            bump_event_bumper=bump_event_bumper,
            bump_event_victim=bump_event_victim,
            bump_event_is_demo=bump_event_is_demo,
        )

    def _new_arena(self) -> tuple[Any, Any, Any]:
        config = self.rs.ArenaConfig()
        config.no_ball_rot = False
        arena = self.rs.Arena(self.rs.GameMode.SOCCAR, tick_rate=120.0, config=config)
        arena.set_car_car_collision(True)
        arena.set_car_ball_collision(True)
        mutator = arena.get_mutator_config()
        mutator.demo_mode = self.rs.DemoMode.NORMAL
        mutator.enable_team_demos = False
        arena.set_mutator_config(mutator)
        car_a = arena.add_car(self.rs.Team.BLUE, self.rs.CarConfig.OCTANE)
        car_b = arena.add_car(self.rs.Team.ORANGE, self.rs.CarConfig.OCTANE)
        return arena, car_a, car_b

    def _set_car_state(
        self, car: Any, snapshot: StateSnapshot, env_index: int, car_index: int
    ) -> None:
        source = self.rs.CarState()
        source.pos = _to_vec(self.rs, snapshot.car_pos[env_index, car_index])
        source.vel = _to_vec(self.rs, snapshot.car_vel[env_index, car_index])
        source.ang_vel = _to_vec(self.rs, snapshot.car_ang_vel[env_index, car_index])
        source.rot_mat = _to_rot_mat(self.rs, snapshot.car_quat[env_index, car_index])
        source.boost = float(snapshot.boost[env_index, car_index])
        source.is_on_ground = bool(snapshot.on_ground[env_index, car_index])
        source.has_jumped = bool(snapshot.has_jumped[env_index, car_index])
        source.is_jumping = bool(snapshot.is_jumping[env_index, car_index])
        source.has_double_jumped = bool(snapshot.has_double_jumped[env_index, car_index])
        source.has_flipped = bool(snapshot.has_flipped[env_index, car_index])
        source.is_flipping = bool(snapshot.is_flipping[env_index, car_index])
        source.jump_time = float(snapshot.jump_time[env_index, car_index])
        source.air_time = float(snapshot.air_time[env_index, car_index])
        source.air_time_since_jump = float(
            snapshot.air_time_since_jump[env_index, car_index]
        )
        source.flip_time = float(snapshot.flip_time[env_index, car_index])
        source.flip_rel_torque = _to_vec(
            self.rs, snapshot.flip_rel_torque[env_index, car_index]
        )
        source.is_supersonic = bool(snapshot.is_supersonic[env_index, car_index])
        source.supersonic_time = float(snapshot.supersonic_time[env_index, car_index])
        source.boosting_time = float(snapshot.boosting_time[env_index, car_index])
        source.handbrake_val = 0.0
        source.last_controls = _previous_controls_at(
            self.rs, snapshot, env_index, car_index
        )
        car.set_state(source)


def _normalize_authority_orders(
    value: str | np.ndarray | None, num_envs: int
) -> np.ndarray:
    if value is None:
        return np.full(num_envs, -1, dtype=np.int32)
    if isinstance(value, str):
        try:
            return np.full(
                num_envs, PHASE_D_NATIVE_BRANCHES.index(value), dtype=np.int32
            )
        except ValueError as exc:
            raise ValueError(f"unknown Phase D native branch: {value}") from exc
    result = np.broadcast_to(np.asarray(value, dtype=np.int32), (num_envs,)).copy()
    if np.any((result < 0) | (result >= len(PHASE_D_NATIVE_BRANCHES))):
        raise ValueError("Phase D native branch entries must be 0 or 1")
    return result


__all__ = [
    "PHASE_D_NATIVE_BRANCHES",
    "ROCKETSIM_BINDING_COMMIT",
    "ROCKETSIM_BINDING_VERSION",
    "ROCKETSIM_PRIMARY_COMMIT",
    "IntegratedBatchOracleFrame",
    "RocketSimIntegratedBatchOracle",
]
