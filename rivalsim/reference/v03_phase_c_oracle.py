"""Isolated pinned RocketSim authority for v0.3 Phase C car/car physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    _import_rocketsim,
    _matrix,
    _previous_controls_at,
    _to_rot_mat,
    _to_vec,
    _vec,
)
from rivalsim.state import StateSnapshot

MAX_CAR_BUMP_EVENTS_PER_TICK = 4
PHASE_C_NATIVE_BRANCHES = ("a_then_b", "b_then_a")
_PHASE_C_INIT_ROOT: str | None = None


@dataclass(slots=True)
class CarCarBatchOracleFrame:
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


class RocketSimCarCarBatchOracle:
    """One fresh two-Octane Soccar arena for every frozen Phase C case."""

    def __init__(
        self,
        state: StateSnapshot,
        collision_root: str,
        *,
        pre_tick_visit_order: str | np.ndarray | None = None,
        max_construction_attempts: int = 1536,
    ):
        global _PHASE_C_INIT_ROOT

        if state.num_envs <= 0:
            raise ValueError("the car/car batch oracle requires at least one world")
        self.rs = _import_rocketsim()
        if _PHASE_C_INIT_ROOT is None:
            self.rs.init(collision_root)
            _PHASE_C_INIT_ROOT = collision_root
        elif collision_root != _PHASE_C_INIT_ROOT:
            raise RuntimeError(
                "RocketSim is process-global and was initialized with a different collision root"
            )
        self.arenas: list[Any] = []
        self.cars: list[tuple[Any, Any]] = []
        self._events: list[list[tuple[int, int, bool]]] = []
        self._initial_car_pos = np.ascontiguousarray(state.car_pos, dtype=np.float32)
        self._initial_car_vel = np.ascontiguousarray(state.car_vel, dtype=np.float32)
        self._initial_car_quat = np.ascontiguousarray(state.car_quat, dtype=np.float32)
        self._initial_car_ang_vel = np.ascontiguousarray(
            state.car_ang_vel, dtype=np.float32
        )
        if not hasattr(self.rs.Arena, "_get_pre_tick_visit_order"):
            raise RuntimeError(
                "Phase C relational authority requires the read-only logical-order "
                "diagnostic build"
            )
        requested_orders = _normalize_authority_orders(
            pre_tick_visit_order, state.num_envs
        )
        self.pre_tick_first_car = np.empty(state.num_envs, dtype=np.int32)
        self.construction_attempts = np.empty(state.num_envs, dtype=np.int32)

        for env_index in range(state.num_envs):
            requested = requested_orders[env_index]
            for _attempt in range(1, max_construction_attempts + 1):
                arena, car_a, car_b = self._new_pair_arena()
                ids = tuple(int(value) for value in arena._get_pre_tick_visit_order())
                logical_ids = (int(car_a.id), int(car_b.id))
                if ids == logical_ids:
                    observed = 0
                elif ids == logical_ids[::-1]:
                    observed = 1
                else:
                    raise RuntimeError(
                        "native logical-order diagnostic did not return the fixed pair"
                    )
                if requested < 0 or observed == requested:
                    break
                # The next assignment releases this complete source arena and
                # begins a new ordinary construction lifecycle. No pointer or
                # bucket value is inspected, retained, or used to synthesize
                # an order.
            else:
                raise RuntimeError(
                    "native source did not produce requested Phase C visitation "
                    f"branch after {max_construction_attempts} fresh arenas"
                )
            self.pre_tick_first_car[env_index] = observed
            self.construction_attempts[env_index] = _attempt
            self._set_car_state(car_a, state, env_index, 0)
            self._set_car_state(car_b, state, env_index, 1)
            car_a.set_controls(self.rs.CarControls())
            car_b.set_controls(self.rs.CarControls())
            events: list[tuple[int, int, bool]] = []
            car_ids = {int(car_a.id): 0, int(car_b.id): 1}
            arena.set_car_bump_callback(_record_bump, (events, car_ids))
            ball = self.rs.BallState()
            ball.pos = self.rs.Vec(-3000.0, -4000.0, 1500.0)
            arena.ball.set_state(ball)
            self.arenas.append(arena)
            self.cars.append((car_a, car_b))
            self._events.append(events)
        self.arena = self.arenas[0]

    def _new_pair_arena(self) -> tuple[Any, Any, Any]:
        config = self.rs.ArenaConfig()
        config.no_ball_rot = False
        arena = self.rs.Arena(
            self.rs.GameMode.SOCCAR,
            tick_rate=120.0,
            config=config,
        )
        arena.set_car_car_collision(True)
        arena.set_car_ball_collision(False)
        mutator = arena.get_mutator_config()
        mutator.demo_mode = self.rs.DemoMode.NORMAL
        mutator.enable_team_demos = False
        arena.set_mutator_config(mutator)
        car_a = arena.add_car(self.rs.Team.BLUE, self.rs.CarConfig.OCTANE)
        car_b = arena.add_car(self.rs.Team.ORANGE, self.rs.CarConfig.OCTANE)
        return arena, car_a, car_b

    @property
    def num_envs(self) -> int:
        return len(self.arenas)

    def step(self) -> None:
        for events in self._events:
            events.clear()
        for arena in self.arenas:
            arena.step(1)

    def frame(self) -> CarCarBatchOracleFrame:
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

        for env_index, cars in enumerate(self.cars):
            id_map = {int(cars[0].id): 0, int(cars[1].id): 1}
            for car_index, car in enumerate(cars):
                source = car.get_state()
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
                car_world_contact[env_index, car_index] = bool(
                    source.has_world_contact
                )
                car_world_contact_normal[env_index, car_index] = _vec(
                    source.world_contact_normal
                )
                car_is_supersonic[env_index, car_index] = bool(
                    source.is_supersonic
                )
                car_supersonic_time[env_index, car_index] = float(
                    source.supersonic_time
                )
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
            events = self._events[env_index]
            if len(events) > MAX_CAR_BUMP_EVENTS_PER_TICK:
                raise RuntimeError("Phase C bump event capacity exceeded")
            bump_event_count[env_index] = len(events)
            for event_index, (bumper, victim, is_demo) in enumerate(events):
                bump_event_bumper[env_index, event_index] = bumper
                bump_event_victim[env_index, event_index] = victim
                bump_event_is_demo[env_index, event_index] = is_demo

        return CarCarBatchOracleFrame(
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
            bump_event_count=bump_event_count,
            bump_event_bumper=bump_event_bumper,
            bump_event_victim=bump_event_victim,
            bump_event_is_demo=bump_event_is_demo,
        )

    def authoritative_snapshot(self) -> StateSnapshot:
        result = StateSnapshot.empty(self.num_envs)
        result.car_pos[:] = self._initial_car_pos
        result.car_vel[:] = self._initial_car_vel
        result.car_quat[:] = self._initial_car_quat
        result.car_ang_vel[:] = self._initial_car_ang_vel
        result.ball_pos[:] = (-3000.0, -4000.0, 1500.0)
        for env_index, cars in enumerate(self.cars):
            for car_index, car in enumerate(cars):
                source = car.get_state()
                result.boost[env_index, car_index] = float(source.boost)
                result.on_ground[env_index, car_index] = int(source.is_on_ground)
                result.is_supersonic[env_index, car_index] = int(
                    source.is_supersonic
                )
                result.supersonic_time[env_index, car_index] = float(
                    source.supersonic_time
                )
        result.validate()
        return result

    def _set_car_state(
        self, car: Any, snapshot: StateSnapshot, env_index: int, car_index: int
    ) -> None:
        source = self.rs.CarState()
        source.pos = _to_vec(self.rs, snapshot.car_pos[env_index, car_index])
        source.vel = _to_vec(self.rs, snapshot.car_vel[env_index, car_index])
        source.ang_vel = _to_vec(
            self.rs, snapshot.car_ang_vel[env_index, car_index]
        )
        source.rot_mat = _to_rot_mat(
            self.rs, snapshot.car_quat[env_index, car_index]
        )
        source.boost = float(snapshot.boost[env_index, car_index])
        source.is_on_ground = bool(snapshot.on_ground[env_index, car_index])
        source.has_jumped = bool(snapshot.has_jumped[env_index, car_index])
        source.is_jumping = bool(snapshot.is_jumping[env_index, car_index])
        source.has_double_jumped = bool(
            snapshot.has_double_jumped[env_index, car_index]
        )
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
        source.is_supersonic = bool(
            snapshot.is_supersonic[env_index, car_index]
        )
        source.supersonic_time = float(
            snapshot.supersonic_time[env_index, car_index]
        )
        source.boosting_time = float(
            snapshot.boosting_time[env_index, car_index]
        )
        source.handbrake_val = 0.0
        source.last_controls = _previous_controls_at(
            self.rs, snapshot, env_index, car_index
        )
        car.set_state(source)


__all__ = [
    "MAX_CAR_BUMP_EVENTS_PER_TICK",
    "PHASE_C_NATIVE_BRANCHES",
    "ROCKETSIM_BINDING_COMMIT",
    "ROCKETSIM_BINDING_VERSION",
    "ROCKETSIM_PRIMARY_COMMIT",
    "CarCarBatchOracleFrame",
    "RocketSimCarCarBatchOracle",
]


def _normalize_authority_orders(
    value: str | np.ndarray | None, num_envs: int
) -> np.ndarray:
    if value is None:
        return np.full(num_envs, -1, dtype=np.int32)
    if isinstance(value, str):
        try:
            return np.full(
                num_envs, PHASE_C_NATIVE_BRANCHES.index(value), dtype=np.int32
            )
        except ValueError as exc:
            raise ValueError(f"unknown Phase C native branch: {value}") from exc
    result = np.broadcast_to(np.asarray(value, dtype=np.int32), (num_envs,)).copy()
    if np.any((result < 0) | (result >= len(PHASE_C_NATIVE_BRANCHES))):
        raise ValueError("Phase C native branch entries must be 0 or 1")
    return result
