"""Pinned RocketSim Python-binding adapter for contact-free parity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.math import quat_to_matrix
from rivalsim.scenarios import Scenario
from rivalsim.state import StateSnapshot

ROCKETSIM_PRIMARY_COMMIT = "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02"
ROCKETSIM_BINDING_COMMIT = "2da51b1dac7b8127127613a5ff30e490bdd70dd8"
ROCKETSIM_BINDING_VERSION = "2.2.1"
_STATIC_WORLD_INIT_ROOT: str | None = None


@dataclass(slots=True)
class OracleFrame:
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_matrix: np.ndarray
    car_ang_vel: np.ndarray
    boost: float
    has_jumped: bool
    is_jumping: bool
    has_double_jumped: bool
    has_flipped: bool
    is_flipping: bool
    jump_time: float
    air_time: float
    air_time_since_jump: float
    flip_time: float
    flip_rel_torque: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_matrix: np.ndarray
    ball_ang_vel: np.ndarray


@dataclass(slots=True)
class StaticWorldOracleFrame:
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_matrix: np.ndarray
    car_ang_vel: np.ndarray
    boost: float
    handbrake_value: float
    on_ground: bool
    wheel_contacts: tuple[bool, bool, bool, bool]
    has_world_contact: bool
    world_contact_normal: np.ndarray


@dataclass(slots=True)
class StaticWorldBatchOracleFrame:
    """Vectorized readback for independent non-colliding cars in one arena."""

    car_pos: np.ndarray
    car_vel: np.ndarray
    car_matrix: np.ndarray
    car_ang_vel: np.ndarray
    boost: np.ndarray
    handbrake_value: np.ndarray
    on_ground: np.ndarray
    wheel_contacts: np.ndarray
    has_world_contact: np.ndarray
    world_contact_normal: np.ndarray


class RocketSimOracle:
    """One isolated RocketSim THE_VOID arena for a deterministic scenario."""

    def __init__(self, state: StateSnapshot):
        if state.num_envs != 1:
            raise ValueError("the oracle adapter accepts one world")
        self.rs = _import_rocketsim()
        config = self.rs.ArenaConfig()
        config.no_ball_rot = False
        self.arena = self.rs.Arena(
            self.rs.GameMode.THE_VOID,
            tick_rate=120.0,
            config=config,
        )
        self.arena.set_car_car_collision(False)
        self.arena.set_car_ball_collision(False)
        self.cars = [
            self.arena.add_car(self.rs.Team.BLUE),
            self.arena.add_car(self.rs.Team.ORANGE),
        ]
        self._set_state(state)

    @classmethod
    def for_scenario(cls, scenario: Scenario) -> RocketSimOracle:
        return cls(scenario.initial)

    def set_controls(self, controls: ControlBatch) -> None:
        if controls.num_envs != 1:
            raise ValueError("the oracle adapter accepts one world")
        for car_index, car in enumerate(self.cars):
            car.set_controls(_to_controls(self.rs, controls, car_index))

    def step(self) -> None:
        self.arena.step(1)

    def frame(self) -> OracleFrame:
        car = self.cars[0].get_state()
        ball = self.arena.ball.get_state()
        return OracleFrame(
            car_pos=_vec(car.pos),
            car_vel=_vec(car.vel),
            car_matrix=_matrix(car.rot_mat),
            car_ang_vel=_vec(car.ang_vel),
            boost=float(car.boost),
            has_jumped=bool(car.has_jumped),
            is_jumping=bool(car.is_jumping),
            has_double_jumped=bool(car.has_double_jumped),
            has_flipped=bool(car.has_flipped),
            is_flipping=bool(car.is_flipping),
            jump_time=float(car.jump_time),
            air_time=float(car.air_time),
            air_time_since_jump=float(car.air_time_since_jump),
            flip_time=float(car.flip_time),
            flip_rel_torque=_vec(car.flip_rel_torque),
            ball_pos=_vec(ball.pos),
            ball_vel=_vec(ball.vel),
            ball_matrix=_matrix(ball.rot_mat),
            ball_ang_vel=_vec(ball.ang_vel),
        )

    def _set_state(self, snapshot: StateSnapshot) -> None:
        for index, car in enumerate(self.cars):
            source = self.rs.CarState()
            source.pos = _to_vec(self.rs, snapshot.car_pos[0, index])
            source.vel = _to_vec(self.rs, snapshot.car_vel[0, index])
            source.ang_vel = _to_vec(self.rs, snapshot.car_ang_vel[0, index])
            source.rot_mat = _to_rot_mat(self.rs, snapshot.car_quat[0, index])
            source.boost = float(snapshot.boost[0, index])
            source.is_on_ground = bool(snapshot.on_ground[0, index])
            source.has_jumped = bool(snapshot.has_jumped[0, index])
            source.is_jumping = bool(snapshot.is_jumping[0, index])
            source.has_double_jumped = bool(snapshot.has_double_jumped[0, index])
            source.has_flipped = bool(snapshot.has_flipped[0, index])
            source.is_flipping = bool(snapshot.is_flipping[0, index])
            source.jump_time = float(snapshot.jump_time[0, index])
            source.air_time = float(snapshot.air_time[0, index])
            source.air_time_since_jump = float(snapshot.air_time_since_jump[0, index])
            source.flip_time = float(snapshot.flip_time[0, index])
            source.flip_rel_torque = _to_vec(self.rs, snapshot.flip_rel_torque[0, index])
            source.is_supersonic = bool(snapshot.is_supersonic[0, index])
            source.supersonic_time = float(snapshot.supersonic_time[0, index])
            source.boosting_time = float(snapshot.boosting_time[0, index])
            previous = ControlBatch(
                snapshot.prev_throttle[:, index : index + 1],
                snapshot.prev_steer[:, index : index + 1],
                snapshot.prev_pitch[:, index : index + 1],
                snapshot.prev_yaw[:, index : index + 1],
                snapshot.prev_roll[:, index : index + 1],
                snapshot.prev_jump[:, index : index + 1],
                snapshot.prev_boost[:, index : index + 1],
                snapshot.prev_handbrake[:, index : index + 1],
            )
            source.last_controls = _to_controls(self.rs, previous, 0)
            car.set_state(source)

        ball = self.rs.BallState()
        ball.pos = _to_vec(self.rs, snapshot.ball_pos[0])
        ball.vel = _to_vec(self.rs, snapshot.ball_vel[0])
        ball.ang_vel = _to_vec(self.rs, snapshot.ball_ang_vel[0])
        ball.rot_mat = _to_rot_mat(self.rs, snapshot.ball_quat[0])
        self.arena.ball.set_state(ball)


class RocketSimStaticWorldOracle:
    """Pinned RocketSim Soccar oracle using the same external CMF directory."""

    def __init__(self, state: StateSnapshot, collision_root: str):
        global _STATIC_WORLD_INIT_ROOT

        if state.num_envs != 1:
            raise ValueError("the oracle adapter accepts one world")
        self.rs = _import_rocketsim()
        if _STATIC_WORLD_INIT_ROOT is None:
            self.rs.init(collision_root)
            _STATIC_WORLD_INIT_ROOT = collision_root
        elif collision_root != _STATIC_WORLD_INIT_ROOT:
            raise RuntimeError(
                "RocketSim is process-global and was initialized with a different collision root"
            )
        config = self.rs.ArenaConfig()
        config.no_ball_rot = False
        self.arena = self.rs.Arena(
            self.rs.GameMode.SOCCAR,
            tick_rate=120.0,
            config=config,
        )
        self.arena.set_car_car_collision(False)
        self.arena.set_car_ball_collision(False)
        self.cars = [
            self.arena.add_car(self.rs.Team.BLUE, self.rs.CarConfig.OCTANE),
            self.arena.add_car(self.rs.Team.ORANGE, self.rs.CarConfig.OCTANE),
        ]
        self._set_state(state)

    def set_controls(self, controls: ControlBatch) -> None:
        if controls.num_envs != 1:
            raise ValueError("the oracle adapter accepts one world")
        for car_index, car in enumerate(self.cars):
            car.set_controls(_to_controls(self.rs, controls, car_index))

    def step(self) -> None:
        self.arena.step(1)

    def frame(self) -> StaticWorldOracleFrame:
        state = self.cars[0].get_state()
        contacts = tuple(bool(value) for value in state.wheels_with_contact)
        return StaticWorldOracleFrame(
            car_pos=_vec(state.pos),
            car_vel=_vec(state.vel),
            car_matrix=_matrix(state.rot_mat),
            car_ang_vel=_vec(state.ang_vel),
            boost=float(state.boost),
            handbrake_value=float(state.handbrake_val),
            on_ground=bool(state.is_on_ground),
            wheel_contacts=(contacts[0], contacts[1], contacts[2], contacts[3]),
            has_world_contact=bool(state.has_world_contact),
            world_contact_normal=_vec(state.world_contact_normal),
        )

    def _set_state(self, snapshot: StateSnapshot) -> None:
        for index, car in enumerate(self.cars):
            source = self.rs.CarState()
            source.pos = _to_vec(self.rs, snapshot.car_pos[0, index])
            source.vel = _to_vec(self.rs, snapshot.car_vel[0, index])
            source.ang_vel = _to_vec(self.rs, snapshot.car_ang_vel[0, index])
            source.rot_mat = _to_rot_mat(self.rs, snapshot.car_quat[0, index])
            source.boost = float(snapshot.boost[0, index])
            source.is_on_ground = bool(snapshot.on_ground[0, index])
            source.has_jumped = bool(snapshot.has_jumped[0, index])
            source.is_jumping = bool(snapshot.is_jumping[0, index])
            source.has_double_jumped = bool(snapshot.has_double_jumped[0, index])
            source.has_flipped = bool(snapshot.has_flipped[0, index])
            source.is_flipping = bool(snapshot.is_flipping[0, index])
            source.jump_time = float(snapshot.jump_time[0, index])
            source.air_time = float(snapshot.air_time[0, index])
            source.air_time_since_jump = float(snapshot.air_time_since_jump[0, index])
            source.flip_time = float(snapshot.flip_time[0, index])
            source.flip_rel_torque = _to_vec(self.rs, snapshot.flip_rel_torque[0, index])
            source.is_supersonic = bool(snapshot.is_supersonic[0, index])
            source.supersonic_time = float(snapshot.supersonic_time[0, index])
            source.boosting_time = float(snapshot.boosting_time[0, index])
            source.handbrake_val = 0.0
            previous = ControlBatch(
                snapshot.prev_throttle[:, index : index + 1],
                snapshot.prev_steer[:, index : index + 1],
                snapshot.prev_pitch[:, index : index + 1],
                snapshot.prev_yaw[:, index : index + 1],
                snapshot.prev_roll[:, index : index + 1],
                snapshot.prev_jump[:, index : index + 1],
                snapshot.prev_boost[:, index : index + 1],
                snapshot.prev_handbrake[:, index : index + 1],
            )
            source.last_controls = _to_controls(self.rs, previous, 0)
            car.set_state(source)

        ball = self.rs.BallState()
        ball.pos = _to_vec(self.rs, snapshot.ball_pos[0])
        ball.vel = _to_vec(self.rs, snapshot.ball_vel[0])
        ball.ang_vel = _to_vec(self.rs, snapshot.ball_ang_vel[0])
        ball.rot_mat = _to_rot_mat(self.rs, snapshot.ball_quat[0])
        self.arena.ball.set_state(ball)


class RocketSimStaticWorldBatchOracle:
    """A batch of isolated one-car Soccar authority arenas.

    Bullet's static constraints are not batch-invariant when many validation
    cars share one arena, even with car/car and car/ball collision disabled.
    Each case therefore owns a fresh arena and is stepped independently, which
    implements the v0.2.2 reset-before-next-case contract literally.
    """

    def __init__(self, state: StateSnapshot, collision_root: str):
        global _STATIC_WORLD_INIT_ROOT

        if state.num_envs <= 0:
            raise ValueError("the batch oracle requires at least one world")
        self.rs = _import_rocketsim()
        if _STATIC_WORLD_INIT_ROOT is None:
            self.rs.init(collision_root)
            _STATIC_WORLD_INIT_ROOT = collision_root
        elif collision_root != _STATIC_WORLD_INIT_ROOT:
            raise RuntimeError(
                "RocketSim is process-global and was initialized with a different collision root"
            )
        self.arenas = []
        self.cars = []
        self._initial_car_pos = np.ascontiguousarray(state.car_pos[:, 0], dtype=np.float32)
        self._initial_car_vel = np.ascontiguousarray(state.car_vel[:, 0], dtype=np.float32)
        self._initial_car_quat = np.ascontiguousarray(state.car_quat[:, 0], dtype=np.float32)
        self._initial_car_ang_vel = np.ascontiguousarray(
            state.car_ang_vel[:, 0], dtype=np.float32
        )
        for env_index in range(state.num_envs):
            config = self.rs.ArenaConfig()
            config.no_ball_rot = False
            arena = self.rs.Arena(
                self.rs.GameMode.SOCCAR,
                tick_rate=120.0,
                config=config,
            )
            arena.set_car_car_collision(False)
            arena.set_car_ball_collision(False)
            car = arena.add_car(self.rs.Team.BLUE, self.rs.CarConfig.OCTANE)
            self._set_car_state(car, state, env_index)
            ball = self.rs.BallState()
            ball.pos = self.rs.Vec(0.0, 0.0, 1500.0)
            arena.ball.set_state(ball)
            self.arenas.append(arena)
            self.cars.append(car)
        # Preserve the single-arena attribute used by older diagnostics.
        self.arena = self.arenas[0]

    @property
    def num_envs(self) -> int:
        return len(self.cars)

    def set_controls(self, controls: ControlBatch) -> None:
        if controls.num_envs != self.num_envs:
            raise ValueError("control batch does not match the batch oracle")
        for env_index, car in enumerate(self.cars):
            car.set_controls(_to_controls_at(self.rs, controls, env_index, 0))

    def step(self) -> None:
        for arena in self.arenas:
            arena.step(1)

    def authoritative_snapshot(self) -> StateSnapshot:
        """Return the exact state read back after RocketSim initialization."""

        result = StateSnapshot.empty(self.num_envs)
        result.car_pos[:, 1] = (0.0, 0.0, 1000.0)
        result.ball_pos[:] = (0.0, 0.0, 1500.0)
        for env_index, car in enumerate(self.cars):
            source = car.get_state()
            # Preserve the exact source fields supplied to CarState. Immediate
            # readback converts Bullet-unit position/velocity back to UU and
            # exposes the transform basis only as RotMat; feeding those lossy
            # views back into the GPU would no longer reconstruct the rigid
            # state that RocketSim actually steps.
            result.car_pos[env_index, 0] = self._initial_car_pos[env_index]
            result.car_vel[env_index, 0] = self._initial_car_vel[env_index]
            result.car_quat[env_index, 0] = self._initial_car_quat[env_index]
            result.car_ang_vel[env_index, 0] = self._initial_car_ang_vel[env_index]
            result.boost[env_index, 0] = float(source.boost)
            result.on_ground[env_index, 0] = int(source.is_on_ground)
            result.has_jumped[env_index, 0] = int(source.has_jumped)
            result.is_jumping[env_index, 0] = int(source.is_jumping)
            result.has_double_jumped[env_index, 0] = int(source.has_double_jumped)
            result.has_flipped[env_index, 0] = int(source.has_flipped)
            result.is_flipping[env_index, 0] = int(source.is_flipping)
            result.jump_time[env_index, 0] = float(source.jump_time)
            result.air_time[env_index, 0] = float(source.air_time)
            result.air_time_since_jump[env_index, 0] = float(source.air_time_since_jump)
            result.flip_time[env_index, 0] = float(source.flip_time)
            result.flip_rel_torque[env_index, 0] = _vec(source.flip_rel_torque)
            result.is_supersonic[env_index, 0] = int(source.is_supersonic)
            result.supersonic_time[env_index, 0] = float(source.supersonic_time)
            result.boosting_time[env_index, 0] = float(source.boosting_time)
        result.validate()
        return result

    def frame(self) -> StaticWorldBatchOracleFrame:
        count = self.num_envs
        car_pos = np.empty((count, 3), dtype=np.float32)
        car_vel = np.empty((count, 3), dtype=np.float32)
        car_matrix = np.empty((count, 3, 3), dtype=np.float32)
        car_ang_vel = np.empty((count, 3), dtype=np.float32)
        boost = np.empty(count, dtype=np.float32)
        handbrake_value = np.empty(count, dtype=np.float32)
        on_ground = np.empty(count, dtype=np.bool_)
        wheel_contacts = np.empty((count, 4), dtype=np.bool_)
        has_world_contact = np.empty(count, dtype=np.bool_)
        world_contact_normal = np.empty((count, 3), dtype=np.float32)
        for env_index, car in enumerate(self.cars):
            source = car.get_state()
            car_pos[env_index] = _vec(source.pos)
            car_vel[env_index] = _vec(source.vel)
            car_matrix[env_index] = _matrix(source.rot_mat)
            car_ang_vel[env_index] = _vec(source.ang_vel)
            boost[env_index] = float(source.boost)
            handbrake_value[env_index] = float(source.handbrake_val)
            on_ground[env_index] = bool(source.is_on_ground)
            wheel_contacts[env_index] = tuple(bool(value) for value in source.wheels_with_contact)
            has_world_contact[env_index] = bool(source.has_world_contact)
            world_contact_normal[env_index] = _vec(source.world_contact_normal)
        return StaticWorldBatchOracleFrame(
            car_pos=car_pos,
            car_vel=car_vel,
            car_matrix=car_matrix,
            car_ang_vel=car_ang_vel,
            boost=boost,
            handbrake_value=handbrake_value,
            on_ground=on_ground,
            wheel_contacts=wheel_contacts,
            has_world_contact=has_world_contact,
            world_contact_normal=world_contact_normal,
        )

    def _set_car_state(self, car: Any, snapshot: StateSnapshot, env_index: int) -> None:
        source = self.rs.CarState()
        source.pos = _to_vec(self.rs, snapshot.car_pos[env_index, 0])
        source.vel = _to_vec(self.rs, snapshot.car_vel[env_index, 0])
        source.ang_vel = _to_vec(self.rs, snapshot.car_ang_vel[env_index, 0])
        source.rot_mat = _to_rot_mat(self.rs, snapshot.car_quat[env_index, 0])
        source.boost = float(snapshot.boost[env_index, 0])
        source.is_on_ground = bool(snapshot.on_ground[env_index, 0])
        source.has_jumped = bool(snapshot.has_jumped[env_index, 0])
        source.is_jumping = bool(snapshot.is_jumping[env_index, 0])
        source.has_double_jumped = bool(snapshot.has_double_jumped[env_index, 0])
        source.has_flipped = bool(snapshot.has_flipped[env_index, 0])
        source.is_flipping = bool(snapshot.is_flipping[env_index, 0])
        source.jump_time = float(snapshot.jump_time[env_index, 0])
        source.air_time = float(snapshot.air_time[env_index, 0])
        source.air_time_since_jump = float(snapshot.air_time_since_jump[env_index, 0])
        source.flip_time = float(snapshot.flip_time[env_index, 0])
        source.flip_rel_torque = _to_vec(self.rs, snapshot.flip_rel_torque[env_index, 0])
        source.is_supersonic = bool(snapshot.is_supersonic[env_index, 0])
        source.supersonic_time = float(snapshot.supersonic_time[env_index, 0])
        source.boosting_time = float(snapshot.boosting_time[env_index, 0])
        source.handbrake_val = 0.0
        source.last_controls = _previous_controls_at(self.rs, snapshot, env_index, 0)
        car.set_state(source)


def binding_metadata() -> dict[str, str]:
    from importlib.metadata import version

    return {
        "package": "rocketsim",
        "resolved_version": version("rocketsim"),
        "binding_commit": ROCKETSIM_BINDING_COMMIT,
        "primary_source_commit": ROCKETSIM_PRIMARY_COMMIT,
        "mode": "GameMode.THE_VOID",
    }


def _import_rocketsim() -> Any:
    try:
        import RocketSim as rs
    except ImportError as error:  # pragma: no cover - optional dependency diagnostic
        raise RuntimeError("install RivalSim with the 'oracle' extra") from error
    return rs


def _to_controls(rs: Any, controls: ControlBatch, car_index: int) -> Any:
    return _to_controls_at(rs, controls, 0, car_index)


def _to_controls_at(rs: Any, controls: ControlBatch, env_index: int, car_index: int) -> Any:
    return rs.CarControls(
        throttle=float(controls.throttle[env_index, car_index]),
        steer=float(controls.steer[env_index, car_index]),
        pitch=float(controls.pitch[env_index, car_index]),
        yaw=float(controls.yaw[env_index, car_index]),
        roll=float(controls.roll[env_index, car_index]),
        jump=bool(controls.jump[env_index, car_index]),
        boost=bool(controls.boost[env_index, car_index]),
        handbrake=bool(controls.handbrake[env_index, car_index]),
    )


def _previous_controls_at(rs: Any, snapshot: StateSnapshot, env_index: int, car_index: int) -> Any:
    return rs.CarControls(
        throttle=float(snapshot.prev_throttle[env_index, car_index]),
        steer=float(snapshot.prev_steer[env_index, car_index]),
        pitch=float(snapshot.prev_pitch[env_index, car_index]),
        yaw=float(snapshot.prev_yaw[env_index, car_index]),
        roll=float(snapshot.prev_roll[env_index, car_index]),
        jump=bool(snapshot.prev_jump[env_index, car_index]),
        boost=bool(snapshot.prev_boost[env_index, car_index]),
        handbrake=bool(snapshot.prev_handbrake[env_index, car_index]),
    )


def _to_vec(rs: Any, value: np.ndarray) -> Any:
    return rs.Vec(float(value[0]), float(value[1]), float(value[2]))


def _to_rot_mat(rs: Any, quat: np.ndarray) -> Any:
    matrix = quat_to_matrix(np.asarray(quat, dtype=np.float32))
    return rs.RotMat(
        _to_vec(rs, matrix[:, 0]),
        _to_vec(rs, matrix[:, 1]),
        _to_vec(rs, matrix[:, 2]),
    )


def _vec(value: Any) -> np.ndarray:
    return np.asarray(value.as_tuple(), dtype=np.float32)


def _matrix(value: Any) -> np.ndarray:
    # The binding returns [forward, right, up] as rows; physics math uses basis columns.
    return np.asarray(value.as_numpy(), dtype=np.float32).reshape(3, 3).T
