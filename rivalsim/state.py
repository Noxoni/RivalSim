"""Host snapshots and GPU-resident batched state allocation."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import warp as wp

from rivalsim.controls import ControlBatch

CAR_VEC3_FIELDS = ("car_pos", "car_vel", "car_ang_vel", "flip_rel_torque")
CAR_FLOAT_FIELDS = (
    "boost",
    "boosting_time",
    "time_since_boosted",
    "jump_time",
    "air_time",
    "air_time_since_jump",
    "flip_time",
    "auto_flip_timer",
    "auto_flip_torque_scale",
    "supersonic_time",
    "prev_throttle",
    "prev_steer",
    "prev_pitch",
    "prev_yaw",
    "prev_roll",
)
CAR_INT_FIELDS = (
    "on_ground",
    "has_jumped",
    "is_jumping",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "is_auto_flipping",
    "is_boosting",
    "sticky_ticks",
    "is_supersonic",
    "prev_jump",
    "prev_boost",
    "prev_handbrake",
)


@dataclass(slots=True)
class StateSnapshot:
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_quat: np.ndarray
    car_ang_vel: np.ndarray
    boost: np.ndarray
    boosting_time: np.ndarray
    time_since_boosted: np.ndarray
    on_ground: np.ndarray
    has_jumped: np.ndarray
    is_jumping: np.ndarray
    has_double_jumped: np.ndarray
    has_flipped: np.ndarray
    is_flipping: np.ndarray
    sticky_ticks: np.ndarray
    jump_time: np.ndarray
    air_time: np.ndarray
    air_time_since_jump: np.ndarray
    flip_time: np.ndarray
    flip_rel_torque: np.ndarray
    auto_flip_timer: np.ndarray
    auto_flip_torque_scale: np.ndarray
    is_auto_flipping: np.ndarray
    is_boosting: np.ndarray
    is_supersonic: np.ndarray
    supersonic_time: np.ndarray
    prev_throttle: np.ndarray
    prev_steer: np.ndarray
    prev_pitch: np.ndarray
    prev_yaw: np.ndarray
    prev_roll: np.ndarray
    prev_jump: np.ndarray
    prev_boost: np.ndarray
    prev_handbrake: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_quat: np.ndarray
    ball_ang_vel: np.ndarray

    @classmethod
    def empty(cls, num_envs: int) -> StateSnapshot:
        car_shape = (num_envs, 2)
        car_vec = (num_envs, 2, 3)
        car_quat = np.zeros((num_envs, 2, 4), dtype=np.float32)
        car_quat[..., 3] = 1.0
        ball_quat = np.zeros((num_envs, 4), dtype=np.float32)
        ball_quat[..., 3] = 1.0
        car_pos = np.zeros(car_vec, dtype=np.float32)
        car_pos[..., 2] = 1000.0
        ball_pos = np.zeros((num_envs, 3), dtype=np.float32)
        ball_pos[..., 2] = 1200.0

        def float_car() -> np.ndarray:
            return np.zeros(car_shape, dtype=np.float32)

        def int_car() -> np.ndarray:
            return np.zeros(car_shape, dtype=np.int32)

        return cls(
            car_pos=car_pos,
            car_vel=np.zeros(car_vec, dtype=np.float32),
            car_quat=car_quat,
            car_ang_vel=np.zeros(car_vec, dtype=np.float32),
            boost=np.full(car_shape, 100.0, dtype=np.float32),
            boosting_time=float_car(),
            time_since_boosted=float_car(),
            on_ground=int_car(),
            has_jumped=int_car(),
            is_jumping=int_car(),
            has_double_jumped=int_car(),
            has_flipped=int_car(),
            is_flipping=int_car(),
            sticky_ticks=int_car(),
            jump_time=float_car(),
            air_time=float_car(),
            air_time_since_jump=float_car(),
            flip_time=float_car(),
            flip_rel_torque=np.zeros(car_vec, dtype=np.float32),
            auto_flip_timer=float_car(),
            auto_flip_torque_scale=float_car(),
            is_auto_flipping=int_car(),
            is_boosting=int_car(),
            is_supersonic=int_car(),
            supersonic_time=float_car(),
            prev_throttle=float_car(),
            prev_steer=float_car(),
            prev_pitch=float_car(),
            prev_yaw=float_car(),
            prev_roll=float_car(),
            prev_jump=int_car(),
            prev_boost=int_car(),
            prev_handbrake=int_car(),
            ball_pos=ball_pos,
            ball_vel=np.zeros((num_envs, 3), dtype=np.float32),
            ball_quat=ball_quat,
            ball_ang_vel=np.zeros((num_envs, 3), dtype=np.float32),
        )

    @classmethod
    def random(cls, num_envs: int, seed: int) -> StateSnapshot:
        state = cls.empty(num_envs)
        rng = np.random.default_rng(seed)
        state.car_pos[..., :2] = rng.uniform(-3000.0, 3000.0, (num_envs, 2, 2))
        state.car_pos[..., 2] = rng.uniform(700.0, 1500.0, (num_envs, 2))
        state.car_vel[...] = rng.uniform(-900.0, 900.0, state.car_vel.shape)
        state.car_ang_vel[...] = rng.uniform(-2.0, 2.0, state.car_ang_vel.shape)
        state.car_quat[...] = _random_quaternions(rng, (num_envs, 2))
        state.boost[...] = rng.uniform(0.0, 100.0, state.boost.shape)
        state.ball_pos[..., :2] = rng.uniform(-3000.0, 3000.0, (num_envs, 2))
        state.ball_pos[..., 2] = rng.uniform(800.0, 1600.0, num_envs)
        state.ball_vel[...] = rng.uniform(-1200.0, 1200.0, state.ball_vel.shape)
        state.ball_ang_vel[...] = rng.uniform(-3.0, 3.0, state.ball_ang_vel.shape)
        state.ball_quat[...] = _random_quaternions(rng, (num_envs,))
        state.validate()
        return state

    @property
    def num_envs(self) -> int:
        return int(self.car_pos.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, item.name).nbytes for item in fields(self))

    def copy(self) -> StateSnapshot:
        return StateSnapshot(*(getattr(self, item.name).copy() for item in fields(self)))

    def validate(self) -> None:
        n = self.num_envs
        expected = {
            **{name: ((n, 2, 3), np.float32) for name in CAR_VEC3_FIELDS},
            **{name: ((n, 2), np.float32) for name in CAR_FLOAT_FIELDS},
            **{name: ((n, 2), np.int32) for name in CAR_INT_FIELDS},
            "car_quat": ((n, 2, 4), np.float32),
            "ball_pos": ((n, 3), np.float32),
            "ball_vel": ((n, 3), np.float32),
            "ball_quat": ((n, 4), np.float32),
            "ball_ang_vel": ((n, 3), np.float32),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"invalid {name}: {value.shape}/{value.dtype}, expected {shape}/{dtype}"
                )
            if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"non-finite state in {name}")

        for name in ("car_quat", "ball_quat"):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if not np.allclose(norms, 1.0, atol=2e-5):
                raise ValueError(f"non-unit orientation in {name}")


class GpuState:
    """Flat Warp arrays; each adjacent pair of car elements forms one world."""

    def __init__(self, snapshot: StateSnapshot, device: str):
        snapshot.validate()
        self.num_envs = snapshot.num_envs
        self.device = device
        for name in CAR_VEC3_FIELDS:
            data = getattr(snapshot, name).reshape(-1, 3)
            setattr(self, name, wp.array(data, dtype=wp.vec3, device=device))
        self.car_quat = wp.array(snapshot.car_quat.reshape(-1, 4), dtype=wp.quat, device=device)
        for name in CAR_FLOAT_FIELDS:
            setattr(
                self,
                name,
                wp.array(getattr(snapshot, name).reshape(-1), dtype=wp.float32, device=device),
            )
        for name in CAR_INT_FIELDS:
            setattr(
                self,
                name,
                wp.array(getattr(snapshot, name).reshape(-1), dtype=wp.int32, device=device),
            )
        # Internal execution hint, intentionally excluded from the public
        # snapshot and frozen logical-state accounting. Static-world wheel
        # queries set it before integration; the void simulator leaves it zero.
        self.air_control_disabled = wp.zeros(
            snapshot.num_envs * 2, dtype=wp.int32, device=device
        )
        self.ball_pos = wp.array(snapshot.ball_pos, dtype=wp.vec3, device=device)
        self.ball_vel = wp.array(snapshot.ball_vel, dtype=wp.vec3, device=device)
        self.ball_quat = wp.array(snapshot.ball_quat, dtype=wp.quat, device=device)
        self.ball_ang_vel = wp.array(snapshot.ball_ang_vel, dtype=wp.vec3, device=device)

    @property
    def car_count(self) -> int:
        return self.num_envs * 2

    def snapshot(self) -> StateSnapshot:
        n = self.num_envs
        values: dict[str, np.ndarray] = {}
        for name in CAR_VEC3_FIELDS:
            values[name] = np.asarray(getattr(self, name).numpy(), dtype=np.float32).reshape(
                n, 2, 3
            )
        values["car_quat"] = np.asarray(self.car_quat.numpy(), dtype=np.float32).reshape(n, 2, 4)
        for name in CAR_FLOAT_FIELDS:
            values[name] = np.asarray(getattr(self, name).numpy(), dtype=np.float32).reshape(n, 2)
        for name in CAR_INT_FIELDS:
            values[name] = np.asarray(getattr(self, name).numpy(), dtype=np.int32).reshape(n, 2)
        values["ball_pos"] = np.asarray(self.ball_pos.numpy(), dtype=np.float32).reshape(n, 3)
        values["ball_vel"] = np.asarray(self.ball_vel.numpy(), dtype=np.float32).reshape(n, 3)
        values["ball_quat"] = np.asarray(self.ball_quat.numpy(), dtype=np.float32).reshape(n, 4)
        values["ball_ang_vel"] = np.asarray(self.ball_ang_vel.numpy(), dtype=np.float32).reshape(
            n, 3
        )
        result = StateSnapshot(**values)
        result.validate()
        return result


class GpuControls:
    def __init__(self, controls: ControlBatch, device: str):
        controls.validate()
        self.num_envs = controls.num_envs
        self.device = device
        for name in ("throttle", "steer", "pitch", "yaw", "roll"):
            data = getattr(controls, name).reshape(-1)
            setattr(self, name, wp.array(data, dtype=wp.float32, device=device))
        for name in ("jump", "boost", "handbrake"):
            data = getattr(controls, name).reshape(-1)
            setattr(self, name, wp.array(data, dtype=wp.int32, device=device))


def previous_controls_from_state(state: StateSnapshot) -> ControlBatch:
    return ControlBatch(
        state.prev_throttle.copy(),
        state.prev_steer.copy(),
        state.prev_pitch.copy(),
        state.prev_yaw.copy(),
        state.prev_roll.copy(),
        state.prev_jump.copy(),
        state.prev_boost.copy(),
        state.prev_handbrake.copy(),
    )


def _random_quaternions(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    u1 = rng.random(shape)
    u2 = rng.random(shape)
    u3 = rng.random(shape)
    result = np.empty((*shape, 4), dtype=np.float32)
    result[..., 0] = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    result[..., 1] = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    result[..., 2] = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    result[..., 3] = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.astype(np.float32, copy=False)
