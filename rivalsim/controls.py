"""Batched controller input representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _continuous(value: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=np.float32), shape).copy()


def _discrete(value: bool | int | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=np.int32), shape).copy()


@dataclass(slots=True)
class ControlBatch:
    """Controller values for exactly two cars per world."""

    throttle: np.ndarray
    steer: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    jump: np.ndarray
    boost: np.ndarray
    handbrake: np.ndarray

    @classmethod
    def zeros(cls, num_envs: int) -> ControlBatch:
        shape = (num_envs, 2)
        return cls(
            *(_continuous(0.0, shape) for _ in range(5)),
            *(_discrete(0, shape) for _ in range(3)),
        )

    @classmethod
    def constant(
        cls,
        num_envs: int,
        *,
        throttle: float = 0.0,
        steer: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        jump: bool = False,
        boost: bool = False,
        handbrake: bool = False,
    ) -> ControlBatch:
        shape = (num_envs, 2)
        return cls(
            _continuous(throttle, shape),
            _continuous(steer, shape),
            _continuous(pitch, shape),
            _continuous(yaw, shape),
            _continuous(roll, shape),
            _discrete(jump, shape),
            _discrete(boost, shape),
            _discrete(handbrake, shape),
        ).clamped()

    @property
    def num_envs(self) -> int:
        return int(self.throttle.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays())

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.throttle,
            self.steer,
            self.pitch,
            self.yaw,
            self.roll,
            self.jump,
            self.boost,
            self.handbrake,
        )

    def copy(self) -> ControlBatch:
        return ControlBatch(*(array.copy() for array in self.arrays()))

    def clamped(self) -> ControlBatch:
        result = self.copy()
        for array in result.arrays()[:5]:
            np.clip(array, -1.0, 1.0, out=array)
        for array in result.arrays()[5:]:
            array[...] = array != 0
        result.validate()
        return result

    def validate(self) -> None:
        shape = (self.num_envs, 2)
        for name in ("throttle", "steer", "pitch", "yaw", "roll"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid continuous control array: {name}")
        for name in ("jump", "boost", "handbrake"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.int32:
                raise ValueError(f"invalid discrete control array: {name}")
