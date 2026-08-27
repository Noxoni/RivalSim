"""Renderer-neutral RivalVis frame and visual interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

type Vec3 = tuple[float, float, float]
type Quaternion = tuple[float, float, float, float]  # RivalSim x, y, z, w.
type Controls = tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class TransformFrame:
    position: Vec3
    quaternion: Quaternion


@dataclass(frozen=True, slots=True)
class BallFrame:
    transform: TransformFrame
    linear_velocity: Vec3
    angular_velocity: Vec3


@dataclass(frozen=True, slots=True)
class CarFrame:
    transform: TransformFrame
    linear_velocity: Vec3
    angular_velocity: Vec3
    boost: float
    speed: float
    on_ground: bool
    wheel_contacts: int
    has_jumped: bool
    is_jumping: bool
    has_double_jumped: bool
    has_flipped: bool
    is_flipping: bool
    dodge_available: bool
    is_supersonic: bool
    is_demoed: bool
    distance_to_ball: float
    touches: int
    reward: float
    controls: Controls


@dataclass(frozen=True, slots=True)
class BoostPadFrame:
    position: Vec3
    is_large: bool
    active: bool


@dataclass(frozen=True, slots=True)
class ViewerFrame:
    physics_tick: int
    policy_decision: int
    regulation_ticks_remaining: int
    blue_score: int
    orange_score: int
    overtime: bool
    kickoff_active: bool
    match_finished: bool
    winner: int | None
    last_touch: int | None
    ball: BallFrame
    cars: tuple[CarFrame, CarFrame]
    boost_pads: tuple[BoostPadFrame, ...]

    @property
    def regulation_seconds_remaining(self) -> float:
        return self.regulation_ticks_remaining / 120.0


def quaternion_forward(quaternion: Quaternion) -> np.ndarray:
    """Return RivalSim's local +X (car forward) axis in world coordinates."""

    x, y, z, w = quaternion
    return np.asarray(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        dtype=np.float64,
    )


def _lerp_vec3(left: Vec3, right: Vec3, alpha: float) -> Vec3:
    return tuple(float(a + (b - a) * alpha) for a, b in zip(left, right, strict=True))


def _nlerp_quaternion(left: Quaternion, right: Quaternion, alpha: float) -> Quaternion:
    """Shortest-arc normalized lerp for visual interpolation only."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    result = a + (b - a) * alpha
    length = float(np.linalg.norm(result))
    if length <= 1.0e-12:
        return left
    return tuple(float(value) for value in result / length)


def _interpolate_transform(
    left: TransformFrame, right: TransformFrame, alpha: float
) -> TransformFrame:
    return TransformFrame(
        position=_lerp_vec3(left.position, right.position, alpha),
        quaternion=_nlerp_quaternion(left.quaternion, right.quaternion, alpha),
    )


def interpolate_viewer_frame(
    previous: ViewerFrame, current: ViewerFrame, alpha: float
) -> ViewerFrame:
    """Interpolate transforms only; gameplay/HUD state always comes from current."""

    alpha = min(1.0, max(0.0, float(alpha)))
    ball = replace(
        current.ball,
        transform=_interpolate_transform(
            previous.ball.transform, current.ball.transform, alpha
        ),
    )
    cars = tuple(
        replace(
            current_car,
            transform=_interpolate_transform(
                previous_car.transform, current_car.transform, alpha
            ),
        )
        for previous_car, current_car in zip(
            previous.cars, current.cars, strict=True
        )
    )
    return replace(current, ball=ball, cars=cars)


__all__ = [
    "BallFrame",
    "BoostPadFrame",
    "CarFrame",
    "Controls",
    "Quaternion",
    "TransformFrame",
    "Vec3",
    "ViewerFrame",
    "interpolate_viewer_frame",
    "quaternion_forward",
]
