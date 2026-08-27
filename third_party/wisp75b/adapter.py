"""Batched CUDA port of pinned Wisp v2-75B inference semantics.

The upstream bot is a scalar RLBot process.  This module keeps its 432-value
observation order, 90-row action table, masks, X mirroring, deterministic model
selection, previous-action history, and 8-tick/7-tick delayed-control state,
while operating on RivalSim's resident tensors.  No policy parameter is
trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import warp as wp

from third_party.wisp75b.source_eta import batched_eta

wp.init()

WISP_UPSTREAM_COMMIT = "58d4ab18fd0c92529b5ae6582ecf1713a6b1887a"
WISP_BOTPACK_COMMIT = "bfc9b79b0ec1599e0ac2f57363dfc28b0bda4f15"
WISP_POLICY_SHA256 = "1BD600A15F43106645DE84B42379FE9AE404ECFB509DC21A2E309480EA17EBF7"
WISP_SHARED_HEAD_SHA256 = "3F7B6B363A72D7CEABA3CDB58BC13E1AE95E07B041B5E94A326C7045BEBD7E42"
WISP_OBS_DIM = 432
WISP_ACTION_COUNT = 90
WISP_TICK_SKIP = 8
WISP_ACTION_DELAY = 7
WISP_PREDICTION_TICKS = (22, 66, 198, 594)

POS_COEF = 1.0 / 5000.0
VEL_COEF = 1.0 / 2300.0
ANG_VEL_COEF = 1.0 / 3.0
BACK_WALL_Y = 5120.0
SIDE_WALL_X = 4096.0
CEILING_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_HALF_WIDTH = 892.755

WISP_BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0), (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0), (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0), (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0), (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0), (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0), (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0), (2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0), (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0), (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0), (0.0, 1024.0, 70.0),
    (-2048.0, 1036.0, 70.0), (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0), (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0), (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0), (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0), (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0), (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0), (0.0, 4240.0, 70.0),
)


# Wisp/RLBot order -> RivalSim/RocketSim order, matched by physical coordinates.
# Wisp's historical (-940, 3310, 70) entry maps to RocketSim's
# (-940, 3308, 70), the same two-unit source discrepancy preserved by Nexto.
RIVALSIM_TO_WISP_PAD_INDICES = (
    6, 7, 8, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 18, 17, 0, 19,
    20, 1, 22, 21, 23, 24, 25, 26, 27, 28, 29, 30, 2, 3, 31, 32, 33,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _reconstruct(path: Path) -> nn.Sequential:
    module = torch.jit.load(str(path), map_location="cpu")
    result = nn.Sequential()
    for child in list(module.modules())[1:]:
        parameters = list(child.named_parameters())
        if len(parameters) == 2:
            weight, bias = parameters[0][1], parameters[1][1]
            if weight.ndim == 2 and bias.ndim == 1:
                layer: nn.Module = nn.Linear(weight.shape[1], weight.shape[0])
            elif weight.ndim == 1 and bias.ndim == 1:
                layer = nn.LayerNorm(weight.shape[0])
            else:
                raise RuntimeError(f"unsupported Wisp layer {weight.shape}/{bias.shape}")
            layer.load_state_dict(child.state_dict())
            result.append(layer)
        elif not parameters:
            result.append(nn.ReLU())
        else:
            raise RuntimeError("unsupported Wisp TorchScript child")
    return result.eval()


def build_action_table(device: torch.device | str) -> torch.Tensor:
    rows: list[list[float]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    rows.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])
    ground_count = len(rows)
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = int(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        rows.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    result = torch.tensor(rows, dtype=torch.float32, device=device)
    if ground_count != 24 or result.shape != (WISP_ACTION_COUNT, 8):
        raise RuntimeError(f"unexpected Wisp action table {result.shape}/{ground_count}")
    return result


@dataclass(frozen=True, slots=True)
class WispStateTensors:
    car_pos: torch.Tensor
    car_vel: torch.Tensor
    car_quat: torch.Tensor
    car_ang_vel: torch.Tensor
    boost: torch.Tensor
    on_ground: torch.Tensor
    is_jumping: torch.Tensor
    has_jumped: torch.Tensor
    has_double_jumped: torch.Tensor
    has_flipped: torch.Tensor
    air_time_since_jump: torch.Tensor
    demoed: torch.Tensor
    pad_cooldown: torch.Tensor
    ball_pos: torch.Tensor
    ball_vel: torch.Tensor
    ball_ang_vel: torch.Tensor
    touch_count: torch.Tensor
    handbrake_value: torch.Tensor

    @classmethod
    def from_bridge(cls, bridge: object) -> "WispStateTensors":
        views = bridge.views
        count = bridge.num_envs
        return cls(
            car_pos=views["car_pos"].reshape(count, 2, 3),
            car_vel=views["car_vel"].reshape(count, 2, 3),
            car_quat=views["car_quat"].reshape(count, 2, 4),
            car_ang_vel=views["car_ang_vel"].reshape(count, 2, 3),
            boost=views["boost"].reshape(count, 2),
            on_ground=views["on_ground"].reshape(count, 2),
            is_jumping=views["is_jumping"].reshape(count, 2),
            has_jumped=views["has_jumped"].reshape(count, 2),
            has_double_jumped=views["has_double_jumped"].reshape(count, 2),
            has_flipped=views["has_flipped"].reshape(count, 2),
            air_time_since_jump=views["air_time_since_jump"].reshape(count, 2),
            demoed=views["car_is_demoed"].reshape(count, 2),
            pad_cooldown=views["pad_cooldown"].reshape(count, 34),
            ball_pos=views["ball_pos"].reshape(count, 3),
            ball_vel=views["ball_vel"].reshape(count, 3),
            ball_ang_vel=views["ball_ang_vel"].reshape(count, 3),
            touch_count=views["rival2.touch_count"].reshape(count, 2),
            handbrake_value=views["handbrake_value"].reshape(count, 2),
        )


def _basis(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z, w = quaternion.unbind(-1)
    forward = torch.stack((1 - 2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)), dim=-1)
    right = torch.stack((2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)), dim=-1)
    up = torch.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), dim=-1)
    return forward, right, up


def _local(vector: torch.Tensor, basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple((vector * axis).sum(-1) for axis in basis), dim=-1)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    squared = vector[..., 0] * vector[..., 0]
    squared = squared + vector[..., 1] * vector[..., 1]
    squared = squared + vector[..., 2] * vector[..., 2]
    return vector / torch.sqrt(squared).unsqueeze(-1).clamp_min(1.0e-7)


def _length3(vector: torch.Tensor) -> torch.Tensor:
    squared = vector[..., 0] * vector[..., 0]
    squared = squared + vector[..., 1] * vector[..., 1]
    squared = squared + vector[..., 2] * vector[..., 2]
    return torch.sqrt(squared)


def _canonical_vector(value: torch.Tensor, invert: torch.Tensor, mirror: torch.Tensor) -> torch.Tensor:
    scale = torch.ones_like(value)
    scale[..., 0] = torch.where(invert, -1.0, 1.0)
    scale[..., 1] = torch.where(invert, -1.0, 1.0)
    result = value * scale
    result[..., 0] = torch.where(mirror, -result[..., 0], result[..., 0])
    return result


def _canonical_physics(
    pos: torch.Tensor,
    vel: torch.Tensor,
    angular: torch.Tensor,
    quaternion: torch.Tensor,
    invert: torch.Tensor,
    mirror: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    forward, right, up = _basis(quaternion)
    cpos = _canonical_vector(pos, invert, mirror)
    cvel = _canonical_vector(vel, invert, mirror)
    cangular = _canonical_vector(angular, invert, torch.zeros_like(mirror))
    cangular[..., 1] = torch.where(mirror, -cangular[..., 1], cangular[..., 1])
    cangular[..., 2] = torch.where(mirror, -cangular[..., 2], cangular[..., 2])
    invert_scale = torch.stack((torch.where(invert, -1.0, 1.0), torch.where(invert, -1.0, 1.0), torch.ones_like(invert, dtype=torch.float32)), -1)
    forward = forward * invert_scale
    right = right * invert_scale
    up = up * invert_scale
    forward = forward * torch.stack((torch.where(mirror, -1.0, 1.0), torch.ones_like(mirror, dtype=torch.float32), torch.ones_like(mirror, dtype=torch.float32)), -1)
    right = right * torch.stack((torch.ones_like(mirror, dtype=torch.float32), torch.where(mirror, -1.0, 1.0), torch.where(mirror, -1.0, 1.0)), -1)
    up = up * torch.stack((torch.where(mirror, -1.0, 1.0), torch.ones_like(mirror, dtype=torch.float32), torch.ones_like(mirror, dtype=torch.float32)), -1)
    return cpos, cvel, cangular, (forward, right, up)


def _corner_distance(pos: torch.Tensor) -> torch.Tensor:
    point = pos[..., :2].abs()
    start = torch.tensor((2944.0, 5120.0), device=pos.device)
    segment = torch.tensor((1152.0, -1152.0), device=pos.device)
    parameter = ((point - start) * segment).sum(-1) / (segment * segment).sum()
    closest = start + parameter.clamp(0.0, 1.0).unsqueeze(-1) * segment
    return torch.linalg.vector_norm(point - closest, dim=-1)


def _sdf_distance(point: torch.Tensor) -> torch.Tensor:
    center = torch.tensor((0.0, 0.0, CEILING_Z / 2), device=point.device)
    semi = torch.tensor((SIDE_WALL_X, BACK_WALL_Y, CEILING_Z / 2), device=point.device)
    q = (point - center).abs() - semi + 280.0
    base = _length3(q.clamp_min(0)) + q.max(-1).values.clamp_max(0)
    inv = 1.0 / math.sqrt(2.0)
    rotated = torch.stack((
        point[..., 0] * inv + point[..., 1] * inv,
        point[..., 0] * -inv + point[..., 1] * inv,
        point[..., 2],
    ), -1)
    corner_semi = torch.tensor((inv*8064.0, inv*8064.0, CEILING_Z/2), device=point.device)
    cq = (rotated - center).abs() - corner_semi + 280.0
    corner = _length3(cq.clamp_min(0)) + cq.max(-1).values.clamp_max(0)
    base_corner = torch.maximum(base, corner) - 280.0
    goal_center = torch.tensor((0.0, 0.0, GOAL_HEIGHT/2), device=point.device)
    goal_semi = torch.tensor((GOAL_HALF_WIDTH, 6000.0, GOAL_HEIGHT/2), device=point.device)
    gq = (point-goal_center).abs()-goal_semi+280.0
    goal = _length3(gq.clamp_min(0))+gq.max(-1).values.clamp_max(0)
    return -torch.minimum(base_corner, goal)


def _landing_normal(pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    p, v = pos.clone(), vel.clone()
    active = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    for _ in range(40):
        active &= _sdf_distance(p) > 0
        next_v = v + torch.tensor((0.0, 0.0, -162.5), device=pos.device)
        p = torch.where(active[:, None], p + next_v * 0.25, p)
        v = torch.where(active[:, None], next_v, v)
    p = p - 0.125 * v
    delta = 0.0004
    axes = torch.eye(3, device=pos.device) * delta
    gradient = torch.stack(tuple(_sdf_distance(p + axis) - _sdf_distance(p - axis) for axis in axes), -1)
    return _normalize(gradient)


def _ball_prediction(pos: torch.Tensor, vel: torch.Tensor, ticks: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-resident kinematic provider at the exact upstream slice horizons.

    RivalSim supplies the authoritative current state; prediction is read-only
    and deliberately excludes cars, matching RLBot's ball-prediction input role.
    Floor/ceiling folding keeps long-horizon features finite without another
    match simulation.
    """
    if not torch.is_tensor(ticks):
        ticks = torch.full((pos.shape[0],), int(ticks), device=pos.device, dtype=torch.float32)
    else:
        ticks = ticks.to(device=pos.device, dtype=torch.float32)
    t = ticks / 120.0
    predicted = pos + vel * t[:, None]
    predicted[:, 2] += -325.0 * t * t
    predicted_velocity = vel.clone()
    predicted_velocity[:, 2] += -650.0 * t
    lower, upper = 91.25, CEILING_Z - 91.25
    span = upper - lower
    phase = torch.remainder(predicted[:, 2] - lower, 2.0 * span)
    descending = phase > span
    predicted[:, 2] = lower + torch.where(descending, 2.0 * span - phase, phase)
    predicted_velocity[:, 2] = torch.where(descending, -predicted_velocity[:, 2], predicted_velocity[:, 2]) * 0.6
    return predicted, predicted_velocity


def _turn_radius(speed: torch.Tensor) -> torch.Tensor:
    v = speed.abs().clamp_max(2499.0)
    curvature = torch.where(v < 500, 0.006900-5.84e-6*v,
        torch.where(v < 1000, 0.005610-3.26e-6*v,
        torch.where(v < 1500, 0.004300-1.95e-6*v,
        torch.where(v < 1750, 0.003025-1.1e-6*v, 0.001800-4e-7*v))))
    return torch.where(v == 0, torch.zeros_like(v), curvature.clamp_min(1e-7).reciprocal())


def _goal_post_between(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_not_goal = first[:, 1].abs() < BACK_WALL_Y
    both_inside = first_not_goal & (second[:, 1].abs() < BACK_WALL_Y)
    inside = torch.where(first_not_goal[:, None], second, first).clone()
    other = torch.where(first_not_goal[:, None], first, second).clone()
    negative = inside[:, 1] < 0
    inside[:, 1] = torch.where(negative, -inside[:, 1], inside[:, 1])
    other[:, 1] = torch.where(negative, -other[:, 1], other[:, 1])
    gy = inside[:, 1] - BACK_WALL_Y
    by = other[:, 1] - BACK_WALL_Y
    left_gx, left_bx = inside[:, 0] + GOAL_HALF_WIDTH, other[:, 0] + GOAL_HALF_WIDTH
    right_gx, right_bx = inside[:, 0] - GOAL_HALF_WIDTH, other[:, 0] - GOAL_HALF_WIDTH
    cross_gz, cross_bz = inside[:, 2] - GOAL_HEIGHT, other[:, 2] - GOAL_HEIGHT
    epsilon = torch.finfo(first.dtype).tiny
    divide = lambda top, bottom: top / torch.where(bottom == 0, torch.full_like(bottom, epsilon), bottom)
    left = (left_bx < 0) & (divide(gy, left_gx) >= divide(by, left_bx))
    right = (right_bx > 0) & (divide(gy, right_gx) <= divide(by, right_bx))
    cross = (cross_bz > 0) & (divide(cross_gz, gy) >= divide(cross_bz, by))
    return ~both_inside & (left | right | cross)


@wp.func
def _wisp_solve_exp(v0: wp.float64, x0: wp.float64, v_inf: wp.float64) -> wp.float64:
    amplitude = v_inf - v0
    denominator = v0
    if denominator < wp.float64(1.0):
        denominator = wp.float64(1.0)
    result = x0 / denominator
    for _iteration in range(2):
        exponential = wp.exp(-result)
        remaining_distance = v_inf * result + (v0 - v_inf) * (wp.float64(1.0) - exponential) - x0
        remaining_velocity = v_inf - amplitude * exponential
        result = result - remaining_distance / remaining_velocity
    return result


@wp.func
def _wisp_solve_quad(v0: wp.float64, x0: wp.float64, acceleration: wp.float64) -> wp.float64:
    denominator = v0
    if denominator < wp.float64(1.0):
        denominator = wp.float64(1.0)
    result = x0 / denominator
    remaining_distance = v0 * result + wp.float64(0.5) * acceleration * result * result - x0
    remaining_velocity = v0 + acceleration * result
    return result - remaining_distance / remaining_velocity


@wp.func
def _wisp_linear_eta_scalar(v0_input: wp.float32, x0_input: wp.float64, boost_input: wp.float64) -> wp.float64:
    v0 = wp.float64(v0_input)
    x0 = wp.float64(x0_input)
    boost_duration = boost_input
    throttle_asymptote = wp.float64(1556.0)
    boost_acceleration = wp.float64(991.66)
    boosted_asymptote = throttle_asymptote + boost_acceleration
    elapsed = wp.float64(0.0)

    if v0 >= wp.float64(2300.0):
        return x0 / wp.float64(2300.0)

    if v0 < wp.float64(1410.0) and boost_duration > wp.float64(0.0):
        destination = _wisp_solve_exp(v0, x0, boosted_asymptote)
        if destination >= wp.float64(0.0) and destination <= boost_duration:
            return destination
        time_limit = wp.log((boosted_asymptote - v0) / (boosted_asymptote - wp.float64(1410.0)))
        segment = time_limit
        if boost_duration < segment:
            segment = boost_duration
        exponential = wp.exp(-segment)
        end_velocity = boosted_asymptote - (boosted_asymptote - v0) * exponential
        segment_distance = boosted_asymptote * segment + (v0 - boosted_asymptote) * (wp.float64(1.0) - exponential)
        x0 = x0 - segment_distance
        v0 = end_velocity
        boost_duration = boost_duration - segment
        elapsed = elapsed + segment

    if v0 < wp.float64(1410.0) and boost_duration <= wp.float64(0.0):
        destination = _wisp_solve_exp(v0, x0, throttle_asymptote)
        time_limit = wp.log((throttle_asymptote - v0) / (throttle_asymptote - wp.float64(1410.0)))
        if destination >= wp.float64(0.0) and destination < time_limit:
            return destination
        exponential = wp.exp(-time_limit)
        end_velocity = throttle_asymptote - (throttle_asymptote - v0) * exponential
        segment_distance = throttle_asymptote * time_limit + (v0 - throttle_asymptote) * (wp.float64(1.0) - exponential)
        x0 = x0 - segment_distance
        v0 = end_velocity
        elapsed = elapsed + time_limit

    if v0 >= wp.float64(1410.0) and boost_duration > wp.float64(0.0):
        destination = _wisp_solve_quad(v0, x0, boost_acceleration)
        if destination >= wp.float64(0.0) and destination <= boost_duration:
            return destination
        time_to_max = (wp.float64(2300.0) - v0) / boost_acceleration
        if time_to_max <= boost_duration:
            x0 = x0 - (v0 * time_to_max + wp.float64(0.5) * boost_acceleration * time_to_max * time_to_max)
            return elapsed + time_to_max + x0 / wp.float64(2300.0)
        x0 = x0 - (v0 * boost_duration + wp.float64(0.5) * boost_acceleration * boost_duration * boost_duration)
        v0 = v0 + boost_acceleration * boost_duration
        boost_duration = wp.float64(0.0)
        elapsed = elapsed + boost_duration

    return elapsed + x0 / v0


@wp.kernel(enable_backward=False)
def _wisp_linear_eta_kernel(
    v0: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float64),
    boost_amount: wp.array(dtype=wp.float32),
    result: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    boost_duration = wp.float64(boost_amount[index]) / wp.float64(33.3)
    result[index] = _wisp_linear_eta_scalar(v0[index], x0[index], boost_duration)


@wp.func
def _wisp_prediction_position(
    position: wp.vec3,
    velocity: wp.vec3,
    prediction_tick: int,
) -> wp.vec3:
    time = wp.float32(prediction_tick) / wp.float32(120.0)
    result = position + velocity * time
    result[2] = result[2] + wp.float32(-325.0) * time * time
    lower = wp.float32(91.25)
    span = wp.float32(2044.0 - 182.5)
    period = wp.float32(2.0) * span
    phase = wp.mod(result[2] - lower, period)
    if phase < wp.float32(0.0):
        phase = phase + period
    if phase > span:
        result[2] = lower + (wp.float32(2.0) * span - phase)
    else:
        result[2] = lower + phase
    return result


@wp.kernel(enable_backward=False)
def _wisp_eta_kernel(
    car_position: wp.array2d(dtype=wp.vec3),
    car_velocity: wp.array2d(dtype=wp.vec3),
    boost_amount: wp.array2d(dtype=wp.float32),
    ball_position: wp.array(dtype=wp.vec3),
    ball_velocity: wp.array(dtype=wp.vec3),
    physical: wp.array(dtype=wp.int64),
    eta_cache: wp.array2d(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
    trace_v0: wp.array3d(dtype=wp.float32),
    trace_x0: wp.array3d(dtype=wp.float64),
    trace_tick: wp.array3d(dtype=wp.int32),
    trace_time: wp.array3d(dtype=wp.float64),
):
    world = wp.tid()
    side = physical[world]
    estimate = eta_cache[world, side]
    for _eta_pass in range(2):
        tick = int(estimate * wp.float64(120.0))
        if tick < 0:
            tick = 0
        if tick > 599:
            tick = 599
        target = _wisp_prediction_position(
            ball_position[world], ball_velocity[world], tick + 1
        )
        delta = target - car_position[world, side]
        # NumPy 1.26 routes these three-element float32 dot products through
        # OpenBLAS SDOT's short tail: each product rounds to float32, the three
        # products accumulate in float64, and the result converts back to
        # float32.  Reproduce that order instead of using CUDA's vec3 dot/FMA.
        squared_sum = wp.float64(wp.float32(delta[0] * delta[0]))
        squared_sum = squared_sum + wp.float64(wp.float32(delta[1] * delta[1]))
        squared_sum = squared_sum + wp.float64(wp.float32(delta[2] * delta[2]))
        distance = wp.sqrt(wp.float32(squared_sum))
        direction = delta / distance
        velocity = car_velocity[world, side]
        velocity_sum = wp.float64(wp.float32(velocity[0] * direction[0]))
        velocity_sum = velocity_sum + wp.float64(wp.float32(velocity[1] * direction[1]))
        velocity_sum = velocity_sum + wp.float64(wp.float32(velocity[2] * direction[2]))
        initial_velocity = wp.float32(velocity_sum)
        target_distance = wp.float64(distance) - wp.float64(136.875)
        boost_duration = (
            wp.float64(boost_amount[world, side]) / wp.float64(33.3)
        )
        estimate = _wisp_linear_eta_scalar(
            initial_velocity, target_distance, boost_duration
        )
        trace_v0[world, side, _eta_pass] = initial_velocity
        trace_x0[world, side, _eta_pass] = target_distance
        trace_tick[world, side, _eta_pass] = tick
        trace_time[world, side, _eta_pass] = estimate
    eta_cache[world, side] = estimate
    result[world] = estimate


def _linear_eta(v0: torch.Tensor, x0: torch.Tensor, boost_amount: torch.Tensor) -> torch.Tensor:
    if v0.dtype != torch.float32 or x0.dtype != torch.float64 or boost_amount.dtype != torch.float32:
        raise TypeError("Wisp ETA inputs must preserve float32/float64/float32 source types")
    result = torch.empty_like(x0)
    stream = wp.stream_from_torch(torch.cuda.current_stream(v0.device))
    wp.launch(
        _wisp_linear_eta_kernel,
        dim=v0.numel(),
        inputs=[wp.from_torch(v0), wp.from_torch(x0), wp.from_torch(boost_amount), wp.from_torch(result)],
        device=str(v0.device),
        stream=stream,
    )
    return result


class WispPolicyAdapter:
    """Frozen batched policy and per-world temporal controller state."""

    def __init__(self, num_worlds: int, *, device: torch.device | str = "cuda:0", model_root: str | Path | None = None):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("production Wisp adapter requires CUDA")
        root = Path(model_root) if model_root is not None else Path(__file__).with_name("models")
        policy_path, shared_path = root / "POLICY.lt", root / "SHARED_HEAD.lt"
        if _sha256(policy_path) != WISP_POLICY_SHA256 or _sha256(shared_path) != WISP_SHARED_HEAD_SHA256:
            raise RuntimeError("pinned Wisp model SHA-256 mismatch")
        self.shared = _reconstruct(shared_path).to(self.device).eval().requires_grad_(False)
        self.policy = _reconstruct(policy_path).to(self.device).eval().requires_grad_(False)
        self.action_table = build_action_table(self.device)
        self._build_masks()
        self.player_index = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)
        self.old_action = torch.zeros((self.num_worlds, 8), device=self.device)
        self.new_action = torch.zeros_like(self.old_action)
        self.previous_action = torch.zeros_like(self.old_action)
        self.ticks = torch.full((self.num_worlds,), -1, dtype=torch.int64, device=self.device)
        self.update_flag = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        # The pinned source's ETA branches on Windows CPU libm results at a
        # one-bit-sensitive 1410 uu/s boundary.  Keep only its compact scalar
        # cache and source-ordered batched solver on the host; observation
        # assembly and policy inference remain resident on CUDA.
        self.eta_cache = np.zeros((self.num_worlds, 2), dtype=np.float64)
        self.eta_trace_v0 = torch.zeros((self.num_worlds, 2, 2), device=self.device)
        self.eta_trace_x0 = torch.zeros((self.num_worlds, 2, 2), dtype=torch.float64, device=self.device)
        self.eta_trace_tick = torch.zeros((self.num_worlds, 2, 2), dtype=torch.int32, device=self.device)
        self.eta_trace_time = torch.zeros((self.num_worlds, 2, 2), dtype=torch.float64, device=self.device)
        self._eta_host_snapshot: tuple[np.ndarray, ...] | None = None
        self._eta_selected = torch.arange(self.num_worlds, device=self.device)
        # Upstream shuffles the padded teammate/opponent lists with Python's
        # process-global RNG on every neural observation.  One-v-one leaves
        # only the opponent's position among three otherwise-zero slots
        # observable.  Use a dedicated checkpointable CUDA RNG for the same
        # uniform slot semantics without a host-side hot-path dependency.
        self.observation_generator = torch.Generator(device=self.device)
        self.observation_generator.manual_seed(2026082703)
        self.opponent_slot = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)
        self.inference_calls = 0

    def _build_masks(self) -> None:
        table = self.action_table
        self.jump_mask = table[:, 5] != 0
        self.boost_mask = table[:, 6] != 0
        self.ground_mask = torch.arange(WISP_ACTION_COUNT, device=self.device) < 24
        self.air_mask = (torch.arange(WISP_ACTION_COUNT, device=self.device) > 24) & ~self.jump_mask
        ground_extra = self.ground_mask & (table[:, 0] == table[:, 6]) & ((table[:, 3] != 0) == (table[:, 7] != 0))
        self.air_mask |= ground_extra

    def set_player_index(self, index: torch.Tensor) -> None:
        if index.shape != (self.num_worlds,) or index.device != self.device:
            raise ValueError("Wisp player index must be one CUDA side per world")
        self.player_index.copy_(index)

    def set_opponent_slot(self, slot: torch.Tensor) -> None:
        if slot.shape != (self.num_worlds,) or slot.device != self.device:
            raise ValueError("Wisp opponent slot must be one CUDA index per world")
        if bool(((slot < 0) | (slot > 2)).any()):
            raise ValueError("Wisp opponent slot must be in [0, 2]")
        self.opponent_slot.copy_(slot)

    def reset(self, reset_mask: torch.Tensor) -> None:
        if reset_mask.shape != (self.num_worlds,) or reset_mask.device != self.device:
            raise ValueError("Wisp reset mask mismatch")
        self.old_action.masked_fill_(reset_mask[:, None], 0)
        self.new_action.masked_fill_(reset_mask[:, None], 0)
        self.previous_action.masked_fill_(reset_mask[:, None], 0)
        self.ticks.copy_(torch.where(reset_mask, torch.full_like(self.ticks, -1), self.ticks))
        self.update_flag.copy_(torch.where(reset_mask, torch.ones_like(self.update_flag), self.update_flag))

    def activate(self, active_mask: torch.Tensor) -> None:
        """Initialize only worlds newly assigned to a Wisp episode."""

        self.reset(active_mask)
        selected = active_mask.detach().cpu().numpy().astype(bool, copy=False)
        self.eta_cache[selected] = 0.0

    def action_mask(self, state: WispStateTensors) -> torch.Tensor:
        rows = torch.arange(self.num_worlds, device=self.device)
        side = self.player_index
        on_ground = state.on_ground[rows, side] != 0
        boost = state.boost[rows, side]
        pos = state.car_pos[rows, side]
        _forward, _right, up = _basis(state.car_quat[rows, side])
        velocity = state.car_vel[rows, side]
        turtled = ~on_ground & (up[:, 2] < -0.8) & (velocity[:, 2].abs() < 50) & (pos[:, 2] < 50)
        available = on_ground | ((state.has_flipped[rows, side] == 0) & (state.has_double_jumped[rows, side] == 0) & (state.air_time_since_jump[rows, side] < 1.25))
        result = torch.where(on_ground[:, None], self.ground_mask, self.air_mask).clone()
        result &= ~((boost == 0)[:, None] & self.boost_mask)
        result |= (available | turtled)[:, None] & self.jump_mask
        empty = ~result.any(-1)
        result[empty] = True
        return result

    def _eta(self, state: WispStateTensors, physical: torch.Tensor, prediction_pos: torch.Tensor) -> torch.Tensor:
        del prediction_pos  # Kept in the signature to mirror the source call site.
        selected = self._eta_selected
        result = torch.zeros(self.num_worlds, dtype=torch.float64, device=self.device)
        if selected.numel() == 0:
            return result
        host = self._eta_host_snapshot
        if host is None:
            host = tuple(
                value.detach().cpu().numpy()
                for value in (
                    state.car_pos,
                    state.car_vel,
                    state.boost,
                    state.ball_pos,
                    state.ball_vel,
                )
            )
        side = physical.index_select(0, selected).detach().cpu().numpy().astype(
            np.int64, copy=False
        )
        selected_host = selected.detach().cpu().numpy()
        selected_cache = self.eta_cache[selected_host].copy()
        output, trace_v0, trace_x0, trace_tick, trace_time = batched_eta(
            host[0], host[1], host[2], host[3], host[4], side, selected_cache
        )
        self.eta_cache[selected_host] = selected_cache
        trace_v0_device = torch.from_numpy(trace_v0).to(self.device)
        trace_x0_device = torch.from_numpy(trace_x0).to(self.device)
        trace_tick_device = torch.from_numpy(trace_tick.astype(np.int32, copy=False)).to(self.device)
        trace_time_device = torch.from_numpy(trace_time).to(self.device)
        selected_physical = physical.index_select(0, selected)
        self.eta_trace_v0[selected, selected_physical] = trace_v0_device
        self.eta_trace_x0[selected, selected_physical] = trace_x0_device
        self.eta_trace_tick[selected, selected_physical] = trace_tick_device
        self.eta_trace_time[selected, selected_physical] = trace_time_device
        result.index_copy_(0, selected, torch.from_numpy(output).to(self.device))
        return result

    def observation(
        self,
        state: WispStateTensors,
        *,
        score_diff: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.num_worlds
        if active_mask is None:
            active_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        if active_mask.shape != (n,) or active_mask.device != self.device:
            raise ValueError("Wisp observation active mask mismatch")
        self._eta_selected = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        rows = torch.arange(n, device=self.device)
        side = self.player_index
        opponent = 1 - side
        invert = side == 1
        world_pos = state.car_pos[rows, side]
        mirror = invert != (world_pos[:, 0] < 0)
        cpos, cvel, cang, basis = _canonical_physics(world_pos, state.car_vel[rows, side], state.car_ang_vel[rows, side], state.car_quat[rows, side], invert, mirror)
        ball_pos = _canonical_vector(state.ball_pos, invert, mirror)
        ball_vel = _canonical_vector(state.ball_vel, invert, mirror)
        ball_ang = _canonical_vector(
            state.ball_ang_vel,
            invert,
            torch.zeros_like(mirror),
        )
        ball_ang[:, 1] = torch.where(mirror, -ball_ang[:, 1], ball_ang[:, 1])
        ball_ang[:, 2] = torch.where(mirror, -ball_ang[:, 2], ball_ang[:, 2])
        features: list[torch.Tensor] = [ball_pos*POS_COEF, ball_vel*VEL_COEF, ball_ang*ANG_VEL_COEF]
        blue_goal = torch.tensor((0.0, -BACK_WALL_Y, GOAL_HEIGHT/2), device=self.device)
        orange_goal = torch.tensor((0.0, BACK_WALL_Y, GOAL_HEIGHT/2), device=self.device)
        my_goal = torch.where(invert[:, None], orange_goal, blue_goal)
        opp_goal = torch.where(invert[:, None], blue_goal, orange_goal)
        features += [(my_goal-ball_pos)*POS_COEF, (opp_goal-ball_pos)*POS_COEF, (ball_pos[:, :2] == 0).all(-1, keepdim=True).float()]
        for tick in WISP_PREDICTION_TICKS:
            p, v = _ball_prediction(state.ball_pos, state.ball_vel, tick+1)
            p, v = _canonical_vector(p, invert, mirror), _canonical_vector(v, invert, mirror)
            features += [p*POS_COEF, v*VEL_COEF, _local(p-cpos, basis)*POS_COEF, _local(v-cvel, basis)*VEL_COEF]
        pad_mapping = torch.tensor(RIVALSIM_TO_WISP_PAD_INDICES, device=self.device)
        timers = state.pad_cooldown.index_select(1, pad_mapping)
        if invert.any():
            timers = torch.where(invert[:, None], timers.flip(1), timers)
        locations = torch.tensor(WISP_BOOST_LOCATIONS, device=self.device)
        mirror_order = torch.argsort(locations[:, 1]*10000-locations[:, 0])
        timers = torch.where(mirror[:, None], timers.index_select(1, mirror_order), timers)
        features.append(torch.where(timers == 0, torch.ones_like(timers), 1/(1+timers)))
        physical_forward, physical_right, _ = _basis(state.car_quat[rows, side])
        soon = world_pos + physical_forward*420.0
        nearest = torch.topk(torch.cdist(soon[:, None], locations[None]).squeeze(1), 5, largest=False).indices
        pads = locations[nearest]
        relative = pads-world_pos[:, None]
        forward_rel = (relative*physical_forward[:, None]).sum(-1)*POS_COEF
        lateral_rel = (relative*physical_right[:, None]).sum(-1)*POS_COEF
        lateral_rel = torch.where(mirror[:, None], -lateral_rel, lateral_rel)
        features.append(torch.stack((forward_rel,lateral_rel),-1).reshape(n,10))
        previous = self.previous_action.clone()
        previous[:, (1,3,4)] = torch.where(mirror[:,None], -previous[:,(1,3,4)], previous[:,(1,3,4)])
        features.append(previous)
        features.append(torch.stack(((BACK_WALL_Y-world_pos[:,1].abs())*POS_COEF,(SIDE_WALL_X-world_pos[:,0].abs())*POS_COEF,_corner_distance(world_pos)*POS_COEF),-1))
        features.append(_local(_landing_normal(world_pos,state.car_vel[rows,side]),basis))
        if score_diff is None: score_diff=torch.zeros(n,device=self.device)
        forward_speed=(state.car_vel[rows,side]*basis[0]).sum(-1)
        features.append(torch.stack((score_diff.clamp(-1,1),_turn_radius(forward_speed.clamp(-2300,2300))/1300.0,(state.touch_count[rows,side]>0).float(),state.handbrake_value[rows,side].float()),-1))
        zeros=torch.zeros((n,51),device=self.device)
        self._eta_host_snapshot = tuple(
            value.index_select(0, self._eta_selected).detach().cpu().numpy()
            for value in (
                state.car_pos,
                state.car_vel,
                state.boost,
                state.ball_pos,
                state.ball_vel,
            )
        )
        features += [self._player_features(state, side, invert, mirror, ball_pos, ball_vel), zeros, zeros]
        opponent_features = self._player_features(state, opponent, invert, mirror, ball_pos, ball_vel)
        self._eta_host_snapshot = None
        features += [
            torch.where((self.opponent_slot == slot)[:, None], opponent_features, zeros)
            for slot in range(3)
        ]
        result=torch.cat(features,-1).contiguous()
        if result.shape != (n,WISP_OBS_DIM): raise RuntimeError(f"Wisp observation shape {result.shape}")
        return result

    def _player_features(self,state:WispStateTensors,physical:torch.Tensor,invert:torch.Tensor,mirror:torch.Tensor,ball_pos:torch.Tensor,ball_vel:torch.Tensor)->torch.Tensor:
        n=self.num_worlds; rows=torch.arange(n,device=self.device)
        pos,vel,ang,basis=_canonical_physics(state.car_pos[rows,physical],state.car_vel[rows,physical],state.car_ang_vel[rows,physical],state.car_quat[rows,physical],invert,mirror)
        dodge_forward=_normalize(basis[0]*torch.tensor((1.0,1.0,0.0),device=self.device)); dodge_right=torch.tensor((0.0,1.0,0.0),device=self.device).expand(n,-1)
        relp,relv=ball_pos-pos,ball_vel-vel
        blue_goal=torch.tensor((0.0,-BACK_WALL_Y,GOAL_HEIGHT/2),device=self.device); orange_goal=torch.tensor((0.0,BACK_WALL_Y,GOAL_HEIGHT/2),device=self.device)
        own_shot=_normalize(blue_goal-ball_pos); opp_shot=_normalize(orange_goal-ball_pos)
        local_ang=_local(ang,basis)
        scalar=lambda x:x.float().unsqueeze(-1)
        on=state.on_ground[rows,physical]!=0
        available=on|((state.has_flipped[rows,physical]==0)&(state.has_double_jumped[rows,physical]==0)&(state.air_time_since_jump[rows,physical]<1.25))
        flip_reset=~on&available&(state.has_jumped[rows,physical]==0)
        demo=state.demoed[rows,physical]!=0
        eta=self._eta(state,physical,state.ball_pos).float()
        blocks=[pos*POS_COEF,basis[0],basis[2],vel*VEL_COEF,ang*ANG_VEL_COEF,local_ang*ANG_VEL_COEF,scalar((basis[0]*vel).sum(-1)*VEL_COEF),_local(relp,basis)*POS_COEF,_local(relv,basis)*VEL_COEF,
            torch.stack(((dodge_forward*relp).sum(-1)*POS_COEF,(dodge_right*relp).sum(-1)*POS_COEF,(dodge_forward*relv).sum(-1)*VEL_COEF,(dodge_right*relv).sum(-1)*VEL_COEF),-1),
            _local(own_shot,basis)*POS_COEF,_local(opp_shot,basis)*POS_COEF,
            torch.stack(((dodge_forward*own_shot).sum(-1)*POS_COEF,(dodge_right*own_shot).sum(-1)*POS_COEF,(dodge_forward*opp_shot).sum(-1)*POS_COEF,(dodge_right*opp_shot).sum(-1)*POS_COEF),-1),
            torch.stack((state.boost[rows,physical]/100.0,on.float(),available.float(),demo.float(),state.is_jumping[rows,physical].float(),flip_reset.float(),(pos[:,1].abs()>BACK_WALL_Y-10).float(),_goal_post_between(state.car_pos[rows,physical],ball_pos).float(),torch.minimum(torch.minimum(SIDE_WALL_X-pos[:,0].abs(),BACK_WALL_Y-pos[:,1].abs()),_corner_distance(pos))*POS_COEF,eta),-1)]
        core=torch.cat(blocks,-1)
        core=torch.where(demo[:,None],torch.zeros_like(core),core)
        result=torch.cat((core,demo.float().unsqueeze(-1),torch.zeros((n,1),device=self.device)),-1)
        if result.shape!=(n,51):raise RuntimeError(f"Wisp player block {result.shape}")
        return result

    @torch.inference_mode()
    def neural_action(
        self,
        state: WispStateTensors,
        active_mask: torch.Tensor | None = None,
        *,
        score_diff: torch.Tensor | None = None,
        shuffle_opponents: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if active_mask is None: active_mask=torch.ones(self.num_worlds,dtype=torch.bool,device=self.device)
        selected=torch.nonzero(active_mask,as_tuple=False).squeeze(-1)
        if shuffle_opponents and selected.numel()>0:
            slots=torch.randint(3,(selected.numel(),),device=self.device,generator=self.observation_generator)
            self.opponent_slot.index_copy_(0,selected,slots)
        obs=self.observation(
            state,
            score_diff=score_diff,
            active_mask=active_mask,
        ); mask=self.action_mask(state)
        index=torch.full((self.num_worlds,),-1,dtype=torch.long,device=self.device); action=torch.zeros((self.num_worlds,8),device=self.device)
        if selected.numel()>0:
            logits=self.policy(self.shared(obs.index_select(0,selected))); masked=logits.masked_fill(~mask.index_select(0,selected),-1.0e10); selected_index=masked.argmax(-1); index.index_copy_(0,selected,selected_index)
            selected_action=self.action_table.index_select(0,selected_index).clone(); mirror=(self.player_index.index_select(0,selected)==1)!=(state.car_pos[selected,self.player_index.index_select(0,selected),0]<0)
            selected_action[:,(1,3,4)]=torch.where(mirror[:,None],-selected_action[:,(1,3,4)],selected_action[:,(1,3,4)]); action.index_copy_(0,selected,selected_action)
            self.inference_calls+=1
        return action,index,obs

    @torch.inference_mode()
    def tick_action(
        self,
        state: WispStateTensors,
        active_mask: torch.Tensor | None = None,
        *,
        score_diff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor,torch.Tensor|None]:
        if active_mask is None:active_mask=torch.ones(self.num_worlds,dtype=torch.bool,device=self.device)
        advancing=active_mask&(self.ticks>=0); self.ticks.add_(advancing.to(self.ticks.dtype))
        compute=active_mask&self.update_flag; indices=None
        if compute.any():
            candidate,all_indices,_=self.neural_action(state,compute,score_diff=score_diff); self.old_action.copy_(torch.where(compute[:,None],self.new_action,self.old_action)); self.new_action.copy_(torch.where(compute[:,None],candidate,self.new_action)); indices=all_indices; self.update_flag&=~compute
        apply_new=(self.ticks>=WISP_ACTION_DELAY-1)|(self.ticks==-1)
        output=torch.where(apply_new[:,None],self.new_action,self.old_action)
        self.previous_action.copy_(torch.where(active_mask[:,None],output,self.previous_action))
        finish=active_mask&((self.ticks>=WISP_TICK_SKIP)|(self.ticks==-1))
        self.ticks.copy_(torch.where(finish,torch.zeros_like(self.ticks),self.ticks))
        self.update_flag|=finish
        return output,indices


__all__=["WISP_ACTION_COUNT","WISP_BOTPACK_COMMIT","WISP_OBS_DIM","WISP_POLICY_SHA256","WISP_SHARED_HEAD_SHA256","WISP_UPSTREAM_COMMIT","WispPolicyAdapter","WispStateTensors","build_action_table"]
