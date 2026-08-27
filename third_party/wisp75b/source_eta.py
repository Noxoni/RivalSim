"""Compiled batched port of Wisp's numerically unstable scalar ETA helper.

The pinned policy's ``eta.py`` branches on the last bit produced by Windows
CPU ``log``/``exp`` at a mathematically exact 1410 uu/s boundary.  CUDA libm
does not reproduce that branch for all float32 inputs.  This module therefore
keeps only that small source routine on the host, compiled as one parallel
batch with the original scalar operation order.  No Python per-world loop is
used by the caller.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _solve_exp_segment(v0: float, x0: float, v_inf: float) -> float:
    amplitude = v_inf - v0
    estimate = x0 / max(v0, 1.0)
    exponential = math.exp(-estimate)
    remaining_distance = v_inf * estimate + (v0 - v_inf) * (1 - exponential) - x0
    remaining_velocity = v_inf - amplitude * exponential
    estimate = estimate - remaining_distance / remaining_velocity
    exponential = math.exp(-estimate)
    remaining_distance = v_inf * estimate + (v0 - v_inf) * (1 - exponential) - x0
    remaining_velocity = v_inf - amplitude * exponential
    return estimate - remaining_distance / remaining_velocity


@njit(cache=True)
def _solve_quad_segment(v0: float, x0: float, acceleration: float) -> float:
    estimate = x0 / max(v0, 1.0)
    remaining_distance = v0 * estimate + 0.5 * acceleration * estimate * estimate - x0
    remaining_velocity = v0 + acceleration * estimate
    return estimate - remaining_distance / remaining_velocity


@njit(cache=True)
def _linear_eta(v0: float, x0: float, boost_duration: float) -> float:
    throttle_asymptote = 1556.0
    boost_acceleration = 991.66
    boosted_asymptote = throttle_asymptote + boost_acceleration
    elapsed = 0.0
    if v0 >= 2300.0:
        return x0 / 2300.0
    if v0 < 1410.0 and boost_duration > 0:
        destination = _solve_exp_segment(v0, x0, boosted_asymptote)
        if destination >= 0.0 and destination <= boost_duration:
            return destination
        time_limit = math.log((boosted_asymptote - v0) / (boosted_asymptote - 1410.0))
        segment = min(time_limit, boost_duration)
        exponential = math.exp(-segment)
        end_velocity = boosted_asymptote - (boosted_asymptote - v0) * exponential
        segment_distance = boosted_asymptote * segment + (v0 - boosted_asymptote) * (
            1 - exponential
        )
        x0 -= segment_distance
        v0 = end_velocity
        boost_duration -= segment
        elapsed += segment
    if v0 < 1410.0 and boost_duration <= 0:
        destination = _solve_exp_segment(v0, x0, throttle_asymptote)
        time_limit = math.log((throttle_asymptote - v0) / (throttle_asymptote - 1410.0))
        if destination >= 0.0 and destination < time_limit:
            return destination
        exponential = math.exp(-time_limit)
        end_velocity = throttle_asymptote - (throttle_asymptote - v0) * exponential
        segment_distance = throttle_asymptote * time_limit + (v0 - throttle_asymptote) * (
            1 - exponential
        )
        x0 -= segment_distance
        v0 = end_velocity
        elapsed += time_limit
    if v0 >= 1410.0 and boost_duration > 0:
        destination = _solve_quad_segment(v0, x0, boost_acceleration)
        if destination >= 0.0 and destination <= boost_duration:
            return destination
        time_to_max = (2300.0 - v0) / boost_acceleration
        if time_to_max <= boost_duration:
            x0 -= v0 * time_to_max + 0.5 * boost_acceleration * time_to_max * time_to_max
            return elapsed + time_to_max + x0 / 2300.0
        x0 -= v0 * boost_duration + 0.5 * boost_acceleration * boost_duration * boost_duration
        v0 += boost_acceleration * boost_duration
        boost_duration = 0.0
        elapsed += boost_duration
    return elapsed + x0 / v0


@njit(cache=True, parallel=True)
def _batched_linear_eta(
    initial_velocity: np.ndarray,
    target_distance: np.ndarray,
    boost_duration: np.ndarray,
) -> np.ndarray:
    output = np.empty(initial_velocity.shape[0], dtype=np.float64)
    for world in prange(initial_velocity.shape[0]):
        output[world] = _linear_eta(
            initial_velocity[world],
            target_distance[world],
            boost_duration[world],
        )
    return output


def _source_inputs(
    car_position: np.ndarray,
    car_velocity: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    physical: np.ndarray,
    tick: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce NumPy 1.26 Vec/norm/dot inputs without scalar Python loops."""

    count = physical.shape[0]
    rows = np.arange(count)
    # Python list indexing is used by the source.  Preserve its negative-index
    # behavior even though ordinary valid Wisp ETAs do not exercise it.
    prediction_tick = np.where(tick < 0, tick + 600, tick)
    time = ((prediction_tick + 1).astype(np.float64) / 120.0).astype(np.float32)
    predicted = (
        ball_position.astype(np.float32, copy=False)
        + ball_velocity.astype(np.float32, copy=False) * time[:, None]
    ).astype(np.float32)
    gravity_delta = (np.float32(-325.0) * time).astype(np.float32)
    gravity_delta = (gravity_delta * time).astype(np.float32)
    predicted[:, 2] = (predicted[:, 2] + gravity_delta).astype(np.float32)

    lower = np.float32(91.25)
    span = np.float32(2044.0 - 182.5)
    # NumPy 1.26 promotes ``2 * np.float32(span)`` to float64.  Explicitly
    # retain that old scalar-promotion rule under the project's NumPy 2 runtime.
    period = 2.0 * float(span)
    phase = np.remainder(
        (predicted[:, 2] - lower).astype(np.float32).astype(np.float64),
        period,
    )
    predicted[:, 2] = np.where(
        phase > float(span),
        float(lower) + (period - phase),
        float(lower) + phase,
    ).astype(np.float32)

    delta = (predicted - car_position[rows, physical]).astype(np.float32)
    squared_components = (delta * delta).astype(np.float32)
    # OpenBLAS' n=3 SDOT path multiplies in float32 and accumulates the three
    # products in double precision on the pinned Windows source environment.
    squared = (
        squared_components[:, 0].astype(np.float64)
        + squared_components[:, 1].astype(np.float64)
        + squared_components[:, 2].astype(np.float64)
    )
    distance = np.sqrt(squared.astype(np.float32)).astype(np.float32)
    direction = (delta / distance[:, None]).astype(np.float32)
    products = (car_velocity[rows, physical] * direction).astype(np.float32)
    initial_velocity = (
        products[:, 0].astype(np.float64)
        + products[:, 1].astype(np.float64)
        + products[:, 2].astype(np.float64)
    ).astype(np.float32)
    # NumPy 1.26 promotes float32 minus a Python float to float64.
    target_distance = distance.astype(np.float64) - 136.875
    return initial_velocity, target_distance


def batched_eta(
    car_position: np.ndarray,
    car_velocity: np.ndarray,
    boost_amount: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    physical: np.ndarray,
    eta_cache: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the pinned two-pass ETA for one selected car per world."""

    count = physical.shape[0]
    rows = np.arange(count)
    physical = physical.astype(np.int64, copy=False)
    estimate = eta_cache[rows, physical].copy()
    trace_v0 = np.empty((count, 2), dtype=np.float32)
    trace_x0 = np.empty((count, 2), dtype=np.float64)
    trace_tick = np.empty((count, 2), dtype=np.int64)
    trace_time = np.empty((count, 2), dtype=np.float64)
    boost_duration = boost_amount[rows, physical].astype(np.float64) / 33.3
    for eta_pass in range(2):
        # Python int truncates toward zero; the source caps only the upper end.
        tick = np.minimum((estimate * 120.0).astype(np.int64), 599)
        initial_velocity, target_distance = _source_inputs(
            car_position,
            car_velocity,
            ball_position,
            ball_velocity,
            physical,
            tick,
        )
        estimate = _batched_linear_eta(
            initial_velocity,
            target_distance,
            boost_duration,
        )
        trace_v0[:, eta_pass] = initial_velocity
        trace_x0[:, eta_pass] = target_distance
        trace_tick[:, eta_pass] = tick
        trace_time[:, eta_pass] = estimate
    eta_cache[rows, physical] = estimate
    return estimate, trace_v0, trace_x0, trace_tick, trace_time


__all__ = ["batched_eta"]
