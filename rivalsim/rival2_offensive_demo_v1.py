"""Physical scenarios and telemetry for useful offensive demolitions.

The module does not modify the production reward.  It separates the literal
authoritative demolition event from the useful offensive outcome that follows:
recovering possession, advancing the ball, or scoring while the defender is
removed.  Rival observations are team-normalized, so positive observation-Y is
goalward for both perspectives; applying a second orange sign here would be an
error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_contracts import OBS_FIELD_NAMES, POSITION_SCALE
from rivalsim.state import StateSnapshot

OFFENSIVE_DEMO_V1_VERSION = "RIVAL2_OFFENSIVE_DEMO_V1"

ROUTE_RECOVER_POSSESSION = 0
ROUTE_OPEN_GOAL = 1
ROUTE_NAMES = ("recover_possession", "open_goal")
DEFAULT_ROUTE_MIX = (0.65, 0.35)

FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class OffensiveDemoScenarioBatch:
    state: StateSnapshot
    route: np.ndarray
    attacker_side: int
    scripted_action: np.ndarray
    initial_defender_lateral_offset_uu: np.ndarray


def build_offensive_demo_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    route: int | None = None,
    route_mix: tuple[float, float] = DEFAULT_ROUTE_MIX,
) -> OffensiveDemoScenarioBatch:
    """Build side-symmetric routes requiring an off-axis defensive removal.

    ``recover_possession`` places the defender between Rival and a reachable
    loose ball. ``open_goal`` places the ball between Rival and a goal-side
    defender so clearing that defender can convert an already moving attack.
    Rival starts goalward rather than aimed at the off-axis defender; the policy
    must deliberately leave the direct ball lane and recover afterward.
    """

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid offensive-demo scenario request")
    if route is not None and route not in range(len(ROUTE_NAMES)):
        raise ValueError("unknown offensive-demo route")
    weights = np.asarray(route_mix, dtype=np.float64)
    if (
        weights.shape != (len(ROUTE_NAMES),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("route mix must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    forward = _yaw_quat(sign * math.pi / 2.0)
    selected = (
        np.full(worlds, route, dtype=np.int32)
        if route is not None
        else rng.choice(
            len(ROUTE_NAMES),
            size=worlds,
            replace=True,
            p=weights / weights.sum(),
        ).astype(np.int32)
    )

    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(0.0)
    scripted_action = np.zeros((worlds, 2, 8), dtype=np.float32)
    lateral_offsets = np.zeros(worlds, dtype=np.float32)

    for world, selected_route in enumerate(selected.tolist()):
        lane_x = float(rng.uniform(-1_000.0, 1_000.0))
        defender_side = -1.0 if rng.random() < 0.5 else 1.0
        defender_offset = defender_side * float(rng.uniform(250.0, 550.0))
        lateral_offsets[world] = defender_offset

        if selected_route == ROUTE_RECOVER_POSSESSION:
            attacker_y = float(rng.uniform(-1_100.0, 1_700.0))
            defender_y = attacker_y + float(rng.uniform(550.0, 950.0))
            ball_y = defender_y + float(rng.uniform(400.0, 850.0))
            ball_speed = float(rng.uniform(80.0, 380.0))
        else:
            attacker_y = float(rng.uniform(900.0, 2_650.0))
            ball_y = attacker_y + float(rng.uniform(300.0, 620.0))
            defender_y = ball_y + float(rng.uniform(300.0, 700.0))
            ball_speed = float(rng.uniform(500.0, 1_050.0))

        state.car_pos[world, attacker_side] = (
            lane_x,
            sign * attacker_y,
            17.0,
        )
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-35.0, 35.0)),
            sign * float(rng.uniform(2_200.0, 2_290.0)),
            0.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.boost[world, attacker_side] = float(rng.uniform(20.0, 75.0))
        state.is_supersonic[world, attacker_side] = 1
        state.supersonic_time[world, attacker_side] = float(rng.uniform(0.4, 1.2))

        state.car_pos[world, other] = (
            lane_x + defender_offset,
            sign * defender_y,
            17.0,
        )
        state.car_vel[world, other] = (
            -defender_side * float(rng.uniform(0.0, 120.0)),
            sign * float(rng.uniform(0.0, 350.0)),
            0.0,
        )
        state.car_quat[world, other] = forward
        state.boost[world, other] = float(rng.uniform(0.0, 40.0))
        # The defender keeps moving but receives no steering script. A later
        # authority may replace this with a frozen policy without changing the
        # physical route definition.
        scripted_action[world, other, 0] = 0.35

        state.ball_pos[world] = (
            lane_x + float(rng.uniform(-90.0, 90.0)),
            sign * ball_y,
            93.15,
        )
        state.ball_vel[world] = (
            float(rng.uniform(-70.0, 70.0)),
            sign * ball_speed,
            0.0,
        )
        state.ball_ang_vel[world, 0] = float(rng.uniform(-1.0, 1.0))

    state.validate()
    return OffensiveDemoScenarioBatch(
        state=state,
        route=selected,
        attacker_side=attacker_side,
        scripted_action=scripted_action,
        initial_defender_lateral_offset_uu=lateral_offsets,
    )


@dataclass(frozen=True, slots=True)
class OffensiveDemoStepEvents:
    opponent_distance_gain_uu: torch.Tensor
    actual_demo: torch.Tensor
    offensive_context_demo: torch.Tensor
    post_demo_touch: torch.Tensor
    post_demo_goalward_progress: torch.Tensor
    post_demo_goal: torch.Tensor
    expired_without_conversion: torch.Tensor


@dataclass(slots=True)
class OffensiveDemoTelemetry:
    actual_demos: int = 0
    offensive_context_demos: int = 0
    post_demo_touches: int = 0
    post_demo_goalward_progress: int = 0
    post_demo_goals: int = 0
    expired_without_conversion: int = 0


class OffensiveDemoOutcomeTracker:
    """Track authoritative demolitions and their bounded offensive conversion."""

    def __init__(
        self,
        route: torch.Tensor,
        *,
        attacker_side: int,
        followup_window_ticks: int = 3 * 120,
        minimum_goalward_progress_uu: float = 300.0,
    ):
        if route.ndim != 1 or attacker_side not in (0, 1):
            raise ValueError("invalid offensive-demo tracker inputs")
        if followup_window_ticks <= 0:
            raise ValueError("follow-up window must be positive")
        if minimum_goalward_progress_uu <= 0.0 or not math.isfinite(
            minimum_goalward_progress_uu
        ):
            raise ValueError("goalward progress threshold must be finite and positive")
        if torch.any((route < 0) | (route >= len(ROUTE_NAMES))):
            raise ValueError("tracker route contains an unknown value")

        self.route = route.to(torch.int64)
        self.attacker_side = int(attacker_side)
        self.device = route.device
        self.followup_window_ticks = int(followup_window_ticks)
        self.minimum_goalward_progress_uu = float(minimum_goalward_progress_uu)
        count = int(route.numel())
        self.demo_seen = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.demo_tick = torch.full(
            (count,), -10_000, dtype=torch.int64, device=self.device
        )
        self.demo_ball_goalward_y = torch.zeros(
            count, dtype=torch.float32, device=self.device
        )
        self.touch_paid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.touch_after_demo = torch.zeros(
            count, dtype=torch.bool, device=self.device
        )
        self.progress_paid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.goal_paid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.expiry_counted = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.telemetry = OffensiveDemoTelemetry()

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.attacker_side, FIELD[field]]

    def _position(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        scale = torch.as_tensor(
            POSITION_SCALE, dtype=torch.float32, device=self.device
        )
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"],
            dim=-1,
        ) * scale

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> OffensiveDemoStepEvents:
        if before.shape != after.shape or before.ndim != 3:
            raise ValueError("offensive-demo observations must align as [N,2,182]")
        count = before.shape[0]
        if count != self.route.numel() or goal_for_attacker.shape != (count,):
            raise ValueError("offensive-demo tracker batch mismatch")
        if active is None:
            active = torch.ones(count, dtype=torch.bool, device=self.device)
        if active.shape != (count,):
            raise ValueError("offensive-demo active mask mismatch")

        # Rival observations are already team-normalized. Positive Y is toward
        # the opponent goal for both agents, so no attacker-side sign belongs
        # in these tests or calculations.
        self_before = self._position(before, "self.position")
        self_after = self._position(after, "self.position")
        opponent_before = self._position(before, "opponent.position")
        opponent_after = self._position(after, "opponent.position")
        ball_before = self._position(before, "ball.position")
        ball_after = self._position(after, "ball.position")

        before_distance = torch.linalg.vector_norm(
            opponent_before - self_before, dim=-1
        )
        after_distance = torch.linalg.vector_norm(
            opponent_after - self_after, dim=-1
        )
        distance_gain = (before_distance - after_distance).clamp(0.0, 40.0)

        opponent_ahead = (opponent_before[:, 1] - self_before[:, 1]) >= 100.0
        ball_ahead = (ball_before[:, 1] - self_before[:, 1]) >= 50.0
        ball_reachable = torch.linalg.vector_norm(
            ball_before - self_before, dim=-1
        ) <= 2_800.0
        defender_before_loose_ball = (
            self.route == ROUTE_RECOVER_POSSESSION
        ) & (ball_before[:, 1] >= opponent_before[:, 1] - 250.0)
        defender_goal_side_of_ball = (self.route == ROUTE_OPEN_GOAL) & (
            opponent_before[:, 1] >= ball_before[:, 1] - 250.0
        )
        offensive_context = (
            active
            & opponent_ahead
            & ball_ahead
            & ball_reachable
            & (defender_before_loose_ball | defender_goal_side_of_ball)
        )

        demo_signal = (
            self._self(after, "lifecycle.opponent_demoed_event") >= 0.5
        )
        actual_demo = active & demo_signal & ~self.demo_seen
        offensive_demo = actual_demo & offensive_context
        self.demo_seen |= actual_demo
        self.demo_tick.copy_(
            torch.where(
                offensive_demo,
                torch.full_like(self.demo_tick, int(tick)),
                self.demo_tick,
            )
        )
        self.demo_ball_goalward_y.copy_(
            torch.where(
                offensive_demo,
                ball_after[:, 1],
                self.demo_ball_goalward_y,
            )
        )

        ticks_after_demo = tick - self.demo_tick
        pending = (
            active
            & (ticks_after_demo >= 1)
            & (ticks_after_demo <= self.followup_window_ticks)
        )
        touch_signal = self._self(after, "lifecycle.self_touch_event") >= 0.5
        post_touch = pending & touch_signal & ~self.touch_paid
        self.touch_after_demo |= post_touch
        self.touch_paid |= post_touch

        progress_since_demo = ball_after[:, 1] - self.demo_ball_goalward_y
        progress_context = self.touch_after_demo | (
            self.route == ROUTE_OPEN_GOAL
        )
        post_progress = (
            pending
            & progress_context
            & (progress_since_demo >= self.minimum_goalward_progress_uu)
            & ~self.progress_paid
        )
        self.progress_paid |= post_progress
        post_goal = pending & goal_for_attacker & ~self.goal_paid
        self.goal_paid |= post_goal

        converted = self.touch_paid | self.progress_paid | self.goal_paid
        expired = (
            active
            & (self.demo_tick > -10_000)
            & (ticks_after_demo == self.followup_window_ticks + 1)
            & ~converted
            & ~self.expiry_counted
        )
        self.expiry_counted |= expired

        self.telemetry.actual_demos += int(actual_demo.sum())
        self.telemetry.offensive_context_demos += int(offensive_demo.sum())
        self.telemetry.post_demo_touches += int(post_touch.sum())
        self.telemetry.post_demo_goalward_progress += int(post_progress.sum())
        self.telemetry.post_demo_goals += int(post_goal.sum())
        self.telemetry.expired_without_conversion += int(expired.sum())
        return OffensiveDemoStepEvents(
            opponent_distance_gain_uu=torch.where(
                offensive_context, distance_gain, torch.zeros_like(distance_gain)
            ),
            actual_demo=actual_demo,
            offensive_context_demo=offensive_demo,
            post_demo_touch=post_touch,
            post_demo_goalward_progress=post_progress,
            post_demo_goal=post_goal,
            expired_without_conversion=expired,
        )


__all__ = [
    "DEFAULT_ROUTE_MIX",
    "FIELD",
    "OFFENSIVE_DEMO_V1_VERSION",
    "ROUTE_NAMES",
    "ROUTE_OPEN_GOAL",
    "ROUTE_RECOVER_POSSESSION",
    "OffensiveDemoOutcomeTracker",
    "OffensiveDemoScenarioBatch",
    "OffensiveDemoStepEvents",
    "OffensiveDemoTelemetry",
    "build_offensive_demo_scenarios",
]
