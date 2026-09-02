"""Goal-directed low-ball pop scenarios and literal physical outcome tracking.

The module is training/evaluation infrastructure.  It pays no reward, defines
no named mechanic, and does not alter Rival's production reward.  A successful
goal is attributed only after a low pop has been followed by an elevated car
contact, so a rolling or unassisted shot cannot satisfy the aerial-chain gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.state import StateSnapshot

GROUND_TO_AIR_GOAL_V3_VERSION = "RIVAL2_GROUND_TO_AIR_GOAL_V3"
PHASE_EASY_FINISH = 0
PHASE_ATTACKING_HALF = 1
PHASE_NAMES = ("easy_finish", "attacking_half")


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


def build_goal_directed_pop_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    phase: int,
) -> StateSnapshot:
    """Create a self-pop opportunity whose physical continuation can score."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid goal-directed pop scenario request")
    if phase not in range(len(PHASE_NAMES)):
        raise ValueError("invalid goal-directed pop phase")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    if phase == PHASE_EASY_FINISH:
        ball_y = (3_450.0, 4_000.0)
        ball_speed = (350.0, 750.0)
        ball_x = (-700.0, 700.0)
    else:
        ball_y = (2_350.0, 3_600.0)
        ball_speed = (500.0, 1_000.0)
        ball_x = (-1_050.0, 1_050.0)

    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    forward = _yaw_quat(sign * math.pi / 2.0)
    reverse = _yaw_quat(-sign * math.pi / 2.0)
    for world in range(worlds):
        x = float(rng.uniform(*ball_x))
        y = float(rng.uniform(*ball_y))
        following_distance = float(rng.uniform(12.0, 45.0))
        lateral_offset = float(rng.uniform(-22.0, 22.0))
        forward_speed = float(rng.uniform(*ball_speed))
        state.ball_pos[world] = (
            x,
            sign * y,
            float(rng.uniform(142.0, 168.0)),
        )
        state.ball_vel[world] = (
            float(rng.uniform(-45.0, 45.0)),
            sign * forward_speed,
            float(rng.uniform(-105.0, -20.0)),
        )
        state.car_pos[world, attacker_side] = (
            x + lateral_offset,
            sign * (y - following_distance),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-30.0, 30.0)),
            sign * float(forward_speed + rng.uniform(-60.0, 110.0)),
            0.0,
        )
        state.car_pos[world, other] = (
            float(rng.uniform(-900.0, 900.0)),
            -sign * float(rng.uniform(3_800.0, 4_500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
    state.validate()
    return state


@dataclass(slots=True)
class GoalDirectedTelemetry:
    attempts: int = 0
    low_pop_touches: int = 0
    elevated_follow_touches: int = 0
    high_follow_touches: int = 0
    second_airborne_touches: int = 0
    third_or_later_airborne_touches: int = 0
    goals_after_elevated_follow: int = 0
    unassisted_or_ground_goals: int = 0
    goalward_velocity_contacts: int = 0
    goalward_velocity_transfer_sum: float = 0.0


class GoalDirectedTracker:
    """Track a causal pop, elevated contacts, repeated contacts, and a goal."""

    def __init__(self, worlds: int, *, attacker_side: int, horizon: int) -> None:
        if worlds <= 0 or attacker_side not in (0, 1) or horizon <= 0:
            raise ValueError("invalid goal-directed tracker request")
        self.worlds = int(worlds)
        self.side = int(attacker_side)
        self.horizon = int(horizon)
        self.initialized = False
        self.telemetry = GoalDirectedTelemetry(attempts=worlds)

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.side, FIELD[field]]

    def _initialize(self, device: torch.device) -> None:
        self.pop_seen = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.elevated_seen = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.high_seen = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.air_touch_count = torch.zeros(self.worlds, dtype=torch.int64, device=device)
        self.last_air_touch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.initialized = True

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        any_goal: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("goal-directed observations must align as [N,2,182]")
        if not self.initialized:
            self._initialize(before.device)
        touch = active & (self._self(after, "lifecycle.self_touch_event") >= 0.5)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height_before = self._self(before, "ball.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        low_pop = touch & ~self.pop_seen & (ball_height_before <= 205.0) & (car_height <= 150.0)
        self.pop_seen |= low_pop
        separated_air_touch = touch & ((tick - self.last_air_touch_tick) >= 4)
        elevated = (
            separated_air_touch
            & self.pop_seen
            & ~low_pop
            & (car_height >= 150.0)
            & (ball_height >= 250.0)
        )
        first_elevated = elevated & ~self.elevated_seen
        high = (
            elevated
            & ~self.high_seen
            & (car_height >= 300.0)
            & (ball_height >= 300.0)
        )
        previous_count = self.air_touch_count.clone()
        self.air_touch_count += elevated.to(torch.int64)
        second = elevated & (previous_count == 1)
        third_or_later = elevated & (previous_count >= 2)
        self.elevated_seen |= elevated
        self.high_seen |= high
        self.last_air_touch_tick.copy_(
            torch.where(
                elevated,
                torch.full_like(self.last_air_touch_tick, tick),
                self.last_air_touch_tick,
            )
        )

        sign = 1.0 if self.side == 0 else -1.0
        before_forward = (
            self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE * sign
        )
        after_forward = (
            self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE * sign
        )
        forward_transfer = after_forward - before_forward
        goalward_contact = elevated & (forward_transfer >= 100.0)
        goal_after_follow = active & goal_for_attacker & self.elevated_seen
        other_goal = active & any_goal & ~goal_after_follow
        done = any_goal | (active & (tick >= self.horizon - 1))

        self.telemetry.low_pop_touches += int(low_pop.sum())
        self.telemetry.elevated_follow_touches += int(first_elevated.sum())
        self.telemetry.high_follow_touches += int(high.sum())
        self.telemetry.second_airborne_touches += int(second.sum())
        self.telemetry.third_or_later_airborne_touches += int(third_or_later.sum())
        self.telemetry.goals_after_elevated_follow += int(goal_after_follow.sum())
        self.telemetry.unassisted_or_ground_goals += int(other_goal.sum())
        self.telemetry.goalward_velocity_contacts += int(goalward_contact.sum())
        self.telemetry.goalward_velocity_transfer_sum += float(
            torch.where(elevated, forward_transfer.clamp(-2_000.0, 2_000.0), 0.0).sum()
        )
        return {
            "low_pop": low_pop,
            "first_elevated": first_elevated,
            "high": high,
            "second": second,
            "third_or_later": third_or_later,
            "goalward_contact": goalward_contact,
            "forward_transfer": forward_transfer,
            "goal_after_follow": goal_after_follow,
            "other_goal": other_goal,
            "done": done,
        }


__all__ = [
    "GROUND_TO_AIR_GOAL_V3_VERSION",
    "PHASE_ATTACKING_HALF",
    "PHASE_EASY_FINISH",
    "PHASE_NAMES",
    "GoalDirectedTelemetry",
    "GoalDirectedTracker",
    "build_goal_directed_pop_scenarios",
]
