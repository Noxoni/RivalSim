"""Fast-aerial initiated, post-launch aerial-option learning primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE

PHASE_MOVING_INTERCEPT = 0
PHASE_GOAL_DIRECTED = 1
PHASE_NAMES = ("moving_intercept", "goal_directed")
FAST_AERIAL_FINAL_TICK = 28


def apply_fast_aerial_initiation(
    learned_action: torch.Tensor,
    option_age: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the authority-frozen option primitive and return its override mask."""

    if learned_action.ndim != 2 or learned_action.shape[1] != 8:
        raise ValueError("aerial option action must be [N,8]")
    if option_age.shape != (learned_action.shape[0],) or active.shape != option_age.shape:
        raise ValueError("aerial option age and active masks must align")
    emitted = learned_action.clone()
    hold = active & (option_age >= 0) & (option_age <= 23)
    release = active & (option_age >= 24) & (option_age <= 27)
    second = active & (option_age == FAST_AERIAL_FINAL_TICK)
    emitted[hold, 0] = 1.0
    emitted[hold, 2] = -1.0
    emitted[hold, 5] = 1.0
    emitted[hold, 6] = 1.0
    emitted[release, 0] = 1.0
    emitted[release, 2] = -1.0
    emitted[release, 5] = 0.0
    emitted[release, 6] = 1.0
    emitted[second] = 0.0
    emitted[second, 0] = 1.0
    emitted[second, 5] = 1.0
    emitted[second, 6] = 1.0
    return emitted, hold | release | second


@dataclass(slots=True)
class AerialTelemetryV2:
    episodes: int = 0
    launches: int = 0
    reached_150uu: int = 0
    reached_250uu: int = 0
    reached_350uu: int = 0
    elevated_contacts: int = 0
    high_contacts: int = 0
    forward_high_contacts: int = 0
    aerial_origin_goals: int = 0
    missed_windows: int = 0
    reward_sum: float = 0.0


class AerialRewardTrackerV2:
    """Physical post-launch progression with high-contact-only intercept success."""

    def __init__(self, worlds: int, *, attacker_side: int, phase: int):
        if worlds <= 0 or attacker_side not in (0, 1):
            raise ValueError("invalid V2 aerial tracker request")
        if phase not in range(len(PHASE_NAMES)):
            raise ValueError("invalid V2 aerial phase")
        self.worlds = int(worlds)
        self.attacker_side = int(attacker_side)
        self.phase = int(phase)
        self.initialized = False
        self.launched: torch.Tensor
        self.reached_150: torch.Tensor
        self.reached_250: torch.Tensor
        self.reached_350: torch.Tensor
        self.elevated_contact: torch.Tensor
        self.high_contact: torch.Tensor
        self.aerial_goal_paid: torch.Tensor
        self.positive_budget: torch.Tensor
        self.telemetry = AerialTelemetryV2(episodes=worlds)

    def _initialize(self, device: torch.device) -> None:
        self.launched = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.reached_150 = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.reached_250 = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.reached_350 = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.elevated_contact = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.high_contact = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.aerial_goal_paid = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.positive_budget = torch.zeros(self.worlds, device=device)
        self.initialized = True

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.attacker_side, FIELD[field]]

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"], dim=-1
        )

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("V2 aerial observations must align as [N,2,182]")
        if not self.initialized:
            self._initialize(before.device)
        ground_before = self._self(before, "self.on_ground") >= 0.5
        ground_after = self._self(after, "self.on_ground") >= 0.5
        launch = active & ground_before & ~ground_after & ~self.launched
        self.launched |= launch

        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        airborne = ~ground_after
        milestone_150 = active & airborne & (car_height >= 150.0) & ~self.reached_150
        milestone_250 = active & airborne & (car_height >= 250.0) & ~self.reached_250
        milestone_350 = active & airborne & (car_height >= 350.0) & ~self.reached_350
        self.reached_150 |= milestone_150
        self.reached_250 |= milestone_250
        self.reached_350 |= milestone_350

        scale = torch.as_tensor(POSITION_SCALE, device=before.device)
        relative_before = self._vector(before, "relative.ball_position") * scale
        relative_after = self._vector(after, "relative.ball_position") * scale
        distance_before = torch.linalg.vector_norm(relative_before, dim=-1)
        distance_after = torch.linalg.vector_norm(relative_after, dim=-1)
        distance_progress = (distance_before - distance_after).clamp(0.0, 40.0)
        vertical_progress = (
            relative_before[:, 2].abs() - relative_after[:, 2].abs()
        ).clamp(0.0, 30.0)
        direction_before = relative_before / distance_before[:, None].clamp_min(1.0)
        direction_after = relative_after / distance_after[:, None].clamp_min(1.0)
        alignment_before = (self._vector(before, "self.forward") * direction_before).sum(-1)
        alignment_after = (self._vector(after, "self.forward") * direction_after).sum(-1)
        alignment_progress = (alignment_after - alignment_before).clamp(0.0, 0.2)

        shaping = active & airborne & (tick > 28)
        reward = torch.where(shaping, distance_progress / 40.0 * 0.03, 0.0)
        reward += torch.where(shaping, vertical_progress / 30.0 * 0.02, 0.0)
        reward += torch.where(shaping, alignment_progress / 0.2 * 0.02, 0.0)
        reward += milestone_150.to(torch.float32) * 0.1
        reward += milestone_250.to(torch.float32) * 0.2
        reward += milestone_350.to(torch.float32) * 0.3

        touch = self._self(after, "lifecycle.self_touch_event") >= 0.5
        elevated = (
            active
            & touch
            & airborne
            & (car_height >= 150.0)
            & (ball_height >= 250.0)
            & ~self.elevated_contact
        )
        high = (
            active
            & touch
            & airborne
            & (car_height >= 300.0)
            & (ball_height >= 300.0)
            & ~self.high_contact
        )
        self.elevated_contact |= elevated
        self.high_contact |= high
        before_forward = (
            self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        after_forward = (
            self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        forward_transfer = (after_forward - before_forward).clamp(0.0, 1600.0)
        forward_high = high & (forward_transfer >= 150.0)
        aerial_goal = (
            active
            & goal_for_attacker
            & self.high_contact
            & ~self.aerial_goal_paid
        )
        self.aerial_goal_paid |= aerial_goal
        reward += elevated.to(torch.float32)
        reward += high.to(torch.float32) * 2.0
        reward += torch.where(elevated, forward_transfer / 1600.0, 0.0)
        reward += aerial_goal.to(torch.float32) * 5.0

        relanded_without_contact = (
            active & self.launched & ground_after & (tick > 40) & ~self.elevated_contact
        )
        missed_low_ball = active & (tick >= 45) & (ball_height < 210.0) & ~self.high_contact
        missed = relanded_without_contact | missed_low_ball
        success = high.clone() if self.phase == PHASE_MOVING_INTERCEPT else aerial_goal.clone()
        done = success | missed
        reward += missed.to(torch.float32) * -0.1
        positive = reward.clamp_min(0.0)
        remaining = (10.0 - self.positive_budget).clamp_min(0.0)
        paid = torch.minimum(positive, remaining)
        self.positive_budget += paid
        reward = torch.where(active, paid + reward.clamp_max(0.0), 0.0)

        self.telemetry.launches += int(launch.sum())
        self.telemetry.reached_150uu += int(milestone_150.sum())
        self.telemetry.reached_250uu += int(milestone_250.sum())
        self.telemetry.reached_350uu += int(milestone_350.sum())
        self.telemetry.elevated_contacts += int(elevated.sum())
        self.telemetry.high_contacts += int(high.sum())
        self.telemetry.forward_high_contacts += int(forward_high.sum())
        self.telemetry.aerial_origin_goals += int(aerial_goal.sum())
        self.telemetry.missed_windows += int(missed.sum())
        self.telemetry.reward_sum += float(reward.sum())
        return reward, done


__all__ = [
    "FAST_AERIAL_FINAL_TICK",
    "PHASE_GOAL_DIRECTED",
    "PHASE_MOVING_INTERCEPT",
    "PHASE_NAMES",
    "AerialRewardTrackerV2",
    "AerialTelemetryV2",
    "apply_fast_aerial_initiation",
]
