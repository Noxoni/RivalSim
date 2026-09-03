"""Natural-match router and physical reward for the protected aerial scorer.

This module is training/deployment infrastructure.  It does not change the
production Gameplay 120 V2 reward and it does not classify named mechanics.
The router consumes only the ordinary 182-field policy observation.  Its
training reward is caused by authoritative contact, goal, and ball-motion
events; raw airtime is deliberately worth zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE

GROUND_TO_AIR_SELFPLAY_V12_VERSION = "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12"

ROUTE_NONE = -1
ROUTE_ASSISTED_LOW_BOUNCE = 0
ROUTE_SOFT_INCOMING_CHIP = 1
ROUTE_RISING_DOUBLE_JUMP = 2
ROUTE_ROOF_CARRY = 3
ROUTE_NAMES = (
    "assisted_low_bounce",
    "soft_incoming_chip",
    "rising_double_jump",
    "roof_carry",
)


@dataclass(frozen=True, slots=True)
class AerialOptionRouterConfig:
    minimum_boost_fraction: float = 0.15
    minimum_forward_alignment: float = 0.55
    minimum_canonical_ball_y_uu: float = -3_000.0
    ownership_margin_uu: float = 200.0
    maximum_planar_distance_uu: float = 480.0
    maximum_option_ticks: int = 420
    minimum_landing_release_tick: int = 72
    cooldown_ticks: int = 90
    maximum_distinct_contacts: int = 6
    contact_separation_ticks: int = 4
    grounded_ball_height_uu: float = 100.0
    airborne_ball_height_uu: float = 130.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_boost_fraction <= 1.0:
            raise ValueError("minimum boost fraction must be in [0,1]")
        if not -1.0 <= self.minimum_forward_alignment <= 1.0:
            raise ValueError("minimum forward alignment must be in [-1,1]")
        if self.ownership_margin_uu < 0.0 or self.maximum_planar_distance_uu <= 0.0:
            raise ValueError("distance gates must be positive")
        if self.maximum_option_ticks <= self.minimum_landing_release_tick:
            raise ValueError("option timeout must follow the landing-release gate")
        if self.cooldown_ticks < 0 or self.maximum_distinct_contacts < 2:
            raise ValueError("invalid lifecycle configuration")
        if self.contact_separation_ticks <= 0:
            raise ValueError("contact separation must be positive")
        if self.grounded_ball_height_uu >= self.airborne_ball_height_uu:
            raise ValueError("airborne threshold must exceed grounded threshold")


@dataclass(frozen=True, slots=True)
class AerialSelfPlayRewardConfig:
    raw_airtime_reward: float = 0.0
    entry_airborne_contact_event: float = 4.0
    high_airborne_contact_event: float = 2.0
    second_goalward_airborne_contact_event: float = 5.0
    third_goalward_airborne_contact_event: float = 4.0
    fourth_goalward_airborne_contact_event: float = 2.0
    fifth_goalward_airborne_contact_event: float = 1.0
    productive_goalward_contact_event: float = 2.0
    minimum_productive_goalward_speed_uu_per_second: float = 600.0
    goalward_speed_neutral_uu_per_second: float = 500.0
    goalward_speed_at_contact_per_uu_per_second: float = 0.003
    forward_velocity_transfer_per_uu_per_second: float = 0.004
    goal_within_contact_budget_event: float = 25.0
    maximum_supplemental_reward_per_attempt: float = 40.0

    def __post_init__(self) -> None:
        if self.raw_airtime_reward != 0.0:
            raise ValueError("raw airtime must remain unrewarded")
        values = (
            self.entry_airborne_contact_event,
            self.high_airborne_contact_event,
            self.second_goalward_airborne_contact_event,
            self.third_goalward_airborne_contact_event,
            self.fourth_goalward_airborne_contact_event,
            self.fifth_goalward_airborne_contact_event,
            self.productive_goalward_contact_event,
            self.minimum_productive_goalward_speed_uu_per_second,
            self.goal_within_contact_budget_event,
            self.maximum_supplemental_reward_per_attempt,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("event rewards and physical thresholds must be positive")


@dataclass(frozen=True, slots=True)
class AerialRouteEligibility:
    eligible: torch.Tensor
    route: torch.Tensor
    ball_height_uu: torch.Tensor
    ball_vertical_speed_uu_per_second: torch.Tensor
    planar_distance_uu: torch.Tensor
    opponent_ball_distance_uu: torch.Tensor
    forward_alignment: torch.Tensor


@dataclass(frozen=True, slots=True)
class AerialRouteSelection:
    active: torch.Tensor
    activated: torch.Tensor
    released: torch.Tensor
    route: torch.Tensor
    eligibility: AerialRouteEligibility


@dataclass(frozen=True, slots=True)
class AerialRouteOutcome:
    supplemental_reward: torch.Tensor
    contact: torch.Tensor
    entry_airborne_contact: torch.Tensor
    high_airborne_contact: torch.Tensor
    second_airborne_contact: torch.Tensor
    productive_goalward_contact: torch.Tensor
    goal_within_contact_budget: torch.Tensor
    contact_budget_exceeded: torch.Tensor
    ball_ground_failure: torch.Tensor


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack(
        [observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1
    )


def aerial_route_eligibility(
    observation: torch.Tensor,
    config: AerialOptionRouterConfig,
) -> AerialRouteEligibility:
    """Classify broad V11-derived entry opportunities from policy-visible state."""

    if observation.ndim != 2 or observation.shape[1] != 182:
        raise ValueError("aerial route expects [N,182] observations")
    scale = torch.as_tensor(
        POSITION_SCALE, dtype=observation.dtype, device=observation.device
    )
    relative = _vector(observation, "relative.ball_position") * scale
    planar = torch.linalg.vector_norm(relative[:, :2], dim=-1)
    planar_direction = relative[:, :2] / planar[:, None].clamp_min(1.0e-6)
    forward = _vector(observation, "self.forward")
    forward = forward / torch.linalg.vector_norm(
        forward, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = (forward[:, :2] * planar_direction).sum(dim=-1)
    ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
    ball_y = observation[:, FIELD["ball.position.y"]] * POSITION_SCALE[1]
    ball_vertical_speed = (
        observation[:, FIELD["ball.linear_velocity.z"]] * BALL_LINEAR_SPEED_SCALE
    )
    ball_position = _vector(observation, "ball.position") * scale
    opponent_position = _vector(observation, "opponent.position") * scale
    # Possession is a ground-entry routing decision, so compare planar races.
    # Including the ball's height would make an opponent parked directly under
    # a rising ball appear artificially far away.
    opponent_ball_distance = torch.linalg.vector_norm(
        (ball_position - opponent_position)[:, :2], dim=-1
    )
    finite = torch.isfinite(observation).all(dim=1)
    common = (
        finite
        & (observation[:, FIELD["self.on_ground"]] >= 0.5)
        & (observation[:, FIELD["self.is_demoed"]] < 0.5)
        & (observation[:, FIELD["self.boost"]] >= config.minimum_boost_fraction)
        & (ball_y >= config.minimum_canonical_ball_y_uu)
        & (planar >= 5.0)
        & (planar <= config.maximum_planar_distance_uu)
        & (alignment >= config.minimum_forward_alignment)
        & (planar <= opponent_ball_distance + config.ownership_margin_uu)
    )

    relative_z = relative[:, 2]
    roof = (
        common
        & (ball_height >= 118.0)
        & (ball_height <= 155.0)
        & (planar <= 150.0)
        & (relative_z >= 80.0)
        & (relative_z <= 155.0)
    )
    rising = (
        common
        & ~roof
        & (ball_height >= 125.0)
        & (ball_height <= 320.0)
        & (ball_vertical_speed >= 75.0)
        & (planar <= 480.0)
    )
    low_bounce = (
        common
        & ~roof
        & ~rising
        & (ball_height >= 105.0)
        & (ball_height <= 225.0)
        & (ball_vertical_speed >= 25.0)
        & (planar <= 430.0)
    )
    chip = (
        common
        & ~roof
        & ~rising
        & ~low_bounce
        & (ball_height >= 90.0)
        & (ball_height <= 125.0)
        & (ball_vertical_speed >= -100.0)
        & (ball_vertical_speed <= 100.0)
        & (planar >= 100.0)
        & (planar <= 430.0)
    )
    route = torch.full(
        (observation.shape[0],),
        ROUTE_NONE,
        dtype=torch.int64,
        device=observation.device,
    )
    route.masked_fill_(low_bounce, ROUTE_ASSISTED_LOW_BOUNCE)
    route.masked_fill_(chip, ROUTE_SOFT_INCOMING_CHIP)
    route.masked_fill_(rising, ROUTE_RISING_DOUBLE_JUMP)
    route.masked_fill_(roof, ROUTE_ROOF_CARRY)
    return AerialRouteEligibility(
        eligible=route >= 0,
        route=route,
        ball_height_uu=ball_height,
        ball_vertical_speed_uu_per_second=ball_vertical_speed,
        planar_distance_uu=planar,
        opponent_ball_distance_uu=opponent_ball_distance,
        forward_alignment=alignment,
    )


class AerialOptionSelfPlayRouter:
    """Persistent direct-policy option latch for flattened player lanes."""

    def __init__(
        self,
        lanes: int,
        *,
        device: str | torch.device,
        router_config: AerialOptionRouterConfig | None = None,
        reward_config: AerialSelfPlayRewardConfig | None = None,
    ) -> None:
        if lanes <= 0:
            raise ValueError("router lanes must be positive")
        self.lanes = int(lanes)
        self.device = torch.device(device)
        self.config = router_config or AerialOptionRouterConfig()
        self.reward_config = reward_config or AerialSelfPlayRewardConfig()
        self.active = torch.zeros(lanes, dtype=torch.bool, device=self.device)
        self.route = torch.full(
            (lanes,), ROUTE_NONE, dtype=torch.int64, device=self.device
        )
        self.age = torch.zeros(lanes, dtype=torch.int64, device=self.device)
        self.cooldown = torch.zeros_like(self.age)
        self.contact_count = torch.zeros_like(self.age)
        self.air_contact_count = torch.zeros_like(self.age)
        self.last_air_contact_tick = torch.full_like(self.age, -10_000)
        self.entry_seen = torch.zeros_like(self.active)
        self.high_seen = torch.zeros_like(self.active)
        self.ever_airborne_car = torch.zeros_like(self.active)
        self.ever_airborne_ball = torch.zeros_like(self.active)
        self.supplemental_paid = torch.zeros(
            lanes, dtype=torch.float32, device=self.device
        )
        self.total_tick = 0
        names = (
            "activations",
            "active_ticks",
            "contacts",
            "entry_airborne_contacts",
            "high_airborne_contacts",
            "second_airborne_contacts",
            "productive_goalward_contacts",
            "goals_within_contact_budget",
            "contact_budget_exceeded",
            "ball_ground_failures",
            "released_reset",
            "released_grounded",
            "released_ball_ground",
            "released_contact_budget",
            "released_timeout",
        )
        self.counters = {
            name: torch.zeros((), dtype=torch.int64, device=self.device)
            for name in names
        }
        self.route_activations = torch.zeros(
            len(ROUTE_NAMES), dtype=torch.int64, device=self.device
        )
        self.reward_sum = torch.zeros((), dtype=torch.float64, device=self.device)

    def _clear(self, mask: torch.Tensor) -> None:
        self.active &= ~mask
        self.route.masked_fill_(mask, ROUTE_NONE)
        self.age.masked_fill_(mask, 0)
        self.contact_count.masked_fill_(mask, 0)
        self.air_contact_count.masked_fill_(mask, 0)
        self.last_air_contact_tick.masked_fill_(mask, -10_000)
        self.entry_seen &= ~mask
        self.high_seen &= ~mask
        self.ever_airborne_car &= ~mask
        self.ever_airborne_ball &= ~mask
        self.supplemental_paid.masked_fill_(mask, 0.0)

    def select(
        self,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> AerialRouteSelection:
        if observation.shape != (self.lanes, 182):
            raise ValueError("router observation shape mismatch")
        if kickoff_active.shape != (self.lanes,) or match_done.shape != (self.lanes,):
            raise ValueError("router lifecycle mask shape mismatch")
        kickoff_active = kickoff_active.to(torch.bool)
        match_done = match_done.to(torch.bool)
        self.cooldown.sub_(1).clamp_min_(0)
        reset_release = self.active & (kickoff_active | match_done)
        grounded_release = (
            self.active
            & self.ever_airborne_car
            & (self.age >= self.config.minimum_landing_release_tick)
            & (observation[:, FIELD["self.on_ground"]] >= 0.5)
        )
        ball_ground_release = (
            self.active
            & self.ever_airborne_ball
            & (
                observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
                <= self.config.grounded_ball_height_uu
            )
        )
        budget_release = self.active & (
            self.contact_count > self.config.maximum_distinct_contacts
        )
        timeout_release = self.active & (self.age >= self.config.maximum_option_ticks)
        grounded_release &= ~reset_release
        ball_ground_release &= ~(reset_release | grounded_release)
        budget_release &= ~(reset_release | grounded_release | ball_ground_release)
        timeout_release &= ~(
            reset_release | grounded_release | ball_ground_release | budget_release
        )
        released = (
            reset_release
            | grounded_release
            | ball_ground_release
            | budget_release
            | timeout_release
        )
        self.counters["released_reset"] += reset_release.sum()
        self.counters["released_grounded"] += grounded_release.sum()
        self.counters["released_ball_ground"] += ball_ground_release.sum()
        self.counters["released_contact_budget"] += budget_release.sum()
        self.counters["released_timeout"] += timeout_release.sum()
        self.cooldown.copy_(
            torch.where(
                released & ~reset_release,
                torch.full_like(self.cooldown, self.config.cooldown_ticks),
                self.cooldown,
            )
        )
        self.cooldown.masked_fill_(reset_release, 0)
        self._clear(released)

        eligibility = aerial_route_eligibility(observation, self.config)
        activated = (
            ~self.active
            & (self.cooldown == 0)
            & ~kickoff_active
            & ~match_done
            & eligibility.eligible
        )
        self.active |= activated
        self.route.copy_(torch.where(activated, eligibility.route, self.route))
        self.age.masked_fill_(activated, 0)
        self.counters["activations"] += activated.sum()
        for route_id in range(len(ROUTE_NAMES)):
            self.route_activations[route_id] += (
                activated & (eligibility.route == route_id)
            ).sum()
        self.counters["active_ticks"] += self.active.sum()
        return AerialRouteSelection(
            active=self.active.clone(),
            activated=activated,
            released=released,
            route=self.route.clone(),
            eligibility=eligibility,
        )

    def observe(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        active_before: torch.Tensor,
        goal_for_lane: torch.Tensor,
    ) -> AerialRouteOutcome:
        if before.shape != after.shape or before.shape != (self.lanes, 182):
            raise ValueError("router transition shape mismatch")
        if active_before.shape != (self.lanes,) or goal_for_lane.shape != (self.lanes,):
            raise ValueError("router outcome mask shape mismatch")
        active_before = active_before.to(torch.bool)
        goal_for_lane = goal_for_lane.to(torch.bool)
        car_airborne = after[:, FIELD["self.on_ground"]] < 0.5
        ball_height = after[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        self.ever_airborne_car |= active_before & car_airborne
        self.ever_airborne_ball |= active_before & (
            ball_height >= self.config.airborne_ball_height_uu
        )
        contact = active_before & (
            after[:, FIELD["lifecycle.self_touch_event"]] >= 0.5
        )
        self.contact_count += contact.to(torch.int64)
        separated = contact & (
            self.total_tick - self.last_air_contact_tick
            >= self.config.contact_separation_ticks
        )
        airborne_contact = separated & car_airborne & (ball_height >= 125.0)
        entry = airborne_contact & ~self.entry_seen
        previous_air_count = self.air_contact_count.clone()
        self.air_contact_count += airborne_contact.to(torch.int64)
        second = airborne_contact & (previous_air_count == 1)
        third = airborne_contact & (previous_air_count == 2)
        fourth = airborne_contact & (previous_air_count == 3)
        fifth = airborne_contact & (previous_air_count == 4)
        self.last_air_contact_tick.copy_(
            torch.where(
                airborne_contact,
                torch.full_like(self.last_air_contact_tick, self.total_tick),
                self.last_air_contact_tick,
            )
        )
        self.entry_seen |= entry
        high = (
            airborne_contact
            & ~self.high_seen
            & (after[:, FIELD["self.position.z"]] * POSITION_SCALE[2] >= 300.0)
            & (ball_height >= 300.0)
        )
        self.high_seen |= high
        before_forward = (
            before[:, FIELD["ball.linear_velocity.y"]] * BALL_LINEAR_SPEED_SCALE
        )
        after_forward = (
            after[:, FIELD["ball.linear_velocity.y"]] * BALL_LINEAR_SPEED_SCALE
        )
        transfer = after_forward - before_forward
        within_budget = self.contact_count <= self.config.maximum_distinct_contacts
        eligible_air_contact = airborne_contact & self.entry_seen & within_budget
        productive = eligible_air_contact & (
            after_forward
            >= self.reward_config.minimum_productive_goalward_speed_uu_per_second
        )
        goal = goal_for_lane & active_before & self.entry_seen & within_budget
        budget_exceeded = contact & ~within_budget
        ball_ground_failure = (
            active_before
            & self.ever_airborne_ball
            & (ball_height <= self.config.grounded_ball_height_uu)
            & ~goal
        )

        reward = torch.zeros(
            self.lanes, dtype=torch.float32, device=self.device
        )
        cfg = self.reward_config
        reward += entry * cfg.entry_airborne_contact_event
        reward += high * cfg.high_airborne_contact_event
        reward += (second & productive) * cfg.second_goalward_airborne_contact_event
        reward += (third & productive) * cfg.third_goalward_airborne_contact_event
        reward += (fourth & productive) * cfg.fourth_goalward_airborne_contact_event
        reward += (fifth & productive) * cfg.fifth_goalward_airborne_contact_event
        reward += productive * cfg.productive_goalward_contact_event
        reward += eligible_air_contact * transfer.clamp(-1_000.0, 1_500.0) * (
            cfg.forward_velocity_transfer_per_uu_per_second
        )
        reward += eligible_air_contact * (
            after_forward - cfg.goalward_speed_neutral_uu_per_second
        ).clamp(-600.0, 1_400.0) * cfg.goalward_speed_at_contact_per_uu_per_second
        reward += goal * cfg.goal_within_contact_budget_event
        remaining = (
            cfg.maximum_supplemental_reward_per_attempt - self.supplemental_paid
        ).clamp_min(0.0)
        positive = torch.minimum(reward.clamp_min(0.0), remaining)
        reward = torch.where(reward > 0.0, positive, reward)
        self.supplemental_paid += positive
        self.reward_sum += reward.sum(dtype=torch.float64)

        for name, mask in (
            ("contacts", contact),
            ("entry_airborne_contacts", entry),
            ("high_airborne_contacts", high),
            ("second_airborne_contacts", second),
            ("productive_goalward_contacts", productive),
            ("goals_within_contact_budget", goal),
            ("contact_budget_exceeded", budget_exceeded),
            ("ball_ground_failures", ball_ground_failure),
        ):
            self.counters[name] += mask.sum()
        self.age += active_before.to(torch.int64)
        self.total_tick += 1
        return AerialRouteOutcome(
            supplemental_reward=reward,
            contact=contact,
            entry_airborne_contact=entry,
            high_airborne_contact=high,
            second_airborne_contact=second,
            productive_goalward_contact=productive,
            goal_within_contact_budget=goal,
            contact_budget_exceeded=budget_exceeded,
            ball_ground_failure=ball_ground_failure,
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "version": GROUND_TO_AIR_SELFPLAY_V12_VERSION,
            "counters": {
                name: int(value.item()) for name, value in self.counters.items()
            },
            "route_activations": {
                name: int(self.route_activations[index].item())
                for index, name in enumerate(ROUTE_NAMES)
            },
            "supplemental_reward_sum": float(self.reward_sum.item()),
            "active_lanes": int(self.active.sum().item()),
        }


__all__ = [
    "GROUND_TO_AIR_SELFPLAY_V12_VERSION",
    "ROUTE_NAMES",
    "AerialOptionRouterConfig",
    "AerialOptionSelfPlayRouter",
    "AerialRouteEligibility",
    "AerialRouteOutcome",
    "AerialRouteSelection",
    "AerialSelfPlayRewardConfig",
    "aerial_route_eligibility",
]
