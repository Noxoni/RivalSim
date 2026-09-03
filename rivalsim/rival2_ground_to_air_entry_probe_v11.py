"""Read-only physical telemetry for the V11 ground-to-air entry families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_entry_v11 import (
    SETUP_NAMES,
    SETUP_RISING_DOUBLE_JUMP,
)
from rivalsim.rival2_ground_to_air_human_bridge_v11 import (
    HumanAerialEnvelopeConfig,
    human_envelope_features,
)
from rivalsim.rival2_ground_to_air_option import FIELD

GROUND_TO_AIR_ENTRY_PROBE_V11_VERSION = "RIVAL2_GROUND_TO_AIR_ENTRY_PROBE_V11"


def _distribution(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    selected = values[mask].detach().to(torch.float64).cpu()
    if selected.numel() == 0:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    quantiles = torch.quantile(
        selected,
        torch.tensor((0.1, 0.5, 0.9), dtype=torch.float64),
    )
    return {
        "count": int(selected.numel()),
        "minimum": float(selected.min()),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "maximum": float(selected.max()),
    }


@dataclass(frozen=True, slots=True)
class EntryProbeStepEvents:
    first_contact: torch.Tensor
    entry_airborne_contact: torch.Tensor
    human_envelope_reached: torch.Tensor
    second_airborne_contact: torch.Tensor
    goal_within_contact_budget: torch.Tensor
    contact_budget_exceeded: torch.Tensor
    ball_ground_failure: torch.Tensor


class GroundToAirEntryProbeV11:
    """Observe native contact onsets without changing reward, state, or action.

    For an already-rising double-jump feed, the first native airborne contact is
    the entry contact. For every other feed the first contact is the setup/carry
    contact and an airborne recontact must be separated by at least
    ``separation_ticks``. This route-aware distinction prevents the old
    low-touch detector from redefining a successful rising-ball entry.
    """

    def __init__(
        self,
        setup: torch.Tensor,
        *,
        attacker_side: int,
        envelope_config: HumanAerialEnvelopeConfig,
        continuation_ticks: int = 240,
        separation_ticks: int = 4,
        maximum_contacts: int = 6,
    ) -> None:
        if setup.ndim != 1 or attacker_side not in (0, 1):
            raise ValueError("invalid V11 entry-probe inputs")
        if torch.any((setup < 0) | (setup >= len(SETUP_NAMES))):
            raise ValueError("entry probe contains an unknown setup")
        if continuation_ticks <= 0 or separation_ticks <= 0 or maximum_contacts <= 0:
            raise ValueError("entry-probe timing and contact limits must be positive")
        envelope_config.validate()
        self.setup = setup.to(torch.int64)
        self.side = int(attacker_side)
        self.device = setup.device
        self.worlds = int(setup.numel())
        self.envelope_config = envelope_config
        self.continuation_ticks = int(continuation_ticks)
        self.separation_ticks = int(separation_ticks)
        self.maximum_contacts = int(maximum_contacts)
        self.initialized = False

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.side, FIELD[field]]

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"],
            dim=-1,
        )

    def _initialize(self, dtype: torch.dtype) -> None:
        self.first_contact_seen = torch.zeros(
            self.worlds, dtype=torch.bool, device=self.device
        )
        self.first_contact_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=self.device
        )
        self.first_contact_airborne = torch.zeros_like(self.first_contact_seen)
        self.entry_seen = torch.zeros_like(self.first_contact_seen)
        self.entry_tick = torch.full_like(self.first_contact_tick, -10_000)
        self.envelope_seen = torch.zeros_like(self.first_contact_seen)
        self.second_seen = torch.zeros_like(self.first_contact_seen)
        self.goal_seen = torch.zeros_like(self.first_contact_seen)
        self.budget_exceeded_seen = torch.zeros_like(self.first_contact_seen)
        self.ball_was_airborne = torch.zeros_like(self.first_contact_seen)
        self.ball_ground_failure_seen = torch.zeros_like(self.first_contact_seen)
        self.contact_count = torch.zeros(
            self.worlds, dtype=torch.int64, device=self.device
        )
        self.first_contact_ball_vertical_transfer = torch.zeros(
            self.worlds, dtype=dtype, device=self.device
        )
        self.first_contact_ball_goalward_transfer = torch.zeros_like(
            self.first_contact_ball_vertical_transfer
        )
        self.entry_car_height = torch.zeros_like(
            self.first_contact_ball_vertical_transfer
        )
        self.entry_ball_height = torch.zeros_like(self.entry_car_height)
        self.entry_car_vertical_speed = torch.zeros_like(self.entry_car_height)
        self.entry_ball_vertical_speed = torch.zeros_like(self.entry_car_height)
        self.entry_distance = torch.zeros_like(self.entry_car_height)
        self.maximum_car_height = torch.full_like(self.entry_car_height, -torch.inf)
        self.maximum_ball_height = torch.full_like(self.entry_car_height, -torch.inf)
        self.maximum_car_vertical_speed = torch.full_like(
            self.entry_car_height, -torch.inf
        )
        self.minimum_distance = torch.full_like(self.entry_car_height, torch.inf)
        self.initialized = True

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        active: torch.Tensor,
        goal_for_attacker: torch.Tensor | None = None,
    ) -> EntryProbeStepEvents:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("V11 entry-probe observations must be [N,2,182]")
        if active.shape != (self.worlds,):
            raise ValueError("V11 entry-probe active mask mismatch")
        if goal_for_attacker is None:
            goal_for_attacker = torch.zeros_like(active)
        if goal_for_attacker.shape != (self.worlds,):
            raise ValueError("V11 entry-probe goal mask mismatch")
        if not self.initialized:
            self._initialize(after.dtype)

        touch = active & (self._self(after, "lifecycle.self_touch_event") >= 0.5)
        airborne = self._self(after, "self.on_ground") < 0.5
        first_seen_before = self.first_contact_seen.clone()
        entry_seen_before = self.entry_seen.clone()
        first_contact = touch & ~first_seen_before
        self.first_contact_seen |= first_contact
        self.first_contact_tick.copy_(
            torch.where(
                first_contact,
                torch.full_like(self.first_contact_tick, int(tick)),
                self.first_contact_tick,
            )
        )
        self.first_contact_airborne |= first_contact & airborne
        self.contact_count += touch.to(torch.int64)

        before_ball_vertical = (
            self._self(before, "ball.linear_velocity.z") * BALL_LINEAR_SPEED_SCALE
        )
        after_ball_vertical = (
            self._self(after, "ball.linear_velocity.z") * BALL_LINEAR_SPEED_SCALE
        )
        before_ball_goalward = (
            self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        after_ball_goalward = (
            self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        self.first_contact_ball_vertical_transfer.copy_(
            torch.where(
                first_contact,
                after_ball_vertical - before_ball_vertical,
                self.first_contact_ball_vertical_transfer,
            )
        )
        self.first_contact_ball_goalward_transfer.copy_(
            torch.where(
                first_contact,
                after_ball_goalward - before_ball_goalward,
                self.first_contact_ball_goalward_transfer,
            )
        )

        rising_first_entry = (
            first_contact
            & airborne
            & (self.setup == SETUP_RISING_DOUBLE_JUMP)
        )
        separated_entry = (
            touch
            & airborne
            & first_seen_before
            & ~entry_seen_before
            & ((int(tick) - self.first_contact_tick) >= self.separation_ticks)
        )
        entry_contact = rising_first_entry | separated_entry
        self.entry_seen |= entry_contact
        self.entry_tick.copy_(
            torch.where(
                entry_contact,
                torch.full_like(self.entry_tick, int(tick)),
                self.entry_tick,
            )
        )

        relative = self._vector(after, "relative.ball_position") * torch.as_tensor(
            POSITION_SCALE, dtype=after.dtype, device=after.device
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        car_vertical_speed = (
            self._self(after, "self.linear_velocity.z") * CAR_LINEAR_SPEED_SCALE
        )
        for target, value in (
            (self.entry_car_height, car_height),
            (self.entry_ball_height, ball_height),
            (self.entry_car_vertical_speed, car_vertical_speed),
            (self.entry_ball_vertical_speed, after_ball_vertical),
            (self.entry_distance, distance),
        ):
            target.copy_(torch.where(entry_contact, value, target))

        age = int(tick) - self.entry_tick
        continuation = (
            active
            & self.entry_seen
            & (age >= 0)
            & (age <= self.continuation_ticks)
        )
        # Do not let a later ordinary bounce masquerade as aerial
        # continuation.  The 120 Hz scenario runner ends an attempt on this
        # event, matching the training-pack semantics requested by the user.
        self.ball_was_airborne |= continuation & (ball_height > 120.0)
        ball_ground_failure = (
            continuation
            & self.ball_was_airborne
            & (ball_height <= 100.0)
            & ~goal_for_attacker
            & ~self.ball_ground_failure_seen
        )
        self.ball_ground_failure_seen |= ball_ground_failure
        self.maximum_car_height = torch.where(
            continuation,
            torch.maximum(self.maximum_car_height, car_height),
            self.maximum_car_height,
        )
        self.maximum_ball_height = torch.where(
            continuation,
            torch.maximum(self.maximum_ball_height, ball_height),
            self.maximum_ball_height,
        )
        self.maximum_car_vertical_speed = torch.where(
            continuation,
            torch.maximum(self.maximum_car_vertical_speed, car_vertical_speed),
            self.maximum_car_vertical_speed,
        )
        self.minimum_distance = torch.where(
            continuation,
            torch.minimum(self.minimum_distance, distance),
            self.minimum_distance,
        )

        envelope = human_envelope_features(
            after,
            side=self.side,
            config=self.envelope_config,
        )["envelope"]
        envelope_event = continuation & envelope & ~self.envelope_seen
        self.envelope_seen |= envelope_event
        second_contact = (
            continuation
            & touch
            & airborne
            & entry_seen_before
            & (age >= self.separation_ticks)
            & ~self.second_seen
        )
        self.second_seen |= second_contact

        budget_event = (
            active
            & (self.contact_count > self.maximum_contacts)
            & ~self.budget_exceeded_seen
        )
        self.budget_exceeded_seen |= budget_event
        goal_event = (
            active
            & goal_for_attacker
            & self.entry_seen
            & (self.contact_count <= self.maximum_contacts)
            & ~self.goal_seen
        )
        self.goal_seen |= goal_event
        return EntryProbeStepEvents(
            first_contact=first_contact,
            entry_airborne_contact=entry_contact,
            human_envelope_reached=envelope_event,
            second_airborne_contact=second_contact,
            goal_within_contact_budget=goal_event,
            contact_budget_exceeded=budget_event,
            ball_ground_failure=ball_ground_failure,
        )

    def telemetry(self) -> dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("V11 entry probe has no observations")
        continued = self.entry_seen & torch.isfinite(self.minimum_distance)
        return {
            "identity": GROUND_TO_AIR_ENTRY_PROBE_V11_VERSION,
            "worlds": self.worlds,
            "continuation_ticks": self.continuation_ticks,
            "separation_ticks": self.separation_ticks,
            "maximum_contacts": self.maximum_contacts,
            "fractions": {
                "first_contact": float(
                    self.first_contact_seen.to(torch.float32).mean().cpu()
                ),
                "first_contact_airborne": float(
                    self.first_contact_airborne.to(torch.float32).mean().cpu()
                ),
                "entry_airborne_contact": float(
                    self.entry_seen.to(torch.float32).mean().cpu()
                ),
                "human_envelope_reached": float(
                    self.envelope_seen.to(torch.float32).mean().cpu()
                ),
                "second_airborne_contact": float(
                    self.second_seen.to(torch.float32).mean().cpu()
                ),
                "goal_within_contact_budget": float(
                    self.goal_seen.to(torch.float32).mean().cpu()
                ),
                "contact_budget_exceeded": float(
                    self.budget_exceeded_seen.to(torch.float32).mean().cpu()
                ),
                "ball_ground_failure": float(
                    self.ball_ground_failure_seen.to(torch.float32).mean().cpu()
                ),
            },
            "first_contact": {
                "ball_vertical_transfer_uu_per_second": _distribution(
                    self.first_contact_ball_vertical_transfer,
                    self.first_contact_seen,
                ),
                "ball_goalward_transfer_uu_per_second": _distribution(
                    self.first_contact_ball_goalward_transfer,
                    self.first_contact_seen,
                ),
            },
            "entry_contact": {
                "car_height_uu": _distribution(
                    self.entry_car_height, self.entry_seen
                ),
                "ball_height_uu": _distribution(
                    self.entry_ball_height, self.entry_seen
                ),
                "car_vertical_speed_uu_per_second": _distribution(
                    self.entry_car_vertical_speed, self.entry_seen
                ),
                "ball_vertical_speed_uu_per_second": _distribution(
                    self.entry_ball_vertical_speed, self.entry_seen
                ),
                "distance_uu": _distribution(self.entry_distance, self.entry_seen),
            },
            "continuation": {
                "maximum_car_height_uu": _distribution(
                    self.maximum_car_height, continued
                ),
                "maximum_ball_height_uu": _distribution(
                    self.maximum_ball_height, continued
                ),
                "maximum_car_vertical_speed_uu_per_second": _distribution(
                    self.maximum_car_vertical_speed, continued
                ),
                "minimum_distance_uu": _distribution(
                    self.minimum_distance, continued
                ),
            },
        }


__all__ = [
    "GROUND_TO_AIR_ENTRY_PROBE_V11_VERSION",
    "EntryProbeStepEvents",
    "GroundToAirEntryProbeV11",
]
