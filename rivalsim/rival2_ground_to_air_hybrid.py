"""Natural-play gate for the controlled ground-to-air scoring option.

The isolated option was validated on low, goalward-moving balls in the
attacking goal area.  This wrapper preserves the original controller byte for
byte and only permits a new latch inside that validated state envelope.  Once
latched, the option is allowed to finish or release under its own lifecycle
rules even if the ball subsequently leaves the entry envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import (
    GroundToAirConfig,
    GroundToAirController,
    GroundToAirStep,
    ground_to_air_eligibility,
)

NATURAL_GROUND_TO_AIR_GATE_VERSION = "RIVAL2_NATURAL_GROUND_TO_AIR_GATE_V1"


@dataclass(frozen=True, slots=True)
class NaturalGroundToAirGateConfig:
    minimum_goalward_ball_y_uu: float = 3_000.0
    maximum_goalward_ball_y_uu: float = 4_400.0
    minimum_goalward_ball_speed_uu_per_second: float = 100.0
    maximum_goalward_ball_speed_uu_per_second: float = 900.0

    def __post_init__(self) -> None:
        if self.minimum_goalward_ball_y_uu >= self.maximum_goalward_ball_y_uu:
            raise ValueError("natural ground-to-air goalward position gate is empty")
        if (
            self.minimum_goalward_ball_speed_uu_per_second
            >= self.maximum_goalward_ball_speed_uu_per_second
        ):
            raise ValueError("natural ground-to-air speed gate is empty")


@dataclass(frozen=True, slots=True)
class NaturalGroundToAirEligibility:
    eligible: torch.Tensor
    physical_eligible: torch.Tensor
    ball_goalward_y_uu: torch.Tensor
    ball_goalward_speed_uu_per_second: torch.Tensor


def natural_ground_to_air_eligibility(
    observation: torch.Tensor,
    *,
    option_config: GroundToAirConfig,
    gate_config: NaturalGroundToAirGateConfig,
) -> NaturalGroundToAirEligibility:
    physical = ground_to_air_eligibility(observation, option_config).eligible
    ball_y = observation[:, FIELD["ball.position.y"]] * POSITION_SCALE[1]
    ball_speed = observation[:, FIELD["ball.linear_velocity.y"]] * BALL_LINEAR_SPEED_SCALE
    eligible = (
        physical
        & (ball_y >= gate_config.minimum_goalward_ball_y_uu)
        & (ball_y <= gate_config.maximum_goalward_ball_y_uu)
        & (ball_speed >= gate_config.minimum_goalward_ball_speed_uu_per_second)
        & (ball_speed <= gate_config.maximum_goalward_ball_speed_uu_per_second)
    )
    return NaturalGroundToAirEligibility(
        eligible=eligible,
        physical_eligible=physical,
        ball_goalward_y_uu=ball_y,
        ball_goalward_speed_uu_per_second=ball_speed,
    )


class NaturalGroundToAirController:
    """Block out-of-envelope activation while preserving active-option state."""

    def __init__(
        self,
        worlds: int,
        *,
        device: str | torch.device,
        option_config: GroundToAirConfig,
        gate_config: NaturalGroundToAirGateConfig,
    ) -> None:
        self.inner = GroundToAirController(
            worlds,
            device=device,
            config=option_config,
        )
        self.option_config = option_config
        self.gate_config = gate_config

    @property
    def active(self) -> torch.Tensor:
        return self.inner.active

    def step(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> tuple[GroundToAirStep, NaturalGroundToAirEligibility]:
        eligibility = natural_ground_to_air_eligibility(
            observation,
            option_config=self.option_config,
            gate_config=self.gate_config,
        )
        # Reuse the established reset input only while inactive.  Active
        # options retain their original low-ball/ground/timeout lifecycle.
        blocked_activation = ~self.inner.active & ~eligibility.eligible
        step = self.inner.step(
            base_action,
            observation,
            kickoff_active=kickoff_active | blocked_activation,
            match_done=match_done,
        )
        return step, eligibility

    def telemetry(self) -> dict[str, object]:
        telemetry = self.inner.telemetry()
        telemetry["natural_gate_version"] = NATURAL_GROUND_TO_AIR_GATE_VERSION
        telemetry["natural_gate_config"] = {
            "minimum_goalward_ball_y_uu": self.gate_config.minimum_goalward_ball_y_uu,
            "maximum_goalward_ball_y_uu": self.gate_config.maximum_goalward_ball_y_uu,
            "minimum_goalward_ball_speed_uu_per_second": (
                self.gate_config.minimum_goalward_ball_speed_uu_per_second
            ),
            "maximum_goalward_ball_speed_uu_per_second": (
                self.gate_config.maximum_goalward_ball_speed_uu_per_second
            ),
        }
        return telemetry


__all__ = [
    "NATURAL_GROUND_TO_AIR_GATE_VERSION",
    "NaturalGroundToAirController",
    "NaturalGroundToAirEligibility",
    "NaturalGroundToAirGateConfig",
    "natural_ground_to_air_eligibility",
]
