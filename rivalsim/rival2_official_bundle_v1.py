"""Single-artifact modular Rival 2 capability controller.

The official artifact preserves each learned policy byte-for-byte inside one
checkpoint.  A small deterministic, observation-only router chooses between
the protected side-specific V23 base, the V3 aerial scorer, and the
side-specific capability-curriculum policy.  It does not average or otherwise
blend incompatible neural-network parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rivalsim.rival2_contracts import (
    CAR_LINEAR_SPEED_SCALE,
    OBS_DIM,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

OFFICIAL_BUNDLE_V1_FORMAT = "RIVAL2_OFFICIAL_CAPABILITY_CHECKPOINT_V1"

MODE_BASE = 0
MODE_AERIAL = 1
MODE_RECOVERY = 2
MODE_DEMO = 3
MODE_NAMES = ("base", "aerial", "recovery", "offensive_demo")

FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


@dataclass(frozen=True, slots=True)
class OfficialCapabilityRouterConfigV1:
    """Conservative physical gates for learned specialist invocation."""

    automatic_aerial_enabled: bool = False
    automatic_recovery_enabled: bool = False
    automatic_offensive_demo_enabled: bool = False
    aerial_maximum_ticks: int = 240
    aerial_release_grounded_after_ticks: int = 24
    aerial_minimum_car_height_uu: float = 95.0
    aerial_maximum_car_height_uu: float = 900.0
    aerial_minimum_ball_height_uu: float = 250.0
    aerial_maximum_ball_height_uu: float = 1200.0
    aerial_minimum_distance_uu: float = 90.0
    aerial_maximum_distance_uu: float = 700.0
    aerial_minimum_relative_ball_y_uu: float = -100.0
    aerial_minimum_boost_fraction: float = 0.10
    aerial_minimum_canonical_ball_y_uu: float = 300.0
    aerial_minimum_opponent_ball_distance_uu: float = 500.0
    pop_minimum_ball_height_uu: float = 100.0
    pop_maximum_ball_height_uu: float = 260.0
    pop_minimum_planar_distance_uu: float = 90.0
    pop_maximum_planar_distance_uu: float = 260.0
    pop_minimum_forward_alignment: float = 0.72
    pop_minimum_boost_fraction: float = 0.40
    pop_minimum_canonical_ball_y_uu: float = 1200.0
    pop_minimum_opponent_ball_distance_uu: float = 800.0
    recovery_maximum_ticks: int = 36
    recovery_minimum_speed_uu_per_second: float = 700.0
    recovery_maximum_car_height_uu: float = 150.0
    recovery_maximum_vertical_speed_uu_per_second: float = -90.0
    recovery_minimum_ball_distance_uu: float = 800.0
    recovery_wall_minimum_abs_x_uu: float = 3800.0
    recovery_wall_minimum_abs_up_x: float = 0.65
    demo_maximum_ticks: int = 90
    demo_minimum_canonical_ball_y_uu: float = 500.0
    demo_minimum_canonical_self_y_uu: float = 0.0
    demo_minimum_opponent_forward_uu: float = 220.0
    demo_maximum_opponent_forward_uu: float = 1200.0
    demo_maximum_opponent_lateral_uu: float = 375.0
    demo_maximum_opponent_distance_uu: float = 1250.0
    demo_minimum_ball_forward_uu: float = 150.0
    specialist_cooldown_ticks: int = 90

    def validate(self) -> None:
        integer_values = (
            self.aerial_maximum_ticks,
            self.aerial_release_grounded_after_ticks,
            self.recovery_maximum_ticks,
            self.demo_maximum_ticks,
            self.specialist_cooldown_ticks,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("official router tick limits must be positive")
        if self.aerial_minimum_distance_uu >= self.aerial_maximum_distance_uu:
            raise ValueError("invalid aerial distance window")
        if self.pop_minimum_planar_distance_uu >= self.pop_maximum_planar_distance_uu:
            raise ValueError("invalid pop distance window")
        if not 0.0 <= self.pop_minimum_forward_alignment <= 1.0:
            raise ValueError("invalid pop alignment")


@dataclass(frozen=True, slots=True)
class OfficialRouterSelection:
    mode: torch.Tensor
    activated: torch.Tensor
    released: torch.Tensor


class OfficialCapabilityRouterV1:
    """Stateful, reset-aware policy router operating on canonical observations."""

    def __init__(
        self,
        lanes: int,
        *,
        device: torch.device | str,
        config: OfficialCapabilityRouterConfigV1 | None = None,
    ) -> None:
        if lanes <= 0:
            raise ValueError("official router requires at least one lane")
        self.lanes = int(lanes)
        self.device = torch.device(device)
        self.config = config or OfficialCapabilityRouterConfigV1()
        self.config.validate()
        self.mode = torch.zeros(self.lanes, dtype=torch.int64, device=self.device)
        self.mode_ticks = torch.zeros_like(self.mode)
        self.cooldown = torch.zeros_like(self.mode)
        self.activations = torch.zeros(len(MODE_NAMES), dtype=torch.int64)
        self.active_ticks = torch.zeros_like(self.activations)
        self.releases = torch.zeros_like(self.activations)

    @staticmethod
    def _scalar(observation: torch.Tensor, name: str) -> torch.Tensor:
        return observation[:, FIELD[name]]

    @classmethod
    def _vector(cls, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [cls._scalar(observation, f"{prefix}.{axis}") for axis in "xyz"], dim=-1
        )

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self.mode.zero_()
            self.mode_ticks.zero_()
            self.cooldown.zero_()
            return
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.lanes,):
            raise ValueError("official router reset mask shape mismatch")
        self.mode.masked_fill_(mask, MODE_BASE)
        self.mode_ticks.masked_fill_(mask, 0)
        self.cooldown.masked_fill_(mask, 0)

    def select(
        self,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> OfficialRouterSelection:
        if observation.shape != (self.lanes, OBS_DIM):
            raise ValueError("official router observation shape mismatch")
        kickoff = kickoff_active.to(device=self.device, dtype=torch.bool)
        done = match_done.to(device=self.device, dtype=torch.bool)
        if kickoff.shape != (self.lanes,) or done.shape != (self.lanes,):
            raise ValueError("official router lifecycle mask shape mismatch")
        reset = kickoff | done
        previous_mode = self.mode.clone()
        self.reset(reset)

        active = self.mode != MODE_BASE
        self.mode_ticks.add_(active.to(torch.int64))
        self.cooldown.sub_(1).clamp_min_(0)

        on_ground = self._scalar(observation, "self.on_ground") >= 0.5
        self_demoed = self._scalar(observation, "self.is_demoed") >= 0.5
        opponent_demoed_event = (
            self._scalar(observation, "lifecycle.opponent_demoed_event") >= 0.5
        )
        ball_position = self._vector(observation, "ball.position") * torch.as_tensor(
            POSITION_SCALE, device=self.device
        )
        self_position = self._vector(observation, "self.position") * torch.as_tensor(
            POSITION_SCALE, device=self.device
        )
        self_velocity = self._vector(
            observation, "self.linear_velocity"
        ) * CAR_LINEAR_SPEED_SCALE
        self_forward = self._vector(observation, "self.forward")
        self_up = self._vector(observation, "self.up")
        relative_ball = self._vector(
            observation, "relative.ball_position"
        ) * torch.as_tensor(POSITION_SCALE, device=self.device)
        relative_opponent = self._vector(
            observation, "relative.opponent_position"
        ) * torch.as_tensor(POSITION_SCALE, device=self.device)
        opponent_position = self_position + relative_opponent
        ball_distance = torch.linalg.vector_norm(relative_ball, dim=-1)
        ball_planar_distance = torch.linalg.vector_norm(relative_ball[:, :2], dim=-1)
        opponent_distance = torch.linalg.vector_norm(relative_opponent, dim=-1)
        opponent_ball_distance = torch.linalg.vector_norm(
            ball_position - opponent_position, dim=-1
        )
        speed = torch.linalg.vector_norm(self_velocity, dim=-1)
        forward_planar = self_forward[:, :2]
        forward_planar = forward_planar / torch.linalg.vector_norm(
            forward_planar, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        ball_planar_direction = relative_ball[:, :2] / ball_planar_distance[:, None].clamp_min(
            1.0e-6
        )
        forward_alignment = (forward_planar * ball_planar_direction).sum(dim=-1)
        boost = self._scalar(observation, "self.boost")

        release_aerial = (self.mode == MODE_AERIAL) & (
            (self.mode_ticks >= self.config.aerial_maximum_ticks)
            | (
                on_ground
                & (self.mode_ticks >= self.config.aerial_release_grounded_after_ticks)
            )
            | (ball_position[:, 2] < 85.0)
            | (ball_distance > 1500.0)
            | self_demoed
        )
        release_recovery = (self.mode == MODE_RECOVERY) & (
            (self.mode_ticks >= self.config.recovery_maximum_ticks)
            | (on_ground & (self.mode_ticks >= 6))
            | self_demoed
        )
        release_demo = (self.mode == MODE_DEMO) & (
            (self.mode_ticks >= self.config.demo_maximum_ticks)
            | opponent_demoed_event
            | (relative_opponent[:, 1] < -250.0)
            | (opponent_distance > 1800.0)
            | self_demoed
        )
        release = release_aerial | release_recovery | release_demo
        for mode in (MODE_AERIAL, MODE_RECOVERY, MODE_DEMO):
            self.releases[mode] += int(((self.mode == mode) & release).sum())
        self.mode.masked_fill_(release, MODE_BASE)
        self.mode_ticks.masked_fill_(release, 0)
        self.cooldown.copy_(
            torch.where(
                release,
                torch.full_like(self.cooldown, self.config.specialist_cooldown_ticks),
                self.cooldown,
            )
        )

        available = (self.mode == MODE_BASE) & (self.cooldown == 0) & ~reset & ~self_demoed
        airborne_aerial = (
            available
            & ~on_ground
            & (self_position[:, 2] >= self.config.aerial_minimum_car_height_uu)
            & (self_position[:, 2] <= self.config.aerial_maximum_car_height_uu)
            & (ball_position[:, 2] >= self.config.aerial_minimum_ball_height_uu)
            & (ball_position[:, 2] <= self.config.aerial_maximum_ball_height_uu)
            & (ball_distance >= self.config.aerial_minimum_distance_uu)
            & (ball_distance <= self.config.aerial_maximum_distance_uu)
            & (relative_ball[:, 1] >= self.config.aerial_minimum_relative_ball_y_uu)
            & (boost >= self.config.aerial_minimum_boost_fraction)
            & (ball_position[:, 1] >= self.config.aerial_minimum_canonical_ball_y_uu)
            & (
                opponent_ball_distance
                >= self.config.aerial_minimum_opponent_ball_distance_uu
            )
        )
        pop_aerial = (
            available
            & on_ground
            & (ball_position[:, 2] >= self.config.pop_minimum_ball_height_uu)
            & (ball_position[:, 2] <= self.config.pop_maximum_ball_height_uu)
            & (ball_planar_distance >= self.config.pop_minimum_planar_distance_uu)
            & (ball_planar_distance <= self.config.pop_maximum_planar_distance_uu)
            & (relative_ball[:, 1] > 0.0)
            & (forward_alignment >= self.config.pop_minimum_forward_alignment)
            & (boost >= self.config.pop_minimum_boost_fraction)
            & (ball_position[:, 1] >= self.config.pop_minimum_canonical_ball_y_uu)
            & (opponent_ball_distance >= self.config.pop_minimum_opponent_ball_distance_uu)
        )
        activate_aerial = self.config.automatic_aerial_enabled & (
            airborne_aerial | pop_aerial
        )

        wall_context = (
            (self_position[:, 0].abs() >= self.config.recovery_wall_minimum_abs_x_uu)
            & (self_up[:, 0].abs() >= self.config.recovery_wall_minimum_abs_up_x)
        )
        floor_context = (
            (self_position[:, 2] <= self.config.recovery_maximum_car_height_uu)
            & (
                self_velocity[:, 2]
                <= self.config.recovery_maximum_vertical_speed_uu_per_second
            )
        )
        activate_recovery = self.config.automatic_recovery_enabled & (
            available
            & ~activate_aerial
            & ~on_ground
            & (speed >= self.config.recovery_minimum_speed_uu_per_second)
            & (ball_distance >= self.config.recovery_minimum_ball_distance_uu)
            & (floor_context | wall_context)
        )

        activate_demo = self.config.automatic_offensive_demo_enabled & (
            available
            & ~activate_aerial
            & ~activate_recovery
            & on_ground
            & (self._scalar(observation, "self.is_supersonic") >= 0.5)
            & (ball_position[:, 1] >= self.config.demo_minimum_canonical_ball_y_uu)
            & (self_position[:, 1] >= self.config.demo_minimum_canonical_self_y_uu)
            & (
                relative_opponent[:, 1]
                >= self.config.demo_minimum_opponent_forward_uu
            )
            & (
                relative_opponent[:, 1]
                <= self.config.demo_maximum_opponent_forward_uu
            )
            & (
                relative_opponent[:, 0].abs()
                <= self.config.demo_maximum_opponent_lateral_uu
            )
            & (opponent_distance <= self.config.demo_maximum_opponent_distance_uu)
            & (relative_ball[:, 1] >= self.config.demo_minimum_ball_forward_uu)
            & (self._scalar(observation, "opponent.is_demoed") < 0.5)
        )

        activated = activate_aerial | activate_recovery | activate_demo
        self.mode.copy_(
            torch.where(
                activate_aerial,
                torch.full_like(self.mode, MODE_AERIAL),
                torch.where(
                    activate_recovery,
                    torch.full_like(self.mode, MODE_RECOVERY),
                    torch.where(
                        activate_demo,
                        torch.full_like(self.mode, MODE_DEMO),
                        self.mode,
                    ),
                ),
            )
        )
        self.mode_ticks.masked_fill_(activated, 0)
        for mode in (MODE_AERIAL, MODE_RECOVERY, MODE_DEMO):
            self.activations[mode] += int(((self.mode == mode) & activated).sum())
            self.active_ticks[mode] += int((self.mode == mode).sum())
        released = (previous_mode != MODE_BASE) & (self.mode == MODE_BASE)
        return OfficialRouterSelection(self.mode.clone(), activated, released)

    def telemetry(self) -> dict[str, Any]:
        return {
            "format": "RIVAL2_OFFICIAL_CAPABILITY_ROUTER_V1_TELEMETRY",
            "config": asdict(self.config),
            "active_lanes": int((self.mode != MODE_BASE).sum()),
            "activations": {
                MODE_NAMES[index]: int(self.activations[index])
                for index in range(len(MODE_NAMES))
            },
            "active_ticks": {
                MODE_NAMES[index]: int(self.active_ticks[index])
                for index in range(len(MODE_NAMES))
            },
            "releases": {
                MODE_NAMES[index]: int(self.releases[index])
                for index in range(len(MODE_NAMES))
            },
        }


def _build_model(component: dict[str, Any], device: torch.device) -> Rival2ActorCritic:
    config = Rival2PolicyConfig(**component["policy_config"])
    if component.get("policy_config_hash") != config.content_hash:
        raise RuntimeError("official component policy configuration hash mismatch")
    model = Rival2ActorCritic(config).to(device)
    model.load_state_dict(component["model"], strict=True)
    model.eval().requires_grad_(False)
    return model


class Rival2OfficialControllerV1:
    """Inference-only controller for one loaded official checkpoint."""

    def __init__(
        self,
        payload: dict[str, Any],
        lanes: int,
        *,
        device: torch.device | str,
    ) -> None:
        if payload.get("format") != OFFICIAL_BUNDLE_V1_FORMAT:
            raise ValueError("unsupported official Rival checkpoint format")
        self.device = torch.device(device)
        self.components = {
            name: _build_model(component, self.device)
            for name, component in payload["components"].items()
        }
        required = {
            "base_blue",
            "base_orange",
            "aerial",
            "capability_blue",
            "capability_orange",
        }
        if set(self.components) != required:
            raise ValueError("official Rival checkpoint component set mismatch")
        config = OfficialCapabilityRouterConfigV1(**payload["router_config"])
        self.router = OfficialCapabilityRouterV1(lanes, device=self.device, config=config)

    def action(
        self,
        observation: torch.Tensor,
        side: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> tuple[torch.Tensor, OfficialRouterSelection]:
        observation = observation.to(self.device)
        side = side.to(device=self.device, dtype=torch.int64)
        if observation.shape != (self.router.lanes, OBS_DIM):
            raise ValueError("official controller observation shape mismatch")
        if side.shape != (self.router.lanes,) or bool(((side < 0) | (side > 1)).any()):
            raise ValueError("official controller side shape/value mismatch")
        with torch.inference_mode():
            actors = {
                name: model(observation)[0] for name, model in self.components.items()
            }
            blue = deterministic_hybrid_action(actors["base_blue"])
            orange = deterministic_hybrid_action(actors["base_orange"])
            base = torch.where((side == 1)[:, None], orange, blue)
            cap_blue = deterministic_hybrid_action(actors["capability_blue"])
            cap_orange = deterministic_hybrid_action(actors["capability_orange"])
            capability = torch.where((side == 1)[:, None], cap_orange, cap_blue)
            aerial = deterministic_hybrid_action(actors["aerial"])
            selection = self.router.select(
                observation,
                kickoff_active=kickoff_active,
                match_done=match_done,
            )
            action = torch.where(
                (selection.mode == MODE_AERIAL)[:, None],
                aerial,
                torch.where(
                    ((selection.mode == MODE_RECOVERY) | (selection.mode == MODE_DEMO))[
                        :, None
                    ],
                    capability,
                    base,
                ),
            )
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("official controller produced nonfinite actions")
        return action, selection


def load_official_checkpoint(
    path: str | Path,
    lanes: int,
    *,
    device: torch.device | str,
) -> tuple[dict[str, Any], Rival2OfficialControllerV1]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload, Rival2OfficialControllerV1(payload, lanes, device=device)


__all__ = [
    "MODE_AERIAL",
    "MODE_BASE",
    "MODE_DEMO",
    "MODE_NAMES",
    "MODE_RECOVERY",
    "OFFICIAL_BUNDLE_V1_FORMAT",
    "OfficialCapabilityRouterConfigV1",
    "OfficialCapabilityRouterV1",
    "OfficialRouterSelection",
    "Rival2OfficialControllerV1",
    "load_official_checkpoint",
]
