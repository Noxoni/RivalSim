"""Read-only diagnostics between a prompt airborne touch and elevated control.

This module deliberately owns no reward.  It observes native 120 Hz contact
onsets and measures the physical continuation after the first separated,
airborne follow touch.  The purpose is to expose the credit gap between a
natural lift and the existing strict elevated-contact gate before a later
training authority is frozen.
"""

from __future__ import annotations

from typing import Any

import torch

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_option import FIELD
from rivalsim.rival2_ground_to_air_touch_geometry import (
    NaturalAerialTouchGeometryProbe,
)

GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_VERSION = (
    "RIVAL2_GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_V1"
)


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


class PromptContinuationProbe:
    """Measure physical continuation after a natural airborne recontact.

    A prompt touch follows the existing geometry contract: a separated native
    touch onset by the attacking car, while airborne, no more than 60 ticks
    after the low setup contact.  All continuation measurements are bounded to
    ``continuation_ticks`` after that event.  No tick is synthesized and no
    state or action is changed.
    """

    def __init__(
        self,
        worlds: int,
        *,
        attacker_side: int,
        continuation_ticks: int = 120,
        close_shell_uu: float = 220.0,
    ) -> None:
        if (
            worlds <= 0
            or attacker_side not in (0, 1)
            or continuation_ticks <= 0
            or close_shell_uu <= 0.0
        ):
            raise ValueError("invalid prompt-continuation probe request")
        self.worlds = int(worlds)
        self.side = int(attacker_side)
        self.continuation_ticks = int(continuation_ticks)
        self.close_shell_uu = float(close_shell_uu)
        self.geometry = NaturalAerialTouchGeometryProbe(
            worlds,
            attacker_side=attacker_side,
        )
        self.initialized = False

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.side, FIELD[field]]

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"],
            dim=-1,
        )

    def _initialize(self, device: torch.device, dtype: torch.dtype) -> None:
        self.prompt_tick = torch.full(
            (self.worlds,),
            -10_000,
            dtype=torch.int64,
            device=device,
        )
        self.last_post_prompt_touch_tick = torch.full_like(
            self.prompt_tick,
            -10_000,
        )
        self.last_observed_tick = -10_000
        self.second_recontact_seen = torch.zeros(
            self.worlds,
            dtype=torch.bool,
            device=device,
        )
        self.bridge_elevated_seen = torch.zeros_like(self.second_recontact_seen)
        self.bridge_high_seen = torch.zeros_like(self.second_recontact_seen)
        self.close_rising_ticks = torch.zeros(
            self.worlds,
            dtype=torch.int64,
            device=device,
        )
        self.close_goalward_ticks = torch.zeros_like(self.close_rising_ticks)
        self.current_close_streak = torch.zeros_like(self.close_rising_ticks)
        self.maximum_close_streak = torch.zeros_like(self.close_rising_ticks)
        self.minimum_distance_uu = torch.full(
            (self.worlds,),
            torch.inf,
            dtype=dtype,
            device=device,
        )
        self.maximum_ball_height_uu = torch.full_like(
            self.minimum_distance_uu,
            -torch.inf,
        )
        self.maximum_car_height_uu = torch.full_like(
            self.minimum_distance_uu,
            -torch.inf,
        )
        self.maximum_ball_vertical_speed = torch.full_like(
            self.minimum_distance_uu,
            -torch.inf,
        )
        self.maximum_ball_goalward_speed = torch.full_like(
            self.minimum_distance_uu,
            -torch.inf,
        )
        self.prompt_ball_vertical_transfer = torch.zeros_like(
            self.minimum_distance_uu
        )
        self.prompt_ball_goalward_transfer = torch.zeros_like(
            self.minimum_distance_uu
        )
        self.prompt_ball_vertical_speed = torch.zeros_like(
            self.minimum_distance_uu
        )
        self.prompt_ball_goalward_speed = torch.zeros_like(
            self.minimum_distance_uu
        )
        self.prompt_car_vertical_speed = torch.zeros_like(
            self.minimum_distance_uu
        )
        self.prompt_distance_uu = torch.zeros_like(self.minimum_distance_uu)
        self.initialized = True

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        active: torch.Tensor,
    ) -> None:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("prompt-continuation observations must be [N,2,182]")
        if active.shape != (self.worlds,):
            raise ValueError("prompt-continuation active mask must align with worlds")
        if not self.initialized:
            self._initialize(before.device, before.dtype)

        if self.geometry.initialized:
            prompt_seen_before = (
                self.geometry.prompt_airborne_follow_seen.clone()
            )
        else:
            prompt_seen_before = torch.zeros(
                self.worlds,
                dtype=torch.bool,
                device=before.device,
            )
        self.geometry.step(before, after, tick=tick, active=active)
        prompt_event = (
            self.geometry.prompt_airborne_follow_seen & ~prompt_seen_before
        )
        self.prompt_tick.copy_(
            torch.where(
                prompt_event,
                torch.full_like(self.prompt_tick, int(tick)),
                self.prompt_tick,
            )
        )
        self.last_post_prompt_touch_tick.copy_(
            torch.where(
                prompt_event,
                torch.full_like(self.last_post_prompt_touch_tick, int(tick)),
                self.last_post_prompt_touch_tick,
            )
        )

        position_scale = torch.as_tensor(
            POSITION_SCALE,
            dtype=after.dtype,
            device=after.device,
        )
        relative = self._vector(after, "relative.ball_position") * position_scale
        distance = torch.linalg.vector_norm(relative, dim=-1)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        car_vertical_speed = (
            self._self(after, "self.linear_velocity.z") * CAR_LINEAR_SPEED_SCALE
        )
        before_ball_vertical = (
            self._self(before, "ball.linear_velocity.z")
            * BALL_LINEAR_SPEED_SCALE
        )
        after_ball_vertical = (
            self._self(after, "ball.linear_velocity.z") * BALL_LINEAR_SPEED_SCALE
        )
        before_ball_goalward = (
            self._self(before, "ball.linear_velocity.y")
            * BALL_LINEAR_SPEED_SCALE
        )
        after_ball_goalward = (
            self._self(after, "ball.linear_velocity.y")
            * BALL_LINEAR_SPEED_SCALE
        )
        self.prompt_ball_vertical_transfer.copy_(
            torch.where(
                prompt_event,
                after_ball_vertical - before_ball_vertical,
                self.prompt_ball_vertical_transfer,
            )
        )
        self.prompt_ball_goalward_transfer.copy_(
            torch.where(
                prompt_event,
                after_ball_goalward - before_ball_goalward,
                self.prompt_ball_goalward_transfer,
            )
        )
        for target, value in (
            (self.prompt_ball_vertical_speed, after_ball_vertical),
            (self.prompt_ball_goalward_speed, after_ball_goalward),
            (self.prompt_car_vertical_speed, car_vertical_speed),
            (self.prompt_distance_uu, distance),
        ):
            target.copy_(torch.where(prompt_event, value, target))

        age = int(tick) - self.prompt_tick
        post_prompt = (
            active
            & self.geometry.prompt_airborne_follow_seen
            & (age >= 0)
            & (age <= self.continuation_ticks)
        )
        airborne = self._self(after, "self.on_ground") < 0.5
        close = post_prompt & airborne & (distance <= self.close_shell_uu)
        close_rising = close & (after_ball_vertical > 0.0)
        close_goalward = close & (after_ball_goalward > 0.0)
        self.close_rising_ticks += close_rising.to(torch.int64)
        self.close_goalward_ticks += close_goalward.to(torch.int64)
        if int(tick) != self.last_observed_tick + 1:
            self.current_close_streak.zero_()
        self.current_close_streak = torch.where(
            close,
            self.current_close_streak + 1,
            torch.zeros_like(self.current_close_streak),
        )
        self.maximum_close_streak = torch.maximum(
            self.maximum_close_streak,
            self.current_close_streak,
        )
        self.last_observed_tick = int(tick)
        self.minimum_distance_uu = torch.where(
            post_prompt,
            torch.minimum(self.minimum_distance_uu, distance),
            self.minimum_distance_uu,
        )
        self.maximum_ball_height_uu = torch.where(
            post_prompt,
            torch.maximum(self.maximum_ball_height_uu, ball_height),
            self.maximum_ball_height_uu,
        )
        self.maximum_car_height_uu = torch.where(
            post_prompt,
            torch.maximum(self.maximum_car_height_uu, car_height),
            self.maximum_car_height_uu,
        )
        self.maximum_ball_vertical_speed = torch.where(
            post_prompt,
            torch.maximum(self.maximum_ball_vertical_speed, after_ball_vertical),
            self.maximum_ball_vertical_speed,
        )
        self.maximum_ball_goalward_speed = torch.where(
            post_prompt,
            torch.maximum(self.maximum_ball_goalward_speed, after_ball_goalward),
            self.maximum_ball_goalward_speed,
        )

        touch = active & (self._self(after, "lifecycle.self_touch_event") >= 0.5)
        separated_recontact = (
            post_prompt
            & touch
            & ((int(tick) - self.last_post_prompt_touch_tick) >= 4)
        )
        self.second_recontact_seen |= separated_recontact
        self.last_post_prompt_touch_tick.copy_(
            torch.where(
                separated_recontact,
                torch.full_like(self.last_post_prompt_touch_tick, int(tick)),
                self.last_post_prompt_touch_tick,
            )
        )
        self.bridge_elevated_seen |= (
            close & (car_height >= 150.0) & (ball_height >= 250.0)
        )
        self.bridge_high_seen |= (
            close & (car_height >= 250.0) & (ball_height >= 300.0)
        )

    def telemetry(self) -> dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("prompt-continuation probe has no observations")
        prompt = self.geometry.prompt_airborne_follow_seen
        post_prompt_observed = prompt & torch.isfinite(self.minimum_distance_uu)
        return {
            "identity": GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_VERSION,
            "worlds": self.worlds,
            "continuation_ticks": self.continuation_ticks,
            "close_shell_uu": self.close_shell_uu,
            "prompt_airborne_follow_fraction": float(
                prompt.to(torch.float32).mean().cpu()
            ),
            "second_recontact_fraction": float(
                self.second_recontact_seen.to(torch.float32).mean().cpu()
            ),
            "bridge_elevated_fraction": float(
                self.bridge_elevated_seen.to(torch.float32).mean().cpu()
            ),
            "bridge_high_fraction": float(
                self.bridge_high_seen.to(torch.float32).mean().cpu()
            ),
            "prompt_contact": {
                "ball_vertical_transfer_uu_per_second": _distribution(
                    self.prompt_ball_vertical_transfer,
                    prompt,
                ),
                "ball_goalward_transfer_uu_per_second": _distribution(
                    self.prompt_ball_goalward_transfer,
                    prompt,
                ),
                "ball_vertical_speed_uu_per_second": _distribution(
                    self.prompt_ball_vertical_speed,
                    prompt,
                ),
                "ball_goalward_speed_uu_per_second": _distribution(
                    self.prompt_ball_goalward_speed,
                    prompt,
                ),
                "car_vertical_speed_uu_per_second": _distribution(
                    self.prompt_car_vertical_speed,
                    prompt,
                ),
                "distance_uu": _distribution(self.prompt_distance_uu, prompt),
            },
            "continuation": {
                "minimum_distance_uu": _distribution(
                    self.minimum_distance_uu,
                    post_prompt_observed,
                ),
                "maximum_ball_height_uu": _distribution(
                    self.maximum_ball_height_uu,
                    post_prompt_observed,
                ),
                "maximum_car_height_uu": _distribution(
                    self.maximum_car_height_uu,
                    post_prompt_observed,
                ),
                "maximum_ball_vertical_speed_uu_per_second": _distribution(
                    self.maximum_ball_vertical_speed,
                    post_prompt_observed,
                ),
                "maximum_ball_goalward_speed_uu_per_second": _distribution(
                    self.maximum_ball_goalward_speed,
                    post_prompt_observed,
                ),
                "close_rising_ticks": _distribution(
                    self.close_rising_ticks.to(torch.float32),
                    prompt,
                ),
                "close_goalward_ticks": _distribution(
                    self.close_goalward_ticks.to(torch.float32),
                    prompt,
                ),
                "maximum_close_streak_ticks": _distribution(
                    self.maximum_close_streak.to(torch.float32),
                    prompt,
                ),
            },
            "source_geometry": self.geometry.telemetry(),
        }


__all__ = [
    "GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_VERSION",
    "PromptContinuationProbe",
]
