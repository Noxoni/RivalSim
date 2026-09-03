"""Read-only native touch geometry for natural ground-to-air diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import FIELD

GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION = "RIVAL2_GROUND_TO_AIR_TOUCH_GEOMETRY_V1"
PROMPT_FOLLOW_MAXIMUM_TICKS = 60


def _summary(values: list[torch.Tensor]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    combined = torch.cat(values).to(torch.float64)
    quantiles = torch.quantile(
        combined,
        torch.tensor((0.1, 0.5, 0.9), dtype=torch.float64),
    )
    return {
        "count": int(combined.numel()),
        "minimum": float(combined.min()),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "maximum": float(combined.max()),
    }


class NaturalAerialTouchGeometryProbe:
    """Measure separated native recontacts without changing reward or control."""

    def __init__(self, worlds: int, *, attacker_side: int) -> None:
        if worlds <= 0 or attacker_side not in (0, 1):
            raise ValueError("invalid touch-geometry probe request")
        self.worlds = int(worlds)
        self.side = int(attacker_side)
        self.initialized = False
        self.values: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {}
            for name in (
                "first_distinct_follow",
                "first_airborne_follow",
                "first_prompt_airborne_follow",
                "first_strict_elevated_follow",
            )
        }

    def _initialize(self, device: torch.device) -> None:
        self.pop_seen = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.any_follow_seen = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.airborne_follow_seen = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.prompt_airborne_follow_seen = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.strict_follow_seen = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.pop_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.last_touch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.initialized = True

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.side, FIELD[field]]

    def _record(
        self,
        category: str,
        mask: torch.Tensor,
        *,
        tick: int,
        car_height: torch.Tensor,
        ball_height: torch.Tensor,
        relative: torch.Tensor,
        on_ground: torch.Tensor,
    ) -> None:
        if not bool(mask.any()):
            return
        index = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        bucket = self.values[category]
        measurements = {
            "tick": torch.full(
                (index.numel(),), tick, dtype=torch.float64, device=index.device
            ),
            "ticks_after_pop": (tick - self.pop_tick.index_select(0, index)).to(
                torch.float64
            ),
            "car_height_uu": car_height.index_select(0, index),
            "ball_height_uu": ball_height.index_select(0, index),
            "relative_x_uu": relative.index_select(0, index)[:, 0],
            "relative_y_uu": relative.index_select(0, index)[:, 1],
            "relative_z_uu": relative.index_select(0, index)[:, 2],
            "distance_uu": torch.linalg.vector_norm(
                relative.index_select(0, index), dim=-1
            ),
            "on_ground": on_ground.index_select(0, index),
        }
        for name, value in measurements.items():
            bucket.setdefault(name, []).append(value.detach().cpu())

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        active: torch.Tensor,
    ) -> None:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("touch-geometry observations must align as [N,2,182]")
        if not self.initialized:
            self._initialize(before.device)
        touch = active & (self._self(after, "lifecycle.self_touch_event") >= 0.5)
        separated = touch & ((tick - self.last_touch_tick) >= 4)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height_before = self._self(before, "ball.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        on_ground = self._self(after, "self.on_ground")
        relative = torch.stack(
            [self._self(after, f"relative.ball_position.{axis}") for axis in "xyz"],
            dim=-1,
        ) * torch.as_tensor(POSITION_SCALE, dtype=after.dtype, device=after.device)

        setup = (
            separated
            & ~self.pop_seen
            & (ball_height_before <= 205.0)
            & (car_height <= 150.0)
        )
        follow = separated & self.pop_seen & ~setup
        first_any = follow & ~self.any_follow_seen
        first_airborne = follow & (on_ground < 0.5) & ~self.airborne_follow_seen
        first_prompt_airborne = (
            follow
            & (on_ground < 0.5)
            & ((tick - self.pop_tick) <= PROMPT_FOLLOW_MAXIMUM_TICKS)
            & ~self.prompt_airborne_follow_seen
        )
        first_strict = (
            follow
            & (car_height >= 150.0)
            & (ball_height >= 250.0)
            & ~self.strict_follow_seen
        )
        self._record(
            "first_distinct_follow",
            first_any,
            tick=tick,
            car_height=car_height,
            ball_height=ball_height,
            relative=relative,
            on_ground=on_ground,
        )
        self._record(
            "first_airborne_follow",
            first_airborne,
            tick=tick,
            car_height=car_height,
            ball_height=ball_height,
            relative=relative,
            on_ground=on_ground,
        )
        self._record(
            "first_prompt_airborne_follow",
            first_prompt_airborne,
            tick=tick,
            car_height=car_height,
            ball_height=ball_height,
            relative=relative,
            on_ground=on_ground,
        )
        self._record(
            "first_strict_elevated_follow",
            first_strict,
            tick=tick,
            car_height=car_height,
            ball_height=ball_height,
            relative=relative,
            on_ground=on_ground,
        )
        self.any_follow_seen |= first_any
        self.airborne_follow_seen |= first_airborne
        self.prompt_airborne_follow_seen |= first_prompt_airborne
        self.strict_follow_seen |= first_strict
        self.pop_seen |= setup
        self.pop_tick.copy_(
            torch.where(setup, torch.full_like(self.pop_tick, tick), self.pop_tick)
        )
        self.last_touch_tick.copy_(
            torch.where(
                separated,
                torch.full_like(self.last_touch_tick, tick),
                self.last_touch_tick,
            )
        )

    def telemetry(self) -> dict[str, Any]:
        categories: dict[str, Any] = {}
        for category, measurements in self.values.items():
            categories[category] = {
                "attempt_fraction": float(
                    {
                        "first_distinct_follow": self.any_follow_seen,
                        "first_airborne_follow": self.airborne_follow_seen,
                        "first_prompt_airborne_follow": (
                            self.prompt_airborne_follow_seen
                        ),
                        "first_strict_elevated_follow": self.strict_follow_seen,
                    }[category]
                    .to(torch.float32)
                    .mean()
                    .cpu()
                ),
                "measurements": {
                    name: _summary(values) for name, values in measurements.items()
                },
            }
        return {
            "identity": GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION,
            "attempts": self.worlds,
            "setup_touch_fraction": float(
                self.pop_seen.to(torch.float32).mean().cpu()
            ),
            "categories": categories,
        }


__all__ = [
    "GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION",
    "PROMPT_FOLLOW_MAXIMUM_TICKS",
    "NaturalAerialTouchGeometryProbe",
]
