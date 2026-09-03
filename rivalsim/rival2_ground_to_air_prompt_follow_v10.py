"""Training-only prompt airborne-follow credit for natural aerial entries.

The event is deliberately geometric rather than a named-mechanic classifier: after
the first low setup touch, it pays exactly once when the same car produces a
separated native touch onset while airborne within the frozen causal window.
Production Gameplay 120 V2 reward is not changed.
"""

from __future__ import annotations

from typing import Any

import torch

from benchmarks.run_rival2_ground_to_air_goal_v3 import (
    GoalDirectedTrainingTracker,
)
from rivalsim.rival2_ground_to_air_touch_geometry import (
    GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION,
    NaturalAerialTouchGeometryProbe,
)

GROUND_TO_AIR_PROMPT_FOLLOW_V10_VERSION = (
    "RIVAL2_GROUND_TO_AIR_PROMPT_FOLLOW_V10"
)


class PromptAerialFollowTrainingTracker(GoalDirectedTrainingTracker):
    """Add bounded prompt-recontact credit to the existing physical curriculum."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.geometry_probe = NaturalAerialTouchGeometryProbe(
            self.worlds, attacker_side=self.side
        )
        self.prompt_airborne_follow_touches = 0
        self.prompt_airborne_follow_reward_sum = 0.0

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.geometry_probe.initialized:
            seen_before = self.geometry_probe.prompt_airborne_follow_seen.clone()
        else:
            seen_before = torch.zeros(
                self.worlds, dtype=torch.bool, device=before.device
            )
        self.geometry_probe.step(
            before,
            after,
            tick=int(kwargs["tick"]),
            active=kwargs["active"],
        )
        prompt_event = (
            self.geometry_probe.prompt_airborne_follow_seen & ~seen_before
        )
        reward, done = super().step(before, after, **kwargs)
        event_value = float(
            self.authority["reward"]["prompt_airborne_follow_event"]
        )
        bonus = prompt_event.to(reward.dtype) * event_value
        reward = reward + bonus
        bonus_sum = float(bonus.sum())
        self.reward_sum += bonus_sum
        self.prompt_airborne_follow_touches += int(prompt_event.sum())
        self.prompt_airborne_follow_reward_sum += bonus_sum
        return reward, done

    def telemetry(self) -> dict[str, Any]:
        result = super().telemetry()
        result.update(
            {
                "prompt_airborne_follow_touches": (
                    self.prompt_airborne_follow_touches
                ),
                "prompt_airborne_follow_reward_sum": (
                    self.prompt_airborne_follow_reward_sum
                ),
                "prompt_airborne_follow_event_value": float(
                    self.authority["reward"]["prompt_airborne_follow_event"]
                ),
                "touch_geometry_probe": self.geometry_probe.telemetry(),
            }
        )
        return result


__all__ = [
    "GROUND_TO_AIR_PROMPT_FOLLOW_V10_VERSION",
    "GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION",
    "PromptAerialFollowTrainingTracker",
]
