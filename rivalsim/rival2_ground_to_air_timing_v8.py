"""Prospective first-jump timing candidates for natural aerial entries."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

GROUND_TO_AIR_TIMING_V8_VERSION = "RIVAL2_GROUND_TO_AIR_TIMING_V8"
PHYSICS_HZ = 120


@dataclass(frozen=True, slots=True)
class TakeoffTimingCandidate:
    name: str
    first_jump_hold_ticks: int
    jump_release_ticks: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("timing candidate name cannot be empty")
        if self.first_jump_hold_ticks <= 0 or self.jump_release_ticks <= 0:
            raise ValueError("jump hold and release ticks must be positive")

    @property
    def second_jump_tick(self) -> int:
        return self.first_jump_hold_ticks + self.jump_release_ticks

    def record(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "first_jump_hold_milliseconds": (
                    self.first_jump_hold_ticks * 1000.0 / PHYSICS_HZ
                ),
                "second_jump_milliseconds_after_launch": (
                    self.second_jump_tick * 1000.0 / PHYSICS_HZ
                ),
            }
        )
        return result


TIMING_CANDIDATES = (
    TakeoffTimingCandidate("inherited_hold_8_release_8", 8, 8),
    TakeoffTimingCandidate("hold_12_release_4", 12, 4),
    TakeoffTimingCandidate("hold_16_release_4", 16, 4),
    TakeoffTimingCandidate("hold_20_release_4", 20, 4),
    TakeoffTimingCandidate("hold_24_release_4", 24, 4),
)


def authority_with_timing(
    authority: dict[str, Any],
    candidate: TakeoffTimingCandidate,
) -> dict[str, Any]:
    """Return an isolated authority view with only jump timing changed."""

    updated = copy.deepcopy(authority)
    updated["option_config"]["first_jump_hold_ticks"] = (
        candidate.first_jump_hold_ticks
    )
    updated["option_config"]["jump_release_ticks"] = candidate.jump_release_ticks
    return updated


__all__ = [
    "GROUND_TO_AIR_TIMING_V8_VERSION",
    "PHYSICS_HZ",
    "TIMING_CANDIDATES",
    "TakeoffTimingCandidate",
    "authority_with_timing",
]
