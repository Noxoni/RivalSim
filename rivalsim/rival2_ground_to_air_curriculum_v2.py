"""Prospective curriculum helpers for the physical ground-to-air option V2.

V2 keeps the literal full-chain validation from V1, but shortens the credit
assignment path during training by retaining the calibrated source-exact
launch until a prospectively scheduled handoff tick.  Successful physical
recontact trajectories may be rehearsed, but no named mechanic label or raw
airtime signal is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUND_TO_AIR_CURRICULUM_V2_VERSION = "RIVAL2_GROUND_TO_AIR_CURRICULUM_V2"


@dataclass(frozen=True, slots=True)
class HandoffStage:
    first_block: int
    handoff_ticks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.first_block <= 0:
            raise ValueError("first block must be positive")
        if not self.handoff_ticks or any(tick < 0 for tick in self.handoff_ticks):
            raise ValueError("handoff ticks must be non-negative")


def validate_handoff_stages(stages: tuple[HandoffStage, ...]) -> None:
    if not stages or stages[0].first_block != 1:
        raise ValueError("handoff schedule must start at block one")
    starts = tuple(stage.first_block for stage in stages)
    if starts != tuple(sorted(set(starts))):
        raise ValueError("handoff stage starts must be strictly increasing")


def handoff_tick_for_block(block: int, stages: tuple[HandoffStage, ...]) -> int:
    """Select a deterministic within-stage handoff tick for one training block."""

    if block <= 0:
        raise ValueError("block must be positive")
    validate_handoff_stages(stages)
    selected = stages[0]
    for candidate in stages[1:]:
        if candidate.first_block > block:
            break
        selected = candidate
    offset = block - selected.first_block
    return selected.handoff_ticks[offset % len(selected.handoff_ticks)]


def learned_handoff_mask(
    *,
    active: torch.Tensor,
    pop_age: torch.Tensor,
    handoff_tick: int,
) -> torch.Tensor:
    """Return worlds controlled by the learned policy after a physical pop."""

    if active.shape != pop_age.shape:
        raise ValueError("active and pop-age tensors must align")
    if handoff_tick < 0:
        raise ValueError("handoff tick cannot be negative")
    return active.to(torch.bool) & (pop_age >= handoff_tick)


def successful_rehearsal_mask(
    reward: torch.Tensor,
    learned_mask: torch.Tensor,
    *,
    event_reward_threshold: float,
    history_ticks: int,
) -> torch.Tensor:
    """Select learned actions leading into the first literal contact event.

    ``reward`` and ``learned_mask`` are time-major ``[T, W]`` tensors.  The
    returned mask includes at most ``history_ticks`` actions ending on each
    world's first event.  It cannot select pre-handoff bootstrap actions.
    """

    if reward.ndim != 2 or learned_mask.shape != reward.shape:
        raise ValueError("rehearsal tensors must align as [time, worlds]")
    if history_ticks <= 0:
        raise ValueError("history ticks must be positive")
    events = (reward >= float(event_reward_threshold)) & learned_mask.to(torch.bool)
    selected = torch.zeros_like(events)
    if reward.shape[0] == 0:
        return selected
    has_event = events.any(dim=0)
    first_event = events.to(torch.int64).argmax(dim=0)
    ticks = torch.arange(reward.shape[0], device=reward.device)[:, None]
    start = (first_event - history_ticks + 1).clamp_min(0)
    window = has_event[None, :] & (ticks >= start[None, :]) & (
        ticks <= first_event[None, :]
    )
    return window & learned_mask.to(torch.bool)


__all__ = [
    "GROUND_TO_AIR_CURRICULUM_V2_VERSION",
    "HandoffStage",
    "handoff_tick_for_block",
    "learned_handoff_mask",
    "successful_rehearsal_mask",
    "validate_handoff_stages",
]
