"""Bounded on-policy correction datasets for aerial intercept DAgger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DAGGER_VERSION = "RIVAL2_AERIAL_INTERCEPT_DAGGER_V1"


@dataclass(frozen=True, slots=True)
class CorrectionTrajectory:
    pack: int
    side: int
    world: int
    round: int
    seed: int
    offset: int
    length: int
    success: bool
    weight: float

    @property
    def category(self) -> tuple[int, int]:
        return self.pack, self.side


class CorrectionDataset:
    """Policy-visited observations paired with frozen-teacher actions."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            raise ValueError("at least one correction round is required")
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        ticks: list[torch.Tensor] = []
        trajectories: list[CorrectionTrajectory] = []
        offset = 0
        for payload in payloads:
            if payload.get("format") != f"{DAGGER_VERSION}_CORRECTIONS":
                raise ValueError("unexpected DAgger correction artifact")
            observation = payload["observation"].contiguous()
            action = payload["action"].contiguous()
            tick = payload["tick"].to(torch.int64).contiguous()
            if observation.ndim != 2 or observation.shape[1] != 182:
                raise ValueError("correction observations must have shape [N,182]")
            if action.shape != (observation.shape[0], 8):
                raise ValueError("correction actions must have shape [N,8]")
            if tick.shape != (observation.shape[0],):
                raise ValueError("correction ticks must align")
            if not bool(torch.isfinite(observation).all() and torch.isfinite(action).all()):
                raise ValueError("correction artifact contains nonfinite values")
            observations.append(observation)
            actions.append(action)
            ticks.append(tick)
            for row in payload["trajectories"]:
                local = dict(row)
                local_start = int(local["offset"])
                local_stop = local_start + int(local["length"])
                local_ticks = tick[local_start:local_stop]
                local["offset"] += offset
                trajectory = CorrectionTrajectory(**local)
                stop = trajectory.offset + trajectory.length
                if (
                    trajectory.length <= 0
                    or local_stop > observation.shape[0]
                    or stop > offset + observation.shape[0]
                ):
                    raise ValueError("invalid correction trajectory boundary")
                if local_ticks.numel() > 1 and not bool((local_ticks[1:] > local_ticks[:-1]).all()):
                    raise ValueError("correction trajectory ticks must be strictly increasing")
                trajectories.append(trajectory)
            offset += observation.shape[0]
        self.observation = torch.cat(observations)
        self.action = torch.cat(actions)
        self.tick = torch.cat(ticks)
        self.trajectories = tuple(trajectories)
        self.categories: dict[tuple[int, int], tuple[int, ...]] = {}
        for index, trajectory in enumerate(self.trajectories):
            self.categories.setdefault(trajectory.category, tuple())
            self.categories[trajectory.category] += (index,)
        expected = [(pack, side) for pack in range(3) for side in range(2)]
        if sorted(self.categories) != expected:
            raise ValueError("corrections require all packs and both physical sides")

    def sample(self, count: int, *, generator: torch.Generator) -> torch.Tensor:
        if count <= 0:
            raise ValueError("correction sample count must be positive")
        categories = sorted(self.categories)
        base, remainder = divmod(count, len(categories))
        selected: list[torch.Tensor] = []
        for category_index, category in enumerate(categories):
            local_count = base + int(category_index < remainder)
            rows = self.categories[category]
            weights = torch.tensor(
                [self.trajectories[index].weight for index in rows],
                dtype=torch.float64,
            )
            chosen = torch.multinomial(
                weights,
                local_count,
                replacement=True,
                generator=generator,
            )
            flat = torch.empty(local_count, dtype=torch.int64)
            for local, trajectory_index in enumerate(rows):
                positions = (chosen == local).nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                trajectory = self.trajectories[trajectory_index]
                within = torch.randint(
                    trajectory.length,
                    (positions.numel(),),
                    generator=generator,
                )
                flat[positions] = trajectory.offset + within
            selected.append(flat)
        result = torch.cat(selected)
        return result[torch.randperm(result.numel(), generator=generator)]


def evenly_spaced_indices(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    """Return at most ``maximum`` ordered rows while retaining both endpoints."""

    if indices.ndim != 1 or maximum <= 0:
        raise ValueError("invalid evenly-spaced sampling request")
    if indices.numel() <= maximum:
        return indices
    positions = (
        torch.linspace(
            0,
            indices.numel() - 1,
            maximum,
            device=indices.device,
        )
        .round()
        .to(torch.int64)
    )
    result = indices.index_select(0, positions)
    if result.numel() > 1 and not bool((result[1:] > result[:-1]).all()):
        raise RuntimeError("evenly-spaced correction rows are not unique")
    return result


__all__ = [
    "DAGGER_VERSION",
    "CorrectionDataset",
    "CorrectionTrajectory",
    "evenly_spaced_indices",
]
