"""Deterministic trajectory storage and sampling for aerial intercept distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DISTILLATION_VERSION = "RIVAL2_AERIAL_INTERCEPT_DISTILLATION_V1"


@dataclass(frozen=True, slots=True)
class TrajectoryIndex:
    pack: int
    side: int
    world: int
    seed: int
    offset: int
    length: int
    first_touch_tick: int
    outcome: str
    weight: float

    @property
    def category(self) -> tuple[int, int]:
        return self.pack, self.side


class DistillationDataset:
    """Flat exact tensors plus whole-trajectory boundaries.

    Sampling is uniform over pack/side categories, weighted over successful
    worlds within a category, and uniform over ticks inside the selected
    trajectory.  A per-block cap prevents one successful world from filling a
    training block.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != f"{DISTILLATION_VERSION}_TRAJECTORIES":
            raise ValueError("unexpected intercept trajectory artifact")
        self.observation = payload["observation"].contiguous()
        self.action = payload["action"].contiguous()
        self.tick = payload["tick"].to(torch.int64).contiguous()
        if self.observation.ndim != 2 or self.observation.shape[1] != 182:
            raise ValueError("distillation observations must have shape [N,182]")
        if self.action.shape != (self.observation.shape[0], 8):
            raise ValueError("distillation actions must have shape [N,8]")
        if self.tick.shape != (self.observation.shape[0],):
            raise ValueError("distillation ticks must align with samples")
        if not bool(torch.isfinite(self.observation).all()):
            raise ValueError("nonfinite distillation observation")
        if not bool(torch.isfinite(self.action).all()):
            raise ValueError("nonfinite distillation action")
        self.trajectories = tuple(TrajectoryIndex(**row) for row in payload["trajectories"])
        if not self.trajectories:
            raise ValueError("distillation artifact has no successful trajectories")
        self.categories: dict[tuple[int, int], tuple[int, ...]] = {}
        for index, trajectory in enumerate(self.trajectories):
            if trajectory.length <= 0 or trajectory.offset < 0:
                raise ValueError("invalid trajectory boundary")
            stop = trajectory.offset + trajectory.length
            if stop > self.observation.shape[0]:
                raise ValueError("trajectory exceeds flat artifact")
            expected = torch.arange(
                int(self.tick[trajectory.offset]),
                int(self.tick[trajectory.offset]) + trajectory.length,
                dtype=torch.int64,
            )
            if not torch.equal(self.tick[trajectory.offset : stop].cpu(), expected):
                raise ValueError("trajectory tick order is not contiguous")
            self.categories.setdefault(trajectory.category, tuple())
            self.categories[trajectory.category] += (index,)
        if sorted(self.categories) != [(pack, side) for pack in range(3) for side in range(2)]:
            raise ValueError("all three packs and both physical sides are required")

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        maximum_samples_per_world: int,
    ) -> torch.Tensor:
        if count <= 0 or maximum_samples_per_world <= 0:
            raise ValueError("sample request must be positive")
        categories = sorted(self.categories)
        base, remainder = divmod(count, len(categories))
        selected: list[torch.Tensor] = []
        for category_index, category in enumerate(categories):
            local_count = base + int(category_index < remainder)
            trajectory_rows = self.categories[category]
            if local_count > len(trajectory_rows) * maximum_samples_per_world:
                raise RuntimeError(
                    "trajectory cap cannot satisfy requested block for "
                    f"pack={category[0]} side={category[1]}"
                )
            weights = torch.tensor(
                [self.trajectories[index].weight for index in trajectory_rows],
                dtype=torch.float64,
            )
            chosen_local = torch.multinomial(
                weights,
                local_count,
                replacement=True,
                generator=generator,
            )
            realized = torch.bincount(chosen_local, minlength=len(trajectory_rows))
            if int(realized.max()) > maximum_samples_per_world:
                # Deterministically round-robin the excess among other valid
                # successful worlds in the same physical category.
                order = torch.randperm(len(trajectory_rows), generator=generator)
                cursor = 0
                for position in range(local_count):
                    candidate = int(chosen_local[position])
                    if int(realized[candidate]) <= maximum_samples_per_world:
                        continue
                    while (
                        int(realized[int(order[cursor % len(order)])]) >= maximum_samples_per_world
                    ):
                        cursor += 1
                        if cursor > local_count * 2:
                            raise RuntimeError("trajectory cap cannot satisfy requested block")
                    replacement = int(order[cursor % len(order)])
                    realized[candidate] -= 1
                    realized[replacement] += 1
                    chosen_local[position] = replacement
                    cursor += 1
            flat = torch.empty(local_count, dtype=torch.int64)
            for local, trajectory_index in enumerate(trajectory_rows):
                positions = (chosen_local == local).nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                trajectory = self.trajectories[trajectory_index]
                tick_offset = torch.randint(
                    trajectory.length,
                    (positions.numel(),),
                    generator=generator,
                )
                flat[positions] = trajectory.offset + tick_offset
            selected.append(flat)
        result = torch.cat(selected)
        permutation = torch.randperm(result.numel(), generator=generator)
        return result[permutation]


def physical_gate(
    physical: list[dict[str, Any]],
    human_validation_rmse: list[float],
    authority: dict[str, Any],
) -> bool:
    if len(physical) != 6 or len(human_validation_rmse) != 2:
        raise ValueError("physical gate requires six pack/side rows and two human rows")
    if any(
        value > authority["optimization"]["human_validation_rmse_max"]
        for value in human_validation_rmse
    ):
        return False
    thresholds = authority["selection"]
    for row in physical:
        if not row.get("finite_observation", False):
            return False
        fractions = row["fractions"]
        pack = row["pack"]
        required = thresholds[pack]
        if fractions["high_touch"] < required["high_touch_fraction_min"]:
            return False
        if pack == "center_pop":
            if fractions["goal"] < required["qualified_goal_fraction_min"]:
                return False
        elif fractions["goalward_first_touch"] < required["goalward_first_touch_fraction_min"]:
            return False
    return True


__all__ = [
    "DISTILLATION_VERSION",
    "DistillationDataset",
    "TrajectoryIndex",
    "physical_gate",
]
