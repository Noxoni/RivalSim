"""Load and restore exact natural aerial-handoff corpus rows."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.lifecycle_state import LifecycleSnapshot
from rivalsim.rival2_env import Rival2Env
from rivalsim.state import StateSnapshot

NATURAL_HANDOFF_CORPUS_V18_FORMAT = (
    "RIVAL2_GROUND_TO_AIR_NATURAL_HANDOFF_CORPUS_V18"
)


def load_natural_handoff_corpus(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != NATURAL_HANDOFF_CORPUS_V18_FORMAT:
        raise ValueError("unsupported natural-handoff corpus format")
    count = int(payload.get("count", -1))
    if count <= 0:
        raise ValueError("natural-handoff corpus is empty")
    if tuple(payload["observation"].shape) != (count, 2, 182):
        raise ValueError("natural-handoff observation shape mismatch")
    if not torch.isfinite(payload["observation"]).all():
        raise ValueError("natural-handoff observations contain nonfinite values")
    for item in fields(StateSnapshot):
        if item.name not in payload["state"]:
            raise ValueError(f"natural-handoff state is missing {item.name}")
    for item in fields(LifecycleSnapshot):
        if item.name not in payload["lifecycle"]:
            raise ValueError(f"natural-handoff lifecycle is missing {item.name}")
    return payload


def normalized_indices(
    payload: dict[str, Any], indices: torch.Tensor | np.ndarray | list[int] | None
) -> torch.Tensor:
    count = int(payload["count"])
    if indices is None:
        return torch.arange(count, dtype=torch.int64)
    result = torch.as_tensor(indices, dtype=torch.int64).flatten()
    if result.numel() == 0 or bool(((result < 0) | (result >= count)).any()):
        raise ValueError("natural-handoff selection is empty or out of bounds")
    return result


def state_snapshot_from_corpus(
    payload: dict[str, Any],
    indices: torch.Tensor | np.ndarray | list[int] | None = None,
) -> StateSnapshot:
    selected = normalized_indices(payload, indices).numpy()
    state = StateSnapshot(
        **{
            item.name: np.ascontiguousarray(payload["state"][item.name][selected])
            for item in fields(StateSnapshot)
        }
    )
    state.validate()
    return state


def _copy_world_rows(destination: Any, source: Any, count: int) -> None:
    target = wp.to_torch(destination).reshape(count, -1)
    value = torch.as_tensor(source, device=target.device).reshape(count, -1)
    target.copy_(value.to(dtype=target.dtype))


def restore_corpus_runtime(
    env: Rival2Env,
    payload: dict[str, Any],
    indices: torch.Tensor | np.ndarray | list[int] | None = None,
) -> torch.Tensor:
    """Restore all captured bridge/lifecycle state and return exact observation."""

    selected = normalized_indices(payload, indices)
    count = int(selected.numel())
    if env.num_envs != count:
        raise ValueError("replay environment size differs from selected corpus rows")
    for name, value in payload["bridge_views"].items():
        if name not in env.bridge.views:
            raise ValueError(f"replay bridge is missing captured view {name}")
        target = env.bridge.views[name].reshape(count, -1)
        source = value.index_select(0, selected).to(
            device=target.device, dtype=target.dtype
        )
        if source.shape != target.shape:
            raise ValueError(f"captured bridge view shape mismatch for {name}")
        target.copy_(source)

    lifecycle = payload["lifecycle"]
    row = selected.numpy()
    direct_lifecycle = (
        "world_tick",
        "episode_tick",
        "blue_score",
        "orange_score",
        "goal_scored",
        "scoring_team",
        "kickoff_reset",
        "kickoff_layout",
        "kickoff_selector",
        "full_reset",
        "reset_required",
        "terminated",
        "truncated",
        "pad_pickup_car",
        "pad_boost_gained",
        "pad_reactivated",
        "demo_respawn_timer",
        "respawn_event",
        "respawn_location",
        "respawn_selector",
    )
    for name in direct_lifecycle:
        _copy_world_rows(
            getattr(env.world.lifecycle, name), lifecycle[name][row], count
        )
    _copy_world_rows(
        env.world.boost_pad_cooldown, lifecycle["pad_cooldown"][row], count
    )
    _copy_world_rows(
        env.world.boost_pad_previous_locked_car,
        lifecycle["pad_previous_locked_car"][row],
        count,
    )
    _copy_world_rows(
        env.world.car_car.car_is_demoed, lifecycle["car_is_demoed"][row], count
    )
    env.observation = env.bridge.observation()
    return env.observation


__all__ = [
    "NATURAL_HANDOFF_CORPUS_V18_FORMAT",
    "load_natural_handoff_corpus",
    "normalized_indices",
    "restore_corpus_runtime",
    "state_snapshot_from_corpus",
]
