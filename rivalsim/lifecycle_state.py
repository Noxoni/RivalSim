"""GPU-resident standard-1v1 lifecycle state for RivalSim v0.4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from rivalsim.kernels.boost_pad import PAD_COUNT

HELD_FLOAT_FIELDS = 29
HELD_INT_FIELDS = 12


def _selector_array(value: int | np.ndarray, count: int, modulo: int) -> np.ndarray:
    result = np.broadcast_to(np.asarray(value, dtype=np.int32), (count,)).copy()
    if np.any((result < 0) | (result >= modulo)):
        raise ValueError(f"selector entries must be in [0, {modulo})")
    return result


@dataclass(slots=True)
class LifecycleSnapshot:
    world_tick: np.ndarray
    episode_tick: np.ndarray
    blue_score: np.ndarray
    orange_score: np.ndarray
    goal_scored: np.ndarray
    scoring_team: np.ndarray
    kickoff_reset: np.ndarray
    kickoff_layout: np.ndarray
    kickoff_selector: np.ndarray
    full_reset: np.ndarray
    reset_required: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    pad_active: np.ndarray
    pad_cooldown: np.ndarray
    pad_previous_locked_car: np.ndarray
    pad_pickup_car: np.ndarray
    pad_reactivated: np.ndarray
    car_is_demoed: np.ndarray
    demo_respawn_timer: np.ndarray
    respawn_event: np.ndarray
    respawn_location: np.ndarray
    respawn_selector: np.ndarray


class LifecycleState:
    """Persistent source/spec state surrounding the accepted v0.3 physics."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        kickoff_selector: int | np.ndarray = 0,
        respawn_selector: int | np.ndarray = 0,
        auto_kickoff: bool = True,
        full_reset_interval_ticks: int = 0,
    ):
        if full_reset_interval_ticks < 0:
            raise ValueError("full_reset_interval_ticks must be non-negative")
        self.num_envs = int(num_envs)
        self.device = device
        self.auto_kickoff_enabled = bool(auto_kickoff)
        self.full_reset_interval_ticks = int(full_reset_interval_ticks)
        kickoff = _selector_array(kickoff_selector, num_envs, 5)
        respawn_world = _selector_array(respawn_selector, num_envs, 4)
        respawn = np.repeat(respawn_world[:, None], 2, axis=1).reshape(-1)

        for name in (
            "world_tick",
            "episode_tick",
            "blue_score",
            "orange_score",
            "goal_scored",
            "kickoff_reset",
            "full_reset",
            "reset_required",
            "terminated",
            "truncated",
            "ball_scored_last",
        ):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.int32, device=device))
        self.scoring_team = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.kickoff_layout = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.kickoff_selector = wp.array(kickoff, dtype=wp.int32, device=device)
        self.auto_kickoff = wp.full(num_envs, int(auto_kickoff), dtype=wp.int32, device=device)
        self.full_reset_interval = wp.full(
            num_envs,
            int(full_reset_interval_ticks),
            dtype=wp.int32,
            device=device,
        )

        pad_capacity = num_envs * PAD_COUNT
        self.pad_cooldown_before = wp.zeros(pad_capacity, dtype=wp.float32, device=device)
        self.pad_pickup_car = wp.zeros(pad_capacity, dtype=wp.int32, device=device)
        self.pad_reactivated = wp.zeros(pad_capacity, dtype=wp.int32, device=device)

        car_capacity = num_envs * 2
        self.demo_respawn_timer = wp.zeros(car_capacity, dtype=wp.float32, device=device)
        self.demo_held_valid = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.demo_request = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.respawn_pending = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.respawn_event = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.respawn_location = wp.full(car_capacity, -1, dtype=wp.int32, device=device)
        self.respawn_selector = wp.array(respawn, dtype=wp.int32, device=device)
        self.held_float = wp.zeros(
            (car_capacity, HELD_FLOAT_FIELDS), dtype=wp.float32, device=device
        )
        self.held_int = wp.zeros((car_capacity, HELD_INT_FIELDS), dtype=wp.int32, device=device)

    @property
    def logical_bytes(self) -> int:
        world_ints = 16
        pad = PAD_COUNT * (4 + 4 + 4)
        car = 2 * (7 * 4 + HELD_FLOAT_FIELDS * 4 + HELD_INT_FIELDS * 4)
        return self.num_envs * (world_ints * 4 + pad + car)

    def snapshot(
        self,
        *,
        pad_cooldown: wp.array,
        pad_previous_locked_car: wp.array,
        car_is_demoed: wp.array,
    ) -> LifecycleSnapshot:
        count = self.num_envs

        def integer(name: str, shape: tuple[int, ...]) -> np.ndarray:
            return np.asarray(getattr(self, name).numpy(), dtype=np.int32).reshape(shape)

        cooldown = np.asarray(pad_cooldown.numpy(), dtype=np.float32).reshape(count, PAD_COUNT)
        previous = np.asarray(pad_previous_locked_car.numpy(), dtype=np.int32).reshape(
            count, PAD_COUNT
        )
        demoed = np.asarray(car_is_demoed.numpy(), dtype=np.int32).reshape(count, 2)
        return LifecycleSnapshot(
            world_tick=integer("world_tick", (count,)),
            episode_tick=integer("episode_tick", (count,)),
            blue_score=integer("blue_score", (count,)),
            orange_score=integer("orange_score", (count,)),
            goal_scored=integer("goal_scored", (count,)),
            scoring_team=integer("scoring_team", (count,)),
            kickoff_reset=integer("kickoff_reset", (count,)),
            kickoff_layout=integer("kickoff_layout", (count,)),
            kickoff_selector=integer("kickoff_selector", (count,)),
            full_reset=integer("full_reset", (count,)),
            reset_required=integer("reset_required", (count,)),
            terminated=integer("terminated", (count,)),
            truncated=integer("truncated", (count,)),
            pad_active=(cooldown == np.float32(0.0)).astype(np.int32),
            pad_cooldown=cooldown,
            pad_previous_locked_car=previous,
            pad_pickup_car=integer("pad_pickup_car", (count, PAD_COUNT)),
            pad_reactivated=integer("pad_reactivated", (count, PAD_COUNT)),
            car_is_demoed=demoed,
            demo_respawn_timer=np.asarray(
                self.demo_respawn_timer.numpy(), dtype=np.float32
            ).reshape(count, 2),
            respawn_event=integer("respawn_event", (count, 2)),
            respawn_location=integer("respawn_location", (count, 2)),
            respawn_selector=integer("respawn_selector", (count, 2)),
        )


__all__ = ["LifecycleSnapshot", "LifecycleState"]
