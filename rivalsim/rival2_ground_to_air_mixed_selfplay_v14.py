"""Mixed natural/self-play curriculum for the protected aerial scorer.

The curriculum keeps ordinary V23 kickoff worlds in the same simulator batch
as a bounded set of authoritative V11 ground-to-air opportunities.  It does
not reward airtime or a named mechanic.  Its only new negative outcome is the
physical failure of an already-started aerial chain: after a first airborne
contact, landing or letting the ball reach the floor before a separated second
contact cancels the one-time entry reward.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_ground_to_air_entry_v11 import (
    DEFENDER_LIVE,
    SETUP_NAMES,
    build_ground_to_air_entry_scenarios,
)
from rivalsim.rival2_ground_to_air_selfplay_training_v12 import (
    AerialOptionSelfPlayTrainerV12,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (
    AerialOptionRouterConfig,
    AerialOptionSelfPlayRouter,
    AerialRouteOutcome,
    AerialSelfPlayRewardConfig,
)
from rivalsim.state import StateSnapshot
from rivalsim.static_world import make_standard_kickoff_state

GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION = (
    "RIVAL2_GROUND_TO_AIR_MIXED_SELFPLAY_V14"
)


@dataclass(frozen=True, slots=True)
class AerialContinuationFailureConfig:
    """Outcome penalties that close the entry-only PPO loophole."""

    landing_before_second_contact: float = -4.0
    ball_ground_before_second_contact: float = -4.0

    def __post_init__(self) -> None:
        if self.landing_before_second_contact >= 0.0:
            raise ValueError("landing failure must have negative reward")
        if self.ball_ground_before_second_contact >= 0.0:
            raise ValueError("ball-ground failure must have negative reward")


@dataclass(frozen=True, slots=True)
class MixedSelfPlayInitialState:
    state: StateSnapshot
    controlled_world: np.ndarray
    attacker_side: np.ndarray
    setup: np.ndarray
    kickoff_selector: np.ndarray


def _overlay_rows(
    destination: StateSnapshot,
    rows: np.ndarray,
    source: StateSnapshot,
) -> None:
    for item in fields(StateSnapshot):
        getattr(destination, item.name)[rows] = getattr(source, item.name)


def build_mixed_selfplay_initial_state(
    worlds: int,
    *,
    seed: int,
    controlled_fraction: float,
    setup_weights: tuple[float, ...],
    difficulty: float,
) -> MixedSelfPlayInitialState:
    """Mix standard kickoff worlds with side-balanced V11 aerial feeds."""

    if worlds < 4:
        raise ValueError("mixed self-play requires at least four worlds")
    if not 0.0 < controlled_fraction < 1.0:
        raise ValueError("controlled fraction must be inside (0,1)")
    controlled_count = round(worlds * controlled_fraction)
    controlled_count = max(2, min(worlds - 2, controlled_count))
    controlled_count -= controlled_count % 2
    rng = np.random.default_rng(seed)
    controlled_rows = np.sort(
        rng.choice(worlds, size=controlled_count, replace=False)
    ).astype(np.int64)
    half = controlled_count // 2
    side_zero_rows = controlled_rows[:half]
    side_one_rows = controlled_rows[half:]
    kickoff_selector = (
        np.arange(worlds, dtype=np.int32) + np.int32(seed % 5)
    ) % 5
    state = make_standard_kickoff_state(worlds, kickoff_selector)
    side_zero = build_ground_to_air_entry_scenarios(
        half,
        seed=seed ^ 0x13579BDF,
        attacker_side=0,
        setup_weights=setup_weights,
        difficulty=difficulty,
        defender_mode=DEFENDER_LIVE,
    )
    side_one = build_ground_to_air_entry_scenarios(
        half,
        seed=seed ^ 0x2468ACE0,
        attacker_side=1,
        setup_weights=setup_weights,
        difficulty=difficulty,
        defender_mode=DEFENDER_LIVE,
    )
    _overlay_rows(state, side_zero_rows, side_zero.state)
    _overlay_rows(state, side_one_rows, side_one.state)
    state.validate()

    controlled = np.zeros(worlds, dtype=np.bool_)
    controlled[controlled_rows] = True
    attacker = np.full(worlds, -1, dtype=np.int8)
    attacker[side_zero_rows] = 0
    attacker[side_one_rows] = 1
    setup = np.full(worlds, -1, dtype=np.int8)
    setup[side_zero_rows] = side_zero.setup.astype(np.int8)
    setup[side_one_rows] = side_one.setup.astype(np.int8)
    return MixedSelfPlayInitialState(
        state=state,
        controlled_world=controlled,
        attacker_side=attacker,
        setup=setup,
        kickoff_selector=kickoff_selector,
    )


class AerialOptionSelfPlayRouterV14(AerialOptionSelfPlayRouter):
    """V12 physical router plus a one-shot failed-continuation outcome."""

    def __init__(
        self,
        lanes: int,
        *,
        device: str | torch.device,
        router_config: AerialOptionRouterConfig,
        reward_config: AerialSelfPlayRewardConfig,
        failure_config: AerialContinuationFailureConfig,
    ) -> None:
        super().__init__(
            lanes,
            device=device,
            router_config=router_config,
            reward_config=reward_config,
        )
        self.failure_config = failure_config
        self.failure_paid = torch.zeros_like(self.active)
        self.counters["landing_before_second_contact"] = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self.counters["ball_ground_before_second_contact"] = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )

    def _clear(self, mask: torch.Tensor) -> None:
        super()._clear(mask)
        self.failure_paid &= ~mask

    def observe(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        active_before: torch.Tensor,
        goal_for_lane: torch.Tensor,
    ) -> AerialRouteOutcome:
        outcome = super().observe(
            before,
            after,
            active_before=active_before,
            goal_for_lane=goal_for_lane,
        )
        active_before = active_before.to(torch.bool)
        goal_for_lane = goal_for_lane.to(torch.bool)
        before_second = self.air_contact_count < 2
        chain_started = self.entry_seen & active_before & before_second
        unpaid = chain_started & ~self.failure_paid & ~goal_for_lane
        ball_failure = unpaid & outcome.ball_ground_failure
        landed = (
            unpaid
            & ~ball_failure
            & self.ever_airborne_car
            & (self.age >= self.config.minimum_landing_release_tick)
            & (after[:, FIELD["self.on_ground"]] >= 0.5)
        )
        penalty = (
            ball_failure.to(torch.float32)
            * self.failure_config.ball_ground_before_second_contact
            + landed.to(torch.float32)
            * self.failure_config.landing_before_second_contact
        )
        failed = ball_failure | landed
        self.failure_paid |= failed
        self.counters["ball_ground_before_second_contact"] += ball_failure.sum()
        self.counters["landing_before_second_contact"] += landed.sum()
        self.reward_sum += penalty.sum(dtype=torch.float64)
        return replace(
            outcome,
            supplemental_reward=outcome.supplemental_reward + penalty,
        )

    def telemetry(self) -> dict[str, Any]:
        result = super().telemetry()
        result["version"] = GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION
        result["continuation_failure_reward"] = {
            "landing_before_second_contact": (
                self.failure_config.landing_before_second_contact
            ),
            "ball_ground_before_second_contact": (
                self.failure_config.ball_ground_before_second_contact
            ),
        }
        return result


class AerialOptionMixedSelfPlayTrainerV14(AerialOptionSelfPlayTrainerV12):
    """V12 option PPO with V14 physical continuation accounting."""

    def __init__(self, *args: Any, failure_config: AerialContinuationFailureConfig, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.failure_config = failure_config
        self.world_count = self.env.num_envs
        self.router = self._new_router()
        self.cumulative_router_counts: dict[str, int] = {}
        self.curriculum_refreshes = 0

    def _new_router(self) -> AerialOptionSelfPlayRouterV14:
        return AerialOptionSelfPlayRouterV14(
            self.env.num_envs * 2,
            device=self.device,
            router_config=self.router_config,
            reward_config=self.reward_config,
            failure_config=self.failure_config,
        )

    def replace_environment(self, env: Any) -> None:
        if env.num_envs != self.world_count or env.device != self.device:
            raise ValueError("replacement environment contract mismatch")
        self.env = env
        self.router = self._new_router()
        self.curriculum_refreshes += 1

    def collect_rollout(self) -> Any:
        rollout = super().collect_rollout()
        for name, value in (self.last_rollout_metrics or {}).get("router", {}).items():
            self.cumulative_router_counts[name] = (
                self.cumulative_router_counts.get(name, 0) + int(value)
            )
        return rollout

    def checkpoint_payload(self, provenance: dict[str, Any]) -> dict[str, Any]:
        payload = super().checkpoint_payload(provenance)
        payload.update(
            {
                "format": f"{GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION}_CHECKPOINT",
                "continuation_failure_config": {
                    "landing_before_second_contact": (
                        self.failure_config.landing_before_second_contact
                    ),
                    "ball_ground_before_second_contact": (
                        self.failure_config.ball_ground_before_second_contact
                    ),
                },
                "cumulative_router_counts": dict(self.cumulative_router_counts),
                "curriculum_refreshes": self.curriculum_refreshes,
            }
        )
        return payload


def mixed_state_summary(batch: MixedSelfPlayInitialState) -> dict[str, Any]:
    controlled = batch.controlled_world
    return {
        "worlds": int(controlled.size),
        "controlled_worlds": int(controlled.sum()),
        "ordinary_kickoff_worlds": int((~controlled).sum()),
        "controlled_by_attacker_side": {
            str(side): int((batch.attacker_side == side).sum()) for side in (0, 1)
        },
        "controlled_by_setup": {
            SETUP_NAMES[index]: int((batch.setup == index).sum())
            for index in range(len(SETUP_NAMES))
        },
    }


__all__ = [
    "GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION",
    "AerialContinuationFailureConfig",
    "AerialOptionMixedSelfPlayTrainerV14",
    "AerialOptionSelfPlayRouterV14",
    "MixedSelfPlayInitialState",
    "build_mixed_selfplay_initial_state",
    "mixed_state_summary",
]
