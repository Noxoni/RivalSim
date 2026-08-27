"""GPU-native Rival 2.0 training in complete five-minute 1v1 matches."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from rivalsim.kernels.rival2 import rival2_begin_decision
from rivalsim.kernels.rival2_full_match import (
    REGULATION_TICKS,
    REWARD_BASE,
    REWARD_GOAL_ONLY,
    REWARD_SCORING,
    rival2_full_match_accumulate_tick,
    rival2_full_match_after_reset,
)
from rivalsim.rival2_contracts import (
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_GOAL_ONLY_VERSION,
    RIVAL2_REWARD_SCORING_V1_VERSION,
    RIVAL2_REWARD_V2_VERSION,
    RIVAL2_REWARD_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import (
    Rival2Env,
    Rival2EpisodeState,
    Rival2TensorBridge,
    Rival2WorldSim,
)
from rivalsim.static_world import CompleteWorldSim


class Rival2FullMatchState(Rival2EpisodeState):
    """Trainer accounting plus persistent regulation/overtime match state."""

    _FULL_VIEW_NAMES = (
        "regulation_ticks_remaining",
        "blue_score",
        "orange_score",
        "overtime",
        "match_done",
        "winner",
        "pending_kickoff_reset",
        "match_goal_count",
        "match_blue_touches",
        "match_orange_touches",
        "kickoff_segment_active",
        "kickoff_segment_ticks",
        "kickoff_segments_total",
        "no_touch_segments_total",
        "completed_matches",
        "completed_blue_wins",
        "completed_orange_wins",
        "completed_overtime_matches",
        "completed_blue_goals",
        "completed_orange_goals",
        "completed_blue_touches",
        "completed_orange_touches",
        "completed_match_goals",
        "completed_match_ticks",
    )

    def __init__(self, num_envs: int, device: str):
        super().__init__(num_envs, device)
        self.physics_reset_mask = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.regulation_ticks_remaining = wp.full(
            num_envs, REGULATION_TICKS, dtype=wp.int32, device=device
        )
        for name in (
            "blue_score",
            "orange_score",
            "overtime",
            "match_done",
            "pending_kickoff_reset",
            "match_goal_count",
            "match_blue_touches",
            "match_orange_touches",
            "kickoff_segment_ticks",
            "kickoff_segments_total",
            "no_touch_segments_total",
            "completed_matches",
            "completed_blue_wins",
            "completed_orange_wins",
            "completed_overtime_matches",
            "completed_blue_goals",
            "completed_orange_goals",
            "completed_blue_touches",
            "completed_orange_touches",
            "completed_match_goals",
            "completed_match_ticks",
        ):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.int32, device=device))
        self.winner = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.kickoff_segment_active = wp.ones(
            num_envs, dtype=wp.int32, device=device
        )

    @property
    def logical_bytes(self) -> int:
        return super().logical_bytes + self.num_envs * 25 * 4

    def torch_views(self) -> dict[str, torch.Tensor]:
        return {
            name: wp.to_torch(getattr(self, name))
            for name in self._FULL_VIEW_NAMES
        }


class Rival2FullMatchWorldSim(Rival2WorldSim):
    """Complete world with match-owned done semantics and goal kickoffs."""

    def __init__(self, *args: Any, reward_mode: int, **kwargs: Any):
        self.reward_mode = int(reward_mode)
        super().__init__(*args, reward_mode=reward_mode, **kwargs)
        self.rival2 = Rival2FullMatchState(self.num_envs, self.device)

    def begin_decision(self) -> None:
        state = self.rival2
        wp.launch(
            rival2_begin_decision,
            dim=self.num_envs,
            inputs=[
                self.state.ball_pos,
                state.interval_tick,
                state.ball_y_before,
                state.ball_y_after,
                state.touch_count,
                state.first_contact_count,
                state.demo_by_count,
                state.demoed_event,
                state.goal_latched,
                state.scoring_team_latched,
                state.terminated,
                state.truncated,
                state.reset_mask,
                state.reward,
                state.kickoff_indicator,
            ],
            device=self.device,
        )

    def _launch_tick(self) -> None:
        # Bypass Rival2WorldSim's short-episode accumulator while retaining the
        # exact accepted CompleteWorldSim physics and lifecycle launch order.
        CompleteWorldSim._launch_tick(self)
        state = self.rival2
        wp.launch(
            rival2_full_match_accumulate_tick,
            dim=self.num_envs,
            inputs=[
                self.reward_mode,
                self.state.ball_pos,
                self.lifecycle.goal_scored,
                self.lifecycle.scoring_team,
                self.car_ball.hit_this_tick,
                self.car_ball_b.hit_this_tick,
                self.car_car.event_count,
                self.car_car.event_bumper,
                self.car_car.event_victim,
                self.car_car.event_is_demo,
                state.interval_tick,
                state.episode_ticks,
                state.no_touch_ticks,
                state.ball_y_before,
                state.ball_y_after,
                state.touch_count,
                state.touch_contact_latched,
                state.demo_by_count,
                state.demoed_event,
                state.goal_latched,
                state.scoring_team_latched,
                state.terminated,
                state.truncated,
                state.reset_mask,
                state.physics_reset_mask,
                state.reward,
                state.regulation_ticks_remaining,
                state.blue_score,
                state.orange_score,
                state.overtime,
                state.match_done,
                state.winner,
                state.pending_kickoff_reset,
                state.match_goal_count,
                state.match_blue_touches,
                state.match_orange_touches,
                state.kickoff_segment_active,
                state.kickoff_segment_ticks,
                state.kickoff_segments_total,
                state.no_touch_segments_total,
                state.completed_matches,
                state.completed_blue_wins,
                state.completed_orange_wins,
                state.completed_overtime_matches,
                state.completed_blue_goals,
                state.completed_orange_goals,
                state.completed_blue_touches,
                state.completed_orange_touches,
                state.completed_match_goals,
                state.completed_match_ticks,
            ],
            device=self.device,
        )

    def apply_interval_resets(self) -> None:
        state = self.rival2
        self._launch_physical_resets(state.physics_reset_mask)
        wp.launch(
            rival2_full_match_after_reset,
            dim=self.num_envs,
            inputs=[
                state.physics_reset_mask,
                state.reset_mask,
                state.episode_ticks,
                state.no_touch_ticks,
                state.kickoff_indicator,
                state.touch_count,
                state.touch_contact_latched,
                state.demo_by_count,
                state.demoed_event,
                state.previous_action,
                state.regulation_ticks_remaining,
                state.blue_score,
                state.orange_score,
                state.overtime,
                state.match_done,
                state.winner,
                state.pending_kickoff_reset,
                state.match_goal_count,
                state.match_blue_touches,
                state.match_orange_touches,
                state.kickoff_segment_active,
                state.kickoff_segment_ticks,
                self.lifecycle.blue_score,
                self.lifecycle.orange_score,
            ],
            device=self.device,
        )


class Rival2FullMatchEnv(Rival2Env):
    """Thirty-hertz PPO interface whose episode is one complete match."""

    def __init__(
        self,
        num_envs: int,
        collision_root: str,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        reward_version: str = RIVAL2_REWARD_V2_VERSION,
        **world_kwargs: Any,
    ):
        if reward_version in (RIVAL2_REWARD_VERSION, RIVAL2_REWARD_V2_VERSION):
            reward_mode = REWARD_BASE
        elif reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION:
            reward_mode = REWARD_GOAL_ONLY
        elif reward_version == RIVAL2_REWARD_SCORING_V1_VERSION:
            reward_mode = REWARD_SCORING
        else:
            raise ValueError(f"unsupported full-match reward: {reward_version}")
        self.reward_version = reward_version
        self.episode_version = RIVAL2_FULL_MATCH_EPISODE_VERSION
        self.contract_hashes = contract_hashes_for_reward(
            reward_version, self.episode_version
        )
        self.world = Rival2FullMatchWorldSim(
            num_envs,
            collision_root,
            device=device,
            seed=seed,
            reward_mode=reward_mode,
            **world_kwargs,
        )
        self.device = torch.device(self.world.device)
        self._activate_torch_stream()
        self.bridge = Rival2TensorBridge(self.world)
        self.full_match_views = self.world.rival2.torch_views()
        self.observation = self.bridge.observation()
        self.decision_count = 0

    def set_reward_version(self, reward_version: str) -> None:
        """Apply an authorized update-boundary reward-only transition."""

        if reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION:
            reward_mode = REWARD_GOAL_ONLY
        elif reward_version == RIVAL2_REWARD_SCORING_V1_VERSION:
            reward_mode = REWARD_SCORING
        elif reward_version in (RIVAL2_REWARD_VERSION, RIVAL2_REWARD_V2_VERSION):
            reward_mode = REWARD_BASE
        else:
            raise ValueError(f"unsupported full-match reward: {reward_version}")
        self.reward_version = reward_version
        self.contract_hashes = contract_hashes_for_reward(
            reward_version, self.episode_version
        )
        self.world.reward_mode = reward_mode

    def start_fresh_matches(self) -> None:
        """Start one new standard match in every lane at an explicit phase boundary."""

        state = self.world.rival2
        state.physics_reset_mask.fill_(1)
        state.reset_mask.fill_(1)
        self.world.apply_interval_resets()
        self.observation = self.bridge.observation()


__all__ = [
    "Rival2FullMatchEnv",
    "Rival2FullMatchState",
    "Rival2FullMatchWorldSim",
]
