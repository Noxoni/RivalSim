from __future__ import annotations

from benchmarks.run_rival2_gameplay_v3_continuation import (
    CHECKPOINT_ITERATIONS,
    CHECKPOINT_OFFSETS,
    FINAL_ITERATION,
    SOURCE_ITERATION,
    aggregate_training_rows,
)
from rivalsim.nexto_short_eval import (
    SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS as NEXTO_SUPPORTED_REWARDS,
)
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_V3_VERSION
from rivalsim.wisp_short_eval import (
    SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS as WISP_SUPPORTED_REWARDS,
)


def _training_row(iteration: int, scale: int) -> dict[str, object]:
    reward_names = (
        "boost_pickups",
        "boost_use",
        "demos",
        "goals",
        "mechanics",
        "progress",
        "saves",
        "speed",
        "supersonic",
        "unnecessary_flip",
    )
    raw_names = (
        "big_pad_pickups",
        "boost_use_player_decisions",
        "completed_player_episodes",
        "completed_player_episodes_hitting_mechanics_budget",
        "demo_events",
        "flip_active_touches",
        "goal_events",
        "mechanics_budget_hit_onsets",
        "mechanics_detected",
        "mechanics_paid",
        "player_decisions",
        "progress_abs_ball_y_uu",
        "progress_nonzero_world_decisions",
        "save_events",
        "small_pad_pickups",
        "speed_abs_normalized_net_units",
        "speed_nonzero_world_decisions",
        "supersonic_player_decisions",
        "touches",
        "unnecessary_flip_contacts",
        "world_decisions",
    )
    mechanics = {
        name: scale
        for name in (
            "ball_reset",
            "breezi",
            "car_reset",
            "half_flip",
            "musty",
            "pinch",
            "pogo",
            "redirect",
            "speedflip",
            "successful_dash",
        )
    }
    rewards = {
        name: {
            "absolute_blue_sum": float(
                {"mechanics": 2, "progress": 10, "unnecessary_flip": 1}.get(name, 3) * scale
            )
        }
        for name in reward_names
    }
    return {
        "ppo_safety_summary": {
            "iteration": iteration,
            "optimizer_step_proposals": 10 * scale,
            "accepted_optimizer_steps": 9 * scale,
            "policy_learning_rate_backoffs": scale,
            "transactional_retries": scale,
            "retention_budget_early_stop": scale == 1,
            "maximum_post_step_minibatch_kl": 0.01 * scale,
            "completed_update_mean_kl": 0.005 * scale,
            "retention_mean_kl": 0.006 * scale,
        },
        "reward_and_behavior_telemetry": {
            "reward_contributions": rewards,
            "absolute_gameplay_reward_sum": 100.0 * scale,
            "raw_counts_and_activity": {name: float(scale) for name in raw_names},
            "mechanics": {"detected": mechanics, "paid": mechanics},
            "ratios": {
                "mechanics_reward_to_absolute_gameplay_reward": 0.02,
                "unnecessary_flip_penalty_to_absolute_gameplay_reward": 0.01,
            },
        },
    }


def test_continuation_uses_prior_30_update_boundary_cadence() -> None:
    assert SOURCE_ITERATION == 489
    assert CHECKPOINT_OFFSETS == (30, 60, 90, 120)
    assert CHECKPOINT_ITERATIONS == (519, 549, 579, 609)
    assert FINAL_ITERATION == 609


def test_compact_evaluators_accept_gameplay_v3_checkpoints() -> None:
    assert RIVAL2_REWARD_GAMEPLAY_V3_VERSION in NEXTO_SUPPORTED_REWARDS
    assert RIVAL2_REWARD_GAMEPLAY_V3_VERSION in WISP_SUPPORTED_REWARDS


def test_training_aggregate_uses_raw_sums_not_mean_of_ratios() -> None:
    result = aggregate_training_rows([_training_row(490, 1), _training_row(491, 2)])

    assert result["accepted_updates"] == 2
    assert result["optimizer_step_proposals"] == 30
    assert result["accepted_optimizer_steps"] == 27
    assert result["retention_budget_early_stop_updates"] == [490]
    assert result["maximum_accepted_minibatch_kl"] == 0.02
    assert result["mechanics_to_absolute_gameplay_reward"] == 0.02
    assert result["bad_flip_to_absolute_gameplay_reward"] == 0.01
    assert result["mechanics_to_progress"] == 0.2
    assert result["bad_flip_to_progress"] == 0.1
    assert result["mechanics"]["detected"]["redirect"] == 3
