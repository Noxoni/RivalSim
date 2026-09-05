import json

import pytest

from benchmarks.report_rival2_ssl_entity_training import contiguous_prefix, summarize
from third_party.nexto.adapter import build_action_table


def test_prefix_requires_exact_order_and_ignores_incomplete_tail():
    lines = [json.dumps({"accepted_updates": i}).encode() + b"\n" for i in (1, 2)]
    assert len(contiguous_prefix([*lines, b'{"accepted_updates":'], 2)) == 2
    with pytest.raises(ValueError):
        contiguous_prefix(lines[::-1], 2)
    with pytest.raises(ValueError):
        contiguous_prefix([*lines, lines[-1]], 3)


def test_count_weighted_summary_and_table_button_consistency():
    table = build_action_table("cpu").tolist()
    rows = []
    for i, (touched, ended) in enumerate(((1, 10), (90, 100)), 1):
        samples = 5898240
        counts = [samples] + [0] * 89
        rows.append(
            dict(
                accepted_updates=i,
                rollout_seconds=6.0,
                ppo_seconds=25.0,
                ppo=dict(
                    optimizer_steps=182,
                    kl_rejections=0,
                    completed_update_mean_kl=0.01,
                    completed_update_sample_kl_max=3.0,
                ),
                training=dict(
                    action_index_counts=counts,
                    trainable_agent_samples=samples,
                    current_selfplay_only=True,
                    kl_telemetry_only=True,
                    jump=samples * table[0][5],
                    boost=samples * table[0][6],
                    handbrake=samples * table[0][7],
                    speed=samples * 500.0,
                    entropy=samples * 3.0,
                    touches=touched,
                    goals=1,
                    resets=ended // 2,
                    no_touch=1,
                    ended_player_episodes=ended,
                    episodes_with_touch=touched,
                    first_touch_age=touched * 2.0,
                    first_touches=touched,
                    goalward_touches=touched,
                    potential_reward_components={"terminal_goal": 0.0},
                ),
            )
        )
    result = summarize(rows, table)
    assert result["ended_player_episode_touch_fraction"] == 91 / 110
    assert result["first_touch_seconds_conditional"] == 2.0
    assert result["max_completed_update_sample_kl"] == 3.0  # Telemetry only.
    rows[0]["training"]["jump"] += 1
    with pytest.raises(AssertionError):
        summarize(rows, table)
