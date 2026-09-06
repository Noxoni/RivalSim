import numpy as np

from benchmarks.diagnose_rival2_direct_skills_learning_signal import (
    distribution,
    independent_gae,
    precedes_event,
)


def test_terminal_kills_value_bootstrap_and_later_episode_credit():
    reward = np.array([0, 10, 100], dtype=np.float32)[:, None]
    value = np.zeros_like(reward)
    following = np.full_like(reward, 30)
    terminal = np.array([0, 1, 0], dtype=bool)[:, None]
    advantage = independent_gae(
        reward, value, following, terminal, np.zeros_like(terminal), 0.9, 0.8
    )
    np.testing.assert_allclose(advantage[:, 0], [27 + 0.72 * 10, 10, 127], atol=1e-5)


def test_truncation_bootstraps_final_state_but_does_not_cross_reset():
    reward = np.array([0, 1, 100], dtype=np.float32)[:, None]
    value = np.zeros_like(reward)
    following = np.full_like(reward, 30)
    truncated = np.array([0, 1, 0], dtype=bool)[:, None]
    advantage = independent_gae(
        reward, value, following, np.zeros_like(truncated), truncated, 0.9, 0.8
    )
    np.testing.assert_allclose(advantage[:, 0], [27 + 0.72 * 28, 28, 127], atol=1e-5)


def test_event_lookahead_is_censored_at_reset_and_rollout_edge():
    event = np.array([0, 0, 0, 0, 1, 0], dtype=bool)[:, None]
    done = np.array([0, 1, 0, 0, 1, 0], dtype=bool)[:, None]
    assert precedes_event(event, done, 5).ravel().tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert precedes_event(event, done, 1).ravel().tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_empty_distribution_does_not_fabricate_a_mean():
    assert distribution(np.array([])) == {"count": 0}
    assert distribution(np.array([-1, 1]))["positive_fraction"] == 0.5
