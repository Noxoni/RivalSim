import numpy as np
import torch

from benchmarks.diagnose_rival2_direct_skills_learning_signal import (
    distribution,
    independent_gae,
    precedes_event,
    stepwise_replay,
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


def test_stepwise_replay_preserves_batch_history_and_reset():
    class TinyRecurrent:
        def __call__(self, obs, hidden, *, reset_before):
            assert obs.shape == (2, 182)
            hidden = hidden.masked_fill(reset_before[None, :, None], 0)
            hidden = hidden + obs[:, :1][None]
            score = hidden[0, :, 0]
            return torch.stack((score, -score), -1), score, hidden

    obs = torch.ones(2, 3, 182)
    reset = torch.tensor([[False, False, True], [True, False, False]])
    value = torch.tensor([[5.0, 6.0, 1.0], [1.0, 2.0, 3.0]])
    logits = torch.stack((value, -value), -1)
    expected = logits.log_softmax(-1)[..., 0]
    data = dict(
        observations=obs,
        initial_hidden=torch.full((1, 2, 1), 4.0),
        reset_before=reset,
        values=value,
        old_log_probability=expected,
        action_indices=torch.zeros(2, 3, dtype=torch.long),
        train_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    replay, report = stepwise_replay(TinyRecurrent(), data)
    assert torch.equal(replay, expected)
    assert report["collected_log_probability_exactly_equal"]
    assert report["absolute_value_error"]["max"] == 0
