from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_human_sequence import (
    RETAINED_OBSERVATION_INDICES,
    ZEROED_OBSERVATION_INDICES,
    project_human_sequence_observation,
)
from rivalsim.rival2_recurrent_policy import (
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
    deterministic_recurrent_action,
)


def _small_model() -> Rival2RecurrentActorCritic:
    torch.manual_seed(7)
    return Rival2RecurrentActorCritic(
        Rival2RecurrentPolicyConfig(encoder_dim=16, hidden_dim=16, post_dim=16)
    )


def test_human_sequence_projection_preserves_only_frozen_shared_fields() -> None:
    source = np.arange(2 * OBS_DIM, dtype=np.float32).reshape(2, OBS_DIM) + 1.0
    projected = project_human_sequence_observation(source)
    assert isinstance(projected, np.ndarray)
    np.testing.assert_array_equal(
        projected[:, RETAINED_OBSERVATION_INDICES],
        source[:, RETAINED_OBSERVATION_INDICES],
    )
    assert np.count_nonzero(projected[:, ZEROED_OBSERVATION_INDICES]) == 0


def test_recurrent_policy_shape_and_reset_match_fresh_hidden() -> None:
    model = _small_model().eval()
    observation = torch.randn(2, 7, OBS_DIM)
    reset = torch.zeros(2, 7, dtype=torch.bool)
    reset[:, 4] = True
    actor, value, hidden = model(observation, reset_before=reset)
    fresh_actor, fresh_value, fresh_hidden = model(observation[:, 4:])
    torch.testing.assert_close(actor[:, 4:], fresh_actor)
    torch.testing.assert_close(value[:, 4:], fresh_value)
    torch.testing.assert_close(hidden, fresh_hidden)
    assert actor.shape == (2, 7, 13)
    assert value.shape == (2, 7)
    assert hidden.shape == (1, 2, 16)


def test_policy_enforces_projection_before_encoder() -> None:
    model = _small_model().eval()
    base = torch.randn(3, 5, OBS_DIM)
    altered = base.clone()
    altered[..., list(ZEROED_OBSERVATION_INDICES)] += 1000.0
    first = model(base)[0]
    second = model(altered)[0]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_deterministic_recurrent_action_uses_hybrid_contract() -> None:
    actor = torch.zeros(1, 13)
    actor[0, :5] = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5])
    actor[0, 10:13] = torch.tensor([-0.1, 0.0, 0.1])
    action = deterministic_recurrent_action(actor)
    torch.testing.assert_close(action[0, :5], torch.tanh(actor[0, :5]))
    torch.testing.assert_close(action[0, 5:], torch.tensor([0.0, 1.0, 1.0]))
