from __future__ import annotations

import torch

from benchmarks.run_rival2_fresh_human_seed_v1 import (
    hard_zero_previous_action_after_adapter,
    neutralize_previous_action_before_adapter,
)
from rivalsim.human_demo.bc_observation_bridge import FieldQuality
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM, OBS_FIELD_NAMES
from rivalsim.rival2_policy import (
    PREVIOUS_ACTION_OBSERVATION_FIELDS,
    PREVIOUS_ACTION_OBSERVATION_INDICES,
    Rival2ActorCritic,
    Rival2PolicyConfig,
)


def test_previous_action_contract_selects_exact_eight_fields() -> None:
    expected = tuple(f"previous_action.{name}" for name in ACTION_NAMES)
    assert expected == PREVIOUS_ACTION_OBSERVATION_FIELDS
    observed = tuple(
        OBS_FIELD_NAMES[index] for index in PREVIOUS_ACTION_OBSERVATION_INDICES
    )
    assert expected == observed
    assert tuple(range(167, 175)) == PREVIOUS_ACTION_OBSERVATION_INDICES


def test_legacy_policy_hash_is_unchanged() -> None:
    assert (
        Rival2PolicyConfig().content_hash
        == "58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136"
    )
    assert (
        Rival2PolicyConfig(zero_previous_action_inputs=True).content_hash
        != Rival2PolicyConfig().content_hash
    )


def test_human_pre_and_post_adapter_masks_are_exact() -> None:
    observation = torch.randn(3, OBS_DIM)
    quality = torch.full(
        (3, OBS_DIM), int(FieldQuality.EXACT_DIRECT), dtype=torch.int64
    )
    unmasked = observation.clone()
    observation, quality = neutralize_previous_action_before_adapter(
        observation, quality
    )
    indices = torch.tensor(PREVIOUS_ACTION_OBSERVATION_INDICES)
    assert torch.count_nonzero(observation.index_select(1, indices)) == 0
    assert torch.all(
        quality.index_select(1, indices) == int(FieldQuality.UNAVAILABLE)
    )
    keep = torch.ones(OBS_DIM, dtype=torch.bool)
    keep[indices] = False
    assert torch.equal(observation[:, keep], unmasked[:, keep])

    reconstructed = torch.randn(3, OBS_DIM)
    original = reconstructed.clone()
    masked = hard_zero_previous_action_after_adapter(reconstructed)
    assert torch.count_nonzero(masked.index_select(1, indices)) == 0
    assert torch.equal(masked[:, keep], original[:, keep])


def test_policy_mask_makes_actor_and_critic_invariant_to_previous_action() -> None:
    torch.manual_seed(7)
    model = Rival2ActorCritic(
        Rival2PolicyConfig(zero_previous_action_inputs=True)
    ).eval()
    left = torch.randn(5, OBS_DIM)
    right = left.clone()
    right[:, list(PREVIOUS_ACTION_OBSERVATION_INDICES)] = torch.randn(5, 8) * 100
    with torch.inference_mode():
        left_actor, left_value = model(left)
        right_actor, right_value = model(right)
    assert torch.equal(left_actor, right_actor)
    assert torch.equal(left_value, right_value)


def test_mask_buffer_does_not_change_checkpoint_tensor_schema() -> None:
    plain = Rival2ActorCritic(Rival2PolicyConfig())
    masked = Rival2ActorCritic(
        Rival2PolicyConfig(zero_previous_action_inputs=True)
    )
    assert tuple(plain.state_dict()) == tuple(masked.state_dict())
