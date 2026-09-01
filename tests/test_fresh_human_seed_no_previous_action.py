from __future__ import annotations

import json

import torch

from benchmarks import run_rival2_fresh_human_seed_no_previous_action_v1 as masked_stage1
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


def test_selected_checkpoint_preserves_mask_and_untrained_critic() -> None:
    payload = torch.load(
        masked_stage1.CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["format"] == masked_stage1.CHECKPOINT_FORMAT
    assert payload["lineage"]["fresh_random_initialization"]
    assert not payload["lineage"]["prior_rival_checkpoint_loaded"]
    assert not payload["critic_trained"]
    assert not payload["ppo_resumable"]
    config = Rival2PolicyConfig(**payload["policy_config"])
    assert config.zero_previous_action_inputs

    torch.manual_seed(masked_stage1.INITIALIZATION_SEED)
    initial = Rival2ActorCritic(
        Rival2PolicyConfig(zero_previous_action_inputs=True)
    )
    with torch.no_grad():
        initial.actor.weight[5:10].zero_()
        initial.actor.bias[5:10].fill_(-1.0)
    for name, value in initial.critic.state_dict().items():
        assert torch.equal(value, payload["model"][f"critic.{name}"])


def test_corrected_closed_loop_evidence_is_complete_and_nonfunctional() -> None:
    result = json.loads(
        (
            masked_stage1.RESULTS / "deterministic_nexto_closed_loop.json"
        ).read_text(encoding="utf-8")
    )
    assert result["execution"]["rival_policy_hz"] == 120
    assert not result["execution"]["gaussian_sampling"]
    assert not result["execution"]["bernoulli_sampling"]
    assert result["execution"]["optimizer_steps"] == 0
    assert result["gameplay"]["rival_touches"] == 0
    assert result["gameplay"]["nexto_goals"] == result["execution"]["episodes"]
    assert not result["behavioral_interpretation"][
        "functional_gameplay_demonstrated"
    ]
