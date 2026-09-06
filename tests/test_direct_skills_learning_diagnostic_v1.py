import pytest
import torch

from benchmarks.analyze_direct_skills_learning_v1 import aggregate, describe, summarize


def test_aggregate_uses_ended_episode_denominator_not_touch_proxy():
    row = dict(branch_updates=3, training=dict(by_reward_role_and_opponent=dict(finishing_nexto=dict(
        goals=2, episode_endings=10, success_endings=2, touches=70, samples=1800))))
    result = aggregate([row], 3, 3)['finishing_nexto']
    assert result['score_fraction_of_ended_episodes'] == .2
    assert result['contacts_per_1800_learner_decisions'] == 70
    with pytest.raises(AssertionError):
        aggregate([row], 3, 5)


def test_empty_diagnostic_subset_not_zero_success():
    assert describe(torch.empty(0)) == dict(count=0, mean=None, min=None, max=None, positive_fraction=None)


def test_masked_opponent_actions_excluded_and_reward_sign_not_invented():
    data = dict(train_mask=torch.tensor([[True, False, True]]),
                opponent_family=torch.ones(1, 3, dtype=torch.long),
                normalized_advantage=torch.tensor([[2., 100., -2.]]),
                advantages=torch.tensor([[4., 200., -4.]]))
    before = {k: torch.zeros(1, 3) for k in ('logp','entropy','max_probability','jump_probability','argmax')}
    after = {k: v.clone() for k,v in before.items()}
    after['logp'][:] = torch.tensor([[.5, 999., -.5]])
    result = summarize(data, before, after, torch.full((1,3), 4), torch.tensor([[True, True, False]]),
                       torch.tensor([[0,1,-1]]), torch.ones(1,3,dtype=torch.bool))['kickoff_nexto']
    assert result['all']['advantage']['count'] == 2
    assert result['all']['advantage_times_logp_change']['mean'] == 1
    assert result['paid_first_touch']['sampled_action_logp_change']['mean'] == .5
    assert result['native_score']['advantage']['count'] == 0
    assert result['native_concede']['advantage']['mean'] == -2
