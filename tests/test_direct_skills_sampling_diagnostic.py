import torch
import pytest

from benchmarks.diagnose_direct_skills_sampling_000650 import SampledEvaluationPolicy,authority
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic


def test_sample_is_exact_training_distribution_and_action_table():
    p=SampledEvaluationPolicy(17)
    x=torch.linspace(-3,3,32*90).reshape(32,90)
    ref=torch.Generator().manual_seed(17)
    _,action,_=p.sample(x/2,ref)
    assert torch.equal(action,p.deterministic(x))
    assert torch.equal(ref.get_state(),p.action_generator.get_state())


def test_sampling_rng_replays_without_altering_weights():
    p=SampledEvaluationPolicy(19)
    before={k:v.clone() for k,v in p.state_dict().items()}
    rng=p.action_generator.get_state()
    a=p.deterministic(torch.zeros(128,90))
    p.action_generator.set_state(rng)
    assert torch.equal(a,p.deterministic(torch.zeros(128,90)))
    assert all(torch.equal(v,p.state_dict()[k]) for k,v in before.items())


def test_forward_logits_hidden_and_state_schema_remain_exact():
    p=SampledEvaluationPolicy(5).eval();q=EntityJointControlActorCritic().eval()
    q.load_state_dict(p.state_dict(),strict=True)
    obs=torch.zeros(4,182);obs[:,9+3:9+6]=1
    with torch.no_grad():
        a,h=p.forward_actor(obs,p.initial_hidden(4))
        b,k=q.forward_actor(obs,q.initial_hidden(4))
    assert torch.equal(a,b) and torch.equal(h,k)
    assert p.state_dict().keys()==q.state_dict().keys()
    assert torch.equal(q.deterministic(b),q.action_table[b.argmax(-1)])


def test_nonfinite_is_not_silently_sampled():
    p=SampledEvaluationPolicy(1)
    with pytest.raises(RuntimeError,match="nonfinite"):
        p.deterministic(torch.full((1,90),float('nan')))


def test_fixed_diagnostic_not_new_acceptance_or_training():
    spec=authority()
    assert spec['optimizer_steps']==0 and not spec['model_selection']
    assert len(set(spec['sampling_seeds']))==3
    assert 'NOT a same-method' in spec['inference']
