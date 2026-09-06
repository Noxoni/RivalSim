import math

import pytest
import torch

from rivalsim.direct_skills_log_barrier_v1 import uniform_log_barrier, sequence_loss
from rivalsim.ssl_entity_training import joint_sequence_loss
from rivalsim.fresh_ground_30hz import ppo_config


def test_uniform_reference_zero_and_analytic_gradient():
    logits = torch.tensor([[0., 0., 0.], [80., -50., -100.]], dtype=torch.float64, requires_grad=True)
    penalty = uniform_log_barrier(logits)
    assert abs(float(penalty[0])) < 1e-12
    penalty.sum().backward()
    assert torch.allclose(logits.grad, logits.softmax(-1) - 1/3, atol=1e-12)
    assert logits.grad[1, 2] == pytest.approx(-1/3)


def test_entropy_alone_vanishes_but_barrier_does_not():
    logits = torch.tensor([100., -100., -100.], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    entropy_gradient = torch.autograd.grad((logp.exp() * logp).sum(), logits, retain_graph=True)[0]
    barrier_gradient = torch.autograd.grad(uniform_log_barrier(logits), logits)[0]
    assert entropy_gradient.norm() < 1e-70
    assert barrier_gradient.norm() > .5


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(5, 7)
        self.critic = torch.nn.Linear(5, 1)

    def forward(self, obs, hidden, reset_before):
        return self.actor(obs), self.critic(obs).squeeze(-1), hidden


def fixture():
    torch.manual_seed(42)
    model = SmallPolicy()
    data = dict(observations=torch.randn(3,4,5), initial_hidden=torch.zeros(1,3,2),
        action_indices=torch.randint(0,7,(3,4)), reset_before=torch.zeros(3,4,dtype=torch.bool),
        train_mask=torch.tensor([[True]*4,[False]*4,[True,True,False,True]]),
        normalized_advantage=torch.randn(3,4), returns=torch.randn(3,4))
    logits,_,_ = model(data['observations'],data['initial_hidden'],data['reset_before'])
    data['old_log_probability'] = logits.detach().log_softmax(-1).gather(-1,data['action_indices'][...,None]).squeeze(-1)
    return model,data,torch.arange(3)


def test_zero_coefficient_preserves_original_loss_and_gradients():
    model,data,index=fixture()
    original,_=joint_sequence_loss(model,data,index,ppo_config())
    changed,_=sequence_loss(model,data,index,ppo_config(),coefficient=0)
    assert torch.equal(original,changed)
    a=torch.autograd.grad(original,tuple(model.parameters()))
    b=torch.autograd.grad(changed,tuple(model.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_exploration_excludes_masked_opponents_and_has_no_critic_gradient():
    model,data,index=fixture()
    total,_=sequence_loss(model,data,index,ppo_config(),coefficient=.03)
    original,_=joint_sequence_loss(model,data,index,ppo_config())
    gradient=torch.autograd.grad(total-original,tuple(model.critic.parameters()))
    assert all(not bool(x.any()) for x in gradient)
    original_logits=model.actor(data['observations'])
    expected=uniform_log_barrier(original_logits)[data['train_mask']].mean()*.03
    assert torch.allclose(total-original,expected)
    with pytest.raises(ValueError):
        sequence_loss(model,data,index,ppo_config(),coefficient=math.inf)
