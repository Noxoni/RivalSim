from __future__ import annotations

import torch

from rivalsim.human_demo.behavior_cloning import (
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
    simulator_retention_objective,
)
from rivalsim.rival2_policy import Rival2PolicyConfig


def test_human_objective_uses_tanh_means_raw_logits_and_teacher_logstd() -> None:
    student = torch.zeros((2, 13), requires_grad=True)
    teacher = torch.zeros((2, 13))
    teacher[:, 5:10] = -0.7
    action = torch.tensor(
        [
            [0.5, -0.5, 0.25, -0.25, 0.0, 1.0, 0.0, 1.0],
            [-0.5, 0.5, -0.25, 0.25, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    result = human_behavior_cloning_objective(
        student,
        teacher,
        action,
        smooth_l1_beta=0.1,
        analog_weight=1.0,
        button_weight=0.25,
        log_std_weight=0.05,
        policy_config=Rival2PolicyConfig(),
    )
    assert result.analog_per_channel.shape == (5,)
    assert result.button_per_channel.shape == (3,)
    assert result.log_std_per_channel.shape == (5,)
    assert result.log_std_retention.item() > 0.0
    result.loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_simulator_retention_has_actor_and_detached_critic_targets() -> None:
    student_actor = torch.zeros((4, 13), requires_grad=True)
    student_value = torch.zeros(4, requires_grad=True)
    teacher_actor = torch.full((4, 13), 0.1, requires_grad=True)
    teacher_value = torch.ones(4, requires_grad=True)
    result = simulator_retention_objective(
        student_actor,
        student_value,
        teacher_actor,
        teacher_value,
        actor_weight=2.0,
        critic_weight=1.0,
    )
    assert result.per_channel_kl.shape == (8,)
    result.loss.backward()
    assert student_actor.grad is not None
    assert student_value.grad is not None
    assert teacher_actor.grad is None
    assert teacher_value.grad is None


def test_mechanic_hierarchy_sampler_is_deterministic_and_bounded() -> None:
    labels = ["common"] * 8 + ["rare"] * 2
    attempts = ["common-a"] * 4 + ["common-b"] * 4 + ["rare-a"] * 2
    first = MechanicHierarchySampler(
        labels,
        attempts,
        uniform_label_fraction=0.1,
        maximum_oversampling_ratio=2.0,
        generator=torch.Generator().manual_seed(17),
    )
    second = MechanicHierarchySampler(
        labels,
        attempts,
        uniform_label_fraction=0.1,
        maximum_oversampling_ratio=2.0,
        generator=torch.Generator().manual_seed(17),
    )
    assert first.maximum_realized_oversampling_ratio <= 2.0
    assert torch.equal(first.sample(128), second.sample(128))


def test_action_metrics_cover_all_eight_channels_exactly() -> None:
    actor = torch.zeros((3, 13))
    actor[:, 10:13] = -1.0
    target = torch.zeros((3, 8))
    metrics = action_metric_summary(actor, target)
    assert metrics["sample_count"] == 3
    assert len(metrics["per_channel"]) == 8
    assert metrics["analog_rmse"] == 0.0
    assert metrics["button_accuracy"] == 1.0
