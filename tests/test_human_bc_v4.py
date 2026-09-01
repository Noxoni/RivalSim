from __future__ import annotations

import copy
import inspect

import pytest
import torch

import rivalsim.human_demo.bc_v4_retention as v4
from benchmarks.run_rival2_human_bc_v4 import _parent_pretraining_baseline_guard
from rivalsim.human_demo.bc_v4_retention import (
    HardTailReplayState,
    V4RetentionPools,
    aligned_role_masks,
    gather_aligned_rows,
    initialize_hard_tail_replay,
    mine_training_hard_tail,
    orientation_score,
    sample_v4_retention_rows,
    v4_combined_validation_eligibility,
    v4_tail_aware_actor_retention_loss,
)
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES
from rivalsim.rival2_policy import Rival2PolicyConfig


def _policy_config() -> Rival2PolicyConfig:
    return Rival2PolicyConfig(hidden_dim=16, hidden_layers=1)


def _encoded(world: int, tick: int, car: int) -> int:
    return world * 256 + tick * 2 + car


def _observations(worlds: int) -> torch.Tensor:
    return torch.zeros(128, worlds, 2, OBS_DIM, dtype=torch.float32)


def _set_row_marker(
    observations: torch.Tensor, encoded: int, value: float
) -> None:
    world, remainder = divmod(encoded, 256)
    tick, car = divmod(remainder, 2)
    observations[tick, world, car, 0] = value


def _orientation_observations(rows: int) -> torch.Tensor:
    observations = torch.zeros(rows, OBS_DIM, dtype=torch.float32)
    observations[:, OBS_FIELD_NAMES.index("self.forward.x")] = 1.0
    observations[:, OBS_FIELD_NAMES.index("self.up.z")] = 1.0
    observations[:, OBS_FIELD_NAMES.index("self.on_ground")] = 1.0
    return observations


def _orientation_authority(*, core_threshold: float = 100.0) -> dict[str, object]:
    core_order = (
        "teacher_orientation_precision",
        "teacher_confidence_normalized_demand",
        "teacher_orientation_mean_magnitude",
        "state_body_angular_rate",
        "state_tilt_fraction",
    )
    return {
        "core_feature_order": list(core_order),
        "core_thresholds": {name: core_threshold for name in core_order},
        "context_thresholds": {
            "airborne_position_z_q35": 0.5,
            "airborne_linear_velocity_z_q35": 0.5,
            "airborne_tilt_q65": 0.25,
            "airborne_body_angular_rate_q65": 0.5,
            "contact_absolute_up_z_q20": 0.1,
            "contact_up_z_q05": -0.5,
        },
    }


def test_aligned_roles_follow_mid_trajectory_reassignment() -> None:
    family = torch.zeros(128, 2, 2, dtype=torch.int64)
    train_mask = torch.ones(128, 2, 2, dtype=torch.bool)
    family[64:, 0, 1] = 1
    train_mask[64:, 0, 1] = False
    family[:, 1, 0] = 1
    train_mask[:, 1, 0] = False
    selected = torch.tensor(
        [
            _encoded(0, 63, 1),
            _encoded(0, 64, 1),
            _encoded(1, 0, 0),
            _encoded(1, 127, 1),
        ],
        dtype=torch.int64,
    )

    roles = aligned_role_masks(selected, family, train_mask)

    assert roles["current_policy_applicable"].tolist() == [True, False, False, True]
    assert roles["counterfactual_opponent"].tolist() == [False, True, True, False]
    assert roles["historical_opponent"].tolist() == [False, True, True, False]


def test_parent_baseline_does_not_require_candidate_distribution_health() -> None:
    passing = {"accepted": True}

    result = _parent_pretraining_baseline_guard(
        complete_retention=passing,
        stress_retention=passing,
        human_distribution=passing,
    )

    assert result == {
        "checks": {
            "complete_retention": True,
            "stress_retention": True,
            "human_distribution": True,
        },
        "accepted": True,
    }


@pytest.mark.parametrize(
    "failed",
    ["complete_retention", "stress_retention", "human_distribution"],
)
def test_parent_baseline_still_fails_closed_on_authoritative_guards(
    failed: str,
) -> None:
    values = {
        "complete_retention": {"accepted": True},
        "stress_retention": {"accepted": True},
        "human_distribution": {"accepted": True},
    }
    values[failed] = {"accepted": False}

    result = _parent_pretraining_baseline_guard(**values)

    assert not result["accepted"]
    assert not result["checks"][failed]


@pytest.mark.parametrize("encoded", [torch.tensor([-1]), torch.tensor([512])])
def test_aligned_metadata_rejects_invalid_encoded_rows(encoded: torch.Tensor) -> None:
    metadata = torch.zeros(128, 2, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="encoded row"):
        gather_aligned_rows(metadata, encoded)


def test_orientation_membership_uses_only_frozen_teacher_orientation_channels() -> None:
    config = _policy_config()
    observations = _orientation_observations(2)
    teacher_actor = torch.zeros(2, config.actor_outputs)
    teacher_actor[:, 5:10] = 2.0
    teacher_actor[0, 0] = 8.0  # Throttle is deliberately not an orientation channel.
    teacher_actor[1, 1] = 3.0  # Steer is an orientation-control channel.
    authority = _orientation_authority()
    authority["core_thresholds"]["teacher_orientation_mean_magnitude"] = 0.5

    first_score, first_member, first_components = orientation_score(
        teacher_actor, observations, authority, policy_config=config
    )
    second_score, second_member, second_components = orientation_score(
        teacher_actor.clone(), observations.clone(), authority, policy_config=config
    )

    assert "student" not in inspect.signature(orientation_score).parameters
    assert first_member.tolist() == [False, True]
    assert first_components["reason_core"].tolist() == [False, True]
    assert torch.equal(first_score, second_score)
    assert torch.equal(first_member, second_member)
    assert all(
        torch.equal(first_components[name], second_components[name])
        for name in first_components
    )


def test_orientation_state_contexts_cover_recovery_and_wall_contact() -> None:
    config = _policy_config()
    observations = _orientation_observations(3)
    teacher_actor = torch.zeros(3, config.actor_outputs)
    teacher_actor[:, 5:10] = 2.0
    authority = _orientation_authority(core_threshold=1e9)
    on_ground = OBS_FIELD_NAMES.index("self.on_ground")
    position_z = OBS_FIELD_NAMES.index("self.position.z")
    velocity_z = OBS_FIELD_NAMES.index("self.linear_velocity.z")
    up_y = OBS_FIELD_NAMES.index("self.up.y")
    up_z = OBS_FIELD_NAMES.index("self.up.z")
    is_flipping = OBS_FIELD_NAMES.index("self.is_flipping")
    front_left = OBS_FIELD_NAMES.index("self.wheel_contact.front_left")

    observations[0, on_ground] = 0.0
    observations[0, position_z] = 0.25
    observations[0, velocity_z] = -0.25
    observations[0, is_flipping] = 1.0
    observations[0, up_z] = 0.0
    observations[0, up_y] = 1.0

    observations[1, front_left] = 1.0
    observations[1, up_z] = 0.0
    observations[1, up_y] = 1.0

    _score, member, components = orientation_score(
        teacher_actor, observations, authority, policy_config=config
    )

    assert member.tolist() == [True, True, False]
    assert components["reason_recovery_landing"].tolist() == [True, False, False]
    assert components["reason_wall_or_ceiling_contact"].tolist() == [False, True, False]


def test_candidate_pool_has_deterministic_unique_exact_quotas() -> None:
    natural = torch.arange(1_000, dtype=torch.int64)
    orientation = natural[:200]
    current = natural[:700]
    historical = natural[700:850]
    fractions = {
        "orientation_sensitive": 0.25,
        "current_policy_applicable": 0.25,
        "historical_opponent": 0.25,
        "natural": 0.25,
    }

    first, first_manifest = v4._build_candidate_pool(
        natural=natural,
        current=current,
        historical=historical,
        orientation=orientation,
        total_rows=200,
        fractions=fractions,
        seed=71,
    )
    second, second_manifest = v4._build_candidate_pool(
        natural=natural,
        current=current,
        historical=historical,
        orientation=orientation,
        total_rows=200,
        fractions=fractions,
        seed=71,
    )
    changed, _manifest = v4._build_candidate_pool(
        natural=natural,
        current=current,
        historical=historical,
        orientation=orientation,
        total_rows=200,
        fractions=fractions,
        seed=72,
    )

    assert torch.equal(first, second)
    assert first_manifest == second_manifest
    assert not torch.equal(first, changed)
    assert first.numel() == torch.unique(first).numel() == 200
    assert first_manifest["segment_rows"] == {
        "orientation_sensitive": 50,
        "current_policy_applicable": 50,
        "historical_opponent": 50,
        "natural": 50,
    }
    assert bool(torch.isin(first[:50], orientation).all())
    assert bool(torch.isin(first[50:100], current).all())
    assert not bool(torch.isin(first[50:100], first[:50]).any())
    assert bool(torch.isin(first[100:150], historical).all())
    assert bool(torch.isin(first[150:], natural).all())


def test_candidate_pool_rejects_invalid_fraction_or_unfillable_quota() -> None:
    natural = torch.arange(20, dtype=torch.int64)
    kwargs = {
        "natural": natural,
        "current": natural[:10],
        "historical": natural[10:15],
        "orientation": natural[:2],
        "total_rows": 10,
        "seed": 3,
    }
    with pytest.raises(ValueError, match="fractions must sum to one"):
        v4._build_candidate_pool(
            **kwargs,
            fractions={
                "orientation_sensitive": 0.1,
                "current_policy_applicable": 0.1,
                "historical_opponent": 0.1,
                "natural": 0.1,
            },
        )
    with pytest.raises(ValueError, match="quota exceeds source pool"):
        v4._build_candidate_pool(
            **kwargs,
            fractions={
                "orientation_sensitive": 0.5,
                "current_policy_applicable": 0.2,
                "historical_opponent": 0.2,
                "natural": 0.1,
            },
        )


class _MarkerPolicy(torch.nn.Module):
    def __init__(self, *, student: bool, config: Rival2PolicyConfig) -> None:
        super().__init__()
        self.student = student
        self.config = config
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor = torch.zeros(
            observation.shape[0],
            self.config.actor_outputs,
            dtype=observation.dtype,
            device=observation.device,
        ) + self.anchor * 0.0
        if self.student:
            actor[:, 0] = observation[:, 0]
        value = torch.zeros(
            observation.shape[0], dtype=observation.dtype, device=observation.device
        )
        return actor, value


def _mine(
    observations: torch.Tensor,
    candidate: torch.Tensor,
    previous: HardTailReplayState,
    *,
    top_k: int,
    max_rows: int,
    mining_round: int,
    lifetime: int = 4,
) -> v4.HardTailMiningResult:
    config = _policy_config()
    return mine_training_hard_tail(
        _MarkerPolicy(student=False, config=config),
        _MarkerPolicy(student=True, config=config),
        observations,
        candidate,
        previous,
        top_k=top_k,
        max_replay_rows=max_rows,
        replay_lifetime_generations=lifetime,
        policy_config=config,
        rows_per_batch=2,
        mining_round=mining_round,
    )


def test_hard_tail_mining_selects_highest_kl_with_encoded_row_tie_break() -> None:
    observations = _observations(1)
    markers = {9: 3.0, 4: 2.0, 7: 2.0, 2: 0.1}
    for encoded, marker in markers.items():
        _set_row_marker(observations, encoded, marker)
    candidate = torch.tensor([9, 4, 7, 2], dtype=torch.int64)
    previous = initialize_hard_tail_replay(
        torch.tensor([99], dtype=torch.int64), generation=0
    )

    result = _mine(
        observations,
        candidate,
        previous,
        top_k=3,
        max_rows=4,
        mining_round=1,
    )

    assert result.replay_rows[:3].tolist() == [9, 4, 7]
    assert result.telemetry["top_k"] == 3
    assert result.telemetry["validation_or_test_rows_inspected"] == 0
    assert result.replay_state.rows.numel() <= 4
    assert result.replay_state.last_seen_generation[:3].tolist() == [1, 1, 1]


def test_hard_tail_replay_enforces_age_and_capacity() -> None:
    observations = _observations(1)
    _set_row_marker(observations, 0, 1.0)
    _set_row_marker(observations, 1, 2.0)
    candidate = torch.tensor([0, 1], dtype=torch.int64)
    previous = initialize_hard_tail_replay(
        torch.tensor([100, 101, 102], dtype=torch.int64),
        scores=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        generation=0,
    )

    result = _mine(
        observations,
        candidate,
        previous,
        top_k=1,
        max_rows=2,
        mining_round=4,
        lifetime=4,
    )

    assert result.replay_state.rows.tolist() == [1]
    assert result.replay_state.last_seen_generation.tolist() == [4]
    assert result.replay_state.scores.dtype == torch.float64
    assert len(result.replay_state.provenance) == 1
    assert result.telemetry["expired_rows"] == 3


def test_hard_tail_mining_fails_closed_on_nonfinite_kl() -> None:
    observations = _observations(1)
    _set_row_marker(observations, 0, float("nan"))
    state = initialize_hard_tail_replay(torch.tensor([1], dtype=torch.int64))
    with pytest.raises(RuntimeError, match=r"nonfinite.*hard-tail"):
        _mine(
            observations,
            torch.tensor([0], dtype=torch.int64),
            state,
            top_k=1,
            max_rows=2,
            mining_round=1,
        )


def _toy_pools() -> V4RetentionPools:
    return V4RetentionPools(
        natural=torch.arange(0, 10, dtype=torch.int64),
        current_policy_applicable=torch.arange(10, 20, dtype=torch.int64),
        historical_opponent=torch.arange(20, 30, dtype=torch.int64),
        low_teacher_variance=torch.arange(30, 40, dtype=torch.int64),
        orientation_sensitive=torch.arange(40, 50, dtype=torch.int64),
        mining_candidate_pool=torch.arange(50, 60, dtype=torch.int64),
        initial_hard_tail_replay=torch.arange(60, 70, dtype=torch.int64),
        low_variance_threshold_log_std=-1.0,
        orientation_authority={},
        manifest={},
    )


def test_retention_sampling_has_exact_deterministic_mixture() -> None:
    pools = _toy_pools()
    replay = torch.arange(60, 70, dtype=torch.int64)
    counts = {
        "natural": 7,
        "current_policy_applicable": 5,
        "historical_opponent": 3,
        "low_teacher_variance": 2,
        "orientation_sensitive": 11,
        "hard_tail_replay": 13,
    }
    first, realized = sample_v4_retention_rows(
        pools, replay, counts, generator=torch.Generator().manual_seed(17)
    )
    second, _realized = sample_v4_retention_rows(
        pools, replay, counts, generator=torch.Generator().manual_seed(17)
    )

    assert torch.equal(first, second)
    assert realized == counts
    assert first.numel() == sum(counts.values())
    offset = 0
    available = {**pools.static_by_name(), "hard_tail_replay": replay}
    for name in (
        "natural",
        "current_policy_applicable",
        "historical_opponent",
        "low_teacher_variance",
        "orientation_sensitive",
        "hard_tail_replay",
    ):
        selected = first[offset : offset + counts[name]]
        assert bool(torch.isin(selected, available[name]).all())
        offset += counts[name]


def _tail_loss(
    teacher: torch.Tensor, student: torch.Tensor
) -> v4.V4TailRetentionLoss:
    return v4_tail_aware_actor_retention_loss(
        teacher,
        student,
        policy_config=_policy_config(),
        mean_kl_coefficient=2.0,
        total_barrier_threshold=0.5,
        total_barrier_temperature=0.05,
        total_barrier_coefficient=4.0,
        total_cvar_fraction=0.01,
        total_cvar_coefficient=4.0,
        orientation_tail_threshold=0.125,
        orientation_tail_temperature=0.0125,
        orientation_cvar_fraction=0.01,
        orientation_cvar_coefficient=4.0,
    )


@pytest.mark.parametrize("actor_index", [1, 2, 3, 4])
def test_orientation_cvar_maps_each_orientation_channel(actor_index: int) -> None:
    config = _policy_config()
    teacher = torch.zeros(1_000, config.actor_outputs, requires_grad=True)
    student = torch.zeros_like(teacher)
    student[-1, actor_index] = 3.0
    student.requires_grad_(True)

    result = _tail_loss(teacher, student)

    assert result.maximum_sample_kl > 0.5
    assert result.maximum_individual_orientation_channel_kl > 0.125
    assert result.total_cvar_barrier > result.total_mean_barrier
    assert result.orientation_cvar_barrier > 0.0
    result.loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert float(student.grad[-1, actor_index].abs()) > 0.0


def test_orientation_cvar_excludes_throttle_but_total_cvar_keeps_tail() -> None:
    config = _policy_config()
    teacher = torch.zeros(1_000, config.actor_outputs)
    student = teacher.clone()
    student[-1, 0] = 3.0

    result = _tail_loss(teacher, student)
    baseline = _tail_loss(teacher, teacher.clone())

    assert result.maximum_sample_kl > 0.5
    assert result.maximum_individual_orientation_channel_kl == 0.0
    assert result.total_cvar_barrier > baseline.total_cvar_barrier
    assert result.orientation_cvar_barrier == pytest.approx(
        float(baseline.orientation_cvar_barrier)
    )


def _retention_metrics(
    *, max_sample: float, max_orientation_channel: float
) -> dict[str, object]:
    mean_channel = {
        name: (0.001 if name not in v4.ORIENTATION_ACTION_NAMES else 0.002)
        for name in v4.ACTION_NAMES
    }
    max_channel = {
        name: (
            max_orientation_channel
            if name in v4.ORIENTATION_ACTION_NAMES
            else 0.01
        )
        for name in v4.ACTION_NAMES
    }
    return {
        "all_perspectives": {
            "mean_kl": 0.005,
            "max_sample_kl": max_sample,
            "mean_channel_kl": mean_channel,
            "max_channel_kl": max_channel,
        },
        "critic": {"rmse": 0.0, "max_absolute_drift": 0.0, "finite": True},
        "actor_finite": True,
    }


def test_complete_and_stress_validation_must_both_be_eligible() -> None:
    hard_guard = {
        "actor_mean_kl": 0.02,
        "actor_max_sample_kl": 2.0,
        "actor_max_channel_kl": 0.02,
        "critic_rmse": 0.075,
        "critic_max_absolute_drift": 0.5,
    }
    margin = {
        "maximum_sample_kl": 1.0,
        "maximum_individual_orientation_channel_kl": 0.5,
    }
    complete_metrics = _retention_metrics(
        max_sample=0.8, max_orientation_channel=0.4
    )
    stress_metrics = _retention_metrics(
        max_sample=1.1, max_orientation_channel=0.4
    )
    eligibility = v4_combined_validation_eligibility(
        complete_metrics,
        stress_metrics,
        hard_guard,
        margin,
    )

    complete = eligibility["complete_validation"]
    stress = eligibility["stress_validation"]
    assert complete["accepted"]
    assert stress["contract_accepted"]
    assert not stress["accepted"]
    assert not eligibility["accepted"]


def test_replay_state_is_immutable_and_shape_validated() -> None:
    source = torch.tensor([5, 7], dtype=torch.int64)
    state = initialize_hard_tail_replay(
        source,
        scores=torch.tensor([0.5, 0.7], dtype=torch.float64),
        generation=3,
        provenance="frozen_test",
    )
    before = copy.deepcopy(state)
    source[0] = 99
    assert torch.equal(state.rows, before.rows)
    assert torch.equal(state.last_seen_generation, before.last_seen_generation)
    assert torch.equal(state.scores, before.scores)
    assert state.provenance == before.provenance
    with pytest.raises(ValueError, match="unique"):
        initialize_hard_tail_replay(torch.tensor([5, 5], dtype=torch.int64))
    with pytest.raises(ValueError, match="scores"):
        initialize_hard_tail_replay(
            torch.tensor([5, 7], dtype=torch.int64), scores=torch.tensor([0.5])
        )
