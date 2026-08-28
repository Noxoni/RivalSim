from __future__ import annotations

import copy

import numpy as np
import torch

from rivalsim.human_demo.bc_observation_bridge import (
    FIELD_QUALITY_SPECS,
    GLOBAL_QUALITY_MASK,
    FieldQuality,
    TrajectoryReconstructionState,
    actor_distribution_distillation_objective,
    bridge_human_frame,
    degrade_simulator_observations,
    field_quality_contract,
)
from rivalsim.human_demo.training_adapter import adapt_frame
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

_FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


def _car(stable_id: str, *, local: bool, team: int) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "team": team,
        "flags": {
            "is_local_human": local,
            "demolished": False,
            "on_ground": False,
            "supersonic": True,
            "jumped": True,
            "double_jumped": False,
            "can_jump": False,
            "has_flip": False,
        },
        "position": (100.0, -200.0, 300.0),
        "rotation": (0, 0, 0),
        "linear_velocity": (400.0, -300.0, 200.0),
        "angular_velocity": (0.25, -0.5, 0.75),
        "boost": 42.0,
        "time_off_ground": 0.625,
        "respawn_time_remaining": 0.0,
        "jump_component": {"active": True, "activity_time": 0.1},
        "dodge_component": {"active": True, "activity_time": 0.19},
        "flip_component": {"active": False, "activity_time": 0.0, "flip_time": 0.0},
        "boost_component": {"active": True, "activity_time": 0.05},
        "wheels": [
            {"index": index, "has_world_contact": index in {0, 2}}
            for index in range(4)
        ],
    }


def _frame(sequence: int) -> dict[str, object]:
    return {
        "sequence": sequence,
        "physics_frame": sequence + 100,
        "rival_action": {
            "throttle": 0.75,
            "steer": -0.25,
            "pitch": 0.5,
            "yaw": -0.125,
            "roll": 0.375,
            "jump": True,
            "boost": False,
            "handbrake": True,
        },
        "ball": {
            "position": (1000.0, 500.0, 100.0),
            "linear_velocity": (800.0, -200.0, 50.0),
            "angular_velocity": (1.0, 2.0, 3.0),
        },
        "cars": [
            _car("human", local=True, team=0),
            _car("opponent", local=False, team=1),
        ],
    }


def _bridge(
    current: dict[str, object],
    *,
    previous: dict[str, object] | None,
    state: TrajectoryReconstructionState | None = None,
    span_start: bool = False,
):
    exact = adapt_frame(
        current,
        session_uuid="session",
        previous_frame=previous,
        lifecycle_boundary_before=previous is None,
    )
    return bridge_human_frame(
        current,
        exact_sample=exact,
        trajectory_state=state or TrajectoryReconstructionState(),
        span_start=span_start,
        lifecycle_boundary_before=previous is None,
    )


def test_field_quality_contract_classifies_all_182_without_promotion() -> None:
    contract = field_quality_contract()

    assert len(FIELD_QUALITY_SPECS) == OBS_DIM == 182
    assert [row.field for row in FIELD_QUALITY_SPECS] == list(OBS_FIELD_NAMES)
    assert contract["counts"] == {
        "approximate_semantically_reconstructed": 34,
        "exact_direct": 16,
        "exactly_derivable": 58,
        "unavailable": 74,
    }
    assert set(GLOBAL_QUALITY_MASK.tolist()) == {0, 1, 2, 3}
    assert not GLOBAL_QUALITY_MASK.flags.writeable


def test_bridge_reconstructs_native_proxies_and_preserves_exact_action() -> None:
    previous = _frame(9)
    current = _frame(10)
    before = copy.deepcopy(current)

    sample = _bridge(current, previous=previous, span_start=True)

    assert sample.bc_usable
    assert not sample.exact_audit_usable
    assert sample.action_unchanged_from_exact_adapter
    np.testing.assert_array_equal(
        sample.action,
        np.asarray((0.75, -0.25, 0.5, -0.125, 0.375, 1, 0, 1), dtype=np.float32),
    )
    assert sample.observation[_FIELD["self.boost"]] == np.float32(0.42)
    assert sample.observation[_FIELD["self.is_jumping"]] == 1.0
    assert sample.observation[_FIELD["self.has_flipped"]] == 1.0
    assert sample.observation[_FIELD["self.is_flipping"]] == 1.0
    assert sample.observation[_FIELD["self.air_time"]] == np.float32(0.5)
    assert sample.observation[_FIELD["self.boosting_time"]] == np.float32(0.5)
    assert sample.quality[_FIELD["self.boost"]] == FieldQuality.APPROXIMATE
    assert sample.quality[_FIELD["ball.position.x"]] == FieldQuality.EXACT_DERIVED
    assert sample.quality[_FIELD["self.on_ground"]] == FieldQuality.EXACT_DIRECT
    assert sample.quality[_FIELD["boost_pad.0.active"]] == FieldQuality.UNAVAILABLE
    assert sample.observation[_FIELD["boost_pad.0.active"]] == 0.0
    assert np.all(sample.quality <= GLOBAL_QUALITY_MASK)
    assert not sample.observation.flags.writeable
    assert not sample.quality.flags.writeable
    assert current == before


def test_previous_action_is_approximate_when_present_and_unavailable_at_boundary() -> None:
    previous = _frame(1)
    current = _frame(2)
    present = _bridge(current, previous=previous)
    boundary = _bridge(current, previous=None)

    for channel in ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"):
        index = _FIELD[f"previous_action.{channel}"]
        assert present.quality[index] == FieldQuality.APPROXIMATE
        assert boundary.quality[index] == FieldQuality.UNAVAILABLE
        assert boundary.observation[index] == 0.0
    assert boundary.bc_usable


def test_freeplay_opponent_is_masked_as_nuisance_but_sample_remains_usable() -> None:
    frame = _frame(2)
    frame["cars"] = [frame["cars"][0]]  # type: ignore[index]
    previous = _frame(1)
    previous["cars"] = [previous["cars"][0]]  # type: ignore[index]

    sample = _bridge(frame, previous=previous)

    assert sample.bc_usable
    for index, field in enumerate(OBS_FIELD_NAMES):
        if field.startswith("opponent.") or field.startswith("relative.opponent"):
            assert sample.quality[index] == FieldQuality.UNAVAILABLE
            assert sample.observation[index] == 0.0
    assert sample.quality[_FIELD["self.position.x"]] == FieldQuality.EXACT_DERIVED


def test_lifecycle_age_reconstruction_is_deterministic_span_local_lower_bound() -> None:
    state_a = TrajectoryReconstructionState()
    state_b = TrajectoryReconstructionState()
    rows_a = []
    rows_b = []
    previous_a = _frame(0)
    previous_b = _frame(0)
    for sequence in range(1, 5):
        current_a = _frame(sequence)
        current_b = _frame(sequence)
        rows_a.append(
            _bridge(
                current_a,
                previous=previous_a,
                state=state_a,
                span_start=sequence == 1,
            ).observation[_FIELD["lifecycle.episode_age"]]
        )
        rows_b.append(
            _bridge(
                current_b,
                previous=previous_b,
                state=state_b,
                span_start=sequence == 1,
            ).observation[_FIELD["lifecycle.episode_age"]]
        )
        previous_a = current_a
        previous_b = current_b

    np.testing.assert_array_equal(rows_a, rows_b)
    assert rows_a[0] == 0.0
    assert rows_a == sorted(rows_a)


def test_simulator_degradation_uses_neutral_only_with_unavailable_mask() -> None:
    source = np.arange(2 * OBS_DIM, dtype=np.float32).reshape(2, OBS_DIM) / 100.0

    degraded, quality = degrade_simulator_observations(source)

    unavailable = GLOBAL_QUALITY_MASK == FieldQuality.UNAVAILABLE
    available = ~unavailable
    np.testing.assert_array_equal(degraded[:, available], source[:, available])
    np.testing.assert_array_equal(degraded[:, unavailable], 0.0)
    np.testing.assert_array_equal(quality[0], GLOBAL_QUALITY_MASK)
    assert not degraded.flags.writeable
    assert not quality.flags.writeable


def test_distillation_interface_is_finite_training_ready_and_nonmutating() -> None:
    torch.manual_seed(17)
    teacher = Rival2ActorCritic(Rival2PolicyConfig())
    student = Rival2ActorCritic(Rival2PolicyConfig())
    student.load_state_dict(teacher.state_dict())
    teacher.eval()
    student.eval()
    true = torch.randn(8, OBS_DIM)
    degraded_np, quality_np = degrade_simulator_observations(true.numpy())
    teacher_before = [parameter.detach().clone() for parameter in teacher.parameters()]
    student_before = [parameter.detach().clone() for parameter in student.parameters()]
    teacher_gradients_before = [parameter.grad for parameter in teacher.parameters()]
    student_gradients_before = [parameter.grad for parameter in student.parameters()]

    result = actor_distribution_distillation_objective(
        teacher,
        student,
        true,
        torch.from_numpy(np.asarray(degraded_np).copy()),
        torch.from_numpy(np.asarray(quality_np).copy()),
    )

    assert result.loss.requires_grad
    assert torch.isfinite(result.loss)
    assert result.loss >= 0
    assert result.per_action_channel_kl.shape == (8,)
    assert result.per_sample_kl.shape == (8,)
    assert result.teacher_actor.shape == result.student_actor.shape == (8, 13)
    assert result.unavailable_fraction == torch.tensor(74 / 182)
    for before, after in zip(teacher_before, teacher.parameters(), strict=True):
        torch.testing.assert_close(before, after)
    for before, after in zip(student_before, student.parameters(), strict=True):
        torch.testing.assert_close(before, after)
    assert all(
        before is after
        for before, after in zip(
            teacher_gradients_before,
            (parameter.grad for parameter in teacher.parameters()),
            strict=True,
        )
    )
    assert all(
        before is after
        for before, after in zip(
            student_gradients_before,
            (parameter.grad for parameter in student.parameters()),
            strict=True,
        )
    )


def test_distillation_interface_rejects_moving_teacher_alias() -> None:
    model = Rival2ActorCritic(Rival2PolicyConfig())
    observations = torch.zeros(2, OBS_DIM)
    degraded, quality = degrade_simulator_observations(observations.numpy())

    with np.testing.assert_raises_regex(
        ValueError, "teacher and student models must be independent"
    ):
        actor_distribution_distillation_objective(
            model,
            model,
            observations,
            torch.from_numpy(np.asarray(degraded).copy()),
            torch.from_numpy(np.asarray(quality).copy()),
        )
