from __future__ import annotations

import copy

import numpy as np
import pytest

from rivalsim.human_demo.training_adapter import (
    action_target,
    adapt_frame,
    contract_identity,
    frames_are_contiguous,
    split_gameplay_regions,
    split_mechanic_candidates,
)
from rivalsim.rival2_contracts import OBS_FIELD_NAMES


def _car(stable_id: str, *, local: bool, team: int) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "team": team,
        "flags": {
            "is_local_human": local,
            "on_ground": True,
            "jumped": False,
            "double_jumped": False,
            "can_jump": True,
            "supersonic": False,
        },
        "position": (100.0, -200.0, 17.0),
        "rotation": (0, 0, 0),
        "linear_velocity": (400.0, -300.0, 0.0),
        "angular_velocity": (0.25, -0.5, 0.75),
        "boost": 33.0,
        "wheels": [
            {"index": index, "has_world_contact": index in {0, 2}}
            for index in range(4)
        ],
    }


def _frame(sequence: int, *, physics_frame: int | None = None) -> dict[str, object]:
    return {
        "sequence": sequence,
        "physics_frame": sequence + 100 if physics_frame is None else physics_frame,
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


def test_action_target_is_exact_ordered_float32_and_read_only() -> None:
    frame = _frame(1)
    before = copy.deepcopy(frame)

    target = action_target(frame)

    np.testing.assert_array_equal(
        target,
        np.asarray((0.75, -0.25, 0.5, -0.125, 0.375, 1, 0, 1), dtype=np.float32),
    )
    assert target.dtype == np.float32
    assert target.shape == (8,)
    assert not target.flags.writeable
    assert frame == before


@pytest.mark.parametrize("channel,value", [("throttle", 1.01), ("jump", 0.5)])
def test_action_target_rejects_noncontract_values(channel: str, value: float) -> None:
    frame = _frame(1)
    frame["rival_action"][channel] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        action_target(frame)


def test_adapter_uses_exact_preceding_action_without_mutation_or_fabrication() -> None:
    previous = _frame(9)
    current = _frame(10)
    before = copy.deepcopy(current)

    sample = adapt_frame(
        current,
        session_uuid="session",
        previous_frame=previous,
    )

    assert sample.previous_action_source_sequence == 9
    for channel, expected in zip(
        ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"),
        action_target(previous),
        strict=True,
    ):
        index = OBS_FIELD_NAMES.index(f"previous_action.{channel}")
        assert sample.exact_field_mask[index]
        assert sample.partial_observation[index] == expected
    assert sample.observation is None
    assert sample.partial_observation.shape == (182,)
    assert sample.exact_field_mask.shape == (182,)
    assert not sample.partial_observation.flags.writeable
    assert not sample.exact_field_mask.flags.writeable
    assert current == before

    boundary = adapt_frame(
        current,
        session_uuid="session",
        previous_frame=previous,
        lifecycle_boundary_before=True,
    )
    assert boundary.previous_action_source_sequence is None
    assert all(
        f"previous_action.{channel}" in boundary.blocked_fields
        for channel in ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
    )


def test_adapter_fails_closed_on_missing_opponent_and_nonunique_wheels() -> None:
    frame = _frame(2)
    frame["cars"] = [frame["cars"][0]]  # type: ignore[index]
    for wheel in frame["cars"][0]["wheels"]:  # type: ignore[index]
        wheel["index"] = 0

    sample = adapt_frame(frame, session_uuid="freeplay", previous_frame=_frame(1))

    assert not sample.usable
    assert "opponent_absent_from_native_freeplay_state" in sample.blocker_reasons
    assert "self_wheel_identity_not_unique" in sample.blocker_reasons
    assert "opponent.position.x" in sample.blocked_fields
    assert "self.wheel_contact.front_left" in sample.blocked_fields
    assert np.isnan(
        sample.partial_observation[OBS_FIELD_NAMES.index("opponent.position.x")]
    )


def test_frame_continuity_requires_both_sequence_and_physics_identity() -> None:
    assert frames_are_contiguous(_frame(1), _frame(2))
    assert not frames_are_contiguous(_frame(1), _frame(2, physics_frame=999))
    assert not frames_are_contiguous(_frame(1), _frame(3))


def test_mechanic_split_is_deterministic_disjoint_and_whole_attempt() -> None:
    candidates = [
        {
            "attempt_id": f"air:{index:02d}",
            "declared_label": "air",
            "session_uuid": "air-session",
            "start_sequence": index * 100,
            "end_sequence": index * 100 + 99,
        }
        for index in range(10)
    ]
    candidates += [
        {
            "attempt_id": f"rare:{index:02d}",
            "declared_label": "rare",
            "session_uuid": "rare-session",
            "start_sequence": index * 10,
            "end_sequence": index * 10 + 9,
        }
        for index in range(3)
    ]

    first = split_mechanic_candidates(candidates)
    second = split_mechanic_candidates(reversed(candidates))

    assert first == second
    assert len(first) == len({row["attempt_id"] for row in first}) == 13
    assert {row["split"] for row in first if row["declared_label"] == "air"} == {
        "train",
        "validation",
        "test",
    }
    assert {row["split"] for row in first if row["declared_label"] == "rare"} == {
        "train",
        "validation",
        "test",
    }
    assert all(row["split_hash"] for row in first)


def test_gameplay_split_uses_only_complete_regions_and_is_deterministic() -> None:
    regions = [
        {
            "region_id": f"region-{index}",
            "start_sequence": index * 100,
            "end_sequence": index * 100 + 99,
            "source_frame_count": 100,
            "boundary_before": ["kickoff_or_round_reset"],
        }
        for index in range(10)
    ]

    first = split_gameplay_regions(regions, session_uuid="gameplay")
    second = split_gameplay_regions(reversed(regions), session_uuid="gameplay")

    assert first == second
    assert sum(row["source_frame_count"] for row in first) == 1000
    assert {row["split"] for row in first} == {"train", "validation", "test"}
    assert all(row["end_sequence"] - row["start_sequence"] + 1 == 100 for row in first)


def test_contract_identity_is_frozen_to_rival_v2_120hz() -> None:
    identity = contract_identity()
    assert identity["observation_version"] == "RIVAL2_OBS_V2_120HZ"
    assert identity["observation_shape"] == [182]
    assert identity["observation_schema_sha256"] == (
        "BF9E141E5A1E5D2F15581C8BBB10F31F11FC5AA6736B327E61C03DD8D2388237"
    )
    assert identity["action_version"] == "RIVAL2_ACTION_V2_120HZ"
    assert identity["action_shape"] == [8]
    assert identity["action_contract_sha256"] == (
        "5E3747CCF9F59BA18D81D07014D60637F7D886907A0F44B0CA681C74F20EF91A"
    )
    assert identity["temporal_reduction"] is None
