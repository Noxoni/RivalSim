from copy import deepcopy

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, training_starts
from rivalsim.sustained_standing_kickoff_v1 import (
    VERSION, standing_training_starts, validate_resume,
)
from rivalsim.fresh_ground_30hz import scenario_hash
from tests.test_fresh_ground_30hz import force_goal


@pytest.mark.parametrize("retired", [False, True])
def test_all_kickoffs_exactly_native_and_everything_else_unchanged(retired):
    old = training_starts(1024, retired)
    new = standing_training_starts(1024, retired)
    rows = np.flatnonzero(new.kickoff_indicator)
    assert set(new.kickoff_layout[rows]) == set(range(5))
    assert set(new.focal_side[rows]) == {0, 1}
    assert set(new.family[rows]) == {0, 4}
    native = make_standard_kickoff_state(len(rows), new.kickoff_layout[rows])
    for key in new.state.__dataclass_fields__:
        a, b = getattr(old.state, key), getattr(new.state, key)
        np.testing.assert_array_equal(b[rows], getattr(native, key))
        np.testing.assert_array_equal(a[new.kickoff_indicator == 0], b[new.kickoff_indicator == 0])
        if key != "car_vel":
            np.testing.assert_array_equal(a, b)
    for key in ("family", "focal_side", "kickoff_indicator", "kickoff_layout", "wall_aerial_variant"):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
    assert np.any(old.state.car_vel[rows] != 0), "Regression fixture must expose the removed assistance"
    assert not new.state.car_vel[rows].any()
    assert not new.state.car_ang_vel[rows].any()
    assert scenario_hash(new) == scenario_hash(standing_training_starts(1024, retired))


def test_resume_cannot_use_an_unbound_old_checkpoint_or_downgrade_amendment():
    package = {"resume": {"accepted_updates": 76, "sha256": "SOURCE"}}
    source = {"accepted_updates": 76}
    validate_resume(source, "SOURCE", package, "AUTHORITY")
    with pytest.raises(AssertionError):
        validate_resume(source, "WRONG", package, "AUTHORITY")
    with pytest.raises(AssertionError):
        validate_resume({"accepted_updates": 75}, "SOURCE", package, "AUTHORITY")
    current = {"accepted_updates": 77, "standing_kickoff_amendment": {
        "version": VERSION, "sha256": "AUTHORITY", "from_update": 76}}
    validate_resume(current, "NEW_CHECKPOINT", package, "AUTHORITY")
    invalid = deepcopy(current)
    invalid["standing_kickoff_amendment"]["sha256"] = "OTHER_AUTHORITY"
    with pytest.raises(AssertionError):
        validate_resume(invalid, "NEW_CHECKPOINT", package, "AUTHORITY")


def test_native_initialization_and_actual_goal_reset_are_stationary():
    torch.set_num_threads(4)
    bank = standing_training_starts(128)
    env = AcquisitionEnv(128, "G:/dev/RLBot-Rival/bot/collision_meshes", device="cuda:0",
                         seed=91, ssl_foundation_scenarios=bank)
    template = env.world.ssl_foundation_reset

    def verify_live():
        indices = template.current_source_index.numpy()
        live = env.world.state.snapshot()
        rows = np.flatnonzero(bank.kickoff_indicator[indices])
        assert len(rows)
        native = make_standard_kickoff_state(len(rows), bank.kickoff_layout[indices[rows]])
        for key in ("car_pos", "car_vel", "car_ang_vel", "car_quat", "boost",
                    "ball_pos", "ball_vel", "ball_ang_vel", "prev_throttle", "prev_jump", "prev_boost"):
            np.testing.assert_array_equal(getattr(live, key)[rows], getattr(native, key))
        assert not hasattr(env, "race") and not hasattr(env, "events")
        assert env.world.gameplay_v3 is None

    verify_live()
    before = wp.to_torch(template.reset_generation).clone()
    for row in range(128):
        force_goal(env, row)
    transition = env.step(torch.zeros(128, 2, 8, device=env.device))
    assert transition.terminated.all() and transition.reset_mask.all()
    assert torch.equal(wp.to_torch(template.reset_generation), before + 1)
    verify_live()
