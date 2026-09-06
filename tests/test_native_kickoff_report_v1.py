import json
import numpy as np

from benchmarks.report_direct_skills_native_kickoff_v1 import reduce_case, action_difference


def source():
    shape = (720, 1, 2)
    data = {"rival_side": np.array([1]), "starting_layout": np.array([4]),
        "pre.goal_count": np.zeros((720, 1), np.int32),
        "applied_controls": np.zeros((*shape, 8), np.float32),
        "policy_action": np.zeros((180, 1, 8), np.float32),
        "pre.car_vel": np.zeros((*shape, 3), np.float32),
        "pre.on_ground": np.ones(shape, np.int32)}
    for key in ("pre.touch_count", "post.touch_count", "post.has_jumped", "post.has_flipped"):
        data[key] = np.zeros(shape, np.int32)
    for c, index in ((0, 100), (1, 110)):
        data["post.touch_count"][index:, 0, c] = 1
        data["pre.touch_count"][index+1:, 0, c] = 1
    return data


def test_cpu_reduction_side_timing_and_serialization():
    d = source()
    d["post.has_flipped"][90:, 0, 0] = 1
    d["post.has_flipped"][120:, 0, 1] = 1
    report = json.loads(json.dumps(reduce_case(d), allow_nan=False))
    assert report["aggregate"]["first_contacts_won"] == 0
    assert report["aggregate"]["rival_pre_contact_flips"] == 0
    assert report["aggregate"]["nexto_pre_contact_flips"] == 1
    assert report["worlds"][0]["rival_contact_lag_ticks"] == 10
    assert report["worlds"][0]["cars"][0]["is_rival"] is True
    assert report["worlds"][0]["cars"][0]["race_ticks"] == 100


def test_similarity_uses_parent_precontact_not_postdivergence():
    parent, child = source(), source()
    child["policy_action"][0, 0, 0] = 1
    child["policy_action"][26, 0, 0] = 1
    report = action_difference(parent, child)
    assert report["pre_parent_first_contact_decisions"] == 25
    assert report["pre_parent_first_contact_different_actions"] == 1
    assert report["full_six_second_different_actions"] == 2
