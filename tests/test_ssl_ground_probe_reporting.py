from benchmarks.report_rival2_ssl_ground_curriculum_probe import decision


def fixture(touches, goals=14):
    return {
        "acquisition_selfplay": {"focal_touch_fraction": touches / 64},
        "finishing_selfplay": {"goals_for": goals},
    }


def test_frozen_probe_rule_requires_both_boundaries_and_parent_improvement():
    control = {0: fixture(16), 20: fixture(16), 30: fixture(15, 16)}
    assert decision({0: fixture(16)}, control)[0] == "INCOMPLETE"
    probe = {0: fixture(16), 20: fixture(20), 30: fixture(20)}
    assert decision(probe, control)[0].startswith("PROMISING")
    probe[20] = fixture(19)
    assert decision(probe, control)[0].startswith("INCONCLUSIVE")
    probe[20] = fixture(20)
    probe[30] = fixture(19)
    assert not decision(probe, control)[1]["acquisition_improves_parent"]
    probe[30] = fixture(20, 12)
    assert not decision(probe, control)[1]["finishing_goal_nonregression"]
