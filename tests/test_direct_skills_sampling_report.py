"""CPU checks of the completed, immutable650 action-selection comparison."""

from benchmarks.report_direct_skills_sampling_000650 import build, spread


def test_three_fixed_seeds_are_all_retained_and_rebuild_exactly():
    a = build()
    assert a == build()
    assert [r['seed'] for r in a['seeds']] == [20260906651, 20260906652, 20260906653]
    assert a['optimizer_steps'] == 0 and not a['model_selection']
    assert all(all(check.values()) for check in a['checks'].values())
    for row in a['seeds']:
        assert len(row['per_match']) == 10
        assert row['finishing']['worlds'] == 64
    for key, value in a['aggregate']['matches'].items():
        values = [row['summary'][key] for row in a['seeds']]
        assert value == spread(values)


def test_spread_keeps_all_seed_values_not_only_best():
    assert spread([2, 9, 1]) == dict(mean=4, minimum=1, maximum=9, values=[2, 9, 1])
