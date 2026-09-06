from benchmarks.report_direct_skills_shooting_000650 import build


def test_completed_trace_reduction_is_exact_and_reproducible():
    a = build()
    assert a == build()
    assert len(a['modes']) == 3 and a['optimizer_steps'] == 0
    assert all(all(row['checks'].values()) for row in a['modes'].values())
