import pytest

from benchmarks.validate_rival2_ssl_entity_match_interface import check_idle_status


def test_dynamic_cuda_check_requires_completed_pilot():
    check_idle_status({"accepted_updates": 100, "status": "completed"})
    for status in ("rollout", "optimizing", "evaluating", "failed", "stopped_at_accepted_boundary"):
        with pytest.raises(RuntimeError, match="do not interrupt training"):
            check_idle_status({"accepted_updates": 100, "status": status})
    for count in (0, 50, 99, 101):
        with pytest.raises(RuntimeError):
            check_idle_status({"accepted_updates": count, "status": "completed"})
