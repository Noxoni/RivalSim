import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jump_timer_audit", ROOT/"benchmarks/audit_entity_native_jump_timer.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_initial_jump_has_zero_simulator_air_since_jump():
    assert audit.timer_probe("Jumping", True, False, False, -1) == (0., "initial_jump_zero")


def test_native_remaining_timer_is_independent_elapsed_basis():
    value, kind = audit.timer_probe("InAir", True, False, False, 1.0)
    assert value == pytest.approx(.2)
    assert kind == "positive_native_remaining"


@pytest.mark.parametrize("phase,jumped,doubled,dodged,remaining", [
    ("OnGround",False,False,False,-1), ("InAir",False,False,False,-1),
    ("InAir",True,False,False,-1), ("InAir",True,False,False,0),
    ("InAir",True,False,False,1.5), ("InAir",True,False,False,float("nan")),
    ("InAir",True,False,True,.5), ("InAir",True,True,False,.5),
    ("Dodging",True,False,True,-1)])
def test_unsupported_or_spent_state_is_not_fabricated(phase,jumped,doubled,dodged,remaining):
    assert audit.timer_probe(phase,jumped,doubled,dodged,remaining) is None


def test_completed_evidence_hashes_and_immutable_parent():
    result = json.loads((audit.OUT/"audit.json").read_text())
    for path, identity in result["source_hashes"].items():
        assert audit.sha(ROOT/path) == identity
    for case in result["cases"].values():
        for path, identity in case["hashes"].items():
            assert audit.sha(ROOT/path) == identity
        assert case["original_action_replay_exact"]
    assert result["optimizer_steps"] == 0
    assert not result["production_changed"]


def test_all_stored_probes_obey_frozen_supported_cases():
    import numpy as np
    result = json.loads((audit.OUT/"audit.json").read_text())
    names = {v:k for k,v in result["phase_codes"].items()}
    for case in result["cases"]:
        data = np.load(audit.OUT/(case+".npz"))
        for row in range(len(data["frame"])):
            for role in range(2):
                actual = audit.timer_probe(names[int(data["phases"][row,role])],
                                          *data["flags"][row,role],
                                          data["remaining"][row,role])
                if actual is None:
                    assert not data["supported"][row,role]
                    assert data["recorded_timer"][row,role] == data["probed_timer"][row,role]
                    assert data["recorded_dodge"][row,role] == data["probed_dodge"][row,role]
                else:
                    assert data["supported"][row,role]
                    assert data["probed_timer"][row,role] == pytest.approx(actual[0],abs=1e-7)
