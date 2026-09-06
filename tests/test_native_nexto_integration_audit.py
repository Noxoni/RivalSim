"""Scope checks for the no-learning installed-opponent audit, not gameplay gates."""
import marshal
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import rlbot.flat as flat
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import audit_native_nexto_integration as audit


def test_source_literal_uses_real_controller_default():
    table = audit.source_kickoff(audit.ROOT/"third_party/nexto/upstream/bot.py")
    actual_default = flat.ControllerState(throttle=1, boost=True, steer=-1)
    np.testing.assert_array_equal(table[44:60, 3], np.full(16, actual_default.yaw))
    assert actual_default.yaw == 0


def test_literal_interpreter_rejects_executable_expression(tmp_path):
    p = tmp_path/"unsafe.py"
    p.write_text('KICKOFF_CONTROLS = unexpected_function()')
    with pytest.raises(ValueError, match="Unexpected kickoff"):
        audit.source_kickoff(p)


def test_table_difference_is_measured_against_literal_not_duplicate_fixture():
    original = audit.source_kickoff(audit.ROOT/"third_party/nexto/upstream/bot.py")
    implemented = audit.method_from_source(audit.ROOT/"third_party/nexto/adapter.py", "build_kickoff_sequence")("cpu").numpy()
    copied_fixture = audit.method_from_source(audit.ROOT/"benchmarks/run_nexto_fidelity.py", "_source_kickoff")
    copied_fixture.__globals__["np"] = np
    np.testing.assert_array_equal(implemented, copied_fixture().astype(np.float32))
    mismatch = np.argwhere(implemented != original)
    np.testing.assert_array_equal(mismatch[:, 0], np.arange(44, 60))
    np.testing.assert_array_equal(mismatch[:, 1], np.full(16, 3))


def test_actual_installed_neural_model_matches_pinned_file():
    archive = audit.CArchiveReader(str(audit.EXE))
    assert archive.extract("nexto-model.pt") == (audit.ROOT/"third_party/nexto/nexto-model.pt").read_bytes()


def test_extracted_scheduler_holds_earlier_observation_action():
    archive = audit.CArchiveReader(str(audit.EXE))
    code = audit.find_code(marshal.loads(archive.extract("bot")), "get_output")
    fn = types.FunctionType(code, {"MatchPhase":flat.MatchPhase,"GameMode":flat.GameMode,"np":np})
    sim_fn = audit.method_from_source(audit.ROOT/"third_party/nexto/adapter.py", "tick_action")
    result = audit.neural_schedule_probe(fn, sim_fn)
    assert result["native_compute_frames"] == [1, 2, 10, 18, 26]
    assert result["simulator_compute_frames"] == [1, 9, 17, 25]
    rows = np.asarray(result["frame_native_action_marker_simulator_action_marker"])
    assert rows[7].tolist() == [8, 2, 1]
    assert rows[8].tolist() == [9, 2, 2]
    assert result == audit.neural_schedule_probe(fn, sim_fn)


def test_code_lookup_does_not_execute_module():
    code = compile('class Nexto:\n def target(self):\n  return 1\nraise RuntimeError("must not execute")\n', "fixture", "exec")
    assert audit.find_code(code,"target").co_name == "target"
    assert audit.find_code(code,"absent") is None


def test_actual_installed_rotation_helper_matches_simulator_basis():
    archive = audit.CArchiveReader(str(audit.EXE)).open_embedded_archive("PYZ.pyz")
    code = audit.find_code(archive.extract("nexto_obs"),"_quats_to_rot_mtx")
    native = types.FunctionType(code,{"np":np})
    q = np.random.default_rng(20260906).normal(size=(100,4))
    q /= np.linalg.norm(q,axis=1,keepdims=True)
    rotation = native(q[:,[3,0,1,2]])
    forward, up = audit.method_from_source(audit.ROOT/"third_party/nexto/adapter.py","_basis")(torch.from_numpy(q))
    np.testing.assert_allclose(forward.numpy(),rotation[:,:,0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(up.numpy(),rotation[:,:,2],atol=1e-14,rtol=0)
