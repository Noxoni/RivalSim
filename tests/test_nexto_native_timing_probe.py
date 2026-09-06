import json
from pathlib import Path
import types

import numpy as np
import torch

from benchmarks.probe_nexto_native_timing import advance_native_clock, make_timing_adapter

ROOT = Path(__file__).resolve().parents[1]


def test_clock_matches_actual_installed_controller_symbolic_trace():
    oracle = json.loads((ROOT/"results/rival2/native_nexto_integration_audit_v1/audit.json").read_text())["scheduler"]
    ticks, update, counter, control = 8, True, 0, 0
    rows, computes = [], []
    for frame in range(1, oracle["frames"] + 1):
        ticks, update, compute, emit = advance_native_clock(ticks, update)
        if compute:
            counter += 1
            computes.append(frame)
        if emit:
            control = counter
        rows.append([frame, control])
    assert computes == oracle["native_compute_frames"]
    assert rows == [r[:2] for r in oracle["frame_native_action_marker_simulator_action_marker"]]


class BaseStub:
    """Only tests wrapper dispatch, pending-history and actual emission separation."""
    def __init__(self):
        self.num_worlds = 1
        self.device = torch.device("cpu")
        self.previous_action = torch.zeros(1, 8)
        self.neural_counter = torch.zeros(1, dtype=torch.long)
        self.kickoff_index = torch.full((1,), -1)
        self.inputs = []

    def neural_action(self, state):
        self.inputs.append(self.previous_action.clone())
        self.previous_action.fill_(len(self.inputs))
        return self.previous_action, None

    def tick_action(self, state, kickoff_active, active_mask=None):
        assert not torch.any(self.neural_counter == 0), "Extra base neural computation"
        if kickoff_active[0]:
            self.previous_action.fill_(.75)
        return self.previous_action, None


def test_pending_neural_action_is_not_emitted_early():
    obj = make_timing_adapter(BaseStub)()
    state = types.SimpleNamespace(ball_pos=torch.zeros(1, 3))
    actual = [obj.tick_action(state, torch.tensor([False]))[0].clone() for _ in range(16)]
    assert [x[0,0].item() for x in actual] == [1]*7 + [2]*8 + [3]
    assert [x[0,0].item() for x in obj.inputs] == [0,1,2]


def test_kickoff_script_updates_pending_and_emitted_action():
    obj = make_timing_adapter(BaseStub)()
    state = types.SimpleNamespace(ball_pos=torch.zeros(1, 3))
    for _ in range(2):
        output, _ = obj.tick_action(state, torch.tensor([True]))
        np.testing.assert_array_equal(output.numpy(), np.full((1,8), .75))
    assert obj.inputs[1][0,0].item() == .75
