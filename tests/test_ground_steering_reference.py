import torch

from benchmarks.diagnose_rival2_ground_steering_reference import pursuit_action
from rivalsim.rival2_contracts import OBS_FIELD_NAMES, POSITION_SCALE


def test_reference_uses_geometry_and_emits_valid_ground_controls_only():
    obs = torch.zeros(3, 182)
    obs[:, OBS_FIELD_NAMES.index("self.forward.x")] = 1
    start = OBS_FIELD_NAMES.index("relative.ball_position.x")
    obs[:, start : start + 3] = torch.tensor(
        [[1000.0, 0, 0], [0, 1000.0, 0], [0, -1000.0, 0]]
    ) / torch.tensor(POSITION_SCALE)
    action = pursuit_action(obs)
    assert action.shape == (3, 8)
    assert torch.equal(action[:, 1], torch.tensor([0.0, 1.0, -1.0]))
    assert torch.equal(action[:, 6], torch.tensor([1.0, 0.0, 0.0]))
    assert action[:, 2:6].count_nonzero() == 0
    assert torch.isfinite(action).all() and action.abs().max() <= 1
