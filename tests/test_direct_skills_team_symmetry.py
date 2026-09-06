import numpy as np
import torch

from benchmarks.audit_rival2_direct_skills_team_symmetry import RESULTS, bridge, half_turn


def fixture_views():
    with np.load(RESULTS / "training_reset_contact_fixture.npz", allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def test_half_turn_swap_is_involutive_in_observation_space():
    torch.set_num_threads(2)
    views = fixture_views()
    count = len(views["final_observation"])
    original = bridge(views, count).observation()
    twice = bridge(half_turn(half_turn(views, count), count), count).observation()
    torch.testing.assert_close(twice, original, rtol=0, atol=1e-6)
    # Negative control: omitting the perspective swap must not pass this fixture.
    once = bridge(half_turn(views, count), count).observation()
    assert (once - original).abs().max() > 0.1


def test_nonuniform_pad_cooldowns_expose_missing_spatial_remap():
    views = fixture_views()
    count = len(views["final_observation"])
    # Synthetic transform unit test only, not extra simulator/evaluation samples.
    views["pad_cooldown"] = torch.linspace(0, 3.3, 34).expand(count, -1).clone()
    original = bridge(views, count).observation()
    rotated = half_turn(views, count)
    correct = bridge(rotated, count).observation().flip(1)
    assert torch.equal(correct[..., 99:167], original[..., 99:167])
    rotated["pad_cooldown"] = views["pad_cooldown"].clone()
    broken = bridge(rotated, count).observation().flip(1)
    assert (broken[..., 99:167] - original[..., 99:167]).abs().max() > 0.1
