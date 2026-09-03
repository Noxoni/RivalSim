from __future__ import annotations

import torch

from rivalsim.rival2_ground_ball_soft_pop import build_ground_ball_soft_pop_scenarios


def test_soft_pop_scenario_is_low_relative_speed_and_mirrored() -> None:
    blue = build_ground_ball_soft_pop_scenarios(8, seed=7, attacker_side=0)
    orange = build_ground_ball_soft_pop_scenarios(8, seed=7, attacker_side=1)
    for state, side, sign in ((blue, 0, 1.0), (orange, 1, -1.0)):
        assert torch.as_tensor(state.ball_pos[:, 2]).eq(92.75).all()
        ball_speed = torch.as_tensor(state.ball_vel[:, 1]) * sign
        car_speed = torch.as_tensor(state.car_vel[:, side, 1]) * sign
        relative = car_speed - ball_speed
        assert relative.min() >= 80.0
        assert relative.max() <= 280.0
        assert (torch.as_tensor(state.ball_pos[:, 1]) * sign > 0.0).all()
