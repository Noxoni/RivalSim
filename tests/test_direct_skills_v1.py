import numpy as np
import torch

from rivalsim.direct_skills_v1 import (
    EVENTS,
    SkillEvents,
    player_roles,
    projected_goal,
    reward_authority,
    scenarios,
)
from rivalsim.fresh_ground_30hz import scenario_hash
from rivalsim.rival2_contracts import OBS_FIELD_NAMES


def observations(worlds=1):
    x = torch.zeros((worlds, 2, 182))

    def put(name, values):
        start = OBS_FIELD_NAMES.index(name + ".x")
        x[..., start : start + 3] = torch.tensor(values)

    # Observation units use physical normalizations, not positions in uu.
    put("ball.position", [0, 0, 93.15 / 2044])
    put("self.position", [0, -150 / 5120, 17 / 2044])
    put("opponent.position", [0, 1200 / 5120, 17 / 2044])
    return x


def inputs(roles=None):
    obs = observations()
    return [
        obs,
        obs.clone(),
        torch.tensor([[1, 0]]),
        torch.tensor([[1, 1]]) if roles is None else roles,
        torch.tensor([False]),
        torch.tensor([False]),
        torch.tensor([-1]),
    ]


def test_native_observation_field_names_and_control_duration():
    tracker, args = SkillEvents(1, "cpu"), inputs()
    reward, events, *_ = tracker.calculate(*args)
    assert events[0, 0, EVENTS.index("first_touch")]
    assert reward[0, 0] == 0.5
    args[2].zero_()
    for _ in range(6):
        reward, events, *_ = tracker.calculate(*args)
        assert not bool(events[..., 1].any())
    reward, events, *_ = tracker.calculate(*args)
    assert events[0, 0, 1]
    assert reward[0, 0] == 1.5
    reward, events, *_ = tracker.calculate(*args)
    assert not bool(events.any())


def test_no_contact_is_not_control_and_simultaneous_contacts_not_invented_order():
    tracker, args = SkillEvents(1, "cpu"), inputs()
    args[2].zero_()
    for _ in range(10):
        _, events, *_ = tracker.calculate(*args)
        assert not bool(events.any())
    args[2].fill_(1)
    tracker.calculate(*args)
    assert not bool(tracker.own_last.any())


def test_natural_reward_lane_has_no_skill_bonus():
    tracker, args = SkillEvents(1, "cpu"), inputs(torch.zeros((1, 2), dtype=torch.long))
    for _ in range(12):
        bonus, events, *_ = tracker.calculate(*args)
        assert torch.equal(bonus, torch.zeros_like(bonus))
        assert not bool(events.any())


def test_true_goal_cannot_pay_timeout_or_post_goal_geometry():
    tracker, args = SkillEvents(1, "cpu"), inputs()
    args[4].fill_(True)
    args[5].fill_(True)
    args[6].fill_(0)
    bonus, events, _, _, success = tracker.calculate(*args)
    assert not bool(events.any()) and not bool(bonus.any())
    assert success[0, 0]


def test_failure_once_and_reset_rearms_only_new_episode():
    tracker, args = SkillEvents(1, "cpu"), inputs()
    args[2].zero_()
    args[5].fill_(True)
    bonus, *_ = tracker.calculate(*args)
    assert torch.equal(bonus, torch.full_like(bonus, -1))
    bonus, *_ = tracker.calculate(*args)
    assert not bool(bonus.any())
    tracker.reset(torch.tensor([True]))
    bonus, *_ = tracker.calculate(*args)
    assert torch.equal(bonus, torch.full_like(bonus, -1))


def test_projected_shot_distinguishes_net_miss_and_retreat():
    p = torch.tensor([[0.0, 2500.0, 93.15], [1500.0, 2500.0, 93.15], [0.0, 2500.0, 93.15]])
    v = torch.tensor([[0.0, 1000.0, 0.0], [0.0, 1000.0, 0.0], [0.0, -1000.0, 0.0]])
    assert projected_goal(p, v, 1).tolist() == [True, False, False]


def test_save_requires_contact_and_sustained_neutralized_threat():
    tracker, args = SkillEvents(1, "cpu"), inputs(torch.full((1, 2), 3))
    vy = OBS_FIELD_NAMES.index("ball.linear_velocity.y")
    by = OBS_FIELD_NAMES.index("ball.position.y")
    args[0][..., by] = -2500 / 5120
    args[1][..., by] = -2500 / 5120
    args[0][..., vy] = -1000 / 6000
    args[1][..., vy] = 1000 / 6000
    for tick in range(8):
        _, events, *_ = tracker.calculate(*args)
        assert bool(events[0, 0, 4]) == (tick == 7)
        args[2].zero_()
    _, events, *_ = tracker.calculate(*args)
    assert not bool(events.any())


def test_approach_cannot_reward_waiting_on_kickoff():
    tracker, args = SkillEvents(1, "cpu"), inputs(torch.full((1, 2), 4))
    args[2].zero_()
    for _ in range(10):
        bonus, *_ = tracker.calculate(*args)
        assert not bool(bonus.any())


def test_roles_swap_attack_defense_not_observations():
    roles = player_roles(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 1, 0, 1]))
    assert roles.tolist() == [[0, 0], [1, 1], [3, 2], [3, 2], [4, 4]]


def test_scenario_determinism_contacts_clear_and_variety():
    a, b = scenarios(512), scenarios(512)
    assert scenario_hash(a) == scenario_hash(b)
    assert set(a.family) == set(range(5))
    d = np.linalg.norm(a.state.car_pos - a.state.ball_pos[:, None], axis=-1)
    assert d.min() > 150
    assert np.isfinite(a.state.car_pos).all()
    natural = a.family == 0
    for parity in (0, 1):
        k = a.kickoff_indicator[natural & (np.arange(512) % 2 == parity)]
        assert 0 < k.sum() < len(k)


def test_defense_is_actual_incoming_ground_shot_not_random_air_ball():
    bank = scenarios(64, family_only=3)
    p = torch.from_numpy(bank.state.ball_pos.copy())
    v = torch.from_numpy(bank.state.ball_vel.copy())
    sign = torch.from_numpy(np.where(bank.focal_side == 0, -1, 1).copy())
    assert (v[:, 1] * sign > 500).all()
    # Some long-range starts enter the <=4s reward threat window later.
    t = (5120 - p[:, 1] * sign) / (v[:, 1] * sign)
    assert ((p[:, 0] + v[:, 0] * t).abs() <= 550.1).all()


def test_reward_authority_no_fake_waiting_reward():
    assert reward_authority()["no_other_direct_rewards"]
    assert "fake" not in reward_authority()["events"]
