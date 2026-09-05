"""Prospective reset-only acquisition/finishing probe; no new reward or controller."""

from __future__ import annotations

import numpy as np

from rivalsim.fresh_ground_30hz import SEED, scenarios
from rivalsim.ssl_foundation_v1 import _yaw_quaternion

VERSION = "RIVAL2_SSL_GROUND_CURRICULUM_PROBE_V1"
PROBE_SEED = 2026091601


def probe_scenarios(worlds, seed=PROBE_SEED):
    """80% achievable ground approaches, 20% unmodified broad ground curriculum.

    Focal cars start 350-850uu from a ground ball in the attacking half, from a
    useful but imperfect side. Half the changed rows have at most .30rad heading
    error; half have at most .70rad. Initial velocity follows the car nose.
    Both cars use the learned policy from tick zero. No action constraints,
    scripted prefixes, contact rewards, task IDs or training-target labels.
    """
    bank = scenarios(worlds, seed)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(worlds)
    changed = order[: int(worlds * 0.80)]
    for ordinal, row in enumerate(changed):
        side = int(bank.focal_side[row])
        other = 1 - side
        sign = 1 if side == 0 else -1
        ball = rng.uniform((-700, 1800), (700, 3200)) * sign
        goal = np.array([0.0, 5120.0 * sign])
        to_goal = goal - ball
        yaw = np.arctan2(to_goal[1], to_goal[0]) + rng.uniform(-0.30, 0.30)
        approach = np.array([np.cos(yaw), np.sin(yaw)])
        position = ball - rng.uniform(350, 850) * approach
        heading_limit = 0.30 if ordinal % 2 == 0 else 0.70
        heading = yaw + rng.uniform(-heading_limit, heading_limit)
        nose = np.array([np.cos(heading), np.sin(heading)])
        state = bank.state
        state.ball_pos[row] = [*ball, 93.15]
        state.ball_vel[row] = [*rng.uniform(-80, 80, 2), 0.0]
        state.ball_ang_vel[row] = 0
        state.car_pos[row, side] = [*position, 17.0]
        state.car_quat[row, side] = _yaw_quaternion(heading)
        state.car_vel[row, side] = [*(rng.uniform(100, 500) * nose), 0.0]
        state.car_ang_vel[row, side] = 0
        # Far enough to allow an achievable attempt, but a real freely acting
        # opponent, not a disabled car or a demonstration prefix.
        opponent_xy = (
            np.array([rng.choice([-1, 1]) * rng.uniform(2200, 3000), rng.uniform(-2400, 500)])
            * sign
        )
        target = ball - opponent_xy
        opponent_heading = np.arctan2(target[1], target[0]) + rng.uniform(-1.5, 1.5)
        state.car_pos[row, other] = [*opponent_xy, 17.0]
        state.car_quat[row, other] = _yaw_quaternion(opponent_heading)
        # One speed for both planar components; avoid sliding momentum.
        opponent_speed = rng.uniform(0, 400)
        state.car_vel[row, other] = [
            opponent_speed * np.cos(opponent_heading),
            opponent_speed * np.sin(opponent_heading),
            0.0,
        ]
        state.car_ang_vel[row, other] = 0
        state.on_ground[row] = 1
        state.boost[row] = rng.uniform(25, 100, 2)
        bank.family[row] = 4
        bank.kickoff_indicator[row] = 0
        bank.kickoff_layout[row] = -1
    bank.state.validate()
    return bank


def specification():
    return dict(
        version=VERSION,
        seed=PROBE_SEED,
        base_training_seed=SEED,
        changed_fraction=0.80,
        untouched_original_fraction=0.20,
        close_approach=dict(
            ball_x=[-700, 700],
            ball_y=[1800, 3200],
            ball_planar_velocity=[-80, 80],
            distance=[350, 850],
            approach_to_goal_angle=[-0.30, 0.30],
            heading_error_bounds=[0.30, 0.70],
            initial_speed=[100, 500],
        ),
        canonical_team_mirroring=True,
        same_policy_both_sides=True,
        reset_only=True,
        constraints_on_actions=False,
        scripted_prefix=False,
        additional_rewards=False,
        model_receives_curriculum_id=False,
        explanation="Test learnability through easier reset geometry; not an SSL capability claim.",
    )
