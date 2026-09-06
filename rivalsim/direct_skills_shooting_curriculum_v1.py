"""Intermediate keeper pressure for half of shooting starts; no reward change."""

from __future__ import annotations

import numpy as np

from rivalsim.direct_skills_v1 import SEED, scenarios as original_scenarios
from rivalsim.ssl_foundation_v1 import _set_coherent_ground_route

VERSION = "RIVAL2_DIRECT_SKILLS_SHOOTING_PRESSURE_V1"
FRACTION = 0.5
ALPHA_RANGE = (0.35, 0.85)


def curriculum_authority():
    return dict(
        version=VERSION, finishing_replacement_fraction=FRACTION,
        unchanged="All non-finishing states and metadata; half original finishing states; "
        "ball/Rival state and all opponent state except position/velocity/quaternion",
        selection="Separate deterministic RNG(seed+65001), shuffled finishing row IDs; "
        "replace floor(finishing_count/2) rows, same total episode-family mixture",
        geometry="Canonical opponent xy interpolates between (alternate-random flank*2200, "
        "ball_y-1300) and original keeper xy; alpha uniform[.35,.85]. Opponent600uu/s "
        "toward own goal, grounded coherent yaw with existing+/-0.2rad heading jitter",
        physical="Reject candidate xy unless inside|x|<3600,|y|<4900 and >=220uu from "
        "ball,>=240uu from Rival. At most32 draws per row; fail if none, no silent fallback",
        alpha_range=list(ALPHA_RANGE), opponent_speed=600,
        reward_change=False, ppo_change=False, task_id_in_observation=False,
        scripted_actions=False, passive_or_disabled_opponent=False,
        rationale="650 scores49/64 initially open and50/64 recovering but5/64 set keeper. "
        "Bridge pressure while retaining hard starts, rather than mostly rewarding solved open nets",
    )


def build(worlds, seed=SEED):
    bank = original_scenarios(worlds, seed)
    rng = np.random.default_rng(seed+65001)
    shooting = np.flatnonzero(bank.family==2)
    selected = np.sort(rng.permutation(shooting)[:len(shooting)//2])
    s = bank.state
    grades, attempts = [], []
    for row in selected:
        focal = bank.focal_side[row]
        opponent = 1-focal
        sign = 1 if focal==0 else -1
        original = s.car_pos[row,opponent,:2].copy()*sign
        ball = s.ball_pos[row,:2]*sign
        own = s.car_pos[row,focal,:2]*sign
        for attempt in range(32):
            alpha = float(rng.uniform(*ALPHA_RANGE))
            recovery = np.array([rng.choice((-1,1))*2200,ball[1]-1300])
            candidate = (1-alpha)*recovery + alpha*original
            if ((np.abs(candidate)<np.array([3600,4900])).all()
                and np.linalg.norm(candidate-ball)>=220
                and np.linalg.norm(candidate-own)>=240):
                break
        else:
            raise ValueError(f"No valid pressure interpolation at row{row}")
        s.car_pos[row,opponent,:2] = candidate*sign
        _set_coherent_ground_route(s,row,opponent,rng,np.array([0,5120])*sign,
                                   (600,600),(0,0))
        grades.append(alpha);attempts.append(attempt+1)
    s.validate()
    audit = dict(worlds=worlds,finishing_count=len(shooting),changed_count=len(selected),
                 changed_rows=selected.tolist(),alpha=grades,draw_attempts=attempts,
                 state_mutations=['opponent.car_pos','opponent.car_vel','opponent.car_quat'])
    return bank,audit


def scenarios(worlds,seed=SEED):
    return build(worlds,seed)[0]
