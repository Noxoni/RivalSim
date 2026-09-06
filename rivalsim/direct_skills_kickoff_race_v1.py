"""Kickoff-only native race outcome, with bounded physical approach exercises.

No action, flip or speed reward. Existing finishing and all other roles remain
unchanged. A race is ordered by actual physics ticks, not a four-tick interval.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from rivalsim.direct_skills_finishing_goal_v2 import (
    FinishingGoalEnv, FinishingGoalEvents, reward_authority as parent_reward,
)
from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios as parent_scenarios
from rivalsim.direct_skills_v1 import SEED
from rivalsim.fresh_ground_30hz import content_hash

VERSION = "RIVAL2_DIRECT_SKILLS_KICKOFF_RACE_V1"
RACE_BONUS = 0.5
MOMENTUM_RANGE = (250.0, 750.0)


def reward_authority():
    return dict(version=VERSION, parent_sha256=content_hash(parent_reward()), changed_role=4,
        race_bonus=RACE_BONUS,
        race="Only the first native contacting player in the entire world episode receives0.5, once. Observe after each physics tick; same-physics-tick bilateral first contact is a tie and pays neither. A later contact in the same four-tick decision cannot steal or split the earlier result.",
        replaced="Kickoff personal-first-touch bonus only. Existing first_touch telemetry means race-first-touch for role4; other roles retain their existing meaning.",
        preserved="Kickoff control gain1.5, controlled advancement0.75, lost-control-0.5, unsuccessful timeout-1, native goals+10/-10. All natural/challenge/finishing/defense rewards and success semantics unchanged. A successful second-touch control play remains worth more than just first contact.",
        terminal="As before, no direct task event on true goal terminal; terminal reward remains the original discounted native goal. Race state clears only at actual episode reset.",
        no_action_or_mechanic_reward=True, no_task_id_or_router=True,
        physics_hz=120, policy_hz=30)


def curriculum_authority():
    return dict(version=VERSION + "_STARTS", parent="Unchanged shooting-pressure V1 source bank",
        selection="Separate RNG(seed+67001), shuffle role4 row IDs and change floor(count/2). No other family altered.",
        changes="Only focal car linear velocity, along its existing native quaternion forward direction, uniform250..750uu/s. Standard positions, ball, boost, orientations and opponent untouched. These are momentum-assisted approach exercises, not genuine standing kickoffs.",
        unchanged_fraction_of_dedicated_kickoffs=0.5, momentum_range=list(MOMENTUM_RANGE),
        nearest_car_admission="Both original standard positions unchanged, preserving Nexto nearest-car kickoff admission. Do not use distance handicaps that disable its routine.",
        family_distribution=dict(natural=0.30, challenge=0.20, finishing=0.20, defense=0.15, kickoff=0.15),
        evaluation="Unchanged original standing starts, no momentum assistance",
        no_scripted_controls=True, no_extra_observation=True)


def scenarios(worlds, seed=SEED):
    bank = parent_scenarios(worlds, seed)
    rng = np.random.default_rng(seed + 67001)
    kickoffs = np.flatnonzero(bank.family == 4)
    selected = np.sort(rng.permutation(kickoffs)[:len(kickoffs)//2])
    side = bank.focal_side[selected]
    q = bank.state.car_quat[selected, side]
    x, y, z, w = np.moveaxis(q, -1, 0)
    direction = np.stack((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)), -1)
    bank.state.car_vel[selected, side] = direction * rng.uniform(*MOMENTUM_RANGE, (len(selected), 1))
    bank.state.validate()
    return bank


@wp.kernel
def observe_first_contact(
    contact_count: wp.array(dtype=wp.int32),
    goal: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    if winner[world] == -1 and goal[world] == 0:
        blue = contact_count[world*2] > 0
        orange = contact_count[world*2+1] > 0
        if blue and orange:
            winner[world] = -2
        elif blue:
            winner[world] = 0
        elif orange:
            winner[world] = 1


class RaceState:
    def __init__(self, worlds, device):
        self.array = wp.full(worlds, -1, dtype=wp.int32, device=str(device))
        self.winner = wp.to_torch(self.array)
        self.paid = torch.zeros(worlds, dtype=torch.bool, device=self.winner.device)

    def reset(self, mask):
        self.winner.masked_fill_(mask, -1)
        self.paid.masked_fill_(mask, False)


class KickoffRaceEvents(FinishingGoalEvents):
    def __init__(self, worlds, device, race):
        super().__init__(worlds, device)
        self.race = race

    def reset(self, mask):
        super().reset(mask)
        self.race.reset(mask)

    def calculate(self, before, after, touches, roles, terminated, truncated, scoring_team):
        old_paid = self.paid[..., 0].clone()
        bonus, events, weighted, approach, success = super().calculate(
            before, after, touches, roles, terminated, truncated, scoring_team)
        kickoff = roles == 4
        race_event = kickoff & ~self.race.paid[:, None] & ~terminated[:, None]
        race_event &= self.race.winner[:, None] == torch.arange(2, device=roles.device)[None]
        payment = race_event.to(before.dtype) * RACE_BONUS
        bonus += torch.where(kickoff, payment - weighted[..., 0], 0)
        weighted[..., 0] = torch.where(kickoff, payment, weighted[..., 0])
        events[..., 0] = torch.where(kickoff, race_event, events[..., 0])
        self.paid[..., 0] = torch.where(kickoff, old_paid | race_event, self.paid[..., 0])
        self.race.paid |= self.race.winner != -1
        return bonus, events, weighted, approach, success


class KickoffRaceEnv(FinishingGoalEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.race = RaceState(self.num_envs, self.device)
        self.events = KickoffRaceEvents(self.num_envs, self.device, self.race)
        self.reward_version = VERSION
        self.contract_hashes = dict(self.contract_hashes, reward=content_hash(reward_authority()))
        original = self.world._launch_tick

        def with_race():
            original()
            wp.launch(observe_first_contact, dim=self.num_envs,
                inputs=[self.world.rival2.touch_count, self.world.rival2.goal_latched, self.race.array],
                device=self.world.device)

        self.world._launch_tick = with_race
