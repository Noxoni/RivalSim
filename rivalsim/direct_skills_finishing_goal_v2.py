"""Graduate finishing to native scoring; preserve every other V1 reward role."""
from __future__ import annotations

import torch

from rivalsim.direct_skills_v1 import DirectSkillsEnv, SkillEvents, reward_authority as original_authority
from rivalsim.fresh_ground_30hz import content_hash

VERSION = 'RIVAL2_DIRECT_SKILLS_FINISHING_GOAL_V2'


def reward_authority():
    return dict(
        version=VERSION,
        parent_reward_authority_sha256=content_hash(original_authority()),
        changed_role=2,
        finishing='Native +10 score / -10 concede, unchanged within-four-tick discount; -1 unsuccessful time-limit ending, once per player episode.',
        success='Finishing only: true native goal scored by this player. A goal/concede terminal never also pays time-limit failure.',
        retired_finishing_rewards=dict(first_touch=0.0,on_target_touch=0.0,approach=0.0),
        telemetry='Existing first_touch and on_target_touch events remain detected/raw, not paid bonuses. last_skill.weighted is zero for them in finishing; approach zero. Existing event names are retained for schema compatibility.',
        other_roles='Exact V1 natural PBRS, challenge, defense and kickoff outputs and tracker state. No extra terminal, action or mechanic reward.',
        physics_hz=120,policy_hz=30,
        transition='No new episode reset or length; true goal first-tick latch and pre-reset truncation bootstrap unchanged.',
        rationale='The frozen parent touches64/64 hard finishing balls but scores6/64. Keeper-ignorant projection is not the missing task outcome.',
        no_new_detector=True,
    )


class FinishingGoalEvents(SkillEvents):
    def calculate(self,before,after,touches,roles,terminated,truncated,scoring_team):
        was_failed = self.paid[...,6].clone()
        bonus, events, weighted, approach, success = super().calculate(
            before,after,touches,roles,terminated,truncated,scoring_team)
        finishing = roles == 2
        failure = finishing & truncated[:,None] & ~terminated[:,None] & ~was_failed
        events[...,6] = torch.where(finishing,failure,events[...,6])
        self.paid[...,6] |= failure
        weighted = weighted.masked_fill(finishing[...,None],0.0)
        weighted[...,6] = torch.where(finishing,-failure.to(before.dtype),weighted[...,6])
        approach = approach.masked_fill(finishing,0.0)
        bonus = torch.where(finishing,-failure.to(before.dtype),bonus)
        winner = scoring_team[:,None] == torch.arange(2,device=roles.device)[None]
        success = torch.where(finishing,terminated[:,None] & winner,success)
        return bonus,events,weighted,approach,success


class FinishingGoalEnv(DirectSkillsEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.events = FinishingGoalEvents(self.num_envs,self.device)
        self.reward_version = VERSION
        self.contract_hashes = dict(self.contract_hashes,reward=content_hash(reward_authority()))
