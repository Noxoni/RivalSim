"""Fresh acquisition-only comparison with one explicit own-first-contact bonus."""
from dataclasses import replace

import torch

from rivalsim.fresh_acquisition_only_v1 import (
    SEED, MAX_UPDATES, WORLDS, training_bank, new_model, authority as previous_authority,
)
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, AcquisitionCollector, FAMILY
from rivalsim.sustained_gameplay_v1 import PHYSICS_GAMMA
from rivalsim.fresh_ground_30hz import content_hash

VERSION = "RIVAL2_FRESH_ACQUISITION_TOUCH_V1"
FIRST_TOUCH_REWARD = 1.0


def reward_authority():
    base = previous_authority()["reward"]
    return dict(version=VERSION+"_REWARD", base=base, first_touch_reward=FIRST_TOUCH_REWARD,
        scope="Only acquisition-start episodes; once per player per episode, on its own native first contact. Opponent-first contact does not remove eligibility. Both players may earn it independently.",
        timing="First contact at physics substep k in [0,3] contributes +1*physics_gamma**k to the 30Hz decision. Before or on goal tick only; never after it.",
        episode="No touch-triggered reset; continue to existing goal or 45s no-contact timeout. Clear paid latches only on real episode resets, not PPO rollouts or evaluations.",
        other_rewards="All existing sustained-gameplay components remain independently payable and unchanged. This bonus is deliberately a direct reward, not potential shaping.",
        learning_mask="Current agents only enter PPO. Nexto reward is excluded by the unchanged train mask.")


def authority():
    result = previous_authority()
    result.update(version=VERSION, reward=reward_authority(),
        comparison="Fresh construction with the same random seed, unchanged starting bank, PPO, opponent mix and probes as fresh_acquisition_only_v1. No prior checkpoint is loaded. Only the first-contact bonus changes the training objective.",
        episode_semantics="Own first contact earns +1 once per player; all original rewards remain. Continue after touch to goal or45s without either car contacting. No touch terminal.",
        control_limit="One matched-seed reward comparison, not multi-seed proof or a general-gameplay promotion.")
    return result


def first_touch_payment(first_tick, first_goal, already_paid, acquisition):
    """Pure tensor boundary: same-goal-tick touch is valid, post-goal is not."""
    valid = ((first_tick >= 0) & (first_tick < 4) & ~already_paid
             & acquisition[:, None]
             & ((first_goal[:, None] < 0) | (first_tick <= first_goal[:, None])))
    payment = (FIRST_TOUCH_REWARD * PHYSICS_GAMMA ** first_tick.clamp_min(0)) * valid
    return valid, payment


class FirstTouchEnv(AcquisitionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_hashes.pop(self.reward_version)
        self.reward_version = VERSION+"_REWARD"
        self.contract_hashes[self.reward_version] = content_hash(reward_authority())
        self.first_touch_paid = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.first_touch_metrics = torch.zeros(5, device=self.device, dtype=torch.float64)

    def _step_impl(self, action, markers=None, tick_action_provider=None):
        # Capture episode membership BEFORE super() resets the physical world.
        acquisition = self.family == FAMILY
        transition = super()._step_impl(action, markers, tick_action_provider)
        first_tick = self.last_native["first_touch_tick"]
        awarded, payment = first_touch_payment(first_tick, self.last_native["first_goal_tick"],
                                             self.first_touch_paid, acquisition)
        self.first_touch_metrics += torch.stack((
            awarded.sum(), (awarded & self.learner).sum(),
            (awarded & ~self.learner).sum(),
            payment.masked_fill(~self.learner, 0).sum(dtype=torch.float64),
            ((first_tick >= 0) & self.first_touch_paid & acquisition[:, None]).sum(),
        )).double()
        self.first_touch_paid |= awarded
        self.first_touch_paid.masked_fill_(transition.reset_mask[:, None], False)
        self.last_native["first_touch_awarded"] = awarded
        self.last_components["first_touch"] = payment
        reward = transition.reward + payment
        self.last_components["total"] = reward
        return replace(transition, reward=reward)


class FirstTouchCollector(AcquisitionCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_component_names = (*self.reward_component_names, "first_touch")

    def collect(self):
        self.env.first_touch_metrics.zero_()
        result = super().collect()
        values = self.env.first_touch_metrics.cpu().tolist()
        self.last_metrics["first_touch_bonus"] = dict(zip((
            "native_awards_all_players", "learner_awards", "nonlearner_awards",
            "learner_reward_sum", "repeat_contact_decisions_without_bonus"), values, strict=True))
        # Check accounting at every rollout before any optimizer entry.
        bonus = self.last_metrics["potential_reward_components"]["first_touch"]
        if abs(bonus-values[3]) > 1e-5 or not (values[1]*PHYSICS_GAMMA**3-1e-5 <= bonus <= values[1]+1e-5):
            raise RuntimeError("First-contact payout accounting failed")
        return result
