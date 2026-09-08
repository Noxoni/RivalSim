"""Fresh full-scenario PPO with explicit first-touch and off-ground-ball bonuses."""
from dataclasses import replace

import numpy as np
import torch
import warp as wp

from rivalsim.fresh_acquisition_only_v1 import SEED, MAX_UPDATES, WORLDS, new_model, authority as old_authority
from rivalsim.sustained_standing_kickoff_v1 import standing_training_starts
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, AcquisitionCollector, NAMES
from rivalsim.sustained_gameplay_v1 import reward_authority as base_reward, PHYSICS_GAMMA
from rivalsim.fresh_ground_30hz import content_hash
from rivalsim.kernels.ball_world import BALL_RADIUS_BT

VERSION = "RIVAL2_FRESH_SUSTAINED_CONTACT_V1"
FIRST_TOUCH_REWARD = 1.0
AIR_TOUCH_REWARD = .25
MAX_AIR_TOUCHES = 10
MIN_BALL_CENTER_BT = BALL_RADIUS_BT + .04  # 2uu clearance above the floor


def training_bank(worlds=WORLDS):
    return standing_training_starts(worlds, retired=False)


def reward_authority():
    return dict(version=VERSION+"_REWARD", base=base_reward(),
        first_touch_reward=FIRST_TOUCH_REWARD,
        first_touch="Each player's own first native contact onset in every episode, independent of opponent first contact, once only. Applies across all six scenario families.",
        air_touch_reward=AIR_TOUCH_REWARD, maximum_paid_air_touches_per_player_episode=MAX_AIR_TOUCHES,
        air_touch="Each distinct native car-ball contact onset when the ball's PRE-CONTACT center is above its physical sphere radius plus2uu floor-clearance tolerance. Ball need not be high and car need not be airborne. A ground touch that pops the ball does not retrospectively qualify. Continuous contact is one event. Maximum2.5 per player per episode.",
        ball_radius_bt=BALL_RADIUS_BT, minimum_ball_center_bt=MIN_BALL_CENTER_BT,
        timing="Read native contact-count increment and each contacting car's captured pre-contact ball position at each120Hz tick. Add bonus*physics_gamma**substep. Before or on goal only, never after. First-touch and airborne-touch may stack with goal/control/potentials.",
        episode="No touch-triggered reset. Retain goal or45s without either player's contact. Clear both payout budgets only on episode reset; no refill at PPO or evaluation boundary.",
        learning_mask="Both current sides learn in self-play; Nexto state/reward is inference-only and excluded from PPO.",
        interpretation="Deliberate direct contact objectives, not potential-only shaping or a named mechanic classifier. No scripted controls or task identifier.")


def authority():
    result = old_authority()
    result.update(version=VERSION, reward=reward_authority(),
        scenarios="Full existing corrected six-family bank:50% acquisition,15% natural,10% challenge,10% finishing,7.5% defense,7.5% kickoff (integer rounding). Includes exact stationary native kickoffs; no assisted kickoff velocity. No family retirement during this comparison.",
        episode_semantics="First own contact+1; distinct off-ground-ball contact+.25, up to10 paid per player. Continue after all contacts; original goals, potentials, control, and inactivity remain.",
        control_limit="User superseded acquisition-only isolation before its first training step. This fresh run changes both reward and scenario distribution; it is not a causal single-factor comparison.",
        source_acquisition_seed=2026090802, source_unique_states=WORLDS,
        initialization="Construct a new random actor/entities/actor GRU and independent critic MLP/GRU, empty Adam, seed2026090810. No trained checkpoint or prior optimizer loaded.")
    return result


@wp.kernel
def contact_bonus_tick(
    native_count: wp.array(dtype=wp.int32),
    pre_ball_a: wp.array(dtype=wp.vec3), pre_ball_b: wp.array(dtype=wp.vec3),
    goal: wp.array(dtype=wp.int32), interval_tick: wp.array(dtype=wp.int32),
    prior_goal: wp.array(dtype=wp.int32), prior_count: wp.array(dtype=wp.int32),
    first_paid: wp.array(dtype=wp.int32), air_paid: wp.array(dtype=wp.int32),
    first_reward: wp.array(dtype=wp.float32), air_reward: wp.array(dtype=wp.float32),
    first_events: wp.array(dtype=wp.int32), air_events: wp.array(dtype=wp.int32),
    air_eligible: wp.array(dtype=wp.int32), repeats: wp.array(dtype=wp.int32),
    discount: float, minimum_height_bt: float,
):
    world = wp.tid()
    k = wp.max(interval_tick[world]-1, 0)
    for side in range(2):
        index = world*2+side
        onset = native_count[index] > prior_count[index]
        if onset and prior_goal[world] == 0:
            if first_paid[index] == 0:
                first_paid[index] = 1
                first_events[index] += 1
                first_reward[index] += wp.pow(discount, float(k))
            else:
                repeats[index] += 1
            p = pre_ball_a[world]
            if side == 1:
                p = pre_ball_b[world]
            if p[2] > minimum_height_bt:
                air_eligible[index] += 1
                if air_paid[index] < 10:
                    air_paid[index] += 1
                    air_events[index] += 1
                    air_reward[index] += .25*wp.pow(discount, float(k))
        prior_count[index] = native_count[index]
    prior_goal[world] = goal[world]


class ContactEnv(AcquisitionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_hashes.pop(self.reward_version)
        self.reward_version = VERSION+"_REWARD"
        self.contract_hashes[self.reward_version] = content_hash(reward_authority())
        self.bonus = {}
        for name in ("prior_count", "first_paid", "air_paid", "first_events", "air_events", "air_eligible", "repeats", "first_reward", "air_reward"):
            dtype = wp.float32 if name.endswith("reward") else wp.int32
            self.bonus[name] = wp.zeros(self.num_envs*2, dtype=dtype, device=str(self.device))
        self.prior_goal = wp.zeros(self.num_envs, dtype=wp.int32, device=str(self.device))
        self.bonus_views = {k:wp.to_torch(v).reshape(self.num_envs, 2) for k,v in self.bonus.items()}
        self.contact_metrics = torch.zeros(9, device=self.device, dtype=torch.float64)
        original = self.world._launch_tick
        def with_contact_bonus():
            original()
            state = self.world.rival2
            wp.launch(contact_bonus_tick, dim=self.num_envs, inputs=[
                state.touch_count, self.world.car_ball.pre_ball_position_bt,
                self.world.car_ball_b.pre_ball_position_bt, state.goal_latched, state.interval_tick,
                self.prior_goal, self.bonus["prior_count"], self.bonus["first_paid"], self.bonus["air_paid"],
                self.bonus["first_reward"], self.bonus["air_reward"], self.bonus["first_events"],
                self.bonus["air_events"], self.bonus["air_eligible"], self.bonus["repeats"],
                PHYSICS_GAMMA, MIN_BALL_CENTER_BT], device=self.world.device)
        self.world._launch_tick = with_contact_bonus

    def _step_impl(self, action, markers=None, tick_action_provider=None):
        self._activate_torch_stream()
        for name, array in self.bonus.items():
            if name not in ("first_paid", "air_paid"):
                array.zero_()
        self.prior_goal.zero_()
        tr = super()._step_impl(action, markers, tick_action_provider)
        b = self.bonus_views
        first, air = b["first_reward"].clone(), b["air_reward"].clone()
        current = self.learner
        self.contact_metrics += torch.stack((
            b["first_events"].masked_fill(~current,0).sum(),
            b["first_events"].masked_fill(current,0).sum(),
            b["air_events"].masked_fill(~current,0).sum(),
            b["air_events"].masked_fill(current,0).sum(),
            b["air_eligible"].masked_fill(~current,0).sum(),
            b["repeats"].masked_fill(~current,0).sum(),
            first.masked_fill(~current,0).sum(dtype=torch.float64),
            air.masked_fill(~current,0).sum(dtype=torch.float64),
            (b["air_paid"] >= MAX_AIR_TOUCHES).masked_fill(~current,False).sum(),
        )).double()
        self.last_components["first_touch"] = first
        self.last_components["air_touch"] = air
        reward = tr.reward + first + air
        self.last_components["total"] = reward
        self.last_native["first_touch_awarded"] = b["first_events"].clone()
        self.last_native["air_touch_awarded"] = b["air_events"].clone()
        b["first_paid"].masked_fill_(tr.reset_mask[:, None], 0)
        b["air_paid"].masked_fill_(tr.reset_mask[:, None], 0)
        return replace(tr, reward=reward)


class ContactCollector(AcquisitionCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_component_names = (*self.reward_component_names, "first_touch", "air_touch")

    def collect(self):
        self.env.contact_metrics.zero_()
        result = super().collect()
        v = self.env.contact_metrics.cpu().tolist()
        self.last_metrics["contact_bonuses"] = dict(zip((
            "learner_first_awards", "excluded_first_awards", "learner_air_awards", "excluded_air_awards",
            "learner_eligible_air_contacts", "repeated_contacts_without_first_bonus",
            "learner_first_reward", "learner_air_reward", "air_budget_capped_agent_decisions"),v,strict=True))
        components = self.last_metrics["potential_reward_components"]
        for key,amount,count,unit in (("first_touch",v[6],v[0],1.),("air_touch",v[7],v[2],.25)):
            if abs(components[key]-amount)>1e-5 or not (count*unit*PHYSICS_GAMMA**3-1e-5<=amount<=count*unit+1e-5):
                raise RuntimeError("Contact payout accounting failed: "+key)
        return result
