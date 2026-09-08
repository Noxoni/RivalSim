"""Fresh sustained-gameplay lane: separate recurrent critic, goal-first reward.

Scenario labels select initial states only. They never select rewards, policy
inputs, controllers, or task-completion resets. No checkpoint is loaded here.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import warp as wp

from rivalsim.fresh_ground_30hz import (
    FreshGroundEnv, WEIGHTS, content_hash, potentials,
)
from rivalsim.recurrent_execution import gru_reset_spans
from rivalsim.rival2_env import Rival2Step
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from rivalsim.ssl_foundation_v1 import _controllability

VERSION = "RIVAL2_SUSTAINED_GAMEPLAY_V1"
POLICY_VERSION = "RIVAL2_ENTITY_JOINT90_SEPARATE_RECURRENT_CRITIC_V1"
SEED = 2026090801
REWARD_HALF_LIFE_SECONDS = 30.0
TRACE_HALF_LIFE_SECONDS = 3.0
GAMMA = 2 ** (-1 / (30 * REWARD_HALF_LIFE_SECONDS))
PHYSICS_GAMMA = GAMMA ** .25
GAE_LAMBDA = 2 ** (-1 / (30 * TRACE_HALF_LIFE_SECONDS)) / GAMMA
IDLE_TICKS = 45 * 120
IDLE_PENALTY = -1.0
CONTROL_RATE = .02  # reward per simulated second, maximal exclusive control
COMPONENTS = (*WEIGHTS, "terminal_goal", "sustained_control", "inactivity", "total")


def ppo_config():
    return Rival2PPOConfig(gamma=GAMMA, gae_lambda=GAE_LAMBDA, learning_rate=1e-4,
                          entropy_coefficient=.001, epochs=2, rollout_horizon=90)


def reward_authority():
    return dict(version=VERSION, goal=10., concede=-10., weights=WEIGHTS,
        potential="Same seven state potentials, combined gamma*Phi(next)-Phi(before). "
                  "True goal successor potential zero; inactivity retains pre-reset potential.",
        gamma=GAMMA, physics_gamma=PHYSICS_GAMMA,
        reward_half_life_seconds=REWARD_HALF_LIFE_SECONDS,
        gae_lambda=GAE_LAMBDA, trace_half_life_seconds=TRACE_HALF_LIFE_SECONDS,
        sustained_control_rate=CONTROL_RATE,
        sustained_control="At each physics tick: native last-exclusive toucher times positive "
            "own-minus-opponent physical controllability. Controllability is proximity "
            "clamp((500-distance)/350,0,1) times velocity match clamp(1-relative_speed/1200,0,1). "
            "Simultaneous touches clear exclusivity. No touch itself earns a bonus. "
            "Contact ownership resets at episode reset; payment continues while control is maintained. "
            "No control payment on or after a goal tick.",
        control_max_discounted_forever=CONTROL_RATE / 120 / (1-PHYSICS_GAMMA),
        idle_ticks=IDLE_TICKS, inactivity_penalty=IDLE_PENALTY,
        inactivity="45 seconds without a native car-ball contact by EITHER player, "
            "including sustained contact, not just a new contact onset. "
            "Goal takes precedence. Apply -1 to both players once at decision boundary. "
            "Truncated, not goal-terminal; bootstrap pre-reset critic value, cut GAE chain "
            "and reset both memories before the replacement scenario.",
        task_success_resets=False, fixed_episode_duration_limit=None,
        limitation="A deliberately small occupancy objective, not purely policy-invariant shaping. "
            "Its maximum discounted lifetime payment is below 1 versus goal 10, but that "
            "does not prove lexicographic goal dominance or eliminate stalling incentives.")


class MemoryCritic(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for din, dout in ((182, 512), (512, 512), (512, 512), (512, 256)):
            layer = nn.Linear(din, dout)
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.memory = nn.GRU(256, 256, batch_first=True)
        self.head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.head.weight, 1.)
        nn.init.zeros_(self.head.bias)

    def forward(self, observation, hidden, reset_before=None, reset_metadata=None):
        step = observation.ndim == 2
        if step:
            observation = observation[:, None]
            if reset_before is not None:
                reset_before = reset_before.reshape(-1, 1)
        encoded = self.features(observation)
        context, state = gru_reset_spans(self.memory, encoded, hidden, reset_before,
                                        reset_metadata)
        value = self.head(context).squeeze(-1)
        return (value[:, 0] if step else value), state


class SustainedPolicy(EntityJointControlActorCritic):
    """Two independent banks [actor, critic], each [1,B,256]. No shared gradients."""
    architecture_version = POLICY_VERSION

    def __init__(self):
        super().__init__()
        self.critic = MemoryCritic()

    def initial_hidden(self, batch_size, *, device=None, dtype=None):
        actor = super().initial_hidden(batch_size, device=device, dtype=dtype)
        return torch.cat((actor, torch.zeros_like(actor)), dim=0)

    def _forward(self, observation, hidden=None, *, reset_before=None,
                 reset_metadata=None, include_value=True):
        if hidden is None:
            hidden = self.initial_hidden(len(observation), device=observation.device,
                                         dtype=observation.dtype)
        if hidden.shape != (2, len(observation), 256):
            raise ValueError("Expected independent actor/critic hidden banks [2,B,256]")
        logits, _, actor_state = super()._forward(observation, hidden[:1],
            reset_before=reset_before, reset_metadata=reset_metadata, include_value=False)
        value, critic_state = (None, hidden[1:])
        if include_value:
            value, critic_state = self.critic(observation, hidden[1:], reset_before, reset_metadata)
        return logits, value, torch.cat((actor_state, critic_state), dim=0)

    def bootstrap_value(self, observation, history_after_current):
        # Do not call a stateless critic or consume the successor twice.
        return self.critic(observation, history_after_current[1:])[0]

    def isolated_value(self, observation, *args, **kwargs):
        raise RuntimeError("This critic requires history. Use bootstrap_value or critic(obs,hidden).")


def fresh_model(seed=SEED):
    torch.manual_seed(seed)
    model = SustainedPolicy()
    # Enable the random recurrent actor branch immediately, with a small output
    # initialization. All actor/entity/critic weights are freshly drawn, not BC.
    nn.init.orthogonal_(model.context_actor.weight, .01)
    nn.init.zeros_(model.context_actor.bias)
    return model


def decision_reward(before, after, first_goal, scoring_team, control_payment, truncated):
    terminal = first_goal >= 0
    previous, successor = potentials(before), potentials(after)
    components = {key: WEIGHTS[key] * (GAMMA * successor[key].masked_fill(
        terminal[:, None], 0) - previous[key]) for key in WEIGHTS}
    side = torch.arange(2, device=before.device)[None]
    outcome = torch.where(side == scoring_team[:, None], 10., -10.)
    components["terminal_goal"] = (outcome * PHYSICS_GAMMA ** first_goal.clamp_min(0)[:, None]
                                    ).masked_fill(~terminal[:, None], 0)
    components["sustained_control"] = control_payment
    components["inactivity"] = (truncated[:, None].expand(-1, 2) * IDLE_PENALTY
                                  * PHYSICS_GAMMA ** 3)
    reward = sum(components.values())
    return reward, dict(components, total=reward)


class SustainedEnv(FreshGroundEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_version = VERSION + "_REWARD"
        self.episode_version = VERSION + "_EPISODES"
        self.backend_contract_hashes = dict(self.contract_hashes)
        self.contract_hashes = {k:v for k,v in self.contract_hashes.items()
                                if k.startswith(("RIVAL2_OBS_", "RIVAL2_ACTION_"))}
        self.contract_hashes[self.reward_version] = content_hash(reward_authority())
        self.contract_hashes[self.episode_version] = content_hash(dict(
            goal="terminated and reset", no_contact_physics_ticks=IDLE_TICKS,
            inactivity="truncated, pre-reset value bootstrap, reset both memories",
            fixed_age_timeout=None, task_completion_reset=False))
        self.last_toucher = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.idle_ticks = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.contact_a = wp.to_torch(self.world.car_ball.hit_this_tick)
        self.contact_b = wp.to_torch(self.world.car_ball_b.hit_this_tick)
        template = self.world.ssl_foundation_reset
        self.focal = wp.to_torch(template.current_focal_side)
        self.family = wp.to_torch(template.current_family)
        self.learner = torch.ones((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.opponent_family = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stat_names = ("world_seconds", "touches_focal", "touches_opponent", "goals_focal",
                           "goals_opponent", "inactivity_resets", "ended_episode_seconds")
        self.statistics = torch.zeros((10, len(self.stat_names)), device=self.device, dtype=torch.float64)

    def _step_impl(self, action, markers=None, tick_action_provider=None):
        if markers is not None:
            raise ValueError("Use wall-clock profiling")
        self._activate_torch_stream()
        before = self.observation
        self.world.begin_decision()
        emitted = self.bridge.set_actions(action).clone()
        views = self.bridge.views
        first_goal = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        prior_touches = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        touches = torch.zeros_like(prior_touches)
        first_touch_tick = torch.full_like(prior_touches, -1)
        control_payment = torch.zeros((self.num_envs, 2), device=self.device)
        for tick in range(4):
            if tick_action_provider is not None:
                self.bridge.set_actions(tick_action_provider(tick))
            self.world.step(1)
            count = views["rival2.touch_count"].reshape(self.num_envs, 2).long()
            delta = (count-prior_touches).clamp_min(0).masked_fill((first_goal >= 0)[:, None], 0)
            first_touch_tick = torch.where((delta > 0) & (first_touch_tick < 0), tick, first_touch_tick)
            touches += delta
            prior_touches.copy_(count)
            contacted = torch.stack((self.contact_a != 0, self.contact_b != 0), -1)
            contacted &= (first_goal < 0)[:, None]
            self.idle_ticks.copy_(torch.where(contacted.any(-1), 0, self.idle_ticks+1))
            owner = torch.where(contacted.sum(-1) == 1, contacted.long().argmax(-1), -1)
            self.last_toucher.copy_(torch.where(contacted.any(-1), owner, self.last_toucher))
            hit_goal = self.goal_latched != 0
            first_goal = torch.where(hit_goal & (first_goal < 0), tick, first_goal)
            ball_p = views["ball_pos"][:, None, :]
            ball_v = views["ball_vel"][:, None, :]
            own_control, _ = _controllability(ball_p, ball_v,
                views["car_pos"].reshape(self.num_envs, 2, 3),
                views["car_vel"].reshape(self.num_envs, 2, 3))
            advantage = (own_control - own_control.flip(-1)).clamp_min(0)
            owns = self.last_toucher[:, None] == torch.arange(2, device=self.device)[None]
            payment = advantage * owns * (CONTROL_RATE / 120 * PHYSICS_GAMMA ** tick)
            control_payment += payment.masked_fill((first_goal >= 0)[:, None], 0)
        # The legacy counter counts onsets, not sustained contact; this lane's
        # clock resets on ANY physical contact, so holding control is not idle.
        views["rival2.no_touch_ticks"].copy_(self.idle_ticks)
        after = self.bridge.observation().clone()
        team = views["rival2.scoring_team_latched"].long().clone()
        terminated = views["rival2.terminated"].bool().clone()
        if not torch.equal(terminated, first_goal >= 0):
            raise RuntimeError("30Hz goal/cadence contract failure")
        age = views["rival2.episode_ticks"].clone()
        idle = views["rival2.no_touch_ticks"].clone()
        # The reusable kernel also emits legacy administrative flags. It never
        # physically resets here. Replace those flags BEFORE any reset consumer.
        truncated = (idle >= IDLE_TICKS) & ~terminated
        reset = terminated | truncated
        views["rival2.truncated"].copy_(truncated)
        views["rival2.reset_mask"].copy_(reset)
        reward, self.last_components = decision_reward(before, after, first_goal, team,
                                                       control_payment, truncated)
        self.last_native = dict(touch_count=touches, first_touch_tick=first_touch_tick,
            first_goal_tick=first_goal, scoring_team=team, episode_ticks=age, no_touch_ticks=idle)
        rows = torch.arange(self.num_envs, device=self.device)
        focal = self.focal.long()
        statistics = torch.stack((torch.full_like(age, 1/30, dtype=torch.float64),
            touches[rows, focal], touches[rows, 1-focal], terminated & (team == focal),
            terminated & (team != focal), truncated, age/120*reset), -1).double()
        self.statistics.index_add_(0, self.family.long()*2+self.opponent_family, statistics)
        self.world.apply_interval_resets()
        self.last_toucher.masked_fill_(reset, -1)
        self.idle_ticks.masked_fill_(reset, 0)
        self.observation = self.bridge.observation()
        self.decision_count += 1
        return Rival2Step(self.observation, after, emitted, reward, terminated, truncated, reset)
