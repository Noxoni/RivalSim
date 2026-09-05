"""Fresh random-weight, 30 Hz ground-learning lane. No checkpoint imports.

The seven *state potentials* are not per-tick behavior payments. Four physics
rewards telescope to one decision reward; goals enter an absorbing reward
state immediately, and administrative truncations retain their value bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import warp as wp

from rivalsim.rival2_contracts import RIVAL2_REWARD_GOAL_ONLY_VERSION
from rivalsim.rival2_env import Rival2Env, Rival2Step
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic, IndependentCriticPolicyConfig
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.ssl_foundation_v1 import (
    SslFoundationScenarioBatch, _physical_vectors, _controllability,
    _set_coherent_ground_route, ssl_foundation_potentials,
)
from rivalsim.state import StateSnapshot, CAR_VEC3_FIELDS, CAR_FLOAT_FIELDS, CAR_INT_FIELDS
from rivalsim.static_world import make_standard_kickoff_state

VERSION = "RIVAL2_FRESH_GROUND_30HZ_V1"
REWARD_VERSION = "RIVAL2_REWARD_FRESH_GROUND_30HZ_V1"
CHECKPOINT_FORMAT = VERSION + "_CHECKPOINT"
GAMMA = 0.995
PHYSICS_GAMMA = GAMMA ** 0.25
GAE_LAMBDA = 2.0 ** (-1.0 / 90.0) / GAMMA
SEED = 2026090501
WEIGHTS = dict(field=1.25, access=.75, control=.75, defense=.50,
               alignment=.30, boost=.15, goal_velocity=.25)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest().upper()


def ppo_config():
    return Rival2PPOConfig(gamma=GAMMA, gae_lambda=GAE_LAMBDA, learning_rate=1e-4,
                          entropy_coefficient=.001, epochs=2, rollout_horizon=90)


def policy_config():
    # Learned exploration, not the tiny fixed noise used around the old V5 actor.
    return IndependentCriticPolicyConfig(log_std_min=math.log(.20), log_std_max=math.log(.90))


@dataclass(frozen=True)
class LearnedExploration:
    distribution_override: None = None

    def as_dict(self):
        return {"kind": "learned_hybrid", "initial_analog_std": .65,
                "analog_std_bounds": [.20, .90], "button_temperature": 1.0,
                "initial_button_probabilities": [.12, .35, .08]}


def fresh_model(seed=SEED):
    """Initialize every trainable tensor here; never import policy/optimizer weights."""
    torch.manual_seed(seed)
    model = IndependentCriticActorCritic(policy_config())
    # The reusable class copies its *random* trunk into its critic. Draw separate
    # critic features instead, and enable the recurrent branch from step zero.
    for layer in model.critic.features.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, math.sqrt(2))
            torch.nn.init.zeros_(layer.bias)
    torch.nn.init.orthogonal_(model.context_actor.weight, .01)
    torch.nn.init.zeros_(model.context_actor.bias)
    with torch.no_grad():
        model.actor.bias[5:10] = math.log(.65)
        model.actor.bias[10:13] = torch.logit(torch.tensor([.12, .35, .08]))
    return model


def potentials(observation):
    base = ssl_foundation_potentials(observation)
    result = {name: getattr(base, name) for name in WEIGHTS if name != "goal_velocity"}
    v = _physical_vectors(observation)
    own_control, own_d = _controllability(v["ball_position"], v["ball_velocity"],
                                         v["self_position"], v["self_velocity"])
    opp_control, opp_d = _controllability(v["ball_position"], v["ball_velocity"],
                                         v["opponent_position"], v["opponent_velocity"])
    own_access, opp_access = torch.exp(-own_d / 2000), torch.exp(-opp_d / 2000)
    # [-.25,1]: own-distance sensitivity persists even far behind the opponent.
    result["access"] = (1 - torch.maximum(own_control, opp_control)) * (
        .75 * own_access + .25 * (own_access - opp_access))
    target = torch.zeros_like(v["ball_position"])
    target[..., 1], target[..., 2] = 5120, 321
    delta = target - v["ball_position"]
    direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1)
    result["goal_velocity"] = ((v["ball_velocity"] * direction).sum(-1) / 3000).clamp(-1, 1)
    return result


def decision_reward(before, after, first_goal_tick, scoring_team):
    """first_goal_tick = 0..3, or -1; aggregate r0+g*r1+g^2*r2+g^3*r3."""
    prior, successor = potentials(before), potentials(after)
    terminal = first_goal_tick >= 0
    components = {key: WEIGHTS[key] * (
        GAMMA * successor[key].masked_fill(terminal[:, None], 0) - prior[key]) for key in WEIGHTS}
    shaping = sum(components.values())
    side = torch.arange(2, device=before.device)[None, :]
    outcome = torch.where(side == scoring_team[:, None], 10., -10.)
    outcome = outcome * PHYSICS_GAMMA ** first_goal_tick.clamp_min(0)[:, None]
    outcome = outcome.masked_fill(~terminal[:, None], 0)
    return shaping + outcome, dict(components, terminal_goal=outcome, total=shaping + outcome)


def scenarios(worlds, seed=SEED + 1, *, family_only=None):
    """50% acquisition, 25% achievable finishing, 15% real kickoff, 10% ground 1v1.

    Both sides are freely controlled from tick zero. No motor/script prefix;
    heading and momentum are coherent but not universally aimed at the ball.
    """
    rng = np.random.default_rng(seed)
    if family_only is None:
        proportions = [(2, .50), (4, .25), (1, .15), (0, .10)]
        family = np.concatenate([np.full(int(worlds*p), f, dtype=np.int8) for f, p in proportions])
        family = np.pad(family, (0, worlds-len(family)), constant_values=2)
        rng.shuffle(family)
    else:
        family = np.full(worlds, family_only, dtype=np.int8)
    focal = np.arange(worlds, dtype=np.int8) % 2
    rng.shuffle(focal)
    state = StateSnapshot.empty(worlds)
    state.on_ground.fill(1)
    state.car_pos[..., 2], state.ball_pos[:, 2] = 17., 93.15
    state.boost[:] = rng.uniform(25, 100, (worlds, 2))
    kickoff = (family == 1).astype(np.int32)
    layouts = np.full(worlds, -1, dtype=np.int32)
    rows = np.flatnonzero(kickoff)
    if len(rows):
        layouts[rows] = (np.arange(len(rows)) + seed) % 5
        ks = make_standard_kickoff_state(len(rows), layouts[rows])
        for key in ks.__dataclass_fields__:
            getattr(state, key)[rows] = getattr(ks, key)
    for row, kind in enumerate(family):
        if kind == 1:
            continue
        ball = state.ball_pos[row]
        ball[:2] = rng.uniform((-1800, -1600), (1800, 1600))
        if kind == 4:
            ball[:2] = rng.uniform((-900, 1800), (900, 3600))
        angle = rng.uniform(-.8, .8) if kind == 4 else rng.uniform(-1.3, 1.3)
        distance = rng.uniform(250, 750) if kind == 4 else rng.uniform(400, 1600)
        direction = np.array([np.sin(angle), np.cos(angle)])
        state.car_pos[row, 0, :2] = ball[:2] - distance * direction
        state.car_pos[row, 1, :2] = rng.uniform((-2400, -3700), (2400, 3800))
        if np.linalg.norm(state.car_pos[row, 1, :2] - ball[:2]) < 1600:
            state.car_pos[row, 1, :2] = [2600 if ball[0] < 0 else -2600, -2600]
        if kind == 0:
            state.car_pos[row, 0, :2] = rng.uniform((-2600, -3800), (2600, 1500))
            if np.linalg.norm(state.car_pos[row, 0, :2] - ball[:2]) < 200:
                state.car_pos[row, 0, 1] -= 400
        state.ball_vel[row, :2] = rng.uniform(-250, 250, 2) if row % 2 else 0
        for car in (0, 1):
            offset = (-.45, .45) if kind == 4 and car == 0 else (-1.5, 1.5)
            _set_coherent_ground_route(state, row, car, rng, ball[:2], (0, 400), offset)
        if np.linalg.norm(state.car_pos[row, 0] - state.car_pos[row, 1]) < 180:
            state.car_pos[row, 1, :2] = [3000 if ball[0] < 0 else -3000, 4300]
            _set_coherent_ground_route(state, row, 1, rng, ball[:2], (0, 400), (-1.5, 1.5))
    for row in np.flatnonzero((focal == 1) & ~kickoff.astype(bool)):
        for key in (*CAR_VEC3_FIELDS, *CAR_FLOAT_FIELDS, *CAR_INT_FIELDS, "car_quat"):
            value = getattr(state, key)
            value[row] = value[row, ::-1].copy()
        for key in ("car_pos", "car_vel", "car_ang_vel", "flip_rel_torque",
                    "ball_pos", "ball_vel", "ball_ang_vel"):
            getattr(state, key)[row, ..., :2] *= -1
        for key in ("car_quat", "ball_quat"):
            x, y, z, w = np.moveaxis(getattr(state, key)[row].copy(), -1, 0)
            getattr(state, key)[row] = np.stack((-y, x, w, -z), axis=-1)
    state.validate()
    return SslFoundationScenarioBatch(state, family, focal, kickoff, layouts,
                                     np.full(worlds, -1, dtype=np.int32))


def scenario_hash(batch):
    digest = hashlib.sha256()
    for key in batch.state.__dataclass_fields__:
        digest.update(key.encode())
        digest.update(getattr(batch.state, key).tobytes())
    for key in ("family", "focal_side", "kickoff_indicator", "kickoff_layout"):
        digest.update(getattr(batch, key).tobytes())
    return digest.hexdigest().upper()


def authority():
    return {"version": VERSION, "initialization": "fresh_random_no_policy_parent",
            "seed": SEED, "worlds": 32768, "physics_hz": 120, "policy_hz": 30,
            "physics_ticks_per_action": 4, "ppo": asdict(ppo_config()),
            "policy": asdict(policy_config()), "critic_lr": 3e-4,
            "exploration": LearnedExploration().as_dict(), "weights": WEIGHTS,
            "reward": REWARD_VERSION, "reward_form": "terminal +/-10 plus gamma*Phi(next)-Phi(now)",
            "physics_discount": PHYSICS_GAMMA, "goal_successor_potential": 0,
            "goal_reward_time": "first scoring physics tick; discount within held action",
            "post_goal_remainder": "absorbing reward; reset once at decision boundary",
            "time_limit_seconds": 30, "no_touch_limit_seconds": 15,
            "truncation": "bootstrap pre-reset final observation; cut GAE trace",
            "access": "(1-max(control))*[.75*exp(-dself/2000)+.25*(exp(-dself/2000)-exp(-dopp/2000))]",
            "goal_velocity": "clamp(dot(ball_velocity,unit(ball->opponent_goal_center))/3000,-1,1)",
            "other_potentials": "unchanged deterministic ssl_foundation_v1 state functions",
            "direct_behavior_rewards": [], "potential_weights_change_during_run": False,
            "scenario_mix": {"acquisition": .50, "finishing": .25, "kickoff": .15, "ground_ongoing": .10},
            "opponents": {"initial_nexto": 0., "after_routine_acquisition": .20,
                          "routine_acquisition": "two consecutive evaluation boundaries: focal touch fraction >=.60, conditional median first touch <=5s, >=1 finishing goal; actual Nexto remains eval-only until then",
                          "frozen_old_policy_probability": 0., "both_selfplay_sides_trainable": True},
            "evaluation_updates": [0, 10, 20, 50], "evaluation_every_after_50": 50,
            "evaluation": {"worlds_per_case": 64, "seconds": 30, "seeds": [SEED+100, SEED+101, SEED+102],
                           "cases": ["acquisition_selfplay", "finishing_selfplay", "standard_kickoff_nexto"],
                           "deterministic_current_policy": True},
            "checkpoint_each_update": True, "permanent_snapshot_every": 50,
            "maximum_updates": None, "deadline": None, "stop": "user interrupt or numerical/corruption/runtime fault",
            "kl": "telemetry_only", "retention_objective": False,
            "resume": "only this lineage; fresh simulator episodes, preserved optimizer and all RNG counters",
            "sequence_microbatch_size": 728, "recurrent_execution": "complete 90-decision sequences; native reset masks"}


class FreshGroundEnv(Rival2Env):
    """Four physics ticks with first-goal timing and read-before-reset telemetry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION, **kwargs)
        assert self.policy_hz == 30 and self.physics_ticks_per_decision == 4
        assert self.world.gameplay_v3 is None and self.world.gameplay_120 is None
        self.reward_version = REWARD_VERSION
        self.contract_hashes = dict(self.contract_hashes, reward=content_hash(authority()))
        self.last_components = None
        self.last_native = None
        self.goal_latched = wp.to_torch(self.world.rival2.goal_latched)

    def _step_impl(self, action, markers=None, tick_action_provider=None):
        if markers is not None:
            raise ValueError("use wall-clock profiling for the isolated 30Hz lane")
        self._activate_torch_stream()
        before = self.observation
        self.world.begin_decision()
        emitted = self.bridge.set_actions(action).clone()
        views = self.bridge.views
        first_goal = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)
        prior_touches = torch.zeros((self.num_envs, 2), dtype=torch.int64, device=self.device)
        touches = torch.zeros_like(prior_touches)
        first_touch_tick = torch.full_like(prior_touches, -1)
        for tick in range(4):
            if tick_action_provider is not None:
                self.bridge.set_actions(tick_action_provider(tick))
            self.world.step(1)
            count = views["rival2.touch_count"].reshape(self.num_envs, 2).to(torch.int64)
            delta = (count - prior_touches).clamp_min(0).masked_fill((first_goal >= 0)[:, None], 0)
            first_touch_tick = torch.where((delta > 0) & (first_touch_tick < 0), tick, first_touch_tick)
            touches += delta
            prior_touches.copy_(count)
            hit = self.goal_latched != 0
            first_goal = torch.where(hit & (first_goal < 0), tick, first_goal)
        after = self.bridge.observation().clone()
        team = views["rival2.scoring_team_latched"].to(torch.int64).clone()
        terminated = views["rival2.terminated"].bool().clone()
        if not torch.equal(terminated, first_goal >= 0):
            raise RuntimeError("30Hz goal/cadence/reset contract failure")
        age = views["rival2.episode_ticks"].clone()
        truncated = (views["rival2.truncated"].bool() | (age >= 3600)) & ~terminated
        reset = terminated | truncated
        views["rival2.truncated"].copy_(truncated)
        views["rival2.reset_mask"].copy_(reset)
        reward, self.last_components = decision_reward(before, after, first_goal, team)
        self.last_native = {"touch_count": touches, "first_touch_tick": first_touch_tick,
                            "first_goal_tick": first_goal, "scoring_team": team, "episode_ticks": age,
                            "no_touch_ticks": views["rival2.no_touch_ticks"].clone()}
        self.world.apply_interval_resets()
        self.observation = self.bridge.observation()
        self.decision_count += 1
        return Rival2Step(self.observation, after, emitted, reward, terminated, truncated, reset)
