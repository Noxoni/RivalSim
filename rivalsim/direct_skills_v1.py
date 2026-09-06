"""Direct, episode-bounded basic-skill objectives plus unmodified natural PBRS.

One entity/joint90 policy; no task ID in observations, router, scripted Rival
actions or mechanic classifiers. Skill predicates are documented task proxies,
not claims of exact possession, save intent, or a recognized fake kickoff.
"""

# Long literal authority descriptions are hashed and kept on one line for review.
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from rivalsim.fresh_ground_30hz import (
    WEIGHTS,
    FreshGroundEnv,
    content_hash,
    decision_reward,
)
from rivalsim.fresh_ground_30hz import (
    scenarios as old_scenarios,
)
from rivalsim.rival2_env import Rival2Step
from rivalsim.ssl_foundation_v1 import _physical_vectors, _set_coherent_ground_route
from rivalsim.state import CAR_FLOAT_FIELDS, CAR_INT_FIELDS, CAR_VEC3_FIELDS
from rivalsim.static_world import make_standard_kickoff_state

VERSION = "RIVAL2_DIRECT_SKILLS_V1"
SEED = 2026090601
NAMES = ("natural", "challenge", "finishing", "defense", "kickoff")
MIX = (0.30, 0.20, 0.20, 0.15, 0.15)
EVENTS = (
    "first_touch",
    "control_gain",
    "forward_control",
    "on_target_touch",
    "shot_clear",
    "lost_control",
    "failed_attempt",
)
EVENT_WEIGHTS = (0.5, 1.5, 0.75, 1.0, 1.5, -0.5, -1.0)


def reward_authority():
    return dict(
        version=VERSION,
        policy_hz=30,
        physics_hz=120,
        scenario_episode_mix=dict(zip(NAMES, MIX, strict=True)),
        natural="Unchanged seven-potential FreshGround reward and +/-10 native goal outcome",
        skills="Only +/-10 native goal outcome, below one-shot task events and bounded approach",
        events=dict(zip(EVENTS, EVENT_WEIGHTS, strict=True)),
        approach="challenge/finishing only before first touch: .5/30 times clamp(speed toward ball/2300,0,1)",
        control="Own native contact since last opponent contact, ball distance<=250uu, relative speed<=500uu/s, opponent distance>=own+150uu, for8 consecutive decisions",
        forward_control="After paid control gain, retain same physical control and advance canonical ball y by400uu",
        shot="Own native contact, planar goal-line projection within750uu of center, predicted height<=550uu, time in(0,4]s, canonical vy>500uu/s",
        clear="Defense only: own native contact turns a previously goal-bound shot into no goal-bound shot; clear bonus after8 safe decisions without opponent touch. Ground/low-ball projection only; task proxy not universal native save event",
        projection="Linearly extrapolate xy to goal plane y=+/-5120; z=max(93.15,z+vz*t-325*t*t). No wall/bounce prediction; initial ground threats validated with native rollout",
        limits="Each event at most once per player episode; no automatic rearming after success; skill episode12s, natural30s/no-touch15s",
        failure="Skill time-limit ending without a role success: -1; true goal never also timeout failure",
        role_success="challenge/kickoff:control gain; finishing:on-target touch or goal; defense:shot clear or control gain or goal",
        lost_control="challenge/kickoff only: after paid control, opponent establishes8-decision control, at most-.5",
        kickoff="Control/advancement rewarded whether first or second to touch. No waiting/fake/speedflip event reward, no scripted Rival prefix, no guarantee fake strategy emerges",
        terminal="Native first-goal physics tick, discounted within four-tick hold; later tick remainder absorbing",
        truncation="Pre-reset final observation bootstrap, GAE cut; skill success does not terminate",
        shaping_policy_invariance="Only natural lane is potential-based. Direct skill bonuses intentionally change the training objective as requested",
        no_other_direct_rewards=True,
        no_task_id_or_router=True,
    )


def _rotate_and_swap(state, row):
    for key in (*CAR_VEC3_FIELDS, *CAR_FLOAT_FIELDS, *CAR_INT_FIELDS, "car_quat"):
        value = getattr(state, key)
        value[row] = value[row, ::-1].copy()
    for key in (
        "car_pos",
        "car_vel",
        "car_ang_vel",
        "flip_rel_torque",
        "ball_pos",
        "ball_vel",
        "ball_ang_vel",
    ):
        getattr(state, key)[row, ..., :2] *= -1
    for key in ("car_quat", "ball_quat"):
        x, y, z, w = np.moveaxis(getattr(state, key)[row].copy(), -1, 0)
        getattr(state, key)[row] = np.stack((-y, x, w, -z), axis=-1)


def scenarios(worlds, seed=SEED, *, family_only=None):
    rng = np.random.default_rng(seed)
    # Start with valid grounded state, then construct each role in blue coordinates.
    batch = old_scenarios(worlds, seed=seed, family_only=0)
    if family_only is None:
        families = np.concatenate(
            [np.full(int(worlds * p), i, np.int32) for i, p in enumerate(MIX)]
        )
        families = np.pad(families, (0, worlds - len(families)), constant_values=0)
        rng.shuffle(families)
    else:
        families = np.full(worlds, family_only, np.int32)
    focal = np.arange(worlds, dtype=np.int32) % 2
    rng.shuffle(focal)
    state = batch.state
    state.car_vel.fill(0)
    state.car_ang_vel.fill(0)
    state.ball_vel.fill(0)
    state.ball_ang_vel.fill(0)
    state.car_pos[..., 2] = 17
    state.ball_pos[:, 2] = 93.15
    state.boost[:] = rng.uniform(25, 100, (worlds, 2))
    kickoff = (families == 4) | ((families == 0) & (rng.random(worlds) < 0.5))
    layouts = np.full(worlds, -1, np.int32)
    for row, family in enumerate(families):
        if kickoff[row]:
            layouts[row] = (row + seed) % 5
            ks = make_standard_kickoff_state(1, layouts[row : row + 1])
            for key in ks.__dataclass_fields__:
                getattr(state, key)[row] = getattr(ks, key)[0]
            continue
        ball = state.ball_pos[row]
        if family == 1:
            ball[:2] = rng.uniform((-1500, -1400), (1500, 1800))
            state.car_pos[row, 0, :2] = ball[:2] + rng.uniform((-350, -650), (350, -250))
            state.car_pos[row, 1, :2] = ball[:2] + rng.uniform((-350, 250), (350, 900))
            state.ball_vel[row, :2] = rng.uniform((-150, -100), (150, 100))
        elif family == 2:
            ball[:2] = rng.uniform((-1100, 2200), (1100, 3800))
            state.car_pos[row, 0, :2] = ball[:2] + rng.uniform((-350, -700), (350, -250))
            state.car_pos[row, 1, :2] = [rng.uniform(-650, 650), rng.uniform(4350, 4900)]
            state.ball_vel[row, :2] = rng.uniform((-100, -100), (100, 200))
        elif family == 3:
            ball[:2] = rng.uniform((-850, -3200), (850, -1600))
            speed = rng.uniform(650, 1400)
            flight_time = (-5120 - ball[1]) / -speed
            state.ball_vel[row, :2] = [(rng.uniform(-550, 550) - ball[0]) / flight_time, -speed]
            state.car_pos[row, 0, :2] = [rng.uniform(-650, 650), rng.uniform(-4800, -4150)]
            state.car_pos[row, 1, :2] = ball[:2] + np.array(
                [rng.uniform(-250, 250), rng.uniform(500, 1000)]
            )
        else:
            ball[:2] = rng.uniform((-2500, -3800), (2500, 3800))
            state.ball_vel[row, :2] = rng.uniform(-900, 900, 2)
            state.car_pos[row, 0, :2] = rng.uniform((-2800, -4000), (2800, 1800))
            state.car_pos[row, 1, :2] = rng.uniform((-2800, -1800), (2800, 4000))
        for car in (0, 1):
            if np.linalg.norm(state.car_pos[row, car, :2] - ball[:2]) < 200:
                state.car_pos[row, car, 0] += 450 if ball[0] < 0 else -450
            offset = (-0.45, 0.45) if family == 2 and car == 0 else (-1.3, 1.3)
            _set_coherent_ground_route(state, row, car, rng, ball[:2], (0, 450), offset)
        if np.linalg.norm(state.car_pos[row, 0] - state.car_pos[row, 1]) < 180:
            state.car_pos[row, 1, 0] = -3000 if ball[0] > 0 else 3000
            _set_coherent_ground_route(state, row, 1, rng, ball[:2], (0, 450), (-1.3, 1.3))
        if focal[row] == 1:
            _rotate_and_swap(state, row)
    state.validate()
    # Metadata is for rewards/analysis only, never appended to the 182 observations.
    return type(batch)(
        state, families, focal, kickoff.astype(np.int32), layouts, np.full(worlds, -1, np.int32)
    )


def player_roles(family, focal):
    roles = family[:, None].expand(-1, 2).clone()
    opposite = torch.arange(2, device=family.device)[None] != focal[:, None]
    roles = torch.where(opposite & (family[:, None] == 2), 3, roles)
    roles = torch.where(opposite & (family[:, None] == 3), 2, roles)
    return roles


def projected_goal(position, velocity, sign):
    vy = velocity[..., 1] * sign
    t = (5120 - position[..., 1] * sign) / vy.clamp_min(1)
    x = position[..., 0] + velocity[..., 0] * t
    z = (position[..., 2] + velocity[..., 2] * t - 325 * t.square()).clamp_min(93.15)
    return (
        (vy > 500) & (t > 0) & (t <= 4) & (x.abs() <= 750) & (z <= 550) & (position[..., 2] <= 550)
    )


class SkillEvents:
    def __init__(self, worlds, device):
        shape = (worlds, 2)
        self.paid = torch.zeros((*shape, len(EVENTS)), dtype=torch.bool, device=device)
        self.own_last = torch.zeros(shape, dtype=torch.bool, device=device)
        self.control_ticks = torch.zeros(shape, dtype=torch.int64, device=device)
        self.clear_ticks = torch.zeros_like(self.control_ticks)
        self.clear_pending = torch.zeros(shape, dtype=torch.bool, device=device)
        self.gain_y = torch.zeros(shape, device=device)

    def reset(self, mask):
        for tensor in (
            self.paid,
            self.own_last,
            self.control_ticks,
            self.clear_ticks,
            self.clear_pending,
            self.gain_y,
        ):
            tensor[mask] = 0

    def calculate(self, before, after, touches, roles, terminated, truncated, scoring_team):
        a, b = _physical_vectors(before), _physical_vectors(after)
        contact, other_contact = touches > 0, touches.flip(1) > 0
        # Simultaneous interval contacts are contested, not invented last-touch ordering.
        self.own_last = torch.where(other_contact, False, self.own_last) | (
            contact & ~other_contact
        )
        distance = (b["ball_position"] - b["self_position"]).norm(dim=-1)
        opponent_distance = (b["ball_position"] - b["opponent_position"]).norm(dim=-1)
        relative_speed = (b["ball_velocity"] - b["self_velocity"]).norm(dim=-1)
        controlled = (
            self.own_last
            & (distance <= 250)
            & (relative_speed <= 500)
            & (opponent_distance >= distance + 150)
        )
        self.control_ticks = torch.where(controlled, self.control_ticks + 1, 0)
        gain = self.control_ticks >= 8
        threat_before = projected_goal(a["ball_position"], a["ball_velocity"], -1)
        threat_after = projected_goal(b["ball_position"], b["ball_velocity"], -1)
        self.clear_pending |= contact & threat_before & ~threat_after
        self.clear_pending &= ~other_contact & ~threat_after
        self.clear_ticks = torch.where(self.clear_pending, self.clear_ticks + 1, 0)
        shot = contact & projected_goal(b["ball_position"], b["ball_velocity"], 1)
        skill, acquire = roles != 0, (roles == 1) | (roles == 4)
        first = contact & skill
        # No first-touch payment in defense; intercept is rewarded as a cleared shot.
        first &= roles != 3
        control_gain = gain & (acquire | (roles == 3))
        new_gain = control_gain & ~self.paid[..., 1]
        self.gain_y = torch.where(new_gain, b["ball_position"][..., 1], self.gain_y)
        forward = (
            acquire
            & self.paid[..., 1]
            & controlled
            & (b["ball_position"][..., 1] - self.gain_y >= 400)
        )
        clear = (roles == 3) & (self.clear_ticks >= 8)
        lost = acquire & self.paid[..., 1] & gain.flip(1)
        winner = torch.arange(2, device=roles.device)[None] == scoring_team[:, None]
        success = torch.where(
            acquire,
            self.paid[..., 1] | control_gain,
            torch.where(
                roles == 2,
                self.paid[..., 3] | shot,
                self.paid[..., 4] | clear | self.paid[..., 1] | control_gain,
            ),
        )
        failure = skill & truncated[:, None] & ~success
        conditions = torch.stack(
            (first, control_gain, forward, (roles == 2) & shot, clear, lost, failure), -1
        )
        events = conditions & ~self.paid & skill[..., None]
        # True native goals are paid once by the terminal contract, not post-goal geometry.
        events &= ~terminated[:, None, None]
        self.paid |= events
        target = a["ball_position"] - a["self_position"]
        direction = target / target.norm(dim=-1, keepdim=True).clamp_min(1)
        approach = ((a["self_velocity"] * direction).sum(-1) / 2300).clamp(0, 1) * (0.5 / 30)
        approach *= ((roles == 1) | (roles == 2)) & ~self.paid[..., 0] & ~terminated[:, None]
        weighted = events.to(before.dtype) * torch.tensor(EVENT_WEIGHTS, device=before.device)
        return (
            weighted.sum(-1) + approach,
            events,
            weighted,
            approach,
            success | (terminated[:, None] & winner),
        )


class DirectSkillsEnv(FreshGroundEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_version = VERSION
        self.contract_hashes = dict(self.contract_hashes, reward=content_hash(reward_authority()))
        self.events = SkillEvents(self.num_envs, self.device)
        reset = self.world.ssl_foundation_reset
        self.family = wp.to_torch(reset.current_family)
        self.focal = wp.to_torch(reset.current_focal_side)
        self.learner = torch.ones((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.opponent_family = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.reset_statistics()

    def reset_statistics(self):
        # rows: five reward roles x two opponent families; raw integer/event and reward sums.
        self.stat_names = (
            "samples",
            "touches",
            "goals",
            "concedes",
            "episode_endings",
            "success_endings",
            *EVENTS,
            "approach_reward",
            "direct_reward",
            "natural_reward",
        )
        self.statistics = torch.zeros(
            (10, len(self.stat_names)), device=self.device, dtype=torch.float64
        )

    def _step_impl(self, action, markers=None, tick_action_provider=None):
        if markers is not None:
            raise ValueError("markers unsupported")
        self._activate_torch_stream()
        before = self.observation
        roles = player_roles(self.family, self.focal)
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
            count = views["rival2.touch_count"].reshape(self.num_envs, 2).long()
            delta = (count - prior_touches).clamp_min(0).masked_fill((first_goal >= 0)[:, None], 0)
            first_touch_tick = torch.where(
                (delta > 0) & (first_touch_tick < 0), tick, first_touch_tick
            )
            touches += delta
            prior_touches.copy_(count)
            first_goal = torch.where((self.goal_latched != 0) & (first_goal < 0), tick, first_goal)
        after = self.bridge.observation().clone()
        team = views["rival2.scoring_team_latched"].long().clone()
        terminated = views["rival2.terminated"].bool().clone()
        if not torch.equal(terminated, first_goal >= 0):
            raise RuntimeError("direct skill goal/cadence/reset failure")
        age = views["rival2.episode_ticks"].clone()
        limit = torch.where(self.family == 0, 3600, 1440)
        truncated = (views["rival2.truncated"].bool() | (age >= limit)) & ~terminated
        reset = terminated | truncated
        views["rival2.truncated"].copy_(truncated)
        views["rival2.reset_mask"].copy_(reset)
        natural, parts = decision_reward(before, after, first_goal, team)
        bonus, events, weighted, approach, success = self.events.calculate(
            before, after, touches, roles, terminated, truncated, team
        )
        natural_mask = roles == 0
        reward = torch.where(natural_mask, natural, parts["terminal_goal"] + bonus)
        # Compatibility with existing collector's component keys; direct totals are
        # separately exposed, never mislabeled as a potential component.
        self.last_components = {k: v * natural_mask for k, v in parts.items() if k in WEIGHTS}
        self.last_components.update(terminal_goal=parts["terminal_goal"], total=reward)
        self.last_native = dict(
            touch_count=touches,
            first_touch_tick=first_touch_tick,
            first_goal_tick=first_goal,
            scoring_team=team,
            episode_ticks=age,
            no_touch_ticks=views["rival2.no_touch_ticks"].clone(),
        )
        self.last_skill = dict(
            roles=roles,
            events=events,
            weighted=weighted,
            approach=approach,
            bonus=bonus,
            success=success,
        )
        winner = torch.arange(2, device=self.device)[None] == team[:, None]
        columns = [
            torch.ones_like(reward),
            touches,
            terminated[:, None] & winner,
            terminated[:, None] & ~winner,
            reset[:, None].expand_as(roles),
            reset[:, None] & success,
            *events.unbind(-1),
            approach,
            bonus,
            natural * natural_mask,
        ]
        stat = torch.stack([v.to(torch.float64) for v in columns], -1)
        group = roles * 2 + self.opponent_family[:, None]
        self.statistics.index_add_(0, group[self.learner], stat[self.learner])
        self.events.reset(reset)
        self.world.apply_interval_resets()
        self.observation = self.bridge.observation()
        self.decision_count += 1
        return Rival2Step(self.observation, after, emitted, reward, terminated, truncated, reset)
