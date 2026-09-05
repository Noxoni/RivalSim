"""Recurrent PPO and deterministic evaluation for the isolated fresh 30Hz lane."""
from __future__ import annotations

import copy
from dataclasses import asdict

import torch
import warp as wp

from rivalsim.fresh_ground_30hz import (
    SEED, VERSION, CHECKPOINT_FORMAT, FreshGroundEnv, LearnedExploration,
    fresh_model, ppo_config, scenarios, scenario_hash, content_hash, authority,
)
from rivalsim.rival2_contracts import OBS_FIELD_NAMES, CAR_LINEAR_SPEED_SCALE
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_unified_policy import deterministic_unified_action
from rivalsim.rival2_recurrent_training import Rival2RecurrentTrainer
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentRolloutBuffer
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

SPEED = OBS_FIELD_NAMES.index("self.linear_velocity.x")
BALL_VY = OBS_FIELD_NAMES.index("ball.linear_velocity.y")


class FreshGroundTrainer(Rival2RecurrentTrainer):
    def __init__(self, env, *, seed=SEED, model=None):
        model = fresh_model(seed) if model is None else model
        super().__init__(env, policy_config=model.config, ppo_config=ppo_config(),
                         phase="fresh_ground_30hz_v1", seed=seed, model=model,
                         source_identity={"kind": "fresh_random", "seed": seed, "parent": None},
                         checkpoint_format=CHECKPOINT_FORMAT, lineage=VERSION)
        actor = [p for name, p in model.named_parameters() if not name.startswith("critic.")]
        self.optimizer = torch.optim.Adam([
            {"params": actor, "lr": 1e-4, "name": "actor_and_recurrent_trunk"},
            {"params": model.critic.parameters(), "lr": 3e-4, "name": "independent_critic"}])
        self.exploration = LearnedExploration()
        self.sequence_microbatch_size = 728
        self.optimize_execution = True
        self.nexto_probability = 0.
        self.competence_streak = 0
        self.opponent_generator = torch.Generator(device=self.device).manual_seed(seed + 2)
        self.is_nexto = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
        self.side = torch.zeros(env.num_envs, device=self.device, dtype=torch.int64)
        self.rows = torch.arange(env.num_envs, device=self.device)
        # No Nexto model or old Rival is instantiated in the initial training phase.
        self.nexto = None
        self.nexto_state = None
        self.assign(torch.ones_like(self.is_nexto))

    def assign(self, mask):
        draw = torch.rand(self.env.num_envs, device=self.device, generator=self.opponent_generator)
        side = torch.randint(2, (self.env.num_envs,), device=self.device, generator=self.opponent_generator)
        self.is_nexto.copy_(torch.where(mask, draw < self.nexto_probability, self.is_nexto))
        self.side.copy_(torch.where(mask, side, self.side))
        if self.nexto_probability and self.nexto is None:
            self.nexto = NextoPolicyAdapter(self.env.num_envs, device=self.device)
            self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        if self.nexto is not None:
            self.nexto.set_player_index(1 - self.side)
            self.nexto.activate(mask & self.is_nexto)

    def accept_evaluation(self, result):
        acquisition, finishing = result["acquisition_selfplay"], result["finishing_selfplay"]
        good = (acquisition["focal_touch_fraction"] >= .60
                and acquisition["median_first_touch_seconds_if_touched"] is not None
                and acquisition["median_first_touch_seconds_if_touched"] <= 5
                and finishing["goals_for"] >= 1)
        self.competence_streak = self.competence_streak + 1 if good else 0
        if self.competence_streak >= 2:
            self.nexto_probability = .20  # activated only at subsequent episode resets

    @torch.no_grad()
    def collect_rollout(self):
        c, n = self.ppo_config, self.env.num_envs
        buffer = Rival2RecurrentRolloutBuffer(c.rollout_horizon, n, self.hidden, self.device,
                                             store_opponent_family=True)
        observation = self.env.observation
        names = ("samples", "touches", "goalward_touches", "goals", "concedes", "no_touch", "time_limit",
                 "speed", "saturation", "first_touches", "first_touch_age", "ended_player_episodes",
                 "episodes_with_touch", "nexto_samples", "resets", "jump", "boost", "handbrake")
        totals = {k: torch.zeros((), device=self.device, dtype=torch.float64) for k in names}
        components = {k: torch.zeros_like(totals["samples"]) for k in (*authority()["weights"], "terminal_goal", "total")}
        self.model.eval()
        for _ in range(c.rollout_horizon):
            reset_before = self.reset_before
            actor, value, hidden = self.model(observation.reshape(-1, 182), self._flat_hidden(),
                                               reset_before=reset_before.reshape(-1))
            sample = sample_hybrid_action(actor, generator=self.policy_generator, config=self.policy_config)
            action = sample.action.reshape(n, 2, 8)
            mask = (~self.is_nexto)[:, None].expand(-1, 2).clone()
            mask[self.rows, self.side] = True
            active = self.is_nexto

            def tick_provider(tick):
                output = action.clone()
                ball = self.nexto_state.ball_pos
                kickoff = (ball[:, 0] == 0) & (ball[:, 1] == 0)
                controls, _ = self.nexto.tick_action(self.nexto_state, kickoff, active_mask=active)
                rows = self.rows[active]
                output[rows, 1-self.side[active]] = controls[active]
                return output

            transition = (self.env.step_with_tick_actions(action, tick_provider)
                          if bool(active.any()) else self.env.step(action))
            native = self.env.last_native
            next_value = self.model.isolated_value(transition.transition_observation.reshape(-1, 182))
            buffer.add(observation=observation, action=action, pre_tanh=sample.pre_tanh.reshape(n, 2, 5),
                       old_log_probability=sample.log_probability.reshape(n, 2), value=value.reshape(n, 2),
                       reward=transition.reward, terminated=transition.terminated[:, None].expand(-1, 2),
                       truncated=transition.truncated[:, None].expand(-1, 2), next_value=next_value.reshape(n, 2),
                       train_mask=mask, reset_before=reset_before,
                       opponent_family=active.to(torch.int64)[:, None].expand(-1, 2))
            touch = native["touch_count"] * mask
            first = (touch > 0) & ~self.episode_has_touch
            self.episode_has_touch |= touch > 0
            ended = transition.reset_mask[:, None] & mask
            totals["samples"] += mask.sum()
            totals["nexto_samples"] += (mask & active[:, None]).sum()
            totals["touches"] += touch.sum()
            totals["goalward_touches"] += (touch * (transition.transition_observation[..., BALL_VY] > 0)).sum()
            totals["first_touches"] += first.sum()
            age = native["episode_ticks"][:, None] - 4 + native["first_touch_tick"] + 1
            totals["first_touch_age"] += age.masked_fill(~first, 0).sum() / 120
            totals["ended_player_episodes"] += ended.sum()
            totals["episodes_with_touch"] += (ended & self.episode_has_touch).sum()
            goals = transition.terminated[:, None] & mask
            winner = native["scoring_team"][:, None] == torch.arange(2, device=self.device)
            totals["goals"] += (goals & winner).sum()
            totals["concedes"] += (goals & ~winner).sum()
            no_touch = transition.truncated & (native["no_touch_ticks"] >= 1800)
            totals["no_touch"] += no_touch.sum()
            totals["time_limit"] += (transition.truncated & ~no_touch).sum()
            totals["resets"] += transition.reset_mask.sum()
            totals["speed"] += observation[..., SPEED:SPEED+3].norm(dim=-1).masked_fill(~mask, 0).sum()
            totals["saturation"] += ((action[..., :5].abs() > .95) & mask[..., None]).sum()
            for j, name in enumerate(("jump", "boost", "handbrake"), start=5):
                totals[name] += action[..., j].masked_fill(~mask, 0).sum()
            for key, component in self.env.last_components.items():
                components[key] += component.masked_fill(~mask, 0).sum(dtype=torch.float64)
            reset = transition.reset_mask[:, None].expand(-1, 2)
            self.hidden = hidden.reshape_as(self.hidden).masked_fill(reset[None, ..., None], 0)
            self.reset_before = reset.clone()
            self.episode_has_touch.masked_fill_(reset, False)
            self.assign(transition.reset_mask)
            observation = transition.observation
        raw = {k: float(v.item()) for k, v in totals.items()}
        samples = int(raw["samples"])
        self.total_agent_samples += samples
        self.physical_physics_ticks_experienced += c.rollout_horizon * n * 4
        minutes = samples / (30 * 60)
        self.last_rollout_metrics = dict(raw, trainable_agent_samples=samples,
            physical_player_minutes=minutes, touches_per_minute=raw["touches"]/max(minutes, 1e-9),
            goalward_touch_fraction=raw["goalward_touches"]/max(raw["touches"], 1),
            mean_first_touch_seconds_if_touched=raw["first_touch_age"]/max(raw["first_touches"], 1),
            ended_player_episode_touch_fraction=raw["episodes_with_touch"]/max(raw["ended_player_episodes"], 1),
            movement_speed=CAR_LINEAR_SPEED_SCALE*raw["speed"]/samples,
            analog_saturation_fraction=raw["saturation"]/(samples*5),
            potential_reward_components={k: float(v.item()) for k, v in components.items()},
            opponent_nexto_probability=self.nexto_probability, physics_hz=120, policy_hz=30,
            goalward_touch_definition="native contacts; ball canonical velocity at decision end (not shot success)")
        return buffer

    def checkpoint_payload(self, **kwargs):
        p = super().checkpoint_payload(**kwargs)
        p.update(authority_sha256=content_hash(authority()),
                 opponents={"nexto_probability": self.nexto_probability, "historical": False, "wisp": False,
                            "current_vs_current_both_sides_trainable": True, "nexto_inference_only": True},
                 competence_streak=self.competence_streak,
                 opponent_rng=self.opponent_generator.get_state(),
                 resume_count=self.resume_count,
                 last_rollout_metrics=copy.deepcopy(self.last_rollout_metrics))
        p["recurrent_state"]["checkpoint_resume_starts_fresh_simulator_kickoff"] = False
        p["recurrent_state"]["checkpoint_resume_starts_fresh_scenario_episodes"] = True
        return p

    def load_checkpoint(self, path):
        # Validate the authority and fresh lineage BEFORE touching any model tensor.
        p = torch.load(path, map_location="cpu", weights_only=False)
        if (p.get("format") != CHECKPOINT_FORMAT or p.get("lineage") != VERSION
                or p.get("authority_sha256") != content_hash(authority())
                or p.get("source", {}).get("parent", "missing") is not None):
            raise ValueError("resume must be this fresh random 30Hz lineage only")
        p = super().load_checkpoint(path)
        self.nexto_probability = p["opponents"]["nexto_probability"]
        self.competence_streak = p["competence_streak"]
        self.opponent_generator.set_state(p["opponent_rng"].cpu())
        self.resume_count = p.get("resume_count", 0) + 1
        self.assign(torch.ones_like(self.is_nexto))
        return p


def make_trainer(collision_root, worlds=32768):
    bank = scenarios(worlds)
    env = FreshGroundEnv(worlds, collision_root, device="cuda:0", seed=SEED,
                         ssl_foundation_scenarios=bank)
    return FreshGroundTrainer(env), bank


@torch.no_grad()
def evaluate(model, collision_root, *, worlds=64, decisions=900):
    """Frozen seeds, one initial episode per world. Ended worlds do not re-enter metrics.

    Acquisition/finishing are bilateral current-policy evaluations (not scripted
    free-ball demos). The kickoff case is actual deterministic Nexto opposition.
    """
    training = model.training
    model.eval()
    result = {}
    for index, (name, family) in enumerate((("acquisition_selfplay", 2), ("finishing_selfplay", 4),
                                           ("standard_kickoff_nexto", 1))):
        bank = scenarios(worlds, SEED+100+index, family_only=family)
        env = FreshGroundEnv(worlds, collision_root, device="cuda:0", seed=SEED+100+index,
                             ssl_foundation_scenarios=bank)
        side = torch.as_tensor(bank.focal_side.astype("int64"), device=env.device)
        rows = torch.arange(worlds, device=env.device)
        hidden = model.initial_hidden(worlds*2, device=env.device)
        reset = torch.ones(worlds*2, dtype=torch.bool, device=env.device)
        alive = torch.ones(worlds, dtype=torch.bool, device=env.device)
        first = torch.full((worlds,), float("nan"), device=env.device)
        touches = torch.zeros(worlds, device=env.device)
        goals = torch.zeros_like(touches)
        concedes = torch.zeros_like(touches)
        no_touch = torch.zeros_like(touches)
        exposure = torch.zeros_like(touches)
        goalward = torch.zeros_like(touches)
        nexto = NextoPolicyAdapter(worlds, device=env.device) if family == 1 else None
        if nexto:
            nexto.set_player_index(1-side)
            nexto.activate(alive)
            ns = NextoStateTensors.from_bridge(env.bridge)
        for tick in range(decisions):
            actor, hidden = model.forward_actor(env.observation.reshape(-1, 182), hidden, reset_before=reset)
            action = deterministic_unified_action(actor).reshape(worlds, 2, 8)
            def provider(_):
                output = action.clone()
                kickoff = (ns.ball_pos[:, 0] == 0) & (ns.ball_pos[:, 1] == 0)
                controls, _ = nexto.tick_action(ns, kickoff, active_mask=alive)
                output[rows, 1-side] = controls
                return output
            tr = env.step_with_tick_actions(action, provider) if nexto else env.step(action)
            native = env.last_native
            contact = native["touch_count"][rows, side] * alive
            first = torch.where((contact > 0) & first.isnan(),
                (tick*4 + native["first_touch_tick"][rows, side] + 1)/120., first)
            touches += contact
            goalward += contact * (tr.transition_observation[rows, side, BALL_VY] > 0)
            goals += alive & tr.terminated & (native["scoring_team"] == side)
            concedes += alive & tr.terminated & (native["scoring_team"] != side)
            no_touch += alive & tr.truncated & (native["no_touch_ticks"] >= 1800)
            exposure += alive / 30
            alive &= ~tr.reset_mask
            reset = tr.reset_mask[:, None].expand(-1, 2).reshape(-1)
            hidden = hidden.masked_fill(reset[None, :, None], 0)
            if not bool(alive.any()):
                break
        touched = torch.isfinite(first)
        result[name] = {"worlds": worlds, "scenario_sha256": scenario_hash(bank),
            "focal_touch_fraction": float(touched.float().mean()),
            "median_first_touch_seconds_if_touched": float(first[touched].median()) if bool(touched.any()) else None,
            "focal_touch_count": int(touches.sum()), "touches_per_minute": float(touches.sum()/exposure.sum()*60),
            "goalward_touch_fraction": float(goalward.sum()/touches.sum().clamp_min(1)),
            "goals_for": int(goals.sum()), "goals_against": int(concedes.sum()),
            "no_touch_truncations": int(no_touch.sum()), "surviving_at_horizon": int(alive.sum()),
            "seconds_of_exposure": float(exposure.sum()), "deterministic": True}
        del env, nexto
    model.train(training)
    return result
