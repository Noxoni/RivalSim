"""Train and select side-specific launch-first aerial option policies."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as v1  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_aerial_option import (  # noqa: E402
    PHASE_EASY_LAUNCH,
    PHASE_GOAL_DIRECTED,
    PHASE_MOVING_INTERCEPT,
    PHASE_NAMES,
    AerialRewardTracker,
    build_aerial_scenarios,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_entropy,
    hybrid_log_probability,
    sample_hybrid_action,
)

AUTHORITY = ROOT / "results/rival2/aerial_option_v1/authority.json"
RESULTS = ROOT / "results/rival2/aerial_option_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/aerial_option_v1"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
AUTHORITY_SHA256 = "656CB57F522E1534C0C4B1B5DB5134DB55928483A080F48F0315CD54D13CBEDF"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-option-v1")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")
SEED = 2_026_090_241
TARGET_LABELS = ("aerialdribble", "groundtoairdribble")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_authority() -> dict[str, Any]:
    if v1.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial-option V1 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    identities = [
        authority["protected_base"]["blue"],
        authority["protected_base"]["orange"],
        authority["negative_evidence"],
        authority["human_launch_rehearsal"]["dataset_manifest"],
        authority["human_launch_rehearsal"]["review_candidates"],
        authority["human_launch_rehearsal"]["observation_adapter"],
    ]
    for identity in identities:
        path = ROOT / identity["path"]
        if v1.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"aerial-option bound input changed: {path}")
    if authority["option_training_boundary"]["critic_used_for_advantage_or_loss"]:
        raise RuntimeError("aerial option must not train through the critic")
    return authority


def make_model(payload: dict[str, Any], device: str) -> Rival2ActorCritic:
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.trunk.requires_grad_(True)
    model.actor.requires_grad_(True)
    model.critic.requires_grad_(False)
    return model


def make_optimizer(
    model: Rival2ActorCritic, authority: dict[str, Any]
) -> torch.optim.AdamW:
    boundary = authority["option_training_boundary"]
    return torch.optim.AdamW(
        [
            {
                "params": model.trunk.parameters(),
                "lr": float(boundary["trunk_learning_rate"]),
            },
            {
                "params": model.actor.parameters(),
                "lr": float(boundary["actor_learning_rate"]),
            },
        ],
        weight_decay=float(boundary["weight_decay"]),
    )


def aerial_validation(model: Rival2ActorCritic, validation: Any, *, device: str) -> dict[str, Any]:
    labels = np.asarray(validation.mechanic_label)
    per_label: dict[str, Any] = {}
    for label in TARGET_LABELS:
        mask = torch.from_numpy(labels == label)
        per_label[label] = v1.evaluate_rows(
            model,
            validation.mechanic_observation[mask],
            validation.mechanic_action[mask],
            device=device,
        )
    score = float(
        np.mean([per_label[label]["complete_action_rmse"] for label in TARGET_LABELS])
    )
    return {
        "labels": per_label,
        "mean_complete_action_rmse": score,
        "finite": all(per_label[label]["finite"] for label in TARGET_LABELS),
    }


def train_human_rehearsal(
    models: list[Rival2ActorCritic],
    teachers: list[Rival2ActorCritic],
    train: Any,
    validation: Any,
    authority: dict[str, Any],
    *,
    device: str,
) -> tuple[list[dict[str, Any]], list[torch.optim.AdamW]]:
    human = authority["human_launch_rehearsal"]
    train_labels = np.asarray(train.mechanic_label)
    mask = torch.from_numpy(np.isin(train_labels, TARGET_LABELS))
    observations = train.mechanic_observation[mask]
    actions = train.mechanic_action[mask]
    labels = train_labels[mask.numpy()].tolist()
    attempts = np.asarray(train.mechanic_attempt)[mask.numpy()].tolist()
    generators = [
        torch.Generator(device="cpu").manual_seed(SEED ^ (0xBC00 + side))
        for side in (0, 1)
    ]
    samplers = [
        MechanicHierarchySampler(
            labels,
            attempts,
            uniform_label_fraction=0.5,
            maximum_oversampling_ratio=8.0,
            generator=generators[side],
        )
        for side in (0, 1)
    ]
    optimizers = [make_optimizer(model, authority) for model in models]
    critic_hashes = [
        human_base.tensor_tree_sha256(
            {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
        )
        for model in models
    ]
    baseline = [aerial_validation(model, validation, device=device) for model in models]
    best = [
        {
            "step": 0,
            "validation": baseline[side],
            "model": copy.deepcopy(models[side].state_dict()),
        }
        for side in (0, 1)
    ]
    curve = RESULTS / "human_rehearsal_curve.jsonl"
    if curve.exists():
        curve.unlink()
    maximum = int(human["maximum_steps"])
    interval = int(human["validation_interval_steps"])
    for step in range(1, maximum + 1):
        side_rows = []
        for side in (0, 1):
            index = samplers[side].sample(int(human["batch_size"]))
            observation = observations.index_select(0, index).to(device)
            action = actions.index_select(0, index).to(device)
            with torch.no_grad():
                teacher_actor, _ = teachers[side](observation)
            student_actor, _ = models[side](observation)
            objective = human_behavior_cloning_objective(
                student_actor,
                teacher_actor,
                action,
                smooth_l1_beta=0.1,
                analog_weight=1.0,
                button_weight=0.25,
                log_std_weight=float(human["log_std_parent_retention_weight"]),
                policy_config=models[side].config,
            )
            optimizers[side].zero_grad(set_to_none=True)
            objective.loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                [*models[side].trunk.parameters(), *models[side].actor.parameters()],
                float(authority["option_training_boundary"]["maximum_gradient_norm"]),
            )
            if not bool(torch.isfinite(objective.loss) and torch.isfinite(gradient)):
                raise RuntimeError("nonfinite aerial human rehearsal")
            optimizers[side].step()
            side_rows.append(
                {
                    "side": side,
                    "loss": float(objective.loss.detach()),
                    "gradient_norm": float(gradient.detach()),
                }
            )
        if step % interval != 0:
            continue
        for side in (0, 1):
            validation_row = aerial_validation(models[side], validation, device=device)
            side_rows[side]["validation"] = validation_row
            if (
                validation_row["finite"]
                and validation_row["mean_complete_action_rmse"]
                < best[side]["validation"]["mean_complete_action_rmse"] - 1.0e-6
            ):
                best[side] = {
                    "step": step,
                    "validation": validation_row,
                    "model": copy.deepcopy(models[side].state_dict()),
                }
        append_jsonl(curve, {"step": step, "sides": side_rows})
        print(
            json.dumps(
                {
                    "stage": "human_launch_rehearsal",
                    "step": step,
                    "scores": [
                        row["validation"]["mean_complete_action_rmse"]
                        for row in side_rows
                    ],
                    "best_steps": [entry["step"] for entry in best],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected: list[dict[str, Any]] = []
    for side in (0, 1):
        models[side].load_state_dict(best[side]["model"], strict=True)
        observed_critic = human_base.tensor_tree_sha256(
            {name: value.detach().cpu() for name, value in models[side].critic.state_dict().items()}
        )
        if observed_critic != critic_hashes[side]:
            raise RuntimeError("aerial human rehearsal changed the critic")
        selected.append(
            {
                "side": side,
                "baseline": baseline[side],
                "step": best[side]["step"],
                "validation": best[side]["validation"],
            }
        )
    # Physical option optimization starts with clean Adam state at the selected
    # supervised boundary; no stale moments from an unselected later BC step.
    return selected, [make_optimizer(model, authority) for model in models]


class OptionRollout:
    def __init__(self, horizon: int, worlds: int, device: str):
        self.horizon = horizon
        self.worlds = worlds
        self.observation = torch.empty((horizon, worlds, 182), device=device)
        self.action = torch.empty((horizon, worlds, 8), device=device)
        self.pre_tanh = torch.empty((horizon, worlds, 5), device=device)
        self.old_log_probability = torch.empty((horizon, worlds), device=device)
        self.reward = torch.zeros((horizon, worlds), device=device)
        self.done = torch.zeros((horizon, worlds), dtype=torch.bool, device=device)
        self.mask = torch.zeros((horizon, worlds), dtype=torch.bool, device=device)

    def discounted_returns(self, gamma: float) -> torch.Tensor:
        result = torch.zeros_like(self.reward)
        carry = torch.zeros(self.worlds, device=self.reward.device)
        for tick in range(self.horizon - 1, -1, -1):
            carry = self.reward[tick] + gamma * carry * (~self.done[tick]).to(torch.float32)
            result[tick] = carry
        return result


def phase_override(authority: dict[str, Any], phase: int) -> HybridDistributionOverride:
    exploration = authority["physical_curriculum"]["exploration"][PHASE_NAMES[phase]]
    return HybridDistributionOverride(
        analog_log_std=math.log(float(exploration["analog_sigma"])),
        button_temperature=float(exploration["button_temperature"]),
    )


def collect_aerial_rollout(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    phase: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
    collision_dir: Path,
) -> tuple[OptionRollout | None, dict[str, Any]]:
    batch = build_aerial_scenarios(
        worlds, seed=seed, attacker_side=side, phase=phase
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    tracker = AerialRewardTracker(worlds, attacker_side=side, phase=phase)
    rollout = None if deterministic else OptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    opponent = 1 - side
    rows = torch.arange(worlds, device=device)
    observation = env.observation
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            if deterministic:
                selected = deterministic_hybrid_action(actor, model.config)
                pre_tanh = actor[:, :5]
                log_probability = hybrid_log_probability(
                    actor,
                    selected,
                    config=model.config,
                    pre_tanh=pre_tanh,
                    distribution_override=distribution,
                )
            else:
                sample = sample_hybrid_action(
                    actor,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution,
                )
                selected = sample.action
                pre_tanh = sample.pre_tanh
                log_probability = sample.log_probability
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], selected, 0.0)
        action[:, opponent] = 0.0
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == side)
        reward, skill_done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            active=active_before,
        )
        terminal = skill_done | transition.terminated | transition.truncated
        if rollout is not None:
            rollout.observation[tick].copy_(observation[:, side])
            rollout.action[tick].copy_(transition.emitted_action[:, side])
            rollout.pre_tanh[tick].copy_(pre_tanh)
            rollout.old_log_probability[tick].copy_(log_probability)
            rollout.reward[tick].copy_(reward)
            rollout.done[tick].copy_(terminal)
            rollout.mask[tick].copy_(active_before)
        saturation += (
            transition.emitted_action[:, side, :5].abs() > 0.95
        ).logical_and(active_before[:, None]).sum(dim=0, dtype=torch.float64)
        action_count += active_before.sum(dtype=torch.float64)
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    # Unfinished worlds terminate at the bounded horizon for return accounting.
    if rollout is not None and bool(active.any()):
        final_tick = min(horizon - 1, tick)
        rollout.done[final_tick] |= active
    torch.cuda.synchronize()
    telemetry = asdict(tracker.telemetry)
    metrics = {
        "side": side,
        "phase": PHASE_NAMES[phase],
        "horizon": horizon,
        "active_worlds_at_end": int(active.sum()),
        "telemetry": telemetry,
        "fractions": {
            "launch": telemetry["launches"] / worlds,
            "reached_100uu": telemetry["reached_100uu"] / worlds,
            "elevated_contact": telemetry["elevated_contacts"] / worlds,
            "high_contact": telemetry["high_contacts"] / worlds,
            "forward_high_contact": telemetry["forward_high_contacts"] / worlds,
            "aerial_origin_goal": telemetry["aerial_origin_goals"] / worlds,
        },
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite_observation": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def option_ppo_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    rollout: OptionRollout,
    *,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    authority: dict[str, Any],
) -> dict[str, Any]:
    physical = authority["physical_curriculum"]
    index = torch.nonzero(rollout.mask.reshape(-1), as_tuple=False).squeeze(-1)
    observation = rollout.observation.reshape(-1, 182).index_select(0, index)
    action = rollout.action.reshape(-1, 8).index_select(0, index)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, index)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, index)
    returns = rollout.discounted_returns(float(physical["discount_gamma"])).reshape(-1).index_select(0, index)
    advantage = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-8)
    sums = {
        "policy_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    proposals = 0
    for _epoch in range(int(physical["epochs"])):
        permutation = torch.randperm(index.numel(), device=index.device, generator=generator)
        for start in range(0, index.numel(), int(physical["minibatch_size"])):
            local = permutation[start : start + int(physical["minibatch_size"])]
            actor, _ = model(observation.index_select(0, local))
            new_log_probability = hybrid_log_probability(
                actor,
                action.index_select(0, local),
                config=model.config,
                pre_tanh=pre_tanh.index_select(0, local),
                distribution_override=distribution,
            )
            old = old_log_probability.index_select(0, local)
            log_ratio = new_log_probability - old
            ratio = torch.exp(log_ratio)
            local_advantage = advantage.index_select(0, local)
            clip = float(physical["ppo_clip"])
            policy_loss = -torch.minimum(
                ratio * local_advantage,
                ratio.clamp(1.0 - clip, 1.0 + clip) * local_advantage,
            ).mean()
            entropy = hybrid_entropy(
                actor, model.config, distribution_override=distribution
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                [*model.trunk.parameters(), *model.actor.parameters()],
                float(authority["option_training_boundary"]["maximum_gradient_norm"]),
            )
            if not bool(torch.isfinite(policy_loss) and torch.isfinite(gradient)):
                raise RuntimeError("nonfinite aerial option PPO")
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (ratio.sub(1.0).abs() > clip).to(torch.float32).mean()
            for name, value in zip(
                sums,
                (policy_loss, entropy, approx_kl, clip_fraction, gradient),
                strict=True,
            ):
                sums[name] += float(value.detach())
            proposals += 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite aerial option model")
    return {
        **{name: value / max(1, proposals) for name, value in sums.items()},
        "steps": proposals,
        "samples": int(index.numel()),
        "return_mean": float(returns.mean()),
        "return_max": float(returns.max()),
    }


def human_auxiliary_update(
    model: Rival2ActorCritic,
    teacher: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    observations: torch.Tensor,
    actions: torch.Tensor,
    sampler: MechanicHierarchySampler,
    *,
    authority: dict[str, Any],
    device: str,
) -> dict[str, float]:
    count = int(authority["physical_curriculum"]["human_auxiliary_samples_per_block"])
    index = sampler.sample(count)
    observation = observations.index_select(0, index).to(device)
    action = actions.index_select(0, index).to(device)
    with torch.no_grad():
        teacher_actor, _ = teacher(observation)
    student_actor, _ = model(observation)
    objective = human_behavior_cloning_objective(
        student_actor,
        teacher_actor,
        action,
        smooth_l1_beta=0.1,
        analog_weight=1.0,
        button_weight=0.25,
        log_std_weight=float(
            authority["human_launch_rehearsal"]["log_std_parent_retention_weight"]
        ),
        policy_config=model.config,
    )
    loss = float(authority["physical_curriculum"]["human_auxiliary_weight"]) * objective.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(
        [*model.trunk.parameters(), *model.actor.parameters()],
        float(authority["option_training_boundary"]["maximum_gradient_norm"]),
    )
    if not bool(torch.isfinite(loss) and torch.isfinite(gradient)):
        raise RuntimeError("nonfinite aerial option human auxiliary")
    optimizer.step()
    return {
        "weighted_loss": float(loss.detach()),
        "raw_loss": float(objective.loss.detach()),
        "gradient_norm": float(gradient.detach()),
    }


def phase_gate(metrics: list[dict[str, Any]], phase: int, authority: dict[str, Any]) -> bool:
    phase_authority = authority["physical_curriculum"]["phases"][phase]
    for row in metrics:
        fractions = row["fractions"]
        if phase == PHASE_EASY_LAUNCH:
            if not (
                fractions["launch"] >= phase_authority["advance"]["launch_fraction_min"]
                and fractions["elevated_contact"]
                >= phase_authority["advance"]["elevated_contact_fraction_min"]
            ):
                return False
        elif phase == PHASE_MOVING_INTERCEPT:
            if not (
                fractions["launch"] >= phase_authority["advance"]["launch_fraction_min"]
                and fractions["high_contact"]
                >= phase_authority["advance"]["high_contact_fraction_min"]
                and fractions["forward_high_contact"]
                >= phase_authority["advance"]["forward_high_contact_fraction_min"]
            ):
                return False
        else:
            if not (
                fractions["high_contact"]
                >= phase_authority["select"]["high_contact_fraction_min"]
                and fractions["forward_high_contact"]
                >= phase_authority["select"]["forward_high_contact_fraction_min"]
                and fractions["aerial_origin_goal"]
                >= phase_authority["select"]["aerial_origin_goal_fraction_min"]
            ):
                return False
        if not row["finite_observation"] or max(row["analog_saturation_fraction"]) >= 0.98:
            return False
    return True


def save_option_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    side: int,
    phase: int,
    block: int,
    human_selection: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_AERIAL_OPTION_V1_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_AERIAL_OPTION_V1",
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "protected_base_sha256": BLUE_SHA256 if side == 0 else ORANGE_SHA256,
        "deployment_side": side,
        "phase": PHASE_NAMES[phase],
        "accepted_phase_block": block,
        "human_rehearsal": human_selection,
        "validation": validation,
        "production_reward_unchanged": True,
        "named_mechanic_classifier_used": False,
        "ppo_resumable_as_general_policy": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": v1.sha256_file(path),
        "model_tensor_sha256": human_base.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "phase": PHASE_NAMES[phase],
        "block": block,
        "side": side,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    sources = [
        torch.load(BLUE, map_location="cpu", weights_only=False),
        torch.load(ORANGE, map_location="cpu", weights_only=False),
    ]
    for side in (0, 1):
        expected = authority["protected_base"]["blue" if side == 0 else "orange"]
        if human_base.tensor_tree_sha256(sources[side]["model"]) != expected[
            "model_tensor_sha256"
        ]:
            raise RuntimeError("aerial option protected parent model changed")
    models = [make_model(source, args.device) for source in sources]
    teachers = [
        make_model(source, args.device).eval().requires_grad_(False) for source in sources
    ]
    human_base.SOURCE = BLUE
    human_base.SOURCE_SHA256 = BLUE_SHA256
    train, validation, _unused_teacher, human_identity = human_base.load_human_data(
        device=args.device
    )
    if human_identity["test_loaded"]:
        raise RuntimeError("aerial option must not open the human test split")
    baseline_human = [
        aerial_validation(model, validation, device=args.device) for model in models
    ]
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(SEED ^ (0xA100 + side))
        for side in (0, 1)
    ]
    baseline_physical: dict[str, Any] = {}
    for phase in range(len(PHASE_NAMES)):
        phase_rows = []
        phase_authority = authority["physical_curriculum"]["phases"][phase]
        for side in (0, 1):
            _unused, metrics = collect_aerial_rollout(
                models[side],
                geometry,
                meshes,
                side=side,
                phase=phase,
                worlds=args.evaluation_worlds_per_side,
                horizon=int(phase_authority["horizon_ticks"]),
                seed=SEED ^ (0xE000 + phase * 16 + side),
                device=args.device,
                generator=generators[side],
                distribution=phase_override(authority, phase),
                deterministic=True,
                collision_dir=args.collision_dir,
            )
            phase_rows.append(metrics)
        baseline_physical[PHASE_NAMES[phase]] = phase_rows
    preflight = {
        "format": "RIVAL2_AERIAL_OPTION_V1_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "human_test_loaded": False,
        "protected_base_unchanged": True,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("aerial option V1 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    human_selection, optimizers = train_human_rehearsal(
        models,
        teachers,
        train,
        validation,
        authority,
        device=args.device,
    )
    train_labels = np.asarray(train.mechanic_label)
    target_mask = torch.from_numpy(np.isin(train_labels, TARGET_LABELS))
    target_observations = train.mechanic_observation[target_mask]
    target_actions = train.mechanic_action[target_mask]
    target_labels = train_labels[target_mask.numpy()].tolist()
    target_attempts = np.asarray(train.mechanic_attempt)[target_mask.numpy()].tolist()
    samplers = [
        MechanicHierarchySampler(
            target_labels,
            target_attempts,
            uniform_label_fraction=0.5,
            maximum_oversampling_ratio=8.0,
            generator=torch.Generator(device="cpu").manual_seed(SEED ^ (0xD100 + side)),
        )
        for side in (0, 1)
    ]
    curve = RESULTS / "physical_training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    selected: dict[str, Any] | None = None
    stop_reason = "maximum_blocks"
    completed_phases: list[str] = []
    for phase in range(len(PHASE_NAMES)):
        phase_authority = authority["physical_curriculum"]["phases"][phase]
        maximum = min(
            int(phase_authority["maximum_blocks"]),
            int(args.maximum_blocks_per_phase),
        )
        distribution = phase_override(authority, phase)
        consecutive = 0
        best_score = (-1.0, -1.0, -1.0)
        best_phase: dict[str, Any] | None = None
        for block in range(1, maximum + 1):
            sides = []
            for side in (0, 1):
                rollout, rollout_metrics = collect_aerial_rollout(
                    models[side],
                    geometry,
                    meshes,
                    side=side,
                    phase=phase,
                    worlds=args.worlds_per_side,
                    horizon=int(phase_authority["horizon_ticks"]),
                    seed=SEED + phase * 1_000_000 + block * 100 + side,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=False,
                    collision_dir=args.collision_dir,
                )
                assert rollout is not None
                ppo = option_ppo_update(
                    models[side],
                    optimizers[side],
                    rollout,
                    generator=generators[side],
                    distribution=distribution,
                    authority=authority,
                )
                del rollout
                auxiliary = human_auxiliary_update(
                    models[side],
                    teachers[side],
                    optimizers[side],
                    target_observations,
                    target_actions,
                    samplers[side],
                    authority=authority,
                    device=args.device,
                )
                sides.append(
                    {
                        "side": side,
                        "rollout": rollout_metrics,
                        "ppo": ppo,
                        "human_auxiliary": auxiliary,
                    }
                )
            row: dict[str, Any] = {
                "phase": PHASE_NAMES[phase],
                "phase_index": phase,
                "block": block,
                "sides": sides,
            }
            interval = int(authority["physical_curriculum"]["evaluation_interval_blocks"])
            if block % interval == 0 or block == maximum:
                evaluation = []
                for side in (0, 1):
                    _unused, metrics = collect_aerial_rollout(
                        models[side],
                        geometry,
                        meshes,
                        side=side,
                        phase=phase,
                        worlds=args.evaluation_worlds_per_side,
                        horizon=int(phase_authority["horizon_ticks"]),
                        seed=SEED ^ (0xE000 + phase * 16 + side),
                        device=args.device,
                        generator=generators[side],
                        distribution=distribution,
                        deterministic=True,
                        collision_dir=args.collision_dir,
                    )
                    evaluation.append(metrics)
                human_validation = [
                    aerial_validation(model, validation, device=args.device)
                    for model in models
                ]
                passed = phase_gate(evaluation, phase, authority)
                row["evaluation"] = {
                    "passed": passed,
                    "physical": evaluation,
                    "human": human_validation,
                }
                minimum_high = min(item["fractions"]["high_contact"] for item in evaluation)
                minimum_forward = min(
                    item["fractions"]["forward_high_contact"] for item in evaluation
                )
                minimum_goal = min(
                    item["fractions"]["aerial_origin_goal"] for item in evaluation
                )
                score = (minimum_goal, minimum_high, minimum_forward)
                records = []
                for side in (0, 1):
                    records.append(
                        save_option_checkpoint(
                            sources[side],
                            models[side],
                            optimizers[side],
                            run_dir
                            / f"{PHASE_NAMES[phase]}_b{block:04d}_{'blue' if side == 0 else 'orange'}.pt",
                            side=side,
                            phase=phase,
                            block=block,
                            human_selection=human_selection[side],
                            validation=row["evaluation"],
                        )
                    )
                row["evaluation"]["checkpoint"] = records
                if score > best_score:
                    best_score = score
                    best_phase = copy.deepcopy(row["evaluation"])
                consecutive = consecutive + 1 if passed else 0
                print(
                    json.dumps(
                        {
                            "stage": "aerial_option_physical",
                            "phase": PHASE_NAMES[phase],
                            "block": block,
                            "passed": passed,
                            "fractions": [item["fractions"] for item in evaluation],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                required = (
                    int(phase_authority.get("advance", {}).get("consecutive_boundaries", 1))
                    if phase != PHASE_GOAL_DIRECTED
                    else 1
                )
                if consecutive >= required:
                    append_jsonl(curve, row)
                    completed_phases.append(PHASE_NAMES[phase])
                    if phase == PHASE_GOAL_DIRECTED:
                        selected = copy.deepcopy(row["evaluation"])
                        stop_reason = "goal_directed_gate_passed"
                    break
            append_jsonl(curve, row)
            rolling = run_dir / "rolling"
            for side in (0, 1):
                save_option_checkpoint(
                    sources[side],
                    models[side],
                    optimizers[side],
                    rolling / ("blue.pt" if side == 0 else "orange.pt"),
                    side=side,
                    phase=phase,
                    block=block,
                    human_selection=human_selection[side],
                    validation=row.get("evaluation", {}),
                )
        else:
            stop_reason = f"{PHASE_NAMES[phase]}_maximum_blocks_without_gate"
        if phase == PHASE_GOAL_DIRECTED and selected is not None:
            break
        if PHASE_NAMES[phase] not in completed_phases:
            selected = best_phase
            break

    result = {
        "format": "RIVAL2_AERIAL_OPTION_V1_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_human": baseline_human,
        "human_rehearsal": human_selection,
        "baseline_physical": baseline_physical,
        "completed_phases": completed_phases,
        "stop_reason": stop_reason,
        "selected": selected,
        "controlled_pass": bool(
            selected is not None
            and selected.get("passed")
            and len(completed_phases) == len(PHASE_NAMES)
        ),
        "base_models_unchanged": True,
        "human_test_loaded": False,
        "promoted": False,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["controlled_pass"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-side", type=int, default=2048)
    parser.add_argument("--evaluation-worlds-per-side", type=int, default=2048)
    parser.add_argument("--maximum-blocks-per-phase", type=int, default=160)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
