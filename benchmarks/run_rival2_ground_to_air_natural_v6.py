"""Learn natural aerial entries with pop-orientation control and balanced PPO."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as aerial_v1  # noqa: E402
from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_PARKED,
    SETUP_NAMES,
    build_natural_ground_to_air_scenarios,
)
from rivalsim.rival2_ground_to_air_option import GroundToAirConfig  # noqa: E402
from rivalsim.rival2_ground_to_air_pop_control_v6 import (  # noqa: E402
    GROUND_TO_AIR_POP_CONTROL_V6_VERSION,
    LearnedPopOrientationConfig,
    LearnedPopOrientationController,
    pop_orientation_channel_mask,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    deterministic_hybrid_action,
    hybrid_distribution_parameters,
    hybrid_entropy,
    sample_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V6"
AUTHORITY = ROOT / "results/rival2/ground_to_air_natural_v6/authority.json"
AUTHORITY_SHA256 = "AFB6D57B3A6A2AB204C132EF1AA37EDE6EFF9C903493D772321E753211358ADA"
RESULTS = ROOT / "results/rival2/ground_to_air_natural_v6"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_natural_v6"
PARENT = natural_v4.PARENT
PARENT_SHA256 = natural_v4.PARENT_SHA256
BLUE = natural_v4.BLUE
ORANGE = natural_v4.ORANGE
BLUE_SHA256 = natural_v4.BLUE_SHA256
ORANGE_SHA256 = natural_v4.ORANGE_SHA256
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v6")
DEFAULT_COLLISION_DIR = natural_v4.DEFAULT_COLLISION_DIR
LOG_TWO = math.log(2.0)
LOG_TWO_PI = math.log(2.0 * math.pi)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    if capability.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("natural V6 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected natural V6 authority format")
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"natural V6 bound input changed: {path}")
    if float(authority["reward"]["raw_airtime_reward"]) != 0.0:
        raise RuntimeError("raw airtime reward is forbidden")
    if int(authority["integrity"]["optimizer_steps_before_authority_commit"]) != 0:
        raise RuntimeError("natural V6 authority is not prospective")
    return authority


def training_strata(authority: dict[str, Any]) -> list[dict[str, Any]]:
    strata = list(authority["scenario"]["training_strata"])
    expected = {
        (setup, defender, side)
        for setup in SETUP_NAMES
        for defender in (DEFENDER_PARKED, DEFENDER_LIVE)
        for side in (0, 1)
    }
    observed = {
        (row["setup"], row["defender_mode"], int(row["side"])) for row in strata
    }
    if observed != expected or len(strata) != len(expected):
        raise RuntimeError("V6 must cover each setup/defender/side stratum exactly once")
    return strata


def hybrid_channel_log_probability(
    actor_output: torch.Tensor,
    action: torch.Tensor,
    *,
    pre_tanh: torch.Tensor,
    config: Any,
    distribution: HybridDistributionOverride,
) -> torch.Tensor:
    """Return five analog and three button log-probability contributions."""

    mean, log_std, logits = hybrid_distribution_parameters(
        actor_output,
        config,
        distribution_override=distribution,
    )
    inv_std = torch.exp(-log_std)
    gaussian = -0.5 * (
        ((pre_tanh - mean) * inv_std).square() + 2.0 * log_std + LOG_TWO_PI
    )
    log_jacobian = 2.0 * (
        LOG_TWO - pre_tanh - F.softplus(-2.0 * pre_tanh)
    )
    analog = gaussian - log_jacobian
    buttons = -F.binary_cross_entropy_with_logits(
        logits,
        action[..., 5:8],
        reduction="none",
    )
    return torch.cat((analog, buttons), dim=-1)


class MaskedOptionRollout(aerial_v1.OptionRollout):
    def __init__(self, horizon: int, worlds: int, device: str):
        super().__init__(horizon, worlds, device)
        self.channel_mask = torch.zeros(
            (horizon, worlds, 8), dtype=torch.bool, device=device
        )


def collect_rollout(
    model: Rival2ActorCritic,
    defenders: dict[int, Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    authority: dict[str, Any],
    side: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
    collision_dir: Path,
    setup: int,
    defender_mode: str,
    attacker_boost_range: tuple[float, float],
    pop_mask_factory: Callable[[Any, LearnedPopOrientationConfig], torch.Tensor]
    | None = None,
) -> tuple[MaskedOptionRollout | None, dict[str, Any]]:
    batch = build_natural_ground_to_air_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
        setup=setup,
        defender_mode=defender_mode,
        live_defender_fraction=1.0 if defender_mode == DEFENDER_LIVE else 0.0,
        attacker_boost_range=attacker_boost_range,
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed ^ side,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    config = GroundToAirConfig(
        **authority["option_config"],
        learned_after_second_jump=True,
    )
    orientation = LearnedPopOrientationConfig(**authority["pop_orientation_control"])
    controller = LearnedPopOrientationController(
        worlds,
        device=device,
        config=config,
        orientation=orientation,
    )
    tracker = goal_v3.GoalDirectedTrainingTracker(
        worlds,
        attacker_side=side,
        horizon=horizon,
        authority=authority,
    )
    rollout = None if deterministic else MaskedOptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    defender_active = torch.as_tensor(
        batch.defender_active, dtype=torch.bool, device=device
    )
    observation = env.observation
    false = torch.zeros(worlds, dtype=torch.bool, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    pop_orientation_ticks = torch.zeros((), dtype=torch.int64, device=device)
    other = 1 - side
    defender = defenders[other]
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            if deterministic:
                learned = deterministic_hybrid_action(actor, model.config)
                pre_tanh = actor[:, :5]
            else:
                sample = sample_hybrid_action(
                    actor,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution,
                )
                learned = sample.action
                pre_tanh = sample.pre_tanh
            defender_actor, _ = defender(observation[:, other])
            defender_action = deterministic_hybrid_action(
                defender_actor, defender.config
            )
        option = controller.step(
            learned,
            observation[:, side],
            kickoff_active=false,
            match_done=~active_before,
        )
        channel_mask = (
            pop_orientation_channel_mask(option)
            if pop_mask_factory is None
            else pop_mask_factory(option, orientation)
        ) & active_before[:, None]
        learned_active = channel_mask.any(dim=1)
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], option.action, 0.0)
        live = active_before & defender_active
        action[:, other] = torch.where(live[:, None], defender_action, 0.0)
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == side)
        reward, skill_done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=active_before & transition.terminated,
            active=active_before,
        )
        terminal = skill_done | transition.terminated | transition.truncated
        if rollout is not None:
            with torch.no_grad():
                old_channel_log_probability = hybrid_channel_log_probability(
                    actor,
                    learned,
                    pre_tanh=pre_tanh,
                    config=model.config,
                    distribution=distribution,
                )
            rollout.observation[tick].copy_(observation[:, side])
            # The sampled action is the latent option action.  During the pop,
            # only its orientation channels reach the simulator; the mask
            # excludes every scripted channel from PPO likelihood.
            rollout.action[tick].copy_(learned)
            rollout.pre_tanh[tick].copy_(pre_tanh)
            rollout.channel_mask[tick].copy_(channel_mask)
            rollout.old_log_probability[tick].copy_(
                (old_channel_log_probability * channel_mask).sum(dim=-1)
            )
            rollout.reward[tick].copy_(reward)
            rollout.done[tick].copy_(terminal)
            rollout.mask[tick].copy_(learned_active)
        saturation += (
            (transition.emitted_action[:, side, :5].abs() > 0.95)
            & learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        analog_sum += (
            transition.emitted_action[:, side, :5] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        button_sum += (
            transition.emitted_action[:, side, 5:] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        action_count += learned_active.sum(dtype=torch.float64)
        pop_orientation_ticks += (option.pop_primitive & active_before).sum()
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    if rollout is not None and bool(active.any()):
        rollout.done[min(tick, horizon - 1)] |= active
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    telemetry = tracker.telemetry()
    fractions = {
        key: telemetry[name] / worlds
        for key, name in (
            ("pop_touch", "low_pop_touches"),
            ("elevated_follow_touch", "elevated_follow_touches"),
            ("high_follow_touch", "high_follow_touches"),
            ("second_airborne_touch", "second_airborne_touches"),
            ("third_airborne_touch", "third_airborne_touches"),
            ("fourth_airborne_touch", "fourth_airborne_touches"),
            ("fifth_airborne_touch", "fifth_airborne_touches"),
            ("contact_budget_exceeded", "contact_budget_exceeded"),
            ("goal_within_contact_budget", "goals_within_contact_budget"),
            ("goal_over_contact_budget", "goals_over_contact_budget"),
            ("sustained_control", "sustained_control_attempts"),
            ("productive_continuation", "productive_continuation_attempts"),
            ("unassisted_or_ground_goal", "unassisted_or_ground_goals"),
            ("goalward_velocity_contact", "goalward_velocity_contacts"),
        )
    }
    metrics = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "setup": SETUP_NAMES[setup],
        "defender_mode": defender_mode,
        "live_defender_fraction": float(defender_active.float().mean().cpu()),
        "attacker_boost_range": list(attacker_boost_range),
        "telemetry": telemetry,
        "fractions": fractions,
        "reward_per_attempt": telemetry["reward_sum"] / worlds,
        "learned_action_ticks": int(action_count),
        "learned_pop_orientation_ticks": int(pop_orientation_ticks),
        "mean_learned_analog_action": (
            analog_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "learned_button_fraction": (
            button_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def prepare_rollout(
    rollout: MaskedOptionRollout,
    *,
    gamma: float,
) -> dict[str, torch.Tensor]:
    index = torch.nonzero(rollout.mask.reshape(-1), as_tuple=False).squeeze(-1)
    if index.numel() == 0:
        raise RuntimeError("balanced V6 rollout has no learned action rows")
    returns = rollout.discounted_returns(gamma).reshape(-1).index_select(0, index)
    advantage = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-8)
    return {
        "observation": rollout.observation.reshape(-1, 182).index_select(0, index),
        "action": rollout.action.reshape(-1, 8).index_select(0, index),
        "pre_tanh": rollout.pre_tanh.reshape(-1, 5).index_select(0, index),
        "channel_mask": rollout.channel_mask.reshape(-1, 8).index_select(0, index),
        "old_log_probability": rollout.old_log_probability.reshape(-1).index_select(0, index),
        "returns": returns,
        "advantage": advantage,
    }


def balanced_masked_ppo_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    rollouts: list[MaskedOptionRollout],
    *,
    authority: dict[str, Any],
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
) -> dict[str, Any]:
    training = authority["training"]
    prepared = [
        prepare_rollout(rollout, gamma=float(training["discount_gamma"]))
        for rollout in rollouts
    ]
    requested = int(training["samples_per_stratum_per_step"])
    sample_count = min(requested, *(row["observation"].shape[0] for row in prepared))
    if sample_count <= 0:
        raise RuntimeError("balanced V6 PPO has no common sample count")
    steps = int(training["balanced_optimizer_steps_per_block"])
    totals = {
        "policy_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    clip = float(training["ppo_clip"])
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_entropy = 0.0
        step_kl = 0.0
        step_clip = 0.0
        for row in prepared:
            local = torch.randperm(
                row["observation"].shape[0],
                device=row["observation"].device,
                generator=generator,
            )[:sample_count]
            actor, _ = model(row["observation"].index_select(0, local))
            mask = row["channel_mask"].index_select(0, local)
            channel_log_probability = hybrid_channel_log_probability(
                actor,
                row["action"].index_select(0, local),
                pre_tanh=row["pre_tanh"].index_select(0, local),
                config=model.config,
                distribution=distribution,
            )
            new_log_probability = (channel_log_probability * mask).sum(dim=-1)
            old = row["old_log_probability"].index_select(0, local)
            log_ratio = new_log_probability - old
            ratio = torch.exp(log_ratio)
            local_advantage = row["advantage"].index_select(0, local)
            policy_loss = -torch.minimum(
                ratio * local_advantage,
                ratio.clamp(1.0 - clip, 1.0 + clip) * local_advantage,
            ).mean()
            (policy_loss / len(prepared)).backward()
            with torch.no_grad():
                step_loss += float(policy_loss)
                step_entropy += float(
                    hybrid_entropy(
                        actor,
                        model.config,
                        distribution_override=distribution,
                    ).mean()
                )
                step_kl += float(((ratio - 1.0) - log_ratio).mean())
                step_clip += float(
                    (ratio.sub(1.0).abs() > clip).to(torch.float32).mean()
                )
        gradient = torch.nn.utils.clip_grad_norm_(
            [*model.trunk.parameters(), *model.actor.parameters()],
            float(training["maximum_gradient_norm"]),
        )
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError("nonfinite balanced V6 PPO gradient")
        optimizer.step()
        divisor = float(len(prepared))
        totals["policy_loss"] += step_loss / divisor
        totals["entropy"] += step_entropy / divisor
        totals["approx_kl"] += step_kl / divisor
        totals["clip_fraction"] += step_clip / divisor
        totals["gradient_norm"] += float(gradient)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite balanced V6 model")
    return {
        **{name: value / steps for name, value in totals.items()},
        "steps": steps,
        "strata": len(prepared),
        "samples_per_stratum_per_step": sample_count,
        "effective_samples_per_step": sample_count * len(prepared),
        "return_mean_by_stratum": [float(row["returns"].mean()) for row in prepared],
        "return_max_by_stratum": [float(row["returns"].max()) for row in prepared],
    }


def validation_rows(
    model: Rival2ActorCritic,
    defenders: dict[int, Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    authority: dict[str, Any],
    worlds: int,
    seed: int,
    device: str,
    generators: list[torch.Generator],
    distribution: HybridDistributionOverride,
    collision_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon = int(authority["episode"]["horizon_ticks"])
    for setup in range(len(SETUP_NAMES)):
        for defender_mode in (DEFENDER_PARKED, DEFENDER_LIVE):
            for side in (0, 1):
                _rollout, metrics = collect_rollout(
                    model,
                    defenders,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=worlds,
                    horizon=horizon,
                    seed=(
                        seed
                        + setup * 100_000
                        + (10_000 if defender_mode == DEFENDER_LIVE else 0)
                    ),
                    device=device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=True,
                    collision_dir=collision_dir,
                    setup=setup,
                    defender_mode=defender_mode,
                    attacker_boost_range=tuple(
                        authority["scenario"]["validation_boost_range"]
                    ),
                )
                rows.append(metrics)
    return rows


def minimum_gate_ratio(rows: list[dict[str, Any]], authority: dict[str, Any]) -> float:
    ratios: list[float] = []
    acceptance = authority["acceptance"]
    for row in rows:
        gate = acceptance["per_defender_mode"][row["defender_mode"]]
        fractions = row["fractions"]
        for name in (
            "pop_touch",
            "elevated_follow_touch",
            "high_follow_touch",
            "productive_continuation",
            "goal_within_contact_budget",
        ):
            ratios.append(float(fractions[name]) / float(gate[f"{name}_fraction_min"]))
        over = float(fractions["contact_budget_exceeded"])
        limit = float(acceptance["contact_budget_exceeded_fraction_max"])
        if over > limit:
            ratios.append(1.0 - (over - limit) / max(limit, 1.0e-12))
    return float(min(ratios))


def selection_key(
    rows: list[dict[str, Any]], authority: dict[str, Any]
) -> tuple[float, float]:
    return minimum_gate_ratio(rows, authority), natural_v4.evaluation_score(rows)


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    block: int,
    evaluation: dict[str, Any],
    disposition: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V6_FRESH_BALANCED_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": VERSION,
        "pop_control_identity": GROUND_TO_AIR_POP_CONTROL_V6_VERSION,
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "ground_to_air_goal_v3_parent_sha256": PARENT_SHA256,
        "protected_v23_defenders": {"blue": BLUE_SHA256, "orange": ORANGE_SHA256},
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "learned_pop_orientation_channels": ["steer", "pitch", "yaw", "roll"],
        "scripted_pop_channels": ["throttle", "jump", "boost", "handbrake"],
        "channel_masked_likelihood": True,
        "equal_stratum_gradient_aggregation": True,
        "maximum_distinct_chain_contacts": 6,
        "disposition": disposition,
        "production_reward_unchanged": True,
        "ppo_resumable_as_general_policy": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": path.as_posix(),
        "sha256": capability.sha256_file(path),
        "model_tensor_sha256": autonomous.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "block": block,
        "disposition": disposition,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    strata = training_strata(authority)
    if capability.sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("controlled aerial scorer parent changed")
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(source, args.device)
    defenders = natural_v4.load_defender_policies(args.device)
    optimizer = natural_v4.make_optimizer(model, authority)
    critic_hash = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(
            int(authority["seeds"]["optimizer_and_exploration"]) ^ side
        )
        for side in (0, 1)
    ]
    update_generator = torch.Generator(device=args.device).manual_seed(
        int(authority["seeds"]["balanced_minibatch"])
    )
    distribution = natural_v4.distribution_override(authority)
    baseline = validation_rows(
        model,
        defenders,
        geometry,
        meshes,
        authority=authority,
        worlds=args.evaluation_worlds_per_row,
        seed=int(authority["seeds"]["validation"]),
        device=args.device,
        generators=generators,
        distribution=distribution,
        collision_dir=args.collision_dir,
    )
    preflight = {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "parent_hash_verified": True,
        "protected_v23_defender_hashes_verified": True,
        "critic_frozen": True,
        "production_reward_unchanged": True,
        "raw_airtime_reward": authority["reward"]["raw_airtime_reward"],
        "pop_orientation_control": authority["pop_orientation_control"],
        "channel_masked_likelihood": True,
        "equal_stratum_gradient_aggregation": True,
        "success_volume_rehearsal": False,
        "strata": strata,
        "optimizer_steps": 0,
        "baseline_validation": baseline,
        "baseline_selection_key": selection_key(baseline, authority),
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("natural V6 training requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()

    best_key = selection_key(baseline, authority)
    best_block = 0
    best_evaluation = copy.deepcopy(baseline)
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_checkpoint: dict[str, Any] | None = None
    stale_boundaries = 0
    consecutive_gate = 0
    stop_reason = "maximum_blocks"
    training = authority["training"]
    maximum_blocks = min(int(training["maximum_blocks"]), int(args.maximum_blocks))
    interval = int(training["evaluation_interval_blocks"])
    horizon = int(authority["episode"]["horizon_ticks"])
    setup_by_name = {name: index for index, name in enumerate(SETUP_NAMES)}

    for block in range(1, maximum_blocks + 1):
        block_model = copy.deepcopy(model.state_dict())
        block_optimizer = copy.deepcopy(optimizer.state_dict())
        rollouts: list[MaskedOptionRollout] = []
        rollout_rows: list[dict[str, Any]] = []
        try:
            for stratum_index, stratum in enumerate(strata):
                setup = setup_by_name[stratum["setup"]]
                defender_mode = stratum["defender_mode"]
                side = int(stratum["side"])
                seed = (
                    int(authority["seeds"]["training"])
                    + block * 100_000
                    + setup * 10_000
                    + (1_000 if defender_mode == DEFENDER_LIVE else 0)
                    + stratum_index * 100
                )
                rollout, metrics = collect_rollout(
                    model,
                    defenders,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=args.worlds_per_stratum,
                    horizon=horizon,
                    seed=seed,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=False,
                    collision_dir=args.collision_dir,
                    setup=setup,
                    defender_mode=defender_mode,
                    attacker_boost_range=tuple(
                        authority["scenario"]["training_boost_range"]
                    ),
                )
                assert rollout is not None
                rollouts.append(rollout)
                rollout_rows.append(metrics)
            ppo = balanced_masked_ppo_update(
                model,
                optimizer,
                rollouts,
                authority=authority,
                generator=update_generator,
                distribution=distribution,
            )
            if not all(
                bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
            ):
                raise FloatingPointError("nonfinite V6 aerial-option parameter")
        except (FloatingPointError, RuntimeError) as error:
            model.load_state_dict(block_model, strict=True)
            optimizer.load_state_dict(block_optimizer)
            stop_reason = f"hard_failure:{type(error).__name__}:{error}"
            append_jsonl(
                RESULTS / "hard_failure.jsonl",
                {"block": block, "created_utc": utc_now(), "reason": stop_reason},
            )
            break
        finally:
            rollouts.clear()
            gc.collect()
            torch.cuda.empty_cache()

        row: dict[str, Any] = {"block": block, "strata": rollout_rows, "ppo": ppo}
        if block % interval == 0 or block == maximum_blocks:
            validation = validation_rows(
                model,
                defenders,
                geometry,
                meshes,
                authority=authority,
                worlds=args.evaluation_worlds_per_row,
                seed=int(authority["seeds"]["validation"]),
                device=args.device,
                generators=generators,
                distribution=distribution,
                collision_dir=args.collision_dir,
            )
            key = selection_key(validation, authority)
            passed = natural_v4.passes_gate(validation, authority)
            improved = (
                key[0] > best_key[0] + 1.0e-9
                or (
                    abs(key[0] - best_key[0]) <= 1.0e-9
                    and key[1]
                    > best_key[1] + float(training["minimum_score_improvement"])
                )
            )
            if improved:
                best_key = key
                best_block = block
                best_evaluation = copy.deepcopy(validation)
                best_model = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                stale_boundaries = 0
            else:
                stale_boundaries += 1
            consecutive_gate = consecutive_gate + 1 if passed else 0
            row["evaluation"] = {
                "validation": validation,
                "selection_key": key,
                "passed": passed,
                "improved": improved,
                "stale_boundaries": stale_boundaries,
            }
            row["rolling_checkpoint"] = save_checkpoint(
                source,
                model,
                optimizer,
                run_dir / "rolling.pt",
                block=block,
                evaluation=row["evaluation"],
                disposition="rolling_diagnostic",
            )
            if improved:
                best_checkpoint = save_checkpoint(
                    source,
                    model,
                    optimizer,
                    run_dir / "best_validation.pt",
                    block=block,
                    evaluation=row["evaluation"],
                    disposition="validation_selected_diagnostic",
                )
            print(
                json.dumps(
                    {
                        "stage": "ground_to_air_natural_v6",
                        "block": block,
                        "selection_key": key,
                        "best_block": best_block,
                        "passed": passed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if consecutive_gate >= int(authority["acceptance"]["consecutive_boundaries"]):
                stop_reason = "validation_gate_passed"
                append_jsonl(curve, row)
                break
            if stale_boundaries >= int(training["plateau_patience_boundaries"]):
                stop_reason = "validation_plateau"
                append_jsonl(curve, row)
                break
        append_jsonl(curve, row)

    model.load_state_dict(best_model, strict=True)
    optimizer.load_state_dict(best_optimizer)
    observed_critic = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    if observed_critic != critic_hash:
        raise RuntimeError("natural V6 aerial option changed the frozen critic")
    validation_pass = natural_v4.passes_gate(best_evaluation, authority)
    test: list[dict[str, Any]] | None = None
    if validation_pass:
        test = validation_rows(
            model,
            defenders,
            geometry,
            meshes,
            authority=authority,
            worlds=args.test_worlds_per_row,
            seed=int(authority["seeds"]["test"]),
            device=args.device,
            generators=generators,
            distribution=distribution,
            collision_dir=args.collision_dir,
        )
    controlled_pass = bool(
        validation_pass
        and test is not None
        and natural_v4.passes_gate(test, authority)
    )
    checkpoints: list[dict[str, Any]] = []
    if controlled_pass:
        checkpoints.append(
            save_checkpoint(
                source,
                model,
                optimizer,
                CHECKPOINTS / "rival2_ground_to_air_natural_v6.pt",
                block=best_block,
                evaluation={"validation": best_evaluation, "test": test},
                disposition="promoted_controlled_option",
            )
        )
    result = {
        "format": f"{VERSION}_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_validation": baseline,
        "best_block": best_block,
        "best_validation": best_evaluation,
        "best_selection_key": best_key,
        "best_diagnostic_checkpoint": best_checkpoint,
        "untouched_test_opened": test is not None,
        "untouched_test": test,
        "stop_reason": stop_reason,
        "controlled_pass": controlled_pass,
        "parent_unchanged": capability.sha256_file(PARENT) == PARENT_SHA256,
        "protected_v23_unchanged": (
            capability.sha256_file(BLUE) == BLUE_SHA256
            and capability.sha256_file(ORANGE) == ORANGE_SHA256
        ),
        "critic_unchanged": True,
        "learned_pop_orientation": True,
        "channel_masked_likelihood": True,
        "equal_stratum_gradient_aggregation": True,
        "success_volume_rehearsal": False,
        "production_reward_unchanged": True,
        "checkpoints": checkpoints,
        "promoted_into_competitive_policy": False,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if controlled_pass else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-stratum", type=int, default=256)
    parser.add_argument("--evaluation-worlds-per-row", type=int, default=256)
    parser.add_argument("--test-worlds-per-row", type=int, default=512)
    parser.add_argument("--maximum-blocks", type=int, default=96)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
