"""Train one canonical post-pop controller with a short-credit curriculum.

V1 proved the low-ball pop but not a reliable follow contact.  V2 initializes
from its better diagnostic side, shares one model across both canonical team
perspectives, moves the learned-control handoff toward the physical contact,
and rehearses only sampled trajectories that actually caused an elevated
recontact.  Full pop-to-follow validation remains the promotion authority.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from dataclasses import asdict
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
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_curriculum_v2 import (  # noqa: E402
    GROUND_TO_AIR_CURRICULUM_V2_VERSION,
    HandoffStage,
    handoff_tick_for_block,
    learned_handoff_mask,
    successful_rehearsal_mask,
)
from rivalsim.rival2_ground_to_air_option import (  # noqa: E402
    GroundToAirConfig,
    GroundToAirController,
    GroundToAirTracker,
    build_ground_to_air_scenarios,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_log_probability,
    sample_hybrid_action,
)

AUTHORITY = ROOT / "results/rival2/ground_to_air_option_v2/authority.json"
# Filled only after the prospective authority is committed.
AUTHORITY_SHA256 = "F2A3F78A4AF474D5FD517209AB41A78E51C6CD70FF51395B72DD513056271855"
RESULTS = ROOT / "results/rival2/ground_to_air_option_v2"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_option_v2"
PARENT = (
    ROOT
    / "checkpoints/rival2/ground_to_air_option_v1_diagnostic/rival2_orange_block80.pt"
)
PARENT_SHA256 = "F353A8B784AD5619CE56C8E8B66D98A4E690F6C40652A629D90345CA753FF28A"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-option-v2")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")


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
        raise RuntimeError("ground-to-air option authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"ground-to-air bound input changed: {path}")
    if authority["reward"]["raw_airtime_reward"] != 0.0:
        raise RuntimeError("ground-to-air authority cannot reward raw airtime")
    return authority


def make_model(payload: dict[str, Any], device: str) -> Rival2ActorCritic:
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.trunk.requires_grad_(True)
    model.actor.requires_grad_(True)
    model.critic.requires_grad_(False)
    return model


def make_optimizer(model: Rival2ActorCritic, authority: dict[str, Any]) -> torch.optim.AdamW:
    training = authority["training"]
    return torch.optim.AdamW(
        [
            {
                "params": model.trunk.parameters(),
                "lr": float(training["trunk_learning_rate"]),
            },
            {
                "params": model.actor.parameters(),
                "lr": float(training["actor_learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )


def distribution_override(authority: dict[str, Any]) -> HybridDistributionOverride:
    exploration = authority["training"]["exploration"]
    return HybridDistributionOverride(
        analog_log_std=float(torch.log(torch.tensor(exploration["analog_sigma"]))),
        button_temperature=float(exploration["button_temperature"]),
    )


class GroundToAirTrainingTracker:
    """Physical reward for maintaining and reacquiring a self-created pop."""

    def __init__(
        self,
        worlds: int,
        *,
        attacker_side: int,
        horizon: int,
        authority: dict[str, Any],
    ) -> None:
        self.worlds = worlds
        self.side = attacker_side
        self.horizon = horizon
        self.authority = authority
        self.physical = GroundToAirTracker(worlds, attacker_side=attacker_side, horizon=horizon)
        self.initialized = False
        self.reward_sum = 0.0
        self.ground_failures = 0
        self.no_pop_failures = 0

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [observation[:, self.side, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"],
            dim=-1,
        )

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            device = before.device
            self.seen_qualified = torch.zeros(self.worlds, dtype=torch.bool, device=device)
            self.seen_rise = torch.zeros(self.worlds, dtype=torch.bool, device=device)
            self.seen_elevated = torch.zeros(self.worlds, dtype=torch.bool, device=device)
            self.seen_high = torch.zeros(self.worlds, dtype=torch.bool, device=device)
            self.seen_second = torch.zeros(self.worlds, dtype=torch.bool, device=device)
            self.initialized = True

        physical_done = self.physical.step(
            before,
            after,
            tick=tick,
            goal_for_attacker=goal_for_attacker,
            active=active,
        )
        qualified = self.physical.qualified_pop & ~self.seen_qualified
        rise = self.physical.rise_250 & ~self.seen_rise
        elevated = self.physical.elevated_follow & ~self.seen_elevated
        high = self.physical.high_follow & ~self.seen_high
        second = self.physical.second_airborne & ~self.seen_second
        self.seen_qualified |= qualified
        self.seen_rise |= rise
        self.seen_elevated |= elevated
        self.seen_high |= high
        self.seen_second |= second

        scale = torch.as_tensor(POSITION_SCALE, dtype=before.dtype, device=before.device)
        relative_before = self._vector(before, "relative.ball_position") * scale
        relative_after = self._vector(after, "relative.ball_position") * scale
        distance_before = torch.linalg.vector_norm(relative_before, dim=-1)
        distance_after = torch.linalg.vector_norm(relative_after, dim=-1)
        horizontal_before = torch.linalg.vector_norm(relative_before[:, :2], dim=-1)
        horizontal_after = torch.linalg.vector_norm(relative_after[:, :2], dim=-1)
        shell = float(self.authority["reward"]["contact_shell_distance_uu"])
        vertical = float(self.authority["reward"]["vertical_standoff_uu"])
        shell_progress = (distance_before - shell).abs() - (distance_after - shell).abs()
        horizontal_progress = horizontal_before - horizontal_after
        vertical_progress = (relative_before[:, 2] - vertical).abs() - (
            relative_after[:, 2] - vertical
        ).abs()
        reward_authority = self.authority["reward"]
        tracking = self.physical.pop_touch & active
        reward = torch.zeros(self.worlds, dtype=torch.float32, device=before.device)
        reward += tracking * (
            shell_progress.clamp(-10.0, 10.0)
            * float(reward_authority["contact_shell_progress_per_uu"])
            + horizontal_progress.clamp(-10.0, 10.0)
            * float(reward_authority["horizontal_tracking_progress_per_uu"])
            + vertical_progress.clamp(-10.0, 10.0)
            * float(reward_authority["vertical_tracking_progress_per_uu"])
        )
        reward += qualified * float(reward_authority["qualified_pop_event"])
        reward += rise * float(reward_authority["ball_rise_event"])
        reward += elevated * float(reward_authority["elevated_follow_touch_event"])
        reward += high * float(reward_authority["high_follow_touch_event"])
        reward += second * float(reward_authority["second_airborne_touch_event"])
        reward += (goal_for_attacker & self.physical.elevated_follow & active) * float(
            reward_authority["goal_after_follow_event"]
        )

        ball_height = after[:, self.side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        ground_failure = (
            active
            & self.physical.pop_touch
            & (
                (tick - self.physical.pop_tick)
                >= int(self.authority["episode"]["ground_failure_after_pop_ticks"])
            )
            & (ball_height <= float(self.authority["episode"]["ground_failure_ball_height_uu"]))
        )
        no_pop_failure = (
            active
            & ~self.physical.pop_touch
            & (tick >= int(self.authority["episode"]["pop_deadline_tick"]))
        )
        failure = ground_failure | no_pop_failure
        reward += failure * float(reward_authority["failure"])
        done = physical_done | failure
        self.ground_failures += int(ground_failure.sum())
        self.no_pop_failures += int(no_pop_failure.sum())
        self.reward_sum += float(reward.sum())
        return reward, done

    def telemetry(self) -> dict[str, Any]:
        result = asdict(self.physical.telemetry)
        result.update(
            {
                "ground_failures": self.ground_failures,
                "no_pop_failures": self.no_pop_failures,
                "reward_sum": self.reward_sum,
            }
        )
        return result


def collect_rollout(
    model: Rival2ActorCritic,
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
    handoff_tick: int | None = None,
) -> tuple[aerial_v1.OptionRollout | None, dict[str, Any]]:
    batch = build_ground_to_air_scenarios(worlds, seed=seed ^ side, attacker_side=side)
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
    # Validation uses the full V1 chain (learned control immediately after the
    # neutral second jump).  Training may hold the calibrated physical source
    # sequence longer so PPO credit begins close to the reacquisition contact.
    config = GroundToAirConfig(
        **authority["bootstrap_config"],
        learned_after_second_jump=handoff_tick is None,
    )
    controller = GroundToAirController(worlds, device=device, config=config)
    tracker = GroundToAirTrainingTracker(
        worlds,
        attacker_side=side,
        horizon=horizon,
        authority=authority,
    )
    rollout = None if deterministic else aerial_v1.OptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    false = torch.zeros(worlds, dtype=torch.bool, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
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
            old_log_probability = hybrid_log_probability(
                actor,
                learned,
                config=model.config,
                pre_tanh=pre_tanh,
                distribution_override=distribution,
            )
        option = controller.step(
            learned,
            observation[:, side],
            kickoff_active=false,
            match_done=~active_before,
        )
        if handoff_tick is None:
            learned_active = active_before & option.learned_control
            emitted_option_action = option.action
        else:
            learned_active = active_before & learned_handoff_mask(
                active=option.active,
                # GroundToAirController advances its public age after building
                # the current action; recover the age used for this tick.
                pop_age=controller.pop_age - option.active.to(torch.int64),
                handoff_tick=handoff_tick,
            )
            emitted_option_action = torch.where(
                learned_active[:, None], learned, option.action
            )
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], emitted_option_action, 0.0)
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
            rollout.old_log_probability[tick].copy_(old_log_probability)
            rollout.reward[tick].copy_(reward)
            rollout.done[tick].copy_(terminal)
            rollout.mask[tick].copy_(learned_active)
        saturation += (
            (transition.emitted_action[:, side, :5].abs() > 0.95) & learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        analog_sum += (
            transition.emitted_action[:, side, :5] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        button_sum += (
            transition.emitted_action[:, side, 5:] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        action_count += learned_active.sum(dtype=torch.float64)
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
            ("pop_touch", "pop_touches"),
            ("qualified_pop", "qualified_pops"),
            ("ball_rise_250", "ball_rise_250"),
            ("elevated_follow_touch", "elevated_follow_touches"),
            ("high_follow_touch", "high_follow_touches"),
            ("second_airborne_touch", "second_airborne_touches"),
            ("goal_after_pop", "goals_after_pop"),
        )
    }
    metrics = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "telemetry": telemetry,
        "fractions": fractions,
        "reward_per_attempt": telemetry["reward_sum"] / worlds,
        "learned_action_ticks": int(action_count),
        "handoff_tick": handoff_tick,
        "mean_learned_analog_action": (analog_sum / action_count.clamp_min(1.0)).cpu().tolist(),
        "learned_button_fraction": (button_sum / action_count.clamp_min(1.0)).cpu().tolist(),
        "analog_saturation_fraction": (saturation / action_count.clamp_min(1.0)).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def handoff_stages(authority: dict[str, Any]) -> tuple[HandoffStage, ...]:
    return tuple(
        HandoffStage(
            first_block=int(row["first_block"]),
            handoff_ticks=tuple(int(value) for value in row["handoff_ticks"]),
        )
        for row in authority["training"]["handoff_schedule"]
    )


def successful_rehearsal_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    rollout: aerial_v1.OptionRollout,
    *,
    authority: dict[str, Any],
    generator: torch.Generator,
) -> dict[str, Any]:
    """Rehearse only actions that physically preceded an elevated recontact."""

    rehearsal = authority["training"]["successful_rehearsal"]
    selected = successful_rehearsal_mask(
        rollout.reward,
        rollout.mask,
        event_reward_threshold=float(rehearsal["event_reward_threshold"]),
        history_ticks=int(rehearsal["history_ticks"]),
    )
    indices = torch.nonzero(selected.reshape(-1), as_tuple=False).flatten()
    maximum = int(rehearsal["maximum_samples_per_rollout"])
    if indices.numel() > maximum:
        order = torch.randperm(indices.numel(), device=indices.device, generator=generator)
        indices = indices.index_select(0, order[:maximum])
    if indices.numel() == 0:
        return {"samples": 0, "loss": 0.0, "gradient_norm": 0.0, "finite": True}
    observation = rollout.observation.reshape(-1, 182).index_select(0, indices)
    target = rollout.action.reshape(-1, 8).index_select(0, indices)
    model.train()
    actor, _ = model(observation)
    analog = F.smooth_l1_loss(
        torch.tanh(actor[:, :5]),
        target[:, :5],
        beta=float(rehearsal["smooth_l1_beta"]),
    )
    buttons = F.binary_cross_entropy_with_logits(actor[:, 10:13], target[:, 5:8])
    loss = float(rehearsal["loss_coefficient"]) * (
        analog + float(rehearsal["button_loss_weight"]) * buttons
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        float(authority["training"]["maximum_gradient_norm"]),
    )
    finite = bool(torch.isfinite(loss) and torch.isfinite(gradient))
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite successful-trajectory rehearsal")
    optimizer.step()
    return {
        "samples": int(indices.numel()),
        "loss": float(loss.detach()),
        "analog_loss": float(analog.detach()),
        "button_loss": float(buttons.detach()),
        "gradient_norm": float(gradient.detach()),
        "finite": True,
    }


def ppo_compatibility(authority: dict[str, Any]) -> dict[str, Any]:
    training = authority["training"]
    return {
        "option_training_boundary": {"maximum_gradient_norm": training["maximum_gradient_norm"]},
        "physical_curriculum": {
            "discount_gamma": training["discount_gamma"],
            "ppo_clip": training["ppo_clip"],
            "epochs": training["epochs"],
            "minibatch_size": training["minibatch_size"],
        },
    }


def evaluation_score(rows: list[dict[str, Any]]) -> float:
    minimum = {
        name: min(row["fractions"][name] for row in rows)
        for name in (
            "elevated_follow_touch",
            "high_follow_touch",
            "second_airborne_touch",
            "goal_after_pop",
        )
    }
    return float(
        8.0 * minimum["goal_after_pop"]
        + 5.0 * minimum["second_airborne_touch"]
        + 3.0 * minimum["high_follow_touch"]
        + minimum["elevated_follow_touch"]
        + 0.05 * sum(row["reward_per_attempt"] for row in rows) / len(rows)
    )


def passes_gate(rows: list[dict[str, Any]], authority: dict[str, Any]) -> bool:
    gate = authority["acceptance"]
    for row in rows:
        fractions = row["fractions"]
        if fractions["pop_touch"] < float(gate["pop_touch_fraction_min"]):
            return False
        if fractions["elevated_follow_touch"] < float(gate["elevated_follow_touch_fraction_min"]):
            return False
        if fractions["high_follow_touch"] < float(gate["high_follow_touch_fraction_min"]):
            return False
        if fractions["second_airborne_touch"] < float(gate["second_airborne_touch_fraction_min"]):
            return False
        if not row["finite"] or max(row["analog_saturation_fraction"]) >= float(
            gate["maximum_analog_saturation_fraction"]
        ):
            return False
    return True


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    block: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_GROUND_TO_AIR_OPTION_V2_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": GROUND_TO_AIR_CURRICULUM_V2_VERSION,
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "diagnostic_parent_sha256": PARENT_SHA256,
        "protected_v23_ancestor_sha256": ORANGE_SHA256,
        "deployment_sides": [0, 1],
        "canonical_shared_policy": True,
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "successful_physical_trajectory_rehearsal": True,
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
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    if capability.sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("V1 diagnostic curriculum parent changed")
    for path, expected in ((BLUE, BLUE_SHA256), (ORANGE, ORANGE_SHA256)):
        if capability.sha256_file(path) != expected:
            raise RuntimeError(f"protected V23 checkpoint changed: {path}")
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = make_model(source, args.device)
    optimizer = make_optimizer(model, authority)
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
    distribution = distribution_override(authority)
    horizon = int(authority["episode"]["horizon_ticks"])

    baseline = [
        collect_rollout(
            model,
            geometry,
            meshes,
            authority=authority,
            side=side,
            worlds=args.evaluation_worlds_per_side,
            horizon=horizon,
            seed=int(authority["seeds"]["validation"]),
            device=args.device,
            generator=generators[side],
            distribution=distribution,
            deterministic=True,
            collision_dir=args.collision_dir,
        )[1]
        for side in (0, 1)
    ]
    preflight = {
        "format": "RIVAL2_GROUND_TO_AIR_OPTION_V2_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "diagnostic_parent_hash_verified": True,
        "protected_v23_hashes_verified": True,
        "canonical_shared_policy": True,
        "critic_frozen": True,
        "production_reward_unchanged": True,
        "raw_airtime_reward": authority["reward"]["raw_airtime_reward"],
        "optimizer_steps": 0,
        "baseline_validation": baseline,
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("ground-to-air option requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    best_score = evaluation_score(baseline)
    best_block = 0
    best_evaluation = copy.deepcopy(baseline)
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    stale_boundaries = 0
    consecutive_gate = 0
    stop_reason = "maximum_blocks"
    compatibility = ppo_compatibility(authority)
    maximum_blocks = min(int(authority["training"]["maximum_blocks"]), args.maximum_blocks)
    interval = int(authority["training"]["evaluation_interval_blocks"])
    stages = handoff_stages(authority)
    for block in range(1, maximum_blocks + 1):
        block_model = copy.deepcopy(model.state_dict())
        block_optimizer = copy.deepcopy(optimizer.state_dict())
        handoff_tick = handoff_tick_for_block(block, stages)
        sides = []
        try:
            for side in (0, 1):
                rollout, rollout_metrics = collect_rollout(
                    model,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=args.worlds_per_side,
                    horizon=horizon,
                    seed=int(authority["seeds"]["training"]) + block * 100,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=False,
                    collision_dir=args.collision_dir,
                    handoff_tick=handoff_tick,
                )
                assert rollout is not None
                ppo = aerial_v1.option_ppo_update(
                    model,
                    optimizer,
                    rollout,
                    generator=generators[side],
                    distribution=distribution,
                    authority=compatibility,
                )
                rehearsal = successful_rehearsal_update(
                    model,
                    optimizer,
                    rollout,
                    authority=authority,
                    generator=generators[side],
                )
                del rollout
                finite_parameters = all(
                    bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
                )
                if not finite_parameters:
                    raise FloatingPointError("nonfinite option-policy parameter")
                sides.append(
                    {
                        "side": side,
                        "rollout": rollout_metrics,
                        "ppo": ppo,
                        "successful_rehearsal": rehearsal,
                    }
                )
        except (FloatingPointError, RuntimeError) as error:
            model.load_state_dict(block_model, strict=True)
            optimizer.load_state_dict(block_optimizer)
            stop_reason = f"hard_failure:{type(error).__name__}:{error}"
            append_jsonl(
                RESULTS / "hard_failure.jsonl",
                {"block": block, "created_utc": utc_now(), "reason": stop_reason},
            )
            break
        row: dict[str, Any] = {
            "block": block,
            "handoff_tick": handoff_tick,
            "sides": sides,
        }
        if block % interval == 0 or block == maximum_blocks:
            validation = [
                collect_rollout(
                    model,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=args.evaluation_worlds_per_side,
                    horizon=horizon,
                    seed=int(authority["seeds"]["validation"]),
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=True,
                    collision_dir=args.collision_dir,
                )[1]
                for side in (0, 1)
            ]
            score = evaluation_score(validation)
            passed = passes_gate(validation, authority)
            improved = score > best_score + float(
                authority["training"]["minimum_score_improvement"]
            )
            if improved:
                best_score = score
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
                "score": score,
                "passed": passed,
                "improved": improved,
                "stale_boundaries": stale_boundaries,
            }
            print(
                json.dumps(
                    {
                        "stage": "ground_to_air_option_v2",
                        "block": block,
                        "handoff_tick": handoff_tick,
                        "score": score,
                        "best_block": best_block,
                        "passed": passed,
                        "fractions": [entry["fractions"] for entry in validation],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            save_checkpoint(
                source,
                model,
                optimizer,
                run_dir / "rolling.pt",
                block=block,
                evaluation=row["evaluation"],
            )
            if consecutive_gate >= int(authority["acceptance"]["consecutive_boundaries"]):
                stop_reason = "validation_gate_passed"
                append_jsonl(curve, row)
                break
            if stale_boundaries >= int(authority["training"]["plateau_patience_boundaries"]):
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
        raise RuntimeError("ground-to-air option changed the frozen critic")

    # Do not open V2's test corpus unless a validation-selected checkpoint has
    # already passed the complete full-chain physical gate.
    validation_pass = passes_gate(best_evaluation, authority)
    test: list[dict[str, Any]] | None = None
    if validation_pass:
        test = [
            collect_rollout(
                model,
                geometry,
                meshes,
                authority=authority,
                side=side,
                worlds=args.test_worlds_per_side,
                horizon=horizon,
                seed=int(authority["seeds"]["test"]),
                device=args.device,
                generator=generators[side],
                distribution=distribution,
                deterministic=True,
                collision_dir=args.collision_dir,
            )[1]
            for side in (0, 1)
        ]
    controlled_pass = bool(
        validation_pass and test is not None and passes_gate(test, authority)
    )
    checkpoints = []
    if controlled_pass:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        checkpoints.append(
            save_checkpoint(
                source,
                model,
                optimizer,
                CHECKPOINTS / "rival2_ground_to_air_option_v2.pt",
                block=best_block,
                evaluation={"validation": best_evaluation, "test": test},
            )
        )
    result = {
        "format": "RIVAL2_GROUND_TO_AIR_OPTION_V2_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_validation": baseline,
        "best_block": best_block,
        "best_validation": best_evaluation,
        "untouched_test_opened": test is not None,
        "untouched_test": test,
        "stop_reason": stop_reason,
        "controlled_pass": controlled_pass,
        "diagnostic_parent_unchanged": capability.sha256_file(PARENT) == PARENT_SHA256,
        "protected_v23_unchanged": (
            capability.sha256_file(BLUE) == BLUE_SHA256
            and capability.sha256_file(ORANGE) == ORANGE_SHA256
        ),
        "critic_unchanged": True,
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
    parser.add_argument("--worlds-per-side", type=int, default=1_024)
    parser.add_argument("--evaluation-worlds-per-side", type=int, default=1_024)
    parser.add_argument("--test-worlds-per-side", type=int, default=2_048)
    parser.add_argument("--maximum-blocks", type=int, default=160)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
