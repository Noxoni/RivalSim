"""Run acquisition-gated ground self-play PPO from exact Unified V5."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_unified_ground_selfplay_ppo_v1 import (  # noqa: E402
    EXPLORATION_CONTRACT,
    EXPLORATION_CONTRACT_HASH,
    SOURCE,
    SOURCE_SHA256,
    append_jsonl,
    exploration_for_update,
    sha256_file,
    source_payload,
    state_dict_sha256,
    write_json,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    CAR_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
    REWARD_ACQUISITION_120_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentPPOCorruption  # noqa: E402
from rivalsim.rival2_recurrent_training import Rival2RecurrentTrainer  # noqa: E402
from rivalsim.rival2_unified_policy import (  # noqa: E402
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)

FORMAT = "RIVAL2_UNIFIED_GROUND_CURRICULUM_PPO_V2"
CHECKPOINT_FORMAT = f"{FORMAT}_CHECKPOINT"
RESULTS = ROOT / "results/rival2/unified_ground_curriculum_ppo_v2"
AUTHORITY = RESULTS / "authority.json"
CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/unified_ground_curriculum_ppo_v2"
    / "rival2_unified_ground_curriculum_ppo_v2.pt"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/unified-ground-curriculum-ppo-v2")
WORLD_COUNT = 32_768
TOTAL_ACCEPTED_UPDATES = 300
SNAPSHOT_INTERVAL = 30
PHASE_A_FIRST_EVALUATION = 30
PHASE_A_EVALUATION_INTERVAL = 15
PHASE_A_MAXIMUM_UPDATES = 120
PHASE_A_POLICY_LR = 5.0e-7
PHASE_B_POLICY_LR = 1.0e-6
CRITIC_LR = 3.0e-4
EVALUATION_WORLDS = 2_048
EVALUATION_TICKS = 1_920
MINIMUM_COMPLETED_AGENT_EPISODES = 2_048
MINIMUM_TOUCHES_PER_MINUTE = 2.0
MINIMUM_EPISODE_TOUCH_FRACTION = 0.5
CONSECUTIVE_PASSING_EVALUATIONS = 2
CAMPAIGN_SEED = 2026090312
EVALUATION_SEED = 2026090313
_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
_NO_TOUCH_AGE_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")
_BALL_VELOCITY_Y_INDEX = OBS_FIELD_NAMES.index("ball.linear_velocity.y")
_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == f"{FORMAT}_AUTHORITY",
        "source": authority.get("source", {}).get("sha256") == SOURCE_SHA256,
        "source_file": sha256_file(SOURCE) == SOURCE_SHA256,
        "exploration": authority.get("exploration", {}).get("contract_sha256")
        == EXPLORATION_CONTRACT_HASH,
        "ppo": authority.get("ppo", {}).get("contract_sha256")
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "ppo_version": authority.get("ppo", {}).get("version")
        == RIVAL2_PPO_120HZ_V1,
        "worlds": authority.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "total_updates": authority.get("campaign", {}).get(
            "accepted_updates_total"
        )
        == TOTAL_ACCEPTED_UPDATES,
        "phase_a_reward": authority.get("phase_a", {}).get(
            "reward_contract_sha256"
        )
        == REWARD_ACQUISITION_120_V1_CONTRACT_HASH,
        "phase_b_reward": authority.get("phase_b", {}).get(
            "reward_contract_sha256"
        )
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "kl_telemetry_only": authority.get("ppo", {}).get("kl_policy")
        == "telemetry_only_no_rejection_or_retry",
        "pure_selfplay": authority.get("opponents", {}).get("current_selfplay")
        == 1.0,
        "abandoned_v1_absent": authority.get("integrity", {}).get(
            "abandoned_v1_update_loaded"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ground curriculum authority mismatch: {checks}")
    return authority


def phase_spec(phase: str) -> tuple[str, float]:
    if phase == "unified_ground_acquisition_v2":
        return RIVAL2_REWARD_ACQUISITION_120_V1_VERSION, PHASE_A_POLICY_LR
    if phase == "unified_ground_gameplay_v2":
        return RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION, PHASE_B_POLICY_LR
    raise ValueError(f"unsupported curriculum phase: {phase}")


def make_env(
    collision_root: Path,
    *,
    worlds: int,
    reward_version: str,
    seed: int,
) -> Rival2Env:
    geometry = ArenaGeometry.load_soccar(collision_root)
    return Rival2Env(
        worlds,
        str(collision_root),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry),
        device="cuda:0",
        seed=seed,
        car_visitation_order="a_then_b",
        reward_version=reward_version,
        episode_version=RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
    )


def configure_optimizer(trainer: Rival2RecurrentTrainer, policy_lr: float) -> None:
    critic_parameters = tuple(trainer.model.critic.parameters())
    critic_ids = {id(parameter) for parameter in critic_parameters}
    policy_parameters = tuple(
        parameter
        for parameter in trainer.model.parameters()
        if id(parameter) not in critic_ids
    )
    trainer.optimizer = torch.optim.Adam(
        [
            {"name": "policy", "params": policy_parameters, "lr": policy_lr},
            {"name": "critic", "params": critic_parameters, "lr": CRITIC_LR},
        ]
    )


def build_trainer(
    collision_root: Path,
    *,
    worlds: int,
    phase: str,
    model_state: dict[str, torch.Tensor] | None = None,
) -> tuple[Rival2RecurrentTrainer, dict[str, Any]]:
    authority = load_authority()
    source = source_payload()
    config = Rival2UnifiedPolicyConfig(**source["policy_config"])
    if source.get("policy_config_sha256") != config.content_hash:
        raise RuntimeError("Unified V5 policy configuration hash mismatch")
    model = Rival2UnifiedActorCritic(config)
    model.load_state_dict(source["model"] if model_state is None else model_state)
    reward_version, policy_lr = phase_spec(phase)
    ppo = replace(
        rival2_ppo_120hz_config(),
        learning_rate=policy_lr,
        epochs=1,
    )
    trainer = Rival2RecurrentTrainer(
        make_env(
            collision_root,
            worlds=worlds,
            reward_version=reward_version,
            seed=CAMPAIGN_SEED,
        ),
        policy_config=config,
        ppo_config=ppo,
        phase=phase,
        source_identity=authority["source"],
        seed=CAMPAIGN_SEED,
        model=model,
        checkpoint_format=CHECKPOINT_FORMAT,
        lineage="Unified Capability V5 -> Ground Curriculum PPO V2",
    )
    configure_optimizer(trainer, policy_lr)
    return trainer, source


def initialize_fresh(
    collision_root: Path,
    *,
    worlds: int,
) -> tuple[Rival2RecurrentTrainer, dict[str, Any]]:
    trainer, source = build_trainer(
        collision_root,
        worlds=worlds,
        phase="unified_ground_acquisition_v2",
    )
    source_model_hash = state_dict_sha256(source["model"])
    loaded_hash = state_dict_sha256(trainer.model.state_dict())
    trainer.phase_transition = {
        "format": f"{FORMAT}_SOURCE_TRANSITION",
        "source_model_tensor_sha256": source_model_hash,
        "loaded_model_tensor_sha256": loaded_hash,
        "model_exact_before_first_update": source_model_hash == loaded_hash,
        "source_optimizer_loaded": False,
        "abandoned_v1_update_loaded": False,
        "fresh_ppo_optimizer": True,
        "fresh_rng_and_counters": True,
        "authority_sha256": sha256_file(AUTHORITY),
    }
    return trainer, source


def optimizer_lrs(trainer: Rival2RecurrentTrainer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(trainer.optimizer.param_groups)
    }


def preflight(
    trainer: Rival2RecurrentTrainer,
    source: dict[str, Any],
    *,
    worlds: int,
    resuming: bool,
) -> dict[str, Any]:
    reward_version, policy_lr = phase_spec(trainer.phase)
    lrs = optimizer_lrs(trainer)
    checks = {
        "source_sha256_exact": sha256_file(SOURCE) == SOURCE_SHA256,
        "source_identity_exact": trainer.source_identity == load_authority()["source"],
        "source_optimizer_not_loaded": trainer.phase_transition is not None
        and trainer.phase_transition.get("source_optimizer_loaded") is False,
        "abandoned_v1_update_not_loaded": trainer.phase_transition is not None
        and trainer.phase_transition.get("abandoned_v1_update_loaded") is False,
        "counter_state_valid": trainer.accepted_updates_total
        == trainer.policy_version
        and (
            trainer.accepted_updates_total > 0
            if resuming
            else trainer.accepted_updates_total == 0
        ),
        "policy_optimizer_lr_exact": lrs.get("policy") == policy_lr,
        "critic_optimizer_lr_exact": lrs.get("critic") == CRITIC_LR,
        "value_loss_isolated": hasattr(trainer.model, "isolated_value"),
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "worlds_exact": worlds == WORLD_COUNT,
        "native_120hz": trainer.env.physics_hz == trainer.env.policy_hz == 120,
        "phase_reward_exact": trainer.env.reward_version == reward_version,
        "runtime_contracts_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(reward_version),
        "named_mechanics_hot_path_absent": trainer.env.world.gameplay_v3 is None,
        "pure_current_selfplay": True,
        "kl_telemetry_only": True,
    }
    return {
        "format": f"{FORMAT}_PREFLIGHT",
        "created_utc": utc_now(),
        "resuming": resuming,
        "phase": trainer.phase,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "format": source.get("format"),
            "model_tensor_sha256": state_dict_sha256(source["model"]),
        },
        "phase_transition": trainer.phase_transition,
        "contracts": dict(trainer.env.contract_hashes),
        "ppo_config": asdict(trainer.ppo_config),
        "optimizer_group_lrs": lrs,
        "exploration_contract": EXPLORATION_CONTRACT,
        "exploration_contract_sha256": EXPLORATION_CONTRACT_HASH,
    }


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().item())
        for name, value in metrics.items()
        if value.numel() == 1
    }


def checkpoint_record(
    trainer: Rival2RecurrentTrainer,
    path: Path,
) -> dict[str, Any]:
    trainer.save_checkpoint(path, include_optimizer=True)
    return {
        "accepted_updates_total": trainer.accepted_updates_total,
        "phase": trainer.phase,
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "path": path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "total_agent_samples": trainer.total_agent_samples,
        "optimizer_group_lrs": optimizer_lrs(trainer),
        "exploration": None
        if trainer.exploration is None
        else trainer.exploration.as_dict(),
    }


@torch.no_grad()
def deterministic_selfplay_evaluation(
    trainer: Rival2RecurrentTrainer,
    collision_root: Path,
) -> dict[str, Any]:
    env = make_env(
        collision_root,
        worlds=EVALUATION_WORLDS,
        reward_version=RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        seed=EVALUATION_SEED + trainer.phase_accepted_updates,
    )
    observation = env.observation
    hidden = trainer.model.initial_hidden(EVALUATION_WORLDS * 2, device=env.device)
    reset_before = torch.ones(
        (EVALUATION_WORLDS, 2), dtype=torch.bool, device=env.device
    )
    episode_touched = torch.zeros_like(reset_before)
    touch_events = torch.zeros((), dtype=torch.int64, device=env.device)
    goalward_touches = torch.zeros((), dtype=torch.int64, device=env.device)
    goals = torch.zeros((), dtype=torch.int64, device=env.device)
    no_touch_resets = torch.zeros((), dtype=torch.int64, device=env.device)
    completed_agent_episodes = torch.zeros((), dtype=torch.int64, device=env.device)
    touched_agent_episodes = torch.zeros((), dtype=torch.int64, device=env.device)
    speed_sum = torch.zeros((), dtype=torch.float64, device=env.device)
    throttle_sum = torch.zeros((), dtype=torch.float64, device=env.device)
    boost_sum = torch.zeros((), dtype=torch.float64, device=env.device)
    trainer.model.eval()
    for _ in range(EVALUATION_TICKS):
        actor, _value, hidden_after = trainer.model(
            observation.reshape(-1, trainer.policy_config.obs_dim),
            hidden,
            reset_before=reset_before.reshape(-1),
        )
        action = deterministic_unified_action(actor).reshape(EVALUATION_WORLDS, 2, 8)
        transition = env.step(action)
        touch = transition.transition_observation[..., _TOUCH_INDEX] > 0.5
        episode_touched.logical_or_(touch)
        touch_events += touch.sum()
        goalward_touches += (
            touch
            & (
                transition.transition_observation[..., _BALL_VELOCITY_Y_INDEX]
                > 0.0
            )
        ).sum()
        goals += transition.terminated.sum()
        no_touch_resets += (
            transition.truncated
            & (
                transition.transition_observation[:, 0, _NO_TOUCH_AGE_INDEX]
                >= 1.0 - 1.0e-6
            )
        ).sum()
        speed_sum += torch.linalg.vector_norm(
            observation[
                ...,
                _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3,
            ],
            dim=-1,
        ).sum(dtype=torch.float64)
        throttle_sum += action[..., 0].sum(dtype=torch.float64)
        boost_sum += action[..., 6].sum(dtype=torch.float64)
        reset_agent = transition.reset_mask[:, None].expand(-1, 2)
        completed_agent_episodes += reset_agent.sum()
        touched_agent_episodes += (episode_touched & reset_agent).sum()
        episode_touched.masked_fill_(reset_agent, False)
        hidden = hidden_after.masked_fill(
            reset_agent.reshape(1, EVALUATION_WORLDS * 2, 1), 0.0
        )
        reset_before = reset_agent
        observation = transition.observation
    player_minutes = EVALUATION_WORLDS * 2 * EVALUATION_TICKS / (120.0 * 60.0)
    action_count = EVALUATION_WORLDS * 2 * EVALUATION_TICKS
    touches = int(touch_events.item())
    completed = int(completed_agent_episodes.item())
    touched = int(touched_agent_episodes.item())
    result = {
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "worlds": EVALUATION_WORLDS,
        "physics_ticks": EVALUATION_TICKS,
        "touch_events": touches,
        "touches_per_minute": touches / player_minutes,
        "goalward_touch_fraction": int(goalward_touches.item()) / max(1, touches),
        "goals": int(goals.item()),
        "no_touch_resets": int(no_touch_resets.item()),
        "completed_agent_episodes": completed,
        "episode_touch_fraction": touched / max(1, completed),
        "mean_speed_uu_per_second": (
            float(speed_sum.item()) / action_count * CAR_LINEAR_SPEED_SCALE
        ),
        "mean_throttle": float(throttle_sum.item()) / action_count,
        "boost_fraction": float(boost_sum.item()) / action_count,
    }
    result["transition_gate_pass"] = (
        result["completed_agent_episodes"] >= MINIMUM_COMPLETED_AGENT_EPISODES
        and result["touches_per_minute"] >= MINIMUM_TOUCHES_PER_MINUTE
        and result["episode_touch_fraction"] >= MINIMUM_EPISODE_TOUCH_FRACTION
    )
    del env
    return result


def transition_to_gameplay(
    trainer: Rival2RecurrentTrainer,
    source: dict[str, Any],
    collision_root: Path,
    *,
    worlds: int,
) -> Rival2RecurrentTrainer:
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in trainer.model.state_dict().items()
    }
    counters = {
        "accepted_updates_total": trainer.accepted_updates_total,
        "policy_version": trainer.policy_version,
        "total_agent_samples": trainer.total_agent_samples,
        "physical_physics_ticks_experienced": (
            trainer.physical_physics_ticks_experienced
        ),
    }
    policy_generator_state = trainer.policy_generator.get_state().cpu()
    shuffle_generator_state = trainer.shuffle_generator.get_state().cpu()
    acquisition_hash = state_dict_sha256(model_state)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    next_trainer, _unused_source = build_trainer(
        collision_root,
        worlds=worlds,
        phase="unified_ground_gameplay_v2",
        model_state=model_state,
    )
    next_trainer.accepted_updates_total = counters["accepted_updates_total"]
    next_trainer.policy_version = counters["policy_version"]
    next_trainer.total_agent_samples = counters["total_agent_samples"]
    next_trainer.physical_physics_ticks_experienced = counters[
        "physical_physics_ticks_experienced"
    ]
    next_trainer.phase_accepted_updates = 0
    next_trainer.policy_generator.set_state(policy_generator_state)
    next_trainer.shuffle_generator.set_state(shuffle_generator_state)
    next_trainer.phase_transition = {
        "format": f"{FORMAT}_PHASE_TRANSITION",
        "created_utc": utc_now(),
        "reward_from": RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        "reward_to": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        "accepted_update_boundary": counters["accepted_updates_total"],
        "acquisition_model_tensor_sha256": acquisition_hash,
        "loaded_gameplay_model_tensor_sha256": state_dict_sha256(
            next_trainer.model.state_dict()
        ),
        "model_exact": acquisition_hash
        == state_dict_sha256(next_trainer.model.state_dict()),
        "fresh_ppo_optimizer": True,
        "source_optimizer_loaded": False,
        "abandoned_v1_update_loaded": False,
        "source": copy.deepcopy(load_authority()["source"]),
    }
    del source
    return next_trainer


def load_resume(
    path: Path,
    collision_root: Path,
    *,
    worlds: int,
) -> tuple[Rival2RecurrentTrainer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint format mismatch")
    trainer, source = build_trainer(
        collision_root,
        worlds=worlds,
        phase=str(payload["phase"]),
    )
    trainer.load_checkpoint(path)
    return trainer, source


def save_snapshot(
    trainer: Rival2RecurrentTrainer,
    run_dir: Path,
    manifest: dict[str, Any],
) -> None:
    record = checkpoint_record(
        trainer,
        run_dir
        / "snapshots"
        / f"ground_curriculum_u{trainer.accepted_updates_total:04d}.pt",
    )
    manifest["snapshots"] = [
        previous
        for previous in manifest["snapshots"]
        if previous["accepted_updates_total"] != trainer.accepted_updates_total
    ] + [record]
    manifest["snapshots"].sort(key=lambda item: item["accepted_updates_total"])


def run(args: argparse.Namespace) -> int:
    if args.worlds != WORLD_COUNT or args.target_updates != TOTAL_ACCEPTED_UPDATES:
        raise ValueError("V2 authority freezes worlds=32768 and accepted updates=300")
    run_dir = Path(args.run_dir).resolve()
    collision_root = Path(args.collision_root).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        trainer, source = load_resume(
            Path(args.resume).resolve(), collision_root, worlds=args.worlds
        )
    else:
        trainer, source = initialize_fresh(collision_root, worlds=args.worlds)
    preflight_payload = preflight(
        trainer,
        source,
        worlds=args.worlds,
        resuming=bool(args.resume),
    )
    write_json(
        RESULTS / ("resume_preflight.json" if args.resume else "preflight.json"),
        preflight_payload,
    )
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"ground curriculum preflight failed: {preflight_payload}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists() and not args.resume:
        raise RuntimeError("training evidence exists; use --resume explicitly")
    manifest_path = RESULTS / "snapshot_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "format": f"{FORMAT}_SNAPSHOT_MANIFEST",
            "source_sha256": SOURCE_SHA256,
            "snapshots": [],
            "phase_a_evaluations": [],
        }
    )
    pass_streak = 0
    if manifest["phase_a_evaluations"]:
        for item in reversed(manifest["phase_a_evaluations"]):
            if not item["transition_gate_pass"]:
                break
            pass_streak += 1
    rolling = run_dir / "rolling.pt"
    hard_failure: dict[str, Any] | None = None
    started = time.monotonic()
    while trainer.accepted_updates_total < TOTAL_ACCEPTED_UPDATES:
        trainer.set_exploration(
            exploration_for_update(trainer.accepted_updates_total + 1)
        )
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_seconds = time.monotonic() - rollout_started
        gameplay = dict(trainer.last_rollout_metrics)
        update_started = time.monotonic()
        try:
            update_metrics = trainer.update(rollout)
        except Rival2RecurrentPPOCorruption as error:
            hard_failure = {
                "format": f"{FORMAT}_HARD_FAILURE",
                "created_utc": utc_now(),
                "phase": trainer.phase,
                "accepted_updates_total": trainer.accepted_updates_total,
                "diagnostics": dict(error.diagnostics),
                "kl_caused_stop": False,
            }
            write_json(RESULTS / "hard_failure.json", hard_failure)
            checkpoint_record(trainer, rolling)
            break
        ppo = scalar_metrics(update_metrics)
        row = {
            "created_utc": utc_now(),
            "phase": trainer.phase,
            "accepted_updates_total": trainer.accepted_updates_total,
            "phase_accepted_updates": trainer.phase_accepted_updates,
            "elapsed_seconds": time.monotonic() - started,
            "rollout_seconds": rollout_seconds,
            "update_seconds": time.monotonic() - update_started,
            "total_agent_samples": trainer.total_agent_samples,
            "exploration": trainer.exploration.as_dict(),
            "optimizer_group_lrs": optimizer_lrs(trainer),
            "ppo": ppo,
            "gameplay": gameplay,
        }
        append_jsonl(curve, row)
        checkpoint_record(trainer, rolling)
        if trainer.accepted_updates_total % SNAPSHOT_INTERVAL == 0:
            save_snapshot(trainer, run_dir, manifest)
            write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "global_update": trainer.accepted_updates_total,
                    "phase": trainer.phase,
                    "phase_update": trainer.phase_accepted_updates,
                    "kl": ppo.get("completed_update_mean_kl"),
                    "touches_per_minute": gameplay["touches_per_minute"],
                    "goals": gameplay["goal_events"],
                    "reward_abs": gameplay["reward_absolute_sum"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if trainer.phase != "unified_ground_acquisition_v2":
            continue
        should_evaluate = (
            trainer.phase_accepted_updates >= PHASE_A_FIRST_EVALUATION
            and trainer.phase_accepted_updates % PHASE_A_EVALUATION_INTERVAL == 0
        )
        if should_evaluate:
            evaluation = deterministic_selfplay_evaluation(trainer, collision_root)
            manifest["phase_a_evaluations"].append(evaluation)
            write_json(
                RESULTS
                / f"phase_a_eval_u{trainer.phase_accepted_updates:04d}.json",
                evaluation,
            )
            write_json(manifest_path, manifest)
            pass_streak = pass_streak + 1 if evaluation["transition_gate_pass"] else 0
            print(json.dumps({"phase_a_evaluation": evaluation}, sort_keys=True))
            if pass_streak >= CONSECUTIVE_PASSING_EVALUATIONS:
                acquisition_checkpoint = checkpoint_record(
                    trainer,
                    run_dir / "phase_a_acquisition_final.pt",
                )
                write_json(
                    RESULTS / "phase_a_summary.json",
                    {
                        "format": f"{FORMAT}_PHASE_A_SUMMARY",
                        "created_utc": utc_now(),
                        "checkpoint": acquisition_checkpoint,
                        "passing_evaluations": pass_streak,
                        "transition": True,
                    },
                )
                trainer = transition_to_gameplay(
                    trainer,
                    source,
                    collision_root,
                    worlds=args.worlds,
                )
                write_json(RESULTS / "phase_transition.json", trainer.phase_transition)
                checkpoint_record(trainer, rolling)
                continue
        if trainer.phase_accepted_updates >= PHASE_A_MAXIMUM_UPDATES:
            hard_failure = {
                "format": f"{FORMAT}_CAPABILITY_BLOCK",
                "created_utc": utc_now(),
                "reason": "deterministic_selfplay_ball_acquisition_gate_not_met",
                "phase_a_updates": trainer.phase_accepted_updates,
                "last_evaluation": manifest["phase_a_evaluations"][-1]
                if manifest["phase_a_evaluations"]
                else None,
                "reward_or_capability_affecting": True,
            }
            write_json(RESULTS / "capability_block.json", hard_failure)
            checkpoint_record(trainer, rolling)
            break

    if hard_failure is None:
        final_record = checkpoint_record(trainer, CHECKPOINT)
        manifest["final"] = final_record
        write_json(manifest_path, manifest)
    else:
        final_record = checkpoint_record(trainer, rolling)
    summary = {
        "format": f"{FORMAT}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "verdict": "BLOCKED" if hard_failure else "PASS",
        "accepted_updates_total": trainer.accepted_updates_total,
        "target_updates": TOTAL_ACCEPTED_UPDATES,
        "phase": trainer.phase,
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "final_checkpoint": final_record,
        "hard_failure": hard_failure,
        "stop_reason": "hard_or_capability_guard" if hard_failure else "target_reached",
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--collision-root",
        default=os.environ.get(
            "RIVALSIM_COLLISION_DIR", "G:/dev/RLBot-Rival/bot/collision_meshes"
        ),
    )
    result.add_argument("--worlds", type=int, default=WORLD_COUNT)
    result.add_argument("--target-updates", type=int, default=TOTAL_ACCEPTED_UPDATES)
    result.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    result.add_argument("--resume")
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
