"""Closed-loop-gated Rival training rooted in the strongest human-derived policy.

This campaign deliberately treats gameplay as the selection authority.  It combines
current-policy self-play, frozen Nexto opponents, and a small bounded replay of the
reviewed human demonstrations.  KL is telemetry only; nonfinite state remains a hard
failure.  The best checkpoint is selected by deterministic Nexto play while retaining
a bounded human-action validation floor.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_human_behavior_cloning_v1 import (  # noqa: E402
    HumanSplit,
    _load_adapter,
    _load_config,
    _load_human_split,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    Rival2MixedPPOSafetyConfig,
    mixed_optimizer_learning_rates,
)
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_CURRENT,
    OPPONENT_NEXTO,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
)
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

SOURCE = ROOT / "checkpoints/rival2/human_bc_ppo_v1/rival2_human_bc_ppo_10h.pt"
SOURCE_SHA256 = "C75F08F91E2FF29D78C15A67B5DE85BDF0B0A548E0268FFDB0CF29C272750067"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v1/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/codex_autonomous_v1"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v1")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")
SEED = 2_026_090_201
WORLD_COUNT = 32_768
TARGET_UPDATES = 300
EVALUATION_INTERVAL = 5
EVALUATION_WORLDS_PER_SIDE = 128
POLICY_LR = 2.0e-6
CRITIC_LR = 1.0e-5
EXPLORATION_SIGMA = 0.10
EXPLORATION_BUTTON_TEMPERATURE = 1.50
HUMAN_REPLAY_STEPS = 8
HUMAN_GAMEPLAY_BATCH = 1024
HUMAN_MECHANIC_BATCH = 512
HUMAN_VALIDATION_GAMEPLAY_RMSE_CEILING = 0.61
HUMAN_VALIDATION_MECHANIC_RMSE_CEILING = 0.62
NEXTO_WIN_TARGET = 0.55


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().item())
        for name, value in metrics.items()
        if value.numel() == 1
    }


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == "RIVAL2_CODEX_AUTONOMOUS_V1_AUTHORITY",
        "source": authority.get("source", {}).get("sha256") == SOURCE_SHA256,
        "reward": authority.get("reward", {}).get("contract_sha256")
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo": authority.get("ppo", {}).get("contract_sha256")
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "worlds": authority.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "kl_telemetry_only": authority.get("ppo", {}).get("kl_telemetry_only") is True,
        "current_probability": authority.get("opponents", {}).get("current") == 0.5,
        "nexto_probability": authority.get("opponents", {}).get("nexto") == 0.5,
        "wisp_probability": authority.get("opponents", {}).get("wisp") == 0.0,
        "human_replay_steps": authority.get("human_replay", {}).get("steps_per_update")
        == HUMAN_REPLAY_STEPS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"campaign authority mismatch: {checks}")
    return authority


def build_trainer(
    collision_dir: Path,
    *,
    worlds: int,
    device: str,
) -> tuple[Rival2OpponentCurriculumTrainer, dict[str, Any]]:
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("functional Human-BC PPO source checkpoint changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    policy_config = Rival2PolicyConfig(**source["policy_config"])
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    kickoff = (np.arange(worlds, dtype=np.int32) + SEED) % 5
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=policy_config,
        ppo_config=rival2_ppo_120hz_config(),
        self_play_config=Rival2SelfPlayConfig(
            historical_chance=0.0,
            historical_pool_bound=1,
        ),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            nexto_probability=0.5,
            wisp_probability=0.0,
            current_probability=0.5,
            historical_probability=0.0,
            seed=SEED ^ 0xC0DE,
        ),
        seed=SEED,
    )
    trainer.model.load_state_dict(source["model"], strict=True)
    trainer.policy_version = int(source["policy_version"])
    trainer.iteration = int(source["iteration"])
    trainer.total_agent_samples = int(source["total_agent_samples"])
    trainer.source_30hz_agent_decision_samples = int(
        source.get("sample_accounting", {}).get("source_30hz_agent_decisions", 0)
    )
    trainer.curriculum_transition = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V1",
        "created_utc": utc_now(),
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "format": source.get("format"),
            "iteration": int(source["iteration"]),
            "policy_version": int(source["policy_version"]),
            "model_tensor_sha256": tensor_tree_sha256(source["model"]),
            "optimizer_loaded": False,
        },
        "selection_authority": (
            "deterministic 256-episode Nexto score with human validation floors"
        ),
        "reward_version": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        "ppo_optimizer": "fresh split Adam at conservative fixed rates",
        "kl_policy": "telemetry_only_no_KL_rejection_or_rollback",
        "human_replay": "reviewed frozen train split only; validation is read-only",
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY),
        },
    }
    trainer.initialize_curriculum_assignments()
    safety = Rival2MixedPPOSafetyConfig(
        initial_policy_learning_rate=POLICY_LR,
        critic_learning_rate=CRITIC_LR,
        minimum_policy_learning_rate=POLICY_LR,
    )
    trainer.enable_safe_mixed_ppo(safety)
    override = HybridDistributionOverride(
        analog_log_std=math.log(EXPLORATION_SIGMA),
        button_temperature=EXPLORATION_BUTTON_TEMPERATURE,
    )
    trainer.set_exploration_override(override)
    return trainer, source


def load_human_data(
    *, device: str
) -> tuple[HumanSplit, HumanSplit, Rival2ActorCritic, dict[str, Any]]:
    config, config_identity = _load_config()
    adapter, _payload, adapter_identity = _load_adapter(config, device)
    source_root = Path(os.environ["APPDATA"]) / "bakkesmod/bakkesmod/data/rival2/human_demos"
    train = _load_human_split(
        "train",
        config=config,
        adapter=adapter,
        source_root=source_root,
        device=device,
    )
    validation = _load_human_split(
        "validation",
        config=config,
        adapter=adapter,
        source_root=source_root,
        device=device,
    )
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    teacher = Rival2ActorCritic(Rival2PolicyConfig(**source["policy_config"])).to(device)
    teacher.load_state_dict(source["model"])
    teacher.eval().requires_grad_(False)
    return train, validation, teacher, {
        "frozen_config": config_identity,
        "adapter": adapter_identity,
        "train_action_sha256": train.action_sha256,
        "validation_action_sha256": validation.action_sha256,
        "train_sequences_sha256": train.source_sequences_sha256,
        "validation_sequences_sha256": validation.source_sequences_sha256,
        "test_loaded": False,
    }


@torch.no_grad()
def precompute_teacher_actor(
    teacher: Rival2ActorCritic,
    observation: torch.Tensor,
    *,
    device: str,
    batch_size: int = 8192,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], batch_size):
        actor, _ = teacher(observation[start : start + batch_size].to(device))
        rows.append(actor.cpu())
    return torch.cat(rows)


@torch.no_grad()
def human_validation(
    model: Rival2ActorCritic,
    validation: HumanSplit,
    *,
    device: str,
) -> dict[str, Any]:
    model.eval()
    output: dict[str, Any] = {}
    for name, observation, action in (
        ("gameplay", validation.gameplay_observation, validation.gameplay_action),
        ("mechanic", validation.mechanic_observation, validation.mechanic_action),
    ):
        actors: list[torch.Tensor] = []
        for start in range(0, observation.shape[0], 8192):
            actor, _ = model(observation[start : start + 8192].to(device))
            actors.append(actor.cpu())
        actor = torch.cat(actors)
        output[name] = action_metric_summary(actor, action)
        output[name]["finite"] = bool(torch.isfinite(actor).all().item())
    output["eligible"] = bool(
        output["gameplay"]["complete_action_rmse"]
        <= HUMAN_VALIDATION_GAMEPLAY_RMSE_CEILING
        and output["mechanic"]["complete_action_rmse"]
        <= HUMAN_VALIDATION_MECHANIC_RMSE_CEILING
        and output["gameplay"]["finite"]
        and output["mechanic"]["finite"]
    )
    return output


def human_replay(
    trainer: Rival2OpponentCurriculumTrainer,
    train: HumanSplit,
    teacher_actor_gameplay: torch.Tensor,
    teacher_actor_mechanic: torch.Tensor,
    mechanic_sampler: MechanicHierarchySampler,
    generator: torch.Generator,
) -> dict[str, float]:
    model = trainer.model
    model.train()
    critic_before = {
        name: value.detach().cpu().clone()
        for name, value in model.critic.state_dict().items()
    }
    sums = {"loss": 0.0, "analog": 0.0, "buttons": 0.0, "log_std": 0.0}
    for _ in range(HUMAN_REPLAY_STEPS):
        gameplay_index = torch.randint(
            train.gameplay_observation.shape[0],
            (HUMAN_GAMEPLAY_BATCH,),
            generator=generator,
        )
        mechanic_index = mechanic_sampler.sample(HUMAN_MECHANIC_BATCH)
        observation = torch.cat(
            (
                train.gameplay_observation.index_select(0, gameplay_index),
                train.mechanic_observation.index_select(0, mechanic_index),
            )
        ).to(trainer.device)
        action = torch.cat(
            (
                train.gameplay_action.index_select(0, gameplay_index),
                train.mechanic_action.index_select(0, mechanic_index),
            )
        ).to(trainer.device)
        teacher_actor = torch.cat(
            (
                teacher_actor_gameplay.index_select(0, gameplay_index),
                teacher_actor_mechanic.index_select(0, mechanic_index),
            )
        ).to(trainer.device)
        student_actor, _ = model(observation)
        objective = human_behavior_cloning_objective(
            student_actor,
            teacher_actor,
            action,
            smooth_l1_beta=0.1,
            analog_weight=1.0,
            button_weight=0.25,
            log_std_weight=0.05,
            policy_config=trainer.policy_config,
        )
        if not bool(torch.isfinite(objective.loss).item()):
            raise Rival2PolicyDisplacementRejected({"reason": "nonfinite_human_replay_loss"})
        trainer.optimizer.zero_grad(set_to_none=True)
        objective.loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [*model.trunk.parameters(), *model.actor.parameters()], 0.5
        )
        if not bool(torch.isfinite(gradient).item()):
            raise Rival2PolicyDisplacementRejected(
                {"reason": "nonfinite_human_replay_gradient"}
            )
        trainer.optimizer.step()
        sums["loss"] += float(objective.loss.detach().item())
        sums["analog"] += float(objective.analog_smooth_l1.detach().item())
        sums["buttons"] += float(objective.button_bce.detach().item())
        sums["log_std"] += float(objective.log_std_retention.detach().item())
    if any(
        not torch.equal(critic_before[name], value.detach().cpu())
        for name, value in model.critic.state_dict().items()
    ):
        raise RuntimeError("human replay changed the critic")
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise Rival2PolicyDisplacementRejected({"reason": "nonfinite_human_replay_parameter"})
    return {name: value / HUMAN_REPLAY_STEPS for name, value in sums.items()}


def checkpoint(
    trainer: Rival2OpponentCurriculumTrainer,
    path: Path,
    *,
    campaign_step: int,
    human_generator: torch.Generator,
    best: dict[str, Any],
) -> dict[str, Any]:
    trainer.curriculum_transition["codex_autonomous_state"] = {
        "campaign_step": int(campaign_step),
        "human_generator_state": human_generator.get_state(),
        "best": copy.deepcopy(best),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "path": path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "campaign_step": int(campaign_step),
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "total_agent_samples": trainer.total_agent_samples,
    }


def run_nexto_evaluation(
    checkpoint_path: Path,
    *,
    campaign_step: int,
    run_dir: Path,
    device: str,
    collision_dir: Path,
    worlds_per_side: int = EVALUATION_WORLDS_PER_SIDE,
) -> dict[str, Any]:
    digest = sha256_file(checkpoint_path)
    label = f"codex_autonomous_u{campaign_step:04d}"
    work_dir = run_dir / "evaluations" / label
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/run_rival2_opponent_curriculum_v1.py"),
        "--collision-dir",
        str(collision_dir),
        "--device",
        device,
        "--work-dir",
        str(work_dir),
        "--evaluation-single-checkpoint",
        str(checkpoint_path),
        "--evaluation-single-sha256",
        digest,
        "--evaluation-single-label",
        label,
        "--evaluation-single-opponent",
        "Nexto",
        "--evaluation-single-mode",
        "deterministic",
        "--evaluation-worlds-per-side",
        str(worlds_per_side),
    ]
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nexto evaluation failed with {completed.returncode}: "
            f"{stderr_path.read_text(encoding='utf-8')[-4000:]}"
        )
    result_path = work_dir / f"evaluation_{label}_nexto_deterministic.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    overall = result["by_rival_side"]["overall"]
    return {
        "campaign_step": int(campaign_step),
        "checkpoint_sha256": digest,
        "episodes": int(overall["episodes"]),
        "goals_for": int(overall["goals_for"]),
        "goals_against": int(overall["goals_against"]),
        "draws": int(overall["no_goal_episodes"]),
        "win_rate": float(overall["goals_for"] / max(overall["episodes"], 1)),
        "touches": int(overall["touches"]["Rival"]),
        "opponent_touches": int(overall["touches"]["Nexto"]),
        "first_touches": int(overall["first_touch"]["Rival"]),
        "no_touch_episodes": int(overall["no_touch_fraction"]["count"]),
        "hard_timeouts": int(overall["hard_timeout_fraction"]["count"]),
        "mean_speed_uu_per_s": float(
            overall["movement_controller"]["Rival"]["mean_speed_uu_per_s"]
        ),
        "full_result": str(result_path),
    }


def candidate_rank(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(evaluation["goals_for"]) - int(evaluation["goals_against"]),
        int(evaluation["goals_for"]),
        int(evaluation["touches"]),
        int(evaluation["first_touches"]),
    )


def preflight(
    trainer: Rival2OpponentCurriculumTrainer,
    source: dict[str, Any],
    human_identity: dict[str, Any],
    *,
    worlds: int,
) -> dict[str, Any]:
    family_count = torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    checks = {
        "source_sha256_exact": sha256_file(SOURCE) == SOURCE_SHA256,
        "source_model_loaded_exact": tensor_tree_sha256(trainer.model.state_dict())
        == tensor_tree_sha256(source["model"]),
        "fresh_optimizer_state_empty": len(trainer.optimizer.state) == 0,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "ppo_contract_exact": trainer.ppo_config.content_hash
        == rival2_ppo_120hz_config().content_hash,
        "world_count": worlds == WORLD_COUNT,
        "physics_and_policy_120hz": trainer.env.physics_hz == 120
        and trainer.env.policy_hz == 120,
        "current_and_nexto_only": family_count[1] == 0 and family_count[3] == 0,
        "both_required_families_present": family_count[OPPONENT_CURRENT] > 0
        and family_count[OPPONENT_NEXTO] > 0,
        "split_learning_rates": mixed_optimizer_learning_rates(trainer.optimizer)
        == {"policy": POLICY_LR, "critic": CRITIC_LR},
        "exploration_override_exact": trainer.exploration_override
        == HybridDistributionOverride(
            analog_log_std=math.log(EXPLORATION_SIGMA),
            button_temperature=EXPLORATION_BUTTON_TEMPERATURE,
        ),
        "named_mechanics_arrays_zero": inventory["named_mechanics_arrays"] == 0,
        "controlled_flick_arrays_zero": inventory["controlled_flick_arrays"] == 0,
        "human_test_not_loaded": human_identity["test_loaded"] is False,
    }
    return {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V1_PREFLIGHT",
        "created_utc": utc_now(),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "source": trainer.curriculum_transition["source"],
        "authority_sha256": sha256_file(AUTHORITY),
        "contracts": dict(trainer.env.contract_hashes),
        "ppo_config": asdict(trainer.ppo_config),
        "mixed_safety": asdict(trainer.mixed_ppo_safety),
        "kl_guard": asdict(
            Rival2KLGuardConfig(
                reject_minibatch_kl=False,
                reject_completed_update_kl=False,
            )
        ),
        "opponent_curriculum": asdict(trainer.opponent_curriculum),
        "realized_initial_family_counts": family_count,
        "human_data": human_identity,
        "worlds": int(worlds),
    }


def run(args: argparse.Namespace) -> int:
    load_authority()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train, validation, teacher, human_identity = load_human_data(device=args.device)
    trainer, source = build_trainer(
        Path(args.collision_dir),
        worlds=int(args.worlds),
        device=args.device,
    )
    preflight_payload = preflight(
        trainer,
        source,
        human_identity,
        worlds=int(args.worlds),
    )
    write_json(RESULTS / "preflight.json", preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"preflight failed: {preflight_payload['checks']}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0

    teacher_actor_gameplay = precompute_teacher_actor(
        teacher, train.gameplay_observation, device=args.device
    )
    teacher_actor_mechanic = precompute_teacher_actor(
        teacher, train.mechanic_observation, device=args.device
    )
    human_generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0xBCCD)
    mechanic_sampler = MechanicHierarchySampler(
        train.mechanic_label,
        train.mechanic_attempt,
        uniform_label_fraction=0.10,
        maximum_oversampling_ratio=4.0,
        generator=human_generator,
    )
    rolling = run_dir / "rolling.pt"
    best_path = CHECKPOINTS / "rival2_codex_autonomous_best.pt"
    state_path = run_dir / "campaign_state.json"
    curve_path = RESULTS / "training_curve.jsonl"

    if args.resume:
        trainer.load_checkpoint(rolling)
        stored = trainer.curriculum_transition["codex_autonomous_state"]
        campaign_step = int(stored["campaign_step"])
        human_generator.set_state(stored["human_generator_state"].cpu())
        best = copy.deepcopy(stored["best"])
    else:
        if curve_path.exists():
            raise RuntimeError("campaign evidence already exists; resume explicitly")
        campaign_step = 0
        baseline_validation = human_validation(
            trainer.model, validation, device=args.device
        )
        initial_record = checkpoint(
            trainer,
            run_dir / "candidate_u0000.pt",
            campaign_step=0,
            human_generator=human_generator,
            best={},
        )
        baseline_evaluation = run_nexto_evaluation(
            run_dir / "candidate_u0000.pt",
            campaign_step=0,
            run_dir=run_dir,
            device=args.device,
            collision_dir=Path(args.collision_dir),
        )
        best = {
            "campaign_step": 0,
            "checkpoint": initial_record,
            "evaluation": baseline_evaluation,
            "human_validation": baseline_validation,
        }
        checkpoint(
            trainer,
            best_path,
            campaign_step=0,
            human_generator=human_generator,
            best=best,
        )
        write_json(RESULTS / "baseline.json", best)

    if trainer.retention_observations is None:
        first_rollout = trainer.collect_rollout()
        trainer.initialize_retention_corpus_from_rollout(
            first_rollout,
            source_identity=trainer.curriculum_transition["source"],
        )
        del first_rollout
        gc.collect()

    kl_guard = Rival2KLGuardConfig(
        reject_minibatch_kl=False,
        reject_completed_update_kl=False,
    )
    target = int(args.target_updates)
    consecutive_regressions = 0
    stop_reason = "target_updates_reached"
    while campaign_step < target:
        campaign_step += 1
        started = time.perf_counter()
        rollout = trainer.collect_rollout()
        try:
            ppo_metrics = trainer.update(rollout, kl_guard=kl_guard)
            replay_metrics = human_replay(
                trainer,
                train,
                teacher_actor_gameplay,
                teacher_actor_mechanic,
                mechanic_sampler,
                human_generator,
            )
        except Rival2PolicyDisplacementRejected as error:
            stop_reason = str(error.diagnostics.get("reason", "nonfinite_guard"))
            write_json(
                RESULTS / "hard_failure.json",
                {
                    "created_utc": utc_now(),
                    "campaign_step": campaign_step,
                    "last_accepted_campaign_step": campaign_step - 1,
                    "diagnostics": error.diagnostics,
                },
            )
            campaign_step -= 1
            break
        torch.cuda.synchronize(trainer.device)
        row = {
            "created_utc": utc_now(),
            "campaign_step": campaign_step,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "total_agent_samples": trainer.total_agent_samples,
            "wall_seconds": time.perf_counter() - started,
            "ppo": scalar_metrics(ppo_metrics),
            "curriculum": trainer.last_rollout_curriculum_metrics,
            "gameplay": trainer.last_rollout_gameplay_metrics,
            "human_replay": replay_metrics,
            "exploration": {
                "analog_sigma": EXPLORATION_SIGMA,
                "button_temperature": EXPLORATION_BUTTON_TEMPERATURE,
            },
        }
        append_jsonl(curve_path, row)
        rolling_record = checkpoint(
            trainer,
            rolling,
            campaign_step=campaign_step,
            human_generator=human_generator,
            best=best,
        )

        if campaign_step % int(args.evaluation_interval) == 0:
            candidate_path = run_dir / f"candidate_u{campaign_step:04d}.pt"
            candidate_record = checkpoint(
                trainer,
                candidate_path,
                campaign_step=campaign_step,
                human_generator=human_generator,
                best=best,
            )
            validation_metrics = human_validation(
                trainer.model, validation, device=args.device
            )
            evaluation = run_nexto_evaluation(
                candidate_path,
                campaign_step=campaign_step,
                run_dir=run_dir,
                device=args.device,
                collision_dir=Path(args.collision_dir),
            )
            boundary = {
                "campaign_step": campaign_step,
                "checkpoint": candidate_record,
                "human_validation": validation_metrics,
                "evaluation": evaluation,
                "eligible": bool(
                    validation_metrics["eligible"]
                    and evaluation["no_touch_episodes"] == 0
                ),
            }
            if boundary["eligible"] and candidate_rank(evaluation) > candidate_rank(
                best["evaluation"]
            ):
                best = copy.deepcopy(boundary)
                checkpoint(
                    trainer,
                    best_path,
                    campaign_step=campaign_step,
                    human_generator=human_generator,
                    best=best,
                )
                consecutive_regressions = 0
            else:
                regression = (
                    candidate_rank(evaluation)[0]
                    < candidate_rank(best["evaluation"])[0] - 24
                    or not validation_metrics["eligible"]
                )
                consecutive_regressions = consecutive_regressions + 1 if regression else 0
            append_jsonl(RESULTS / "evaluation_boundaries.jsonl", boundary)
            write_json(RESULTS / "best.json", best)
            print(
                json.dumps(
                    {
                        "step": campaign_step,
                        "nexto": [evaluation["goals_for"], evaluation["goals_against"]],
                        "touches": evaluation["touches"],
                        "human_gameplay_rmse": validation_metrics["gameplay"][
                            "complete_action_rmse"
                        ],
                        "best_step": best["campaign_step"],
                        "best_nexto": [
                            best["evaluation"]["goals_for"],
                            best["evaluation"]["goals_against"],
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if evaluation["win_rate"] >= NEXTO_WIN_TARGET and boundary["eligible"]:
                stop_reason = "nexto_win_target_reached"
                break
            if consecutive_regressions >= 3:
                stop_reason = "three_consecutive_closed_loop_regressions"
                break
        elif campaign_step == 1 or campaign_step % 2 == 0:
            print(
                json.dumps(
                    {
                        "step": campaign_step,
                        "seconds": round(row["wall_seconds"], 3),
                        "touches_per_minute": row["gameplay"]["touches_per_minute"],
                        "nexto_rollout_wins": row["curriculum"]["rival_wins"]["nexto"],
                        "nexto_rollout_losses": row["curriculum"]["opponent_wins"][
                            "nexto"
                        ],
                        "kl": row["ppo"].get("completed_update_mean_kl"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        write_json(
            state_path,
            {
                "format": "RIVAL2_CODEX_AUTONOMOUS_V1_STATE",
                "updated_utc": utc_now(),
                "campaign_step": campaign_step,
                "rolling": rolling_record,
                "best": best,
                "stop_reason": None,
            },
        )
        del rollout
        gc.collect()

    final = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V1_RESULT",
        "created_utc": utc_now(),
        "campaign_step": campaign_step,
        "stop_reason": stop_reason,
        "best": best,
        "rolling": checkpoint(
            trainer,
            rolling,
            campaign_step=campaign_step,
            human_generator=human_generator,
            best=best,
        ),
        "target_updates": target,
        "nexto_win_target": NEXTO_WIN_TARGET,
    }
    write_json(RESULTS / "result.json", final)
    write_json(state_path, final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0 if not stop_reason.startswith("nonfinite") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds", type=int, default=WORLD_COUNT)
    parser.add_argument("--target-updates", type=int, default=TARGET_UPDATES)
    parser.add_argument("--evaluation-interval", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
