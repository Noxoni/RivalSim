"""Closed-loop-gated Nexto distillation with immutable human replay.

This is a bounded recovery experiment from the strongest functional human-derived
checkpoint.  It collects the exact controller actions emitted by the pinned Nexto
adapter on native RivalSim states, then jointly supervises those states and the
reviewed human gameplay/mechanic demonstrations.  No PPO or reward optimization is
performed here.  The critic is frozen, and deterministic closed-loop play remains
the promotion authority.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as base  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_opponent_curriculum import NEXTO_ACTING_VERSION  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402


SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v4/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "172BA59786A2E08EB6DC95CFE29F20C21826F7CB9429FF3C89F4D7C4F4BD9E10"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v8/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v8"
CHECKPOINTS = ROOT / "checkpoints/rival2/codex_autonomous_v8"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v8")
COLLECTION_WORLDS = 8_192
SEED = 2_026_090_223
SUPERVISED_STEPS = 64
BOUNDARIES = (1, 2, 4, 8, 16, 32, 64)
NEXTO_BATCH = 4_096
HUMAN_GAMEPLAY_BATCH = 2_048
HUMAN_MECHANIC_BATCH = 2_048
ACTOR_LR = 2.0e-6
GRADIENT_CLIP = 0.5
VALIDATION_WORLD_FRACTION = 0.125


def configure_base() -> None:
    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    base.AUTHORITY = AUTHORITY
    base.RESULTS = RESULTS
    base.CHECKPOINTS = CHECKPOINTS
    base.DEFAULT_RUN_DIR = DEFAULT_RUN_DIR
    base.SEED = SEED
    base.WORLD_COUNT = COLLECTION_WORLDS
    base.POLICY_LR = ACTOR_LR
    base.CRITIC_LR = 1.0e-5
    base.POLICY_TRAINING_BOUNDARY = "full"
    base.CURRENT_OPPONENT_PROBABILITY = 0.0
    base.NEXTO_OPPONENT_PROBABILITY = 1.0
    base.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V8"


def load_authority() -> dict[str, Any]:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": payload.get("format") == "RIVAL2_CODEX_AUTONOMOUS_V8_AUTHORITY",
        "source": payload.get("source", {}).get("sha256") == SOURCE_SHA256,
        "collection_worlds": payload.get("nexto_distillation", {}).get("collection_worlds")
        == COLLECTION_WORLDS,
        "supervised_steps": payload.get("nexto_distillation", {}).get("maximum_steps")
        == SUPERVISED_STEPS,
        "boundaries": payload.get("nexto_distillation", {}).get("validation_boundaries")
        == list(BOUNDARIES),
        "human_mix": payload.get("human_replay", {}).get("gameplay_batch")
        == HUMAN_GAMEPLAY_BATCH
        and payload.get("human_replay", {}).get("mechanic_batch")
        == HUMAN_MECHANIC_BATCH,
        "no_ppo": payload.get("optimization", {}).get("ppo_steps") == 0,
        "critic_frozen": payload.get("optimization", {}).get("critic") == "frozen",
    }
    if not all(checks.values()):
        raise RuntimeError(f"V8 authority mismatch: {checks}")
    return payload


def save_candidate(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    path: Path,
    *,
    step: int,
    authority_sha256: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + int(step)
    payload["iteration"] = int(source["iteration"])
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V8_NEXTO_DISTILLATION",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": authority_sha256,
        },
        "supervised_step": int(step),
        "critic_frozen": True,
        "ppo_steps": 0,
        "selection": "deterministic Nexto score subject to human validation floors",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "bytes": path.stat().st_size,
        "supervised_step": int(step),
        "model_tensor_sha256": base.tensor_tree_sha256(payload["model"]),
    }


@torch.no_grad()
def supervision_metrics(
    model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    batch_size: int = 16_384,
) -> dict[str, Any]:
    actors: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], batch_size):
        actor, _value = model(observation[start : start + batch_size])
        actors.append(actor.cpu())
    return action_metric_summary(torch.cat(actors), action.cpu())


def collect_nexto_supervision(
    *,
    collision_dir: Path,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    trainer, source = base.build_trainer(
        collision_dir,
        worlds=COLLECTION_WORLDS,
        device=device,
    )
    rollout = trainer.collect_rollout()
    flat_version = rollout.policy_version.reshape(-1)
    selected = flat_version == NEXTO_ACTING_VERSION
    if int(selected.sum().item()) != COLLECTION_WORLDS * rollout.horizon:
        raise RuntimeError("Nexto supervision count does not match one opponent per world/tick")
    observation = rollout.observations.reshape(-1, rollout.observations.shape[-1])[selected].clone()
    action = rollout.actions.reshape(-1, 8)[selected].clone()
    world = (
        torch.arange(COLLECTION_WORLDS, device=device)
        .view(1, COLLECTION_WORLDS, 1)
        .expand(rollout.horizon, COLLECTION_WORLDS, 2)
        .reshape(-1)[selected]
        .clone()
    )
    validation_world_start = int(COLLECTION_WORLDS * (1.0 - VALIDATION_WORLD_FRACTION))
    train_mask = world < validation_world_start
    if not bool(torch.isfinite(observation).all().item()) or not bool(
        torch.isfinite(action).all().item()
    ):
        raise RuntimeError("nonfinite Nexto supervision corpus")
    if bool((action[:, :5].abs() > 1.0).any().item()) or bool(
        ((action[:, 5:] != 0.0) & (action[:, 5:] != 1.0)).any().item()
    ):
        raise RuntimeError("Nexto supervision actions violate Rival action bounds")
    telemetry = {
        "horizon": int(rollout.horizon),
        "worlds": COLLECTION_WORLDS,
        "samples": int(observation.shape[0]),
        "train_samples": int(train_mask.sum().item()),
        "validation_samples": int((~train_mask).sum().item()),
        "validation_world_start": validation_world_start,
        "action_mean": action.mean(dim=0).cpu().tolist(),
        "action_nonzero_fraction": (action != 0.0).to(torch.float32).mean(dim=0).cpu().tolist(),
        "rollout_curriculum": copy.deepcopy(trainer.last_rollout_curriculum_metrics),
    }
    del rollout, trainer, selected, flat_version, world
    gc.collect()
    torch.cuda.empty_cache()
    return observation, action, train_mask, telemetry, source


def run(args: argparse.Namespace) -> int:
    configure_base()
    authority = load_authority()
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V8 source checkpoint hash mismatch")
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V8 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    nexto_observation, nexto_action, nexto_train_mask, corpus, source = (
        collect_nexto_supervision(
            collision_dir=Path(args.collision_dir),
            device=args.device,
        )
    )
    train_index = torch.nonzero(nexto_train_mask, as_tuple=False).squeeze(-1)
    validation_index = torch.nonzero(~nexto_train_mask, as_tuple=False).squeeze(-1)
    del nexto_train_mask

    policy_config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(policy_config).to(args.device)
    model.load_state_dict(source["model"], strict=True)
    model.critic.requires_grad_(False)
    teacher.eval().requires_grad_(False)
    critic_before = {
        name: value.detach().cpu().clone() for name, value in model.critic.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        [*model.trunk.parameters(), *model.actor.parameters()],
        lr=ACTOR_LR,
        weight_decay=0.0,
    )
    generator = torch.Generator(device=args.device).manual_seed(SEED ^ 0x8A8A)
    human_generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0xBCCD)
    mechanic_sampler = MechanicHierarchySampler(
        train.mechanic_label,
        train.mechanic_attempt,
        uniform_label_fraction=0.10,
        maximum_oversampling_ratio=4.0,
        generator=human_generator,
    )

    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V8_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": base.sha256_file(SOURCE),
        "source_model_tensor_sha256": base.tensor_tree_sha256(source["model"]),
        "human_identity": human_identity,
        "nexto_corpus": corpus,
        "checks": {
            "source_exact": base.sha256_file(SOURCE) == SOURCE_SHA256,
            "nexto_only_collection": set(corpus["rollout_curriculum"]["world_decisions"])
            == {"current", "historical", "nexto", "wisp"}
            and corpus["rollout_curriculum"]["world_decisions"]["nexto"]
            == COLLECTION_WORLDS * int(corpus["horizon"]),
            "human_test_not_loaded": human_identity["test_loaded"] is False,
            "critic_frozen": not any(p.requires_grad for p in model.critic.parameters()),
            "fresh_actor_optimizer": len(optimizer.state) == 0,
            "ppo_steps_zero": authority["optimization"]["ppo_steps"] == 0,
        },
    }
    if not all(preflight["checks"].values()):
        preflight["verdict"] = "FAIL"
        base.write_json(RESULTS / "preflight.json", preflight)
        raise RuntimeError(f"V8 preflight failed: {preflight['checks']}")
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    authority_sha = base.sha256_file(AUTHORITY)
    baseline_human = base.human_validation(model, validation, device=args.device)
    baseline_nexto_imitation = supervision_metrics(
        model,
        nexto_observation.index_select(0, validation_index),
        nexto_action.index_select(0, validation_index),
    )
    baseline_evaluation = base.run_nexto_evaluation(
        SOURCE,
        campaign_step=0,
        run_dir=run_dir,
        device=args.device,
        collision_dir=Path(args.collision_dir),
    )
    best = {
        "supervised_step": 0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": base.tensor_tree_sha256(source["model"]),
        },
        "evaluation": baseline_evaluation,
        "human_validation": baseline_human,
        "nexto_imitation_validation": baseline_nexto_imitation,
    }
    base.write_json(RESULTS / "baseline.json", best)

    losses: list[dict[str, float]] = []
    for step in range(1, SUPERVISED_STEPS + 1):
        nexto_local = torch.randint(
            train_index.shape[0],
            (NEXTO_BATCH,),
            device=args.device,
            generator=generator,
        )
        nexto_index = train_index.index_select(0, nexto_local)
        gameplay_index = torch.randint(
            train.gameplay_observation.shape[0],
            (HUMAN_GAMEPLAY_BATCH,),
            generator=human_generator,
        )
        mechanic_index = mechanic_sampler.sample(HUMAN_MECHANIC_BATCH)
        observation = torch.cat(
            (
                nexto_observation.index_select(0, nexto_index),
                train.gameplay_observation.index_select(0, gameplay_index).to(args.device),
                train.mechanic_observation.index_select(0, mechanic_index).to(args.device),
            )
        )
        action = torch.cat(
            (
                nexto_action.index_select(0, nexto_index),
                train.gameplay_action.index_select(0, gameplay_index).to(args.device),
                train.mechanic_action.index_select(0, mechanic_index).to(args.device),
            )
        )
        with torch.no_grad():
            teacher_actor, _teacher_value = teacher(observation)
        model.train()
        student_actor, _student_value = model(observation)
        objective = human_behavior_cloning_objective(
            student_actor,
            teacher_actor,
            action,
            smooth_l1_beta=0.1,
            analog_weight=1.0,
            button_weight=0.25,
            log_std_weight=0.05,
            policy_config=policy_config,
        )
        if not bool(torch.isfinite(objective.loss).item()):
            raise RuntimeError("nonfinite V8 supervised loss")
        optimizer.zero_grad(set_to_none=True)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*model.trunk.parameters(), *model.actor.parameters()], GRADIENT_CLIP
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("nonfinite V8 supervised gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
            raise RuntimeError("nonfinite V8 parameter")
        loss_row = {
            "step": float(step),
            "loss": float(objective.loss.detach().item()),
            "analog": float(objective.analog_smooth_l1.detach().item()),
            "buttons": float(objective.button_bce.detach().item()),
            "log_std": float(objective.log_std_retention.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
        }
        losses.append(loss_row)
        base.append_jsonl(RESULTS / "training_curve.jsonl", loss_row)
        if step not in BOUNDARIES:
            continue

        candidate_path = run_dir / f"candidate_s{step:04d}.pt"
        candidate = save_candidate(
            source,
            model,
            candidate_path,
            step=step,
            authority_sha256=authority_sha,
        )
        human = base.human_validation(model, validation, device=args.device)
        nexto_imitation = supervision_metrics(
            model,
            nexto_observation.index_select(0, validation_index),
            nexto_action.index_select(0, validation_index),
        )
        evaluation = base.run_nexto_evaluation(
            candidate_path,
            campaign_step=step,
            run_dir=run_dir,
            device=args.device,
            collision_dir=Path(args.collision_dir),
        )
        eligible = bool(human["eligible"] and evaluation["no_touch_episodes"] == 0)
        boundary = {
            "supervised_step": step,
            "checkpoint": candidate,
            "human_validation": human,
            "nexto_imitation_validation": nexto_imitation,
            "evaluation": evaluation,
            "eligible": eligible,
        }
        base.append_jsonl(RESULTS / "evaluation_boundaries.jsonl", boundary)
        if eligible and base.candidate_rank(evaluation) > base.candidate_rank(best["evaluation"]):
            promoted_path = CHECKPOINTS / "rival2_codex_autonomous_best.pt"
            promoted = save_candidate(
                source,
                model,
                promoted_path,
                step=step,
                authority_sha256=authority_sha,
            )
            best = copy.deepcopy(boundary)
            best["checkpoint"] = promoted
            base.write_json(RESULTS / "best.json", best)
        print(
            json.dumps(
                {
                    "step": step,
                    "nexto": [evaluation["goals_for"], evaluation["goals_against"]],
                    "touches": evaluation["touches"],
                    "nexto_imitation_rmse": nexto_imitation["complete_action_rmse"],
                    "human_gameplay_rmse": human["gameplay"]["complete_action_rmse"],
                    "best_step": best["supervised_step"],
                    "best_nexto": [
                        best["evaluation"]["goals_for"],
                        best["evaluation"]["goals_against"],
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    critic_exact = all(
        torch.equal(critic_before[name], value.detach().cpu())
        for name, value in model.critic.state_dict().items()
    )
    if not critic_exact:
        raise RuntimeError("V8 changed the frozen critic")
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V8_RESULT",
        "authority_sha256": authority_sha,
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "supervised_steps": SUPERVISED_STEPS,
        "ppo_steps": 0,
        "critic_byte_identical": critic_exact,
        "nexto_corpus": corpus,
        "best": best,
        "target_nexto_win_rate": 0.55,
        "target_reached": float(best["evaluation"]["win_rate"]) >= 0.55,
    }
    base.write_json(RESULTS / "result.json", result)
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, default=base.DEFAULT_COLLISION_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
