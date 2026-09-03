"""Joint natural/aerial student-state correction for unified Rival V3."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import evaluate_rival2_unified_capability_v2 as physical  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as aerial  # noqa: E402
from benchmarks import run_rival2_unified_capability_distillation_v1 as v1  # noqa: E402
from benchmarks import run_rival2_unified_capability_distillation_v3 as v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic  # noqa: E402

AUTHORITY = ROOT / "results/rival2/unified_capability_distillation_v4/authority.json"
RESULTS = ROOT / "results/rival2/unified_capability_distillation_v4"
CHECKPOINT = (
    ROOT / "checkpoints/rival2/unified_capability_distillation_v4/rival2_unified_capability_v4.pt"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/unified-capability-distillation-v4")
FAMILY_STUDENT_NATURAL = "student_natural_v3"
FAMILY_STUDENT_AERIAL = "student_aerial_v3"
FAMILIES = (
    v1.FAMILY_NATURAL,
    FAMILY_STUDENT_NATURAL,
    v1.FAMILY_AERIAL,
    FAMILY_STUDENT_AERIAL,
    v1.FAMILY_DEMO,
    v1.FAMILY_FLOOR,
    v1.FAMILY_WALL,
)


def configure_shared_paths() -> None:
    for module in (v1, v3):
        module.AUTHORITY = AUTHORITY
        module.RESULTS = RESULTS
        module.CHECKPOINT = CHECKPOINT


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V4_AUTHORITY":
        raise RuntimeError("unified capability V4 authority format mismatch")
    if authority["integrity"]["optimizer_steps_before_authority_commit"] != 0:
        raise RuntimeError("V4 authority does not preserve prospective optimization")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{AUTHORITY.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != AUTHORITY.read_bytes():
        raise RuntimeError("V4 authority is not byte-identical to the committed object")
    for identity in [*authority["sources"].values(), authority["parent"]]:
        path = ROOT / identity["path"]
        observed = v1.sha256_file(path)
        if observed != identity["sha256"]:
            raise RuntimeError(f"source hash mismatch for {path}: {observed}")
    return authority


def student_aerial_path(run_dir: Path, split: str) -> Path:
    return run_dir / "corpora" / split / "student_aerial.pt"


def collect_student_aerial(
    parent: Rival2UnifiedActorCritic,
    *,
    authority: dict[str, Any],
    collision_root: Path,
    worlds_per_side: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    goal_authority = aerial.load_authority()
    geometry = ArenaGeometry.load_soccar(collision_root / "soccar")
    meshes = WarpArenaMeshes(geometry, device)
    distribution = aerial.distribution_override(goal_authority)
    observations: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    sides: list[torch.Tensor] = []
    source_metrics: list[dict[str, Any]] = []
    for side in (0, 1):
        generator = torch.Generator(device=device).manual_seed(seed ^ (0xDA00 + side))
        rollout, metrics = aerial.collect_rollout(
            physical.StatefulUnifiedAdapter(parent),
            geometry,
            meshes,
            authority=goal_authority,
            side=side,
            worlds=worlds_per_side,
            horizon=int(authority["corpora"]["validation"]["aerial_horizon_ticks"]),
            seed=seed,
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
            collision_dir=collision_root / "soccar",
            phase=1,
            record_deterministic=True,
        )
        assert rollout is not None
        observations.append(rollout.observation.detach().cpu())
        valid.append(rollout.mask.detach().cpu())
        sides.append(torch.full((worlds_per_side,), side, dtype=torch.int64))
        source_metrics.append(metrics)
        del rollout
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "format": "RIVAL2_UNIFIED_STUDENT_AERIAL_CORPUS_V1",
        "kind": FAMILY_STUDENT_AERIAL,
        "seed": seed,
        "observation": torch.cat(observations, dim=1),
        "valid": torch.cat(valid, dim=1),
        "side": torch.cat(sides),
        "source_metrics": source_metrics,
        "source_policy_sha256": authority["parent"]["sha256"],
        "teacher_target": authority["sources"]["aerial_teacher"],
    }


def build_all_corpora(
    run_dir: Path,
    authority: dict[str, Any],
    teachers: v1.FrozenTeachers,
    parent: Rival2UnifiedActorCritic,
    *,
    collision_root: Path,
    device: str,
) -> dict[str, Any]:
    manifest = v3.build_all_corpora(
        run_dir,
        authority,
        teachers,
        parent,
        collision_root=collision_root,
        device=device,
    )
    # V3 names this family by its source version. The V4 authority makes the
    # source explicit, so only the dictionary key changes; corpus bytes do not.
    for split in ("train", "validation"):
        standard_path = v3.student_natural_path(run_dir, split)
        spec = authority["corpora"][split]
        corpus = collect_student_aerial(
            parent,
            authority=authority,
            collision_root=collision_root,
            worlds_per_side=int(spec["student_aerial_worlds_per_side"]),
            seed=int(authority["seeds"][f"{split}_student_aerial"]),
            device=device,
        )
        row = v1.save_corpus(student_aerial_path(run_dir, split), corpus)
        manifest["splits"][split]["student_aerial"] = row
        manifest["splits"][split]["student_natural"]["path"] = standard_path.as_posix()
        del corpus
    manifest["format"] = "RIVAL2_UNIFIED_CORPUS_MANIFEST_V4"
    manifest["student_natural_policy"] = copy.deepcopy(authority["parent"])
    manifest["student_aerial_policy"] = copy.deepcopy(authority["parent"])
    v1.write_json(run_dir / "corpora" / "manifest.json", manifest)
    v1.write_json(RESULTS / "corpus_manifest.json", manifest)
    return manifest


def load_pools(run_dir: Path, split: str, sequence_ticks: int) -> dict[str, v1.SequencePool]:
    pools = v3.load_pools(run_dir, split, sequence_ticks)
    pools[FAMILY_STUDENT_NATURAL] = pools.pop(v3.FAMILY_STUDENT_NATURAL)
    aerial_payload = torch.load(
        student_aerial_path(run_dir, split), map_location="cpu", weights_only=False
    )
    pools[FAMILY_STUDENT_AERIAL] = v1.SequencePool.from_payload(
        aerial_payload, sequence_ticks=sequence_ticks
    )
    return pools


def target_actor(
    teachers: v1.FrozenTeachers,
    family: str,
    observation: torch.Tensor,
    side: torch.Tensor,
) -> torch.Tensor:
    if family == FAMILY_STUDENT_NATURAL:
        teacher_family = v1.FAMILY_NATURAL
    elif family == FAMILY_STUDENT_AERIAL:
        teacher_family = v1.FAMILY_AERIAL
    else:
        teacher_family = family
    return v1.teacher_actor(teachers, teacher_family, observation, side)


@torch.no_grad()
def validate(
    model: Rival2UnifiedActorCritic,
    teachers: v1.FrozenTeachers,
    pools: dict[str, v1.SequencePool],
    *,
    authority: dict[str, Any],
    device: str,
    samples_per_family: int = 256,
) -> dict[str, Any]:
    optimization = authority["optimization"]
    generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"]) ^ 0x5151
    )
    metrics: dict[str, Any] = {}
    for family in FAMILIES:
        observation, side = pools[family].sample(
            samples_per_family, generator=generator, device=device
        )
        actor, _value, _hidden = model(observation)
        target = target_actor(teachers, family, observation, side)
        _loss, row = v1.family_loss(
            actor,
            target,
            burn_in=int(authority["corpora"]["burn_in_ticks"]),
            button_weight=float(optimization["button_weight"]),
            log_std_weight=float(optimization["log_std_weight"]),
        )
        metrics[family] = row
    weights = optimization["family_weights"]
    metrics["weighted_expected_action_rmse"] = sum(
        float(weights[family]) * metrics[family]["expected_action_rmse"] for family in FAMILIES
    )
    metrics["finite"] = all(math.isfinite(float(metrics[family]["loss"])) for family in FAMILIES)
    return metrics


def eligible(metrics: dict[str, Any], authority: dict[str, Any]) -> bool:
    if not metrics["finite"]:
        return False
    selection = authority["selection"]
    checks = (
        (v1.FAMILY_NATURAL, "natural_expected_action_rmse_max"),
        (FAMILY_STUDENT_NATURAL, "student_natural_expected_action_rmse_max"),
        (v1.FAMILY_AERIAL, "aerial_expected_action_rmse_max"),
        (FAMILY_STUDENT_AERIAL, "student_aerial_expected_action_rmse_max"),
    )
    if any(
        metrics[family]["expected_action_rmse"] > float(selection[key]) for family, key in checks
    ):
        return False
    limit = float(selection["capability_family_expected_action_rmse_max"])
    return all(
        metrics[family]["expected_action_rmse"] <= limit
        for family in (v1.FAMILY_DEMO, v1.FAMILY_FLOOR, v1.FAMILY_WALL)
    )


def save_checkpoint(
    model: Rival2UnifiedActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    authority: dict[str, Any],
    parent_payload: dict[str, Any],
    step: int,
    validation: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V4",
        "created_utc": v1.utc_now(),
        "model": state,
        "policy_config": asdict(model.config),
        "policy_config_sha256": model.config.content_hash,
        "optimizer": {
            "format": "RIVAL2_UNIFIED_CONTEXT_ONLY_ADAMW_V1",
            "state": optimizer.state_dict(),
        },
        "accepted_supervised_steps": step,
        "cumulative_supervised_steps": int(parent_payload["cumulative_supervised_steps"]) + step,
        "parent": copy.deepcopy(authority["parent"]),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": v1.sha256_file(AUTHORITY),
        },
        "sources": copy.deepcopy(authority["sources"]),
        "contracts": copy.deepcopy(authority["contracts"]),
        "observation_version": parent_payload["observation_version"],
        "action_version": parent_payload["action_version"],
        "reward_version": parent_payload["reward_version"],
        "episode_version": parent_payload["episode_version"],
        "contract_hashes": copy.deepcopy(parent_payload["contract_hashes"]),
        "physics_hz": 120,
        "policy_hz": 120,
        "validation": validation,
        "baseline_validation": baseline,
        "runtime_router": False,
        "task_identifier_input": False,
        "ppo_resumable": False,
        "model_tensor_sha256": v1.tensor_tree_sha256(state),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CHECKPOINT)
    return {
        "path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": v1.sha256_file(CHECKPOINT),
        "bytes": CHECKPOINT.stat().st_size,
        "accepted_supervised_steps": step,
        "cumulative_supervised_steps": payload["cumulative_supervised_steps"],
        "model_tensor_sha256": payload["model_tensor_sha256"],
    }


def run(args: argparse.Namespace) -> int:
    configure_shared_paths()
    authority = load_authority()
    parent_payload, model = v3.load_parent(authority, args.device)
    _teacher_payloads, teachers = v1.load_teachers(authority, args.device)
    preflight = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V4_PREFLIGHT",
        "created_utc": v1.utc_now(),
        "authority_sha256": v1.sha256_file(AUTHORITY),
        "parent_sha256": v1.sha256_file(ROOT / authority["parent"]["path"]),
        "parent_model_tensor_sha256": v1.tensor_tree_sha256(parent_payload["model"]),
        "frozen_base": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith(("trunk.", "actor.", "critic."))
        ),
        "deterministic_aerial_recording_supported": True,
        "runtime_router": False,
        "task_identifier_input": False,
        "optimizer_steps": 0,
        "verdict": "PASS",
    }
    if not preflight["frozen_base"]:
        raise RuntimeError("unified V4 base is not frozen")
    v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_corpora:
        raise RuntimeError("unified V4 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_corpora:
        build_all_corpora(
            run_dir,
            authority,
            teachers,
            model,
            collision_root=args.collision_root,
            device=args.device,
        )
    sequence_ticks = int(authority["corpora"]["sequence_ticks"])
    train_pools = load_pools(run_dir, "train", sequence_ticks)
    validation_pools = load_pools(run_dir, "validation", sequence_ticks)

    config = authority["optimization"]
    optimizer = torch.optim.AdamW(
        model.context_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    baseline = validate(model, teachers, validation_pools, authority=authority, device=args.device)
    v1.write_json(RESULTS / "baseline.json", baseline)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = float("inf")
    best_step = 0
    stale = 0
    stop_reason = "maximum_accepted_steps"
    batch_count = int(config["batch_sequences_per_family"])
    weights = config["family_weights"]
    maximum_steps = min(int(config["maximum_accepted_steps"]), args.max_steps)
    interval = int(config["validation_interval_steps"])
    generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"]) ^ 0xA5A5
    )
    model.train()
    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=args.device)
        train_metrics: dict[str, Any] = {}
        for family in FAMILIES:
            observation, side = train_pools[family].sample(
                batch_count, generator=generator, device=args.device
            )
            actor, _value, _hidden = model(observation)
            target = target_actor(teachers, family, observation, side)
            local, row = v1.family_loss(
                actor,
                target,
                burn_in=int(authority["corpora"]["burn_in_ticks"]),
                button_weight=float(config["button_weight"]),
                log_std_weight=float(config["log_std_weight"]),
            )
            total = total + float(weights[family]) * local
            train_metrics[family] = row
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.context_parameters, float(config["maximum_gradient_norm"])
        )
        if not bool(torch.isfinite(total) and torch.isfinite(gradient)):
            raise RuntimeError("nonfinite unified V4 loss or gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.context_parameters):
            raise RuntimeError("nonfinite unified V4 context parameter")
        if step % interval != 0 and step != maximum_steps:
            continue
        model.eval()
        validation = validate(
            model, teachers, validation_pools, authority=authority, device=args.device
        )
        model.train()
        score = float(validation["weighted_expected_action_rmse"])
        is_eligible = eligible(validation, authority)
        improved = is_eligible and score < (best_score - float(config["minimum_score_improvement"]))
        if improved:
            best_score = score
            best_step = step
            best_metrics = copy.deepcopy(validation)
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            stale = 0
        else:
            stale += 1
        v1.append_jsonl(
            curve,
            {
                "step": step,
                "created_utc": v1.utc_now(),
                "train_loss": float(total.detach()),
                "gradient_norm": float(gradient.detach()),
                "train": train_metrics,
                "validation": validation,
                "eligible": is_eligible,
                "selected": improved,
                "stale_boundaries": stale,
            },
        )
        print(
            f"step={step} score={score:.6f} "
            f"natural={validation[v1.FAMILY_NATURAL]['expected_action_rmse']:.6f} "
            f"natural_dagger={validation[FAMILY_STUDENT_NATURAL]['expected_action_rmse']:.6f} "
            f"aerial={validation[v1.FAMILY_AERIAL]['expected_action_rmse']:.6f} "
            f"aerial_dagger={validation[FAMILY_STUDENT_AERIAL]['expected_action_rmse']:.6f} "
            f"demo={validation[v1.FAMILY_DEMO]['expected_action_rmse']:.6f} "
            f"floor={validation[v1.FAMILY_FLOOR]['expected_action_rmse']:.6f} "
            f"wall={validation[v1.FAMILY_WALL]['expected_action_rmse']:.6f} "
            f"eligible={is_eligible} selected={improved}",
            flush=True,
        )
        if stale >= int(config["plateau_patience_boundaries"]):
            stop_reason = "validation_plateau"
            break

    if best_state is None or best_optimizer is None or best_metrics is None:
        result = {
            "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V4_RESULT",
            "created_utc": v1.utc_now(),
            "verdict": "BLOCKED",
            "reason": "no validation-eligible V4 checkpoint",
            "baseline": baseline,
            "stop_reason": stop_reason,
        }
        v1.write_json(RESULTS / "result.json", result)
        return 2
    model.load_state_dict(best_state, strict=True)
    optimizer.load_state_dict(best_optimizer)
    checkpoint = save_checkpoint(
        model,
        optimizer,
        authority=authority,
        parent_payload=parent_payload,
        step=best_step,
        validation=best_metrics,
        baseline=baseline,
    )
    result = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V4_RESULT",
        "created_utc": v1.utc_now(),
        "verdict": "TRAINED_PENDING_PHYSICAL_EVALUATION",
        "runtime_router": False,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "baseline": baseline,
        "selected_validation": best_metrics,
        "checkpoint": checkpoint,
    }
    v1.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--collision-root", type=Path, default=physical.COLLISION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-corpora", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
