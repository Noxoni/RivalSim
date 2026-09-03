"""Train a single recurrent Rival policy from a capability-baked feed-forward base.

This is the prospective correction to unified distillation V1.  Capability V2
changed only the actor head by a very small amount, so its Blue/Orange actor
heads are averaged directly into one frozen base.  One recurrent residual then
learns the materially larger aerial teacher while rehearsing natural and
capability states.  No expert, route, scenario id, or task id is available to
the deployed policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_unified_capability_distillation_v1 as v1
from rivalsim.rival2_policy import Rival2ActorCritic, deterministic_hybrid_action
from rivalsim.rival2_unified_policy import (
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)

AUTHORITY = ROOT / "results/rival2/unified_capability_distillation_v2/authority.json"
RESULTS = ROOT / "results/rival2/unified_capability_distillation_v2"
CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/unified_capability_distillation_v2/rival2_unified_capability_v2.pt"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/unified-capability-distillation-v2")


def configure_v1_paths() -> None:
    """Point the shared deterministic corpus helpers at the V2 authority."""

    v1.AUTHORITY = AUTHORITY
    v1.RESULTS = RESULTS
    v1.CHECKPOINT = CHECKPOINT


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V2_AUTHORITY":
        raise RuntimeError("unified capability V2 authority format mismatch")
    if authority["integrity"]["optimizer_steps_before_authority_commit"] != 0:
        raise RuntimeError("V2 authority does not preserve the prospective boundary")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{AUTHORITY.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != AUTHORITY.read_bytes():
        raise RuntimeError("V2 authority is not byte-identical to the committed Git object")
    for source in authority["sources"].values():
        path = ROOT / source["path"]
        observed = v1.sha256_file(path)
        if observed != source["sha256"]:
            raise RuntimeError(f"source hash mismatch for {path}: {observed}")
    return authority


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).to(torch.float64))
    denominator = torch.linalg.vector_norm(right.to(torch.float64)).clamp_min(1e-30)
    return float(numerator / denominator)


def make_capability_baked_base(
    payloads: dict[str, Any], teachers: v1.FrozenTeachers
) -> tuple[Rival2ActorCritic, dict[str, Any], dict[str, Any]]:
    """Return the single frozen feed-forward base authorized for V2."""

    base_blue = payloads["base_blue"]
    base_orange = payloads["base_orange"]
    capability_blue = payloads["capability_blue"]
    capability_orange = payloads["capability_orange"]

    # The common trunk and critic are a required structural precondition.  The
    # averaged actor is the only baked capability change.
    for prefix in ("trunk.", "critic."):
        names = [name for name in base_blue["model"] if name.startswith(prefix)]
        if not all(
            torch.equal(base_blue["model"][name], base_orange["model"][name])
            and torch.equal(base_blue["model"][name], capability_blue["model"][name])
            and torch.equal(base_blue["model"][name], capability_orange["model"][name])
            for name in names
        ):
            raise RuntimeError(f"{prefix[:-1]} tensors differ across V23/capability parents")

    baked_state = {
        name: value.detach().cpu().clone() for name, value in base_blue["model"].items()
    }
    actor_diagnostics: dict[str, Any] = {}
    for name in ("actor.weight", "actor.bias"):
        blue = capability_blue["model"][name].detach().cpu()
        orange = capability_orange["model"][name].detach().cpu()
        baked_state[name] = (blue + orange) * 0.5
        actor_diagnostics[name] = {
            "blue_orange_relative_l2": relative_l2(blue, orange),
            "baked_vs_v23_blue_relative_l2": relative_l2(
                baked_state[name], base_blue["model"][name].detach().cpu()
            ),
        }

    baked = Rival2ActorCritic(teachers.base_blue.config)
    baked.load_state_dict(baked_state, strict=True)
    baked.eval().requires_grad_(False)

    baked_payload = copy.deepcopy(base_blue)
    baked_payload["model"] = baked_state
    baked_payload["format"] = "RIVAL2_CAPABILITY_BAKED_BASE_V1"
    baked_payload["base_construction"] = {
        "trunk": "byte-identical V23/capability trunk",
        "critic": "byte-identical V23/capability critic",
        "actor": "elementwise arithmetic mean of Capability V2 Blue/Orange actor heads",
        "source_sha256": copy.deepcopy(
            {
                name: identity["sha256"]
                for name, identity in json.loads(AUTHORITY.read_text(encoding="utf-8"))[
                    "sources"
                ].items()
            }
        ),
    }
    return baked, baked_payload, actor_diagnostics


def eligible(metrics: dict[str, Any], authority: dict[str, Any]) -> bool:
    if not metrics["finite"]:
        return False
    selection = authority["selection"]
    if metrics[v1.FAMILY_NATURAL]["expected_action_rmse"] > float(
        selection["natural_expected_action_rmse_max"]
    ):
        return False
    if metrics[v1.FAMILY_AERIAL]["expected_action_rmse"] > float(
        selection["aerial_expected_action_rmse_max"]
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
    baked_payload: dict[str, Any],
    step: int,
    validation: dict[str, Any],
    baseline: dict[str, Any],
    actor_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    model_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V2",
        "created_utc": v1.utc_now(),
        "model": model_state,
        "policy_config": asdict(model.config),
        "policy_config_sha256": model.config.content_hash,
        "optimizer": {
            "format": "RIVAL2_UNIFIED_CONTEXT_ONLY_ADAMW_V1",
            "state": optimizer.state_dict(),
        },
        "accepted_supervised_steps": step,
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": v1.sha256_file(AUTHORITY),
        },
        "sources": copy.deepcopy(authority["sources"]),
        "base_construction": copy.deepcopy(baked_payload["base_construction"]),
        "base_actor_diagnostics": actor_diagnostics,
        "contracts": copy.deepcopy(authority["contracts"]),
        "observation_version": baked_payload["observation_version"],
        "action_version": baked_payload["action_version"],
        "reward_version": baked_payload["reward_version"],
        "episode_version": baked_payload["episode_version"],
        "contract_hashes": copy.deepcopy(baked_payload["contract_hashes"]),
        "physics_hz": 120,
        "policy_hz": 120,
        "validation": validation,
        "baseline_validation": baseline,
        "runtime_router": False,
        "task_identifier_input": False,
        "ppo_resumable": False,
        "base_model_tensor_sha256": v1.tensor_tree_sha256(baked_payload["model"]),
        "model_tensor_sha256": v1.tensor_tree_sha256(model_state),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CHECKPOINT)
    return {
        "path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": v1.sha256_file(CHECKPOINT),
        "bytes": CHECKPOINT.stat().st_size,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "accepted_supervised_steps": step,
    }


def run(args: argparse.Namespace) -> int:
    configure_v1_paths()
    authority = load_authority()
    payloads, teachers = v1.load_teachers(authority, args.device)
    baked, baked_payload, actor_diagnostics = make_capability_baked_base(
        payloads, teachers
    )
    baked = baked.to(args.device)
    model = Rival2UnifiedActorCritic(Rival2UnifiedPolicyConfig()).to(args.device)
    model.load_feedforward_parent(baked)
    model.freeze_base()
    model.train()

    generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"])
    )
    parity_observation = torch.randn((4096, 182), generator=generator).to(args.device)
    with torch.no_grad():
        baked_actor, baked_value = baked(parity_observation)
        actor, value, _hidden = model(parity_observation)
    parity = {
        "actor_exact": bool(torch.equal(actor, baked_actor)),
        "value_exact": bool(torch.equal(value, baked_value)),
        "action_exact": bool(
            torch.equal(
                deterministic_unified_action(actor),
                deterministic_hybrid_action(baked_actor),
            )
        ),
    }
    if not all(parity.values()):
        raise RuntimeError(f"zero-residual capability-baked parity failed: {parity}")
    preflight = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V2_PREFLIGHT",
        "created_utc": v1.utc_now(),
        "authority_sha256": v1.sha256_file(AUTHORITY),
        "source_hashes_verified": True,
        "contracts_identical": True,
        "common_trunk_exact": True,
        "common_critic_exact": True,
        "base_actor_construction": "mean Capability V2 Blue/Orange actor",
        "base_actor_diagnostics": actor_diagnostics,
        "runtime_router": False,
        "task_identifier_input": False,
        "parent_parity": parity,
        "baked_base_tensor_sha256": v1.tensor_tree_sha256(baked_payload["model"]),
        "optimizer_steps": 0,
        "verdict": "PASS",
    }
    v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_corpora:
        raise RuntimeError("unified V2 distillation requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_corpora:
        v1.build_corpora(
            run_dir,
            authority,
            teachers,
            collision_root=args.collision_root,
            device=args.device,
        )
    sequence_ticks = int(authority["corpora"]["sequence_ticks"])
    train_pools = v1.load_pools(run_dir, "train", sequence_ticks)
    validation_pools = v1.load_pools(run_dir, "validation", sequence_ticks)

    optimizer_config = authority["optimization"]
    optimizer = torch.optim.AdamW(
        model.context_parameters,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    baseline = v1.validate(
        model, teachers, validation_pools, authority=authority, device=args.device
    )
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
    batch_count = int(optimizer_config["batch_sequences_per_family"])
    weights = optimizer_config["family_weights"]
    maximum_steps = min(int(optimizer_config["maximum_accepted_steps"]), args.max_steps)
    interval = int(optimizer_config["validation_interval_steps"])
    train_generator = torch.Generator(device="cpu").manual_seed(
        int(authority["seeds"]["optimizer"]) ^ 0xA5A5
    )
    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float32, device=args.device)
        train_metrics: dict[str, Any] = {}
        for family in v1.FAMILIES:
            observation, side = train_pools[family].sample(
                batch_count, generator=train_generator, device=args.device
            )
            student, _value, _hidden = model(observation)
            target = v1.teacher_actor(teachers, family, observation, side)
            local, row = v1.family_loss(
                student,
                target,
                burn_in=int(authority["corpora"]["burn_in_ticks"]),
                button_weight=float(optimizer_config["button_weight"]),
                log_std_weight=float(optimizer_config["log_std_weight"]),
            )
            total = total + float(weights[family]) * local
            train_metrics[family] = row
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.context_parameters,
            float(optimizer_config["maximum_gradient_norm"]),
        )
        if not bool(torch.isfinite(total) and torch.isfinite(gradient)):
            raise RuntimeError("nonfinite unified V2 distillation loss or gradient")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.context_parameters
        ):
            raise RuntimeError("nonfinite unified V2 recurrent context parameter")
        if step % interval != 0 and step != maximum_steps:
            continue
        model.eval()
        validation = v1.validate(
            model, teachers, validation_pools, authority=authority, device=args.device
        )
        model.train()
        score = float(validation["weighted_expected_action_rmse"])
        is_eligible = eligible(validation, authority)
        improved = is_eligible and score < (
            best_score - float(optimizer_config["minimum_score_improvement"])
        )
        if improved:
            best_score = score
            best_step = step
            best_metrics = copy.deepcopy(validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
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
        natural_rmse = validation[v1.FAMILY_NATURAL]["expected_action_rmse"]
        print(
            f"step={step} score={score:.6f} natural={natural_rmse:.6f} "
            f"aerial={validation[v1.FAMILY_AERIAL]['expected_action_rmse']:.6f} "
            f"demo={validation[v1.FAMILY_DEMO]['expected_action_rmse']:.6f} "
            f"floor={validation[v1.FAMILY_FLOOR]['expected_action_rmse']:.6f} "
            f"wall={validation[v1.FAMILY_WALL]['expected_action_rmse']:.6f} "
            f"eligible={is_eligible} selected={improved}",
            flush=True,
        )
        if stale >= int(optimizer_config["plateau_patience_boundaries"]):
            stop_reason = "validation_plateau"
            break

    if best_state is None or best_optimizer is None or best_metrics is None:
        result = {
            "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V2_RESULT",
            "created_utc": v1.utc_now(),
            "verdict": "BLOCKED",
            "reason": "no validation-eligible unified V2 checkpoint",
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
        baked_payload=baked_payload,
        step=best_step,
        validation=best_metrics,
        baseline=baseline,
        actor_diagnostics=actor_diagnostics,
    )
    result = {
        "format": "RIVAL2_UNIFIED_CAPABILITY_DISTILLATION_V2_RESULT",
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
    parser.add_argument("--collision-root", type=Path, default=v1.DEFAULT_COLLISION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-corpora", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
