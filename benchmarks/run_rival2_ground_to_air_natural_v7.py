"""Train a narrow steer/yaw/roll pop residual over the proven fixed pitch."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v6 as natural_v6  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_PARKED,
    SETUP_NAMES,
)
from rivalsim.rival2_ground_to_air_pop_control_v7 import (  # noqa: E402
    GROUND_TO_AIR_POP_CONTROL_V7_VERSION,
    active_pop_orientation_channel_mask,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
)

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V7"
AUTHORITY = ROOT / "results/rival2/ground_to_air_natural_v7/authority.json"
AUTHORITY_SHA256 = "6192B823A13993A0C883D83E35AC128F3C2D964BA172F36589356E91E6FCE2A6"
RESULTS = ROOT / "results/rival2/ground_to_air_natural_v7"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_natural_v7"
PARENT = natural_v4.PARENT
PARENT_SHA256 = natural_v4.PARENT_SHA256
BLUE = natural_v4.BLUE
ORANGE = natural_v4.ORANGE
BLUE_SHA256 = natural_v4.BLUE_SHA256
ORANGE_SHA256 = natural_v4.ORANGE_SHA256
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v7")
DEFAULT_COLLISION_DIR = natural_v4.DEFAULT_COLLISION_DIR


def load_authority() -> dict[str, Any]:
    if capability.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("natural V7 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected natural V7 authority format")
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"natural V7 bound input changed: {path}")
    if float(authority["reward"]["raw_airtime_reward"]) != 0.0:
        raise RuntimeError("raw airtime reward is forbidden")
    if int(authority["integrity"]["optimizer_steps_before_authority_commit"]) != 0:
        raise RuntimeError("natural V7 authority is not prospective")
    return authority


def collect_rollout(*args: Any, **kwargs: Any) -> Any:
    return natural_v6.collect_rollout(
        *args,
        **kwargs,
        pop_mask_factory=active_pop_orientation_channel_mask,
    )


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
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V7_FRESH_BALANCED_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": VERSION,
        "pop_control_identity": GROUND_TO_AIR_POP_CONTROL_V7_VERSION,
        "created_utc": natural_v6.utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "ground_to_air_goal_v3_parent_sha256": PARENT_SHA256,
        "protected_v23_defenders": {"blue": BLUE_SHA256, "orange": ORANGE_SHA256},
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "fixed_pop_pitch": True,
        "learned_pop_orientation_channels": ["steer", "yaw", "roll"],
        "scripted_pop_channels": ["throttle", "pitch", "jump", "boost", "handbrake"],
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
    strata = natural_v6.training_strata(authority)
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
        "created_utc": natural_v6.utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "parent_hash_verified": True,
        "protected_v23_defender_hashes_verified": True,
        "critic_frozen": True,
        "production_reward_unchanged": True,
        "raw_airtime_reward": authority["reward"]["raw_airtime_reward"],
        "pop_orientation_control": authority["pop_orientation_control"],
        "fixed_pop_pitch": True,
        "causal_pop_channels": ["steer", "yaw", "roll"],
        "channel_masked_likelihood": True,
        "equal_stratum_gradient_aggregation": True,
        "success_volume_rehearsal": False,
        "strata": strata,
        "optimizer_steps": 0,
        "baseline_validation": baseline,
        "baseline_selection_key": natural_v6.selection_key(baseline, authority),
        "verdict": "PASS",
    }
    natural_v6.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("natural V7 training requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    best_key = natural_v6.selection_key(baseline, authority)
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
        rollouts: list[natural_v6.MaskedOptionRollout] = []
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
            ppo = natural_v6.balanced_masked_ppo_update(
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
                raise FloatingPointError("nonfinite V7 aerial-option parameter")
        except (FloatingPointError, RuntimeError) as error:
            model.load_state_dict(block_model, strict=True)
            optimizer.load_state_dict(block_optimizer)
            stop_reason = f"hard_failure:{type(error).__name__}:{error}"
            natural_v6.append_jsonl(
                RESULTS / "hard_failure.jsonl",
                {"block": block, "created_utc": natural_v6.utc_now(), "reason": stop_reason},
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
            key = natural_v6.selection_key(validation, authority)
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
                        "stage": "ground_to_air_natural_v7",
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
                natural_v6.append_jsonl(curve, row)
                break
            if stale_boundaries >= int(training["plateau_patience_boundaries"]):
                stop_reason = "validation_plateau"
                natural_v6.append_jsonl(curve, row)
                break
        natural_v6.append_jsonl(curve, row)

    model.load_state_dict(best_model, strict=True)
    optimizer.load_state_dict(best_optimizer)
    observed_critic = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    if observed_critic != critic_hash:
        raise RuntimeError("natural V7 aerial option changed the frozen critic")
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
                CHECKPOINTS / "rival2_ground_to_air_natural_v7.pt",
                block=best_block,
                evaluation={"validation": best_evaluation, "test": test},
                disposition="promoted_controlled_option",
            )
        )
    result = {
        "format": f"{VERSION}_RESULT",
        "created_utc": natural_v6.utc_now(),
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
        "fixed_pop_pitch": True,
        "learned_pop_orientation_channels": ["steer", "yaw", "roll"],
        "channel_masked_likelihood": True,
        "equal_stratum_gradient_aggregation": True,
        "success_volume_rehearsal": False,
        "production_reward_unchanged": True,
        "checkpoints": checkpoints,
        "promoted_into_competitive_policy": False,
    }
    natural_v6.write_json(RESULTS / "result.json", result)
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
    parser.add_argument("--maximum-blocks", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
