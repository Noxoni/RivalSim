"""Train natural aerial entries with family/defender-local PPO advantages.

V4 proved the incoming-chip route but globally normalized a mixed rollout, so
that high-return family dominated low-bounce and defended matched-dribble
entries.  V5 restarts from the accepted controlled scorer and gives every
setup/defender/side stratum one independent rollout and optimizer update.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_PARKED,
    GROUND_TO_AIR_NATURAL_V4_VERSION,
    SETUP_NAMES,
)

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V5"
AUTHORITY = ROOT / "results/rival2/ground_to_air_natural_v5/authority.json"
AUTHORITY_SHA256 = "4EDFC3CE48DCD7BB9906DC0C0197445D49DD0E41B6ABB414211800C190080F96"
RESULTS = ROOT / "results/rival2/ground_to_air_natural_v5"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_natural_v5"
PARENT = natural_v4.PARENT
PARENT_SHA256 = natural_v4.PARENT_SHA256
BLUE = natural_v4.BLUE
ORANGE = natural_v4.ORANGE
BLUE_SHA256 = natural_v4.BLUE_SHA256
ORANGE_SHA256 = natural_v4.ORANGE_SHA256
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v5")
DEFAULT_COLLISION_DIR = natural_v4.DEFAULT_COLLISION_DIR


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
        raise RuntimeError("natural V5 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected natural V5 authority format")
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"natural V5 bound input changed: {path}")
    if float(authority["reward"]["raw_airtime_reward"]) != 0.0:
        raise RuntimeError("raw airtime reward is forbidden")
    if int(authority["integrity"]["optimizer_steps_before_authority_commit"]) != 0:
        raise RuntimeError("natural V5 authority is not prospective")
    return authority


def training_strata(authority: dict[str, Any]) -> list[dict[str, Any]]:
    strata = list(authority["scenario"]["training_strata"])
    expected = {
        (setup, defender)
        for setup in SETUP_NAMES
        for defender in (DEFENDER_PARKED, DEFENDER_LIVE)
    }
    observed = {(row["setup"], row["defender_mode"]) for row in strata}
    if observed != expected or len(strata) != len(expected):
        raise RuntimeError("V5 training strata must cover each setup/defender pair once")
    weights = {float(row["optimizer_weight"]) for row in strata}
    if weights != {1.0}:
        raise RuntimeError("V5 training strata must have equal optimizer weight")
    return strata


def save_checkpoint(
    source: dict[str, Any],
    model: torch.nn.Module,
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
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V5_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": VERSION,
        "scenario_identity": GROUND_TO_AIR_NATURAL_V4_VERSION,
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "ground_to_air_goal_v3_parent_sha256": PARENT_SHA256,
        "protected_v23_defenders": {
            "blue": BLUE_SHA256,
            "orange": ORANGE_SHA256,
        },
        "canonical_shared_option": True,
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "family_defender_local_advantages": True,
        "equal_stratum_optimizer_weight": True,
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
    distribution = natural_v4.distribution_override(authority)
    baseline = natural_v4.validation_rows(
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
        "family_defender_local_advantages": True,
        "strata": strata,
        "optimizer_steps": 0,
        "baseline_validation": baseline,
        "baseline_score": natural_v4.evaluation_score(baseline),
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("natural V5 training requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()

    best_score = natural_v4.evaluation_score(baseline)
    best_block = 0
    best_evaluation = copy.deepcopy(baseline)
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_checkpoint: dict[str, Any] | None = None
    stale_boundaries = 0
    consecutive_gate = 0
    stop_reason = "maximum_blocks"
    training = authority["training"]
    compatibility = goal_v3.ppo_compatibility(authority)
    maximum_blocks = min(int(training["maximum_blocks"]), int(args.maximum_blocks))
    interval = int(training["evaluation_interval_blocks"])
    horizon = int(authority["episode"]["horizon_ticks"])
    setup_by_name = {name: index for index, name in enumerate(SETUP_NAMES)}

    for block in range(1, maximum_blocks + 1):
        block_model = copy.deepcopy(model.state_dict())
        block_optimizer = copy.deepcopy(optimizer.state_dict())
        block_rows: list[dict[str, Any]] = []
        try:
            # Rotate the fixed equal-weight order, without changing membership or count.
            rotated = strata[(block - 1) % len(strata) :] + strata[: (block - 1) % len(strata)]
            for stratum_index, stratum in enumerate(rotated):
                setup = setup_by_name[stratum["setup"]]
                defender_mode = stratum["defender_mode"]
                for side in (0, 1):
                    seed = (
                        int(authority["seeds"]["training"])
                        + block * 100_000
                        + setup * 10_000
                        + (1_000 if defender_mode == DEFENDER_LIVE else 0)
                        + stratum_index * 100
                    )
                    rollout, rollout_metrics = natural_v4.collect_rollout(
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
                        live_defender_fraction=(
                            1.0 if defender_mode == DEFENDER_LIVE else 0.0
                        ),
                        attacker_boost_range=tuple(
                            authority["scenario"]["training_boost_range"]
                        ),
                    )
                    assert rollout is not None
                    ppo = natural_v4.aerial_v1.option_ppo_update(
                        model,
                        optimizer,
                        rollout,
                        generator=generators[side],
                        distribution=distribution,
                        authority=compatibility,
                    )
                    rehearsal = goal_v3.successful_rehearsal_update(
                        model,
                        optimizer,
                        rollout,
                        authority=authority,
                        generator=generators[side],
                    )
                    goal_rehearsal = goal_v3.successful_rehearsal_update(
                        model,
                        optimizer,
                        rollout,
                        authority=authority,
                        generator=generators[side],
                        rehearsal_key="goal_rehearsal",
                    )
                    del rollout
                    if not all(
                        bool(torch.isfinite(parameter).all())
                        for parameter in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite V5 aerial-option parameter")
                    block_rows.append(
                        {
                            "setup": stratum["setup"],
                            "defender_mode": defender_mode,
                            "side": side,
                            "rollout": rollout_metrics,
                            "ppo": ppo,
                            "successful_rehearsal": rehearsal,
                            "goal_rehearsal": goal_rehearsal,
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

        row: dict[str, Any] = {"block": block, "strata": block_rows}
        if block % interval == 0 or block == maximum_blocks:
            validation = natural_v4.validation_rows(
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
            score = natural_v4.evaluation_score(validation)
            passed = natural_v4.passes_gate(validation, authority)
            improved = score > best_score + float(training["minimum_score_improvement"])
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
            rolling = save_checkpoint(
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
            row["rolling_checkpoint"] = rolling
            print(
                json.dumps(
                    {
                        "stage": "ground_to_air_natural_v5",
                        "block": block,
                        "score": score,
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
        gc.collect()
        torch.cuda.empty_cache()

    model.load_state_dict(best_model, strict=True)
    optimizer.load_state_dict(best_optimizer)
    observed_critic = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    if observed_critic != critic_hash:
        raise RuntimeError("natural V5 aerial option changed the frozen critic")
    validation_pass = natural_v4.passes_gate(best_evaluation, authority)
    test: list[dict[str, Any]] | None = None
    if validation_pass:
        test = natural_v4.validation_rows(
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
                CHECKPOINTS / "rival2_ground_to_air_natural_v5.pt",
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
        "best_score": best_score,
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
        "family_defender_local_advantages": True,
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
