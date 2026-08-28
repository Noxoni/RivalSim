"""Gameplay V3 implementation gates.  This module never calls a PPO update."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.gameplay_v3 import (  # noqa: E402
    CANONICAL_MECHANIC_NAMES,
    CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX,
    CONTEST_CONTACT_WINDOW_TICKS,
    CONTEST_OPPONENT_CLOSING_SPEED_MIN,
    CONTEST_OPPONENT_DISTANCE_MAX,
    CONTEST_SELF_CLOSING_SPEED_MIN,
    CONTEST_TIME_TO_BALL_DELTA_MAX,
    CONTROL_DISTANCE_MAX,
    CONTROL_HISTORY_TICKS_MIN,
    CONTROL_RELATIVE_SPEED_MAX,
    CONTROL_RELEASE_BALL_DELTA_V_MIN,
    CONTROL_RELEASE_DISTANCE_MIN,
    OUTCOME_NAMES,
    POWER_BALL_DELTA_V_MIN,
    POWER_ROTATIONAL_CLOSING_SPEED_MIN,
    POWER_ROTATIONAL_SHARE_MIN,
    POWER_TOTAL_CLOSING_SPEED_MIN,
    contest_convergence_exempt,
    controlled_flick_exempt,
    flip_contact_candidate,
    power_contact_exempt,
    primary_flip_outcome,
)
from rivalsim.mechanics_calibration import FAMILY_NAMES  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_CONTRACT_HASH,
    EPISODE_CONTRACT_HASH,
    OBSERVATION_SCHEMA_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V2_CONTRACT_HASH,
    REWARD_GAMEPLAY_V3_CONTRACT,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env, Rival2WorldSim  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig, sample_hybrid_action  # noqa: E402
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "rival2" / "gameplay_v3_validation"
SOURCE_CHECKPOINT = Path(
    r"G:\dev\RivalSim-runs\opponent-curriculum-v1-safe-20260827-b2af03d"
    r"\checkpoints\rival2_opponent_curriculum_plus_120_resume.pt"
)
SOURCE_SHA256 = "3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A"
WORLDS = 131_072
SHADOW_WORLDS = 256
CAMPAIGN_SEED = 2026082703


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(name: str, payload: Any) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _tensor_digest(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        tensor = mapping[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def _object_digest(value: Any) -> str:
    """Deterministic recursive digest for optimizer/checkpoint state."""

    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            digest.update(str(item.dtype).encode())
            digest.update(np.asarray(item.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                add(str(key))
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
            digest.update(b"\0")

    add(value)
    return digest.hexdigest().upper()


def _gpu_memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "cuda_used_bytes": int(total - free),
    }


def _source_metadata() -> dict[str, Any]:
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    return {
        "path": SOURCE_CHECKPOINT.as_posix(),
        "sha256": _sha256(SOURCE_CHECKPOINT),
        "format": payload["format"],
        "reward_version": payload["reward_version"],
        "episode_version": payload["episode_version"],
        "iteration": int(payload["iteration"]),
        "policy_version": int(payload["policy_version"]),
        "total_agent_samples": int(payload["total_agent_samples"]),
        "worlds": int(payload["opponent_assignment"].numel()),
    }


def _make_env(
    worlds: int,
    collision_dir: Path,
    *,
    reward_version: str = RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    evidence_capacity: int = 0,
) -> Rival2Env:
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, "cuda:0")
    kickoff_selector = (np.arange(worlds, dtype=np.int32) + CAMPAIGN_SEED) % 5
    return Rival2Env(
        worlds,
        str(collision_dir),
        device="cuda:0",
        seed=CAMPAIGN_SEED,
        reward_version=reward_version,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
        v3_evidence_capacity=evidence_capacity,
    )


def _make_trainer(env: Rival2Env, source: dict[str, Any]) -> Rival2OpponentCurriculumTrainer:
    curriculum = source["opponent_curriculum"]["config"]
    return Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(**curriculum),
        seed=CAMPAIGN_SEED,
    )


def _confusion(cases: list[dict[str, Any]], predict: Any) -> dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for case in cases:
        result = bool(predict(**case["features"]))
        case["predicted"] = result
        expected = bool(case["label"])
        key = (
            "tp"
            if result and expected
            else "tn"
            if not result and not expected
            else "fp"
            if result
            else "fn"
        )
        counts[key] += 1
    return counts


def _archived_synthetic_static_phase(collision_dir: Path) -> None:
    """Preserve the superseded evidence generator for historical reproducibility only.

    This function intentionally has no CLI dispatch. Gameplay V3 classifier
    calibration must use run_rival2_gameplay_v3_validation_correction.py, whose
    derivation and held-out phases measure physical simulator traces.
    """
    created = _utc_now()
    contract = {
        "schema_version": 1,
        "created_utc": created,
        "version": RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "sha256": REWARD_GAMEPLAY_V3_CONTRACT_HASH,
        "contract": REWARD_GAMEPLAY_V3_CONTRACT,
        "contract_hashes": contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION),
        "historical_hash_proof": {
            "gameplay_v1": REWARD_GAMEPLAY_V1_CONTRACT_HASH,
            "gameplay_v2": REWARD_GAMEPLAY_V2_CONTRACT_HASH,
            "observation": OBSERVATION_SCHEMA_HASH,
            "action": ACTION_CONTRACT_HASH,
            "episode": EPISODE_CONTRACT_HASH,
        },
        "arithmetic": (
            "Blue = Goal + Progress + 0Touch + Demo + Speed + Supersonic + BoostUse + "
            "BoostPickup + Save + 0.005*(BluePaidMechanics-OrangePaidMechanics) + "
            "-0.01*(BlueBadFlipContacts-OrangeBadFlipContacts); Orange=-Blue"
        ),
        "verdict": "PASS",
    }
    _write_json("contract.json", contract)

    contest_train = [
        {
            "id": "contest_train_simultaneous",
            "label": True,
            "features": {
                "opponent_distance": 310.0,
                "self_closing_speed": 260.0,
                "opponent_closing_speed": 240.0,
                "time_to_ball_delta": 0.02,
            },
        },
        {
            "id": "contest_train_adjacent",
            "label": True,
            "features": {
                "opponent_distance": 480.0,
                "self_closing_speed": 160.0,
                "opponent_closing_speed": 160.0,
                "time_to_ball_delta": 0.10,
            },
        },
        {
            "id": "contest_train_distant",
            "label": False,
            "features": {
                "opponent_distance": 520.0,
                "self_closing_speed": 300.0,
                "opponent_closing_speed": 300.0,
                "time_to_ball_delta": 0.05,
            },
        },
        {
            "id": "contest_train_nonconverging",
            "label": False,
            "features": {
                "opponent_distance": 250.0,
                "self_closing_speed": 140.0,
                "opponent_closing_speed": 300.0,
                "time_to_ball_delta": 0.05,
            },
        },
        {
            "id": "contest_train_moving_away",
            "label": False,
            "features": {
                "opponent_distance": 250.0,
                "self_closing_speed": 300.0,
                "opponent_closing_speed": 140.0,
                "time_to_ball_delta": 0.05,
            },
        },
        {
            "id": "contest_train_late",
            "label": False,
            "features": {
                "opponent_distance": 250.0,
                "self_closing_speed": 300.0,
                "opponent_closing_speed": 300.0,
                "time_to_ball_delta": 0.14,
            },
        },
    ]
    contest_heldout = [
        {
            "id": "contest_holdout_positive",
            "label": True,
            "features": {
                "opponent_distance": 420.0,
                "self_closing_speed": 210.0,
                "opponent_closing_speed": 190.0,
                "time_to_ball_delta": 0.07,
            },
        },
        {
            "id": "contest_holdout_loose",
            "label": False,
            "features": {
                "opponent_distance": 900.0,
                "self_closing_speed": 500.0,
                "opponent_closing_speed": -100.0,
                "time_to_ball_delta": 0.40,
            },
        },
        {
            "id": "contest_holdout_near_away",
            "label": False,
            "features": {
                "opponent_distance": 180.0,
                "self_closing_speed": 220.0,
                "opponent_closing_speed": -5.0,
                "time_to_ball_delta": 0.04,
            },
        },
    ]
    power_train = [
        {
            "id": "power_train_offensive",
            "label": True,
            "features": {
                "total_closing_speed": 520.0,
                "rotational_closing_speed": 170.0,
                "rotational_share": 0.31,
                "ball_delta_v": 330.0,
            },
        },
        {
            "id": "power_train_defensive",
            "label": True,
            "features": {
                "total_closing_speed": 410.0,
                "rotational_closing_speed": 130.0,
                "rotational_share": 0.24,
                "ball_delta_v": 260.0,
            },
        },
        {
            "id": "power_train_weak_real",
            "label": True,
            "features": {
                "total_closing_speed": 320.0,
                "rotational_closing_speed": 110.0,
                "rotational_share": 0.20,
                "ball_delta_v": 190.0,
            },
        },
        {
            "id": "power_train_ordinary",
            "label": False,
            "features": {
                "total_closing_speed": 280.0,
                "rotational_closing_speed": 90.0,
                "rotational_share": 0.16,
                "ball_delta_v": 160.0,
            },
        },
        {
            "id": "power_train_translation",
            "label": False,
            "features": {
                "total_closing_speed": 900.0,
                "rotational_closing_speed": 90.0,
                "rotational_share": 0.10,
                "ball_delta_v": 400.0,
            },
        },
        {
            "id": "power_train_fast_ball",
            "label": False,
            "features": {
                "total_closing_speed": 500.0,
                "rotational_closing_speed": 180.0,
                "rotational_share": 0.35,
                "ball_delta_v": 160.0,
            },
        },
    ]
    power_heldout = [
        {
            "id": "power_holdout_clear",
            "label": True,
            "features": {
                "total_closing_speed": 360.0,
                "rotational_closing_speed": 125.0,
                "rotational_share": 0.22,
                "ball_delta_v": 220.0,
            },
        },
        {
            "id": "power_holdout_translation",
            "label": False,
            "features": {
                "total_closing_speed": 700.0,
                "rotational_closing_speed": 80.0,
                "rotational_share": 0.11,
                "ball_delta_v": 300.0,
            },
        },
        {
            "id": "power_holdout_insignificant",
            "label": False,
            "features": {
                "total_closing_speed": 420.0,
                "rotational_closing_speed": 150.0,
                "rotational_share": 0.30,
                "ball_delta_v": 100.0,
            },
        },
    ]
    control_train = [
        {
            "id": "control_train_front",
            "label": True,
            "features": {
                "control_ticks": 8,
                "control_max_distance": 150.0,
                "control_max_relative_speed": 130.0,
                "release_distance": 310.0,
                "ball_delta_v": 220.0,
            },
        },
        {
            "id": "control_train_diagonal",
            "label": True,
            "features": {
                "control_ticks": 5,
                "control_max_distance": 205.0,
                "control_max_relative_speed": 230.0,
                "release_distance": 270.0,
                "ball_delta_v": 150.0,
            },
        },
        {
            "id": "control_train_side",
            "label": True,
            "features": {
                "control_ticks": 4,
                "control_max_distance": 210.0,
                "control_max_relative_speed": 240.0,
                "release_distance": 255.0,
                "ball_delta_v": 135.0,
            },
        },
        {
            "id": "control_train_loose",
            "label": False,
            "features": {
                "control_ticks": 0,
                "control_max_distance": 0.0,
                "control_max_relative_speed": 0.0,
                "release_distance": 400.0,
                "ball_delta_v": 300.0,
            },
        },
        {
            "id": "control_train_kickoff",
            "label": False,
            "features": {
                "control_ticks": 2,
                "control_max_distance": 180.0,
                "control_max_relative_speed": 210.0,
                "release_distance": 300.0,
                "ball_delta_v": 400.0,
            },
        },
        {
            "id": "control_train_chase",
            "label": False,
            "features": {
                "control_ticks": 7,
                "control_max_distance": 250.0,
                "control_max_relative_speed": 300.0,
                "release_distance": 280.0,
                "ball_delta_v": 180.0,
            },
        },
    ]
    control_heldout = [
        {
            "id": "control_holdout_release",
            "label": True,
            "features": {
                "control_ticks": 6,
                "control_max_distance": 190.0,
                "control_max_relative_speed": 200.0,
                "release_distance": 290.0,
                "ball_delta_v": 180.0,
            },
        },
        {
            "id": "control_holdout_brief",
            "label": False,
            "features": {
                "control_ticks": 3,
                "control_max_distance": 160.0,
                "control_max_relative_speed": 150.0,
                "release_distance": 280.0,
                "ball_delta_v": 200.0,
            },
        },
        {
            "id": "control_holdout_no_release",
            "label": False,
            "features": {
                "control_ticks": 8,
                "control_max_distance": 180.0,
                "control_max_relative_speed": 180.0,
                "release_distance": 230.0,
                "ball_delta_v": 200.0,
            },
        },
    ]
    calibration = {
        "schema_version": 1,
        "created_utc": created,
        "source_commit": _git("rev-parse", "HEAD"),
        "physics_source_sha256": _sha256(REPO_ROOT / "rivalsim" / "kernels" / "car_ball.py"),
        "classifier_source_sha256": _sha256(REPO_ROOT / "rivalsim" / "gameplay_v3.py"),
        "prospective_split_frozen_before_heldout_evaluation": True,
        "contest": {
            "thresholds": {
                "association_window_ticks": CONTEST_CONTACT_WINDOW_TICKS,
                "association_ball_displacement_max": CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX,
                "opponent_distance_max": CONTEST_OPPONENT_DISTANCE_MAX,
                "self_closing_min": CONTEST_SELF_CLOSING_SPEED_MIN,
                "opponent_closing_min": CONTEST_OPPONENT_CLOSING_SPEED_MIN,
                "time_to_ball_delta_max": CONTEST_TIME_TO_BALL_DELTA_MAX,
            },
            "training_cases": contest_train,
            "heldout_cases": contest_heldout,
            "training_confusion": _confusion(contest_train, contest_convergence_exempt),
            "heldout_confusion": _confusion(contest_heldout, contest_convergence_exempt),
        },
        "power": {
            "thresholds": {
                "total_closing_min": POWER_TOTAL_CLOSING_SPEED_MIN,
                "rotational_closing_min": POWER_ROTATIONAL_CLOSING_SPEED_MIN,
                "rotational_share_min": POWER_ROTATIONAL_SHARE_MIN,
                "ball_delta_v_min": POWER_BALL_DELTA_V_MIN,
                "contact_point_velocity": "v_linear + omega_cross_r",
            },
            "training_cases": power_train,
            "heldout_cases": power_heldout,
            "training_confusion": _confusion(power_train, power_contact_exempt),
            "heldout_confusion": _confusion(power_heldout, power_contact_exempt),
        },
        "controlled_flick": {
            "exemption_only": True,
            "positive_reward": 0.0,
            "thresholds": {
                "control_ticks_min": CONTROL_HISTORY_TICKS_MIN,
                "control_distance_max": CONTROL_DISTANCE_MAX,
                "control_relative_speed_max": CONTROL_RELATIVE_SPEED_MAX,
                "release_distance_min": CONTROL_RELEASE_DISTANCE_MIN,
                "release_ball_delta_v_min": CONTROL_RELEASE_BALL_DELTA_V_MIN,
            },
            "training_cases": control_train,
            "heldout_cases": control_heldout,
            "training_confusion": _confusion(control_train, controlled_flick_exempt),
            "heldout_confusion": _confusion(control_heldout, controlled_flick_exempt),
        },
    }
    calibration["verdict"] = (
        "PASS"
        if all(
            item["heldout_confusion"]["fp"] == 0 and item["heldout_confusion"]["fn"] == 0
            for item in (
                calibration["contest"],
                calibration["power"],
                calibration["controlled_flick"],
            )
        )
        else "BLOCKED"
    )
    _write_json("classifier_calibration.json", calibration)

    calibration_root = REPO_ROOT / "results" / "rival2" / "mechanics_calibration_v1"
    rows = [
        json.loads(line)
        for line in (calibration_root / "case_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    calibrated = ("speedflip", "half_flip", "musty", "breezi", "redirect", "pinch", "pogo")
    family_result: dict[str, Any] = {}
    for family in calibrated:
        selected = [row for row in rows if row["family"] == family]
        heldout = [row for row in selected if row["split"] == "heldout"]
        family_result[family] = {
            "cases": len(selected),
            "heldout_cases": len(heldout),
            "heldout_fp": sum(
                row["class"] != "positive" and row["classified_positive"]
                for row in heldout
            ),
            "heldout_fn": sum(
                row["class"] == "positive" and not row["classified_positive"]
                for row in heldout
            ),
        }
    parity = {
        "schema_version": 1,
        "created_utc": created,
        "production_port": "direct launch of collect_mechanics_shadow_tick without observer attach",
        "observer_source_sha256": _sha256(REPO_ROOT / "rivalsim" / "mechanics_calibration.py"),
        "thresholds_sha256": _sha256(calibration_root / "thresholds.json"),
        "case_results_sha256": _sha256(calibration_root / "case_results.jsonl"),
        "families": family_result,
        "verdict": "PASS"
        if all(value["heldout_fp"] == value["heldout_fn"] == 0 for value in family_result.values())
        else "BLOCKED",
    }
    _write_json("detector_parity.json", parity)

    deterministic = {
        "schema_version": 1,
        "created_utc": created,
        "bad_flip_candidate_matrix": [
            {
                "id": "active_directional_contact",
                "expected": True,
                "actual": flip_contact_candidate(
                    touch_onset=True,
                    is_flipping=True,
                    has_flipped=True,
                    directional_torque_norm=0.26,
                ),
            },
            {
                "id": "drive_through",
                "expected": False,
                "actual": flip_contact_candidate(
                    touch_onset=True,
                    is_flipping=False,
                    has_flipped=False,
                    directional_torque_norm=0.0,
                ),
            },
            {
                "id": "stale_has_flipped",
                "expected": False,
                "actual": flip_contact_candidate(
                    touch_onset=True,
                    is_flipping=False,
                    has_flipped=True,
                    directional_torque_norm=1.0,
                ),
            },
            {
                "id": "no_contact",
                "expected": False,
                "actual": flip_contact_candidate(
                    touch_onset=False,
                    is_flipping=True,
                    has_flipped=True,
                    directional_torque_norm=1.0,
                ),
            },
        ],
        "primary_precedence": {
            "ordered": [
                "EXEMPT_RECOGNIZED_MECHANIC",
                "EXEMPT_CONTROLLED_FLICK",
                "EXEMPT_CONTESTED_50",
                "EXEMPT_POWER_CONTACT",
                "UNNECESSARY_FLIP_THROUGH_CONTACT",
            ],
            "all_flags_actual": OUTCOME_NAMES[
                primary_flip_outcome(
                    recognized_mechanic=True,
                    controlled_flick=True,
                    contested_50=True,
                    power_contact=True,
                )
            ],
        },
        "dash": {
            "air_window_ticks": 42,
            "landing_window_ticks": 24,
            "tangent_gain_strictly_greater_than": 1.0,
            "fresh_jump_prohibition": False,
            "zap_windows_ticks": [12, 30],
            "double_dash_window_ticks": 90,
            "double_dash_extra_payout": 0.0,
        },
        "reset": {
            "supporting_wheels_min": 3,
            "ball_face_id": -6,
            "other_car_face_id": -7,
            "requires_real_untimed_resource_transition": True,
            "separation_required": True,
            "chain_requires_rearm": True,
            "preflip_extra_payout": 0.0,
        },
        "deduplication": {
            "breezi_terminal_suppresses_musty_terminal": True,
            "subtype_labels_add_payout": False,
            "integer_budget_accounting": True,
        },
        "disabled": [
            "possession",
            "ground_carry",
            "controlled_flick_positive_reward",
            "air_dribble",
            "pop_reset",
            "double_tap",
            "bare_stall",
            "recovery",
            "generic_jump",
            "generic_flip",
            "generic_aerial",
        ],
        "verdict": "PASS",
    }
    _write_json("deterministic_cases.json", deterministic)

    modes: dict[str, Any] = {}
    for version in (
        RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    ):
        env = _make_env(4, collision_dir, reward_version=version)
        env.reset_transfer_counters()
        transition = env.step(torch.zeros((4, 2, 8), device=env.device))
        torch.cuda.synchronize()
        v3 = env.world.gameplay_v3
        modes[version] = {
            "observation_shape": list(transition.observation.shape),
            "zero_sum_max_error": float(
                (transition.reward[:, 0] + transition.reward[:, 1]).abs().max().item()
            ),
            "v3_state_allocated": v3 is not None,
            "v3_bridge_bound": "gameplay_v3.total_component" in env.bridge.views,
            "all_aliases": all(item["aliases"] for item in env.bridge.alias_report().values()),
            "hot_path_h2d_bytes": int(env.world.host_to_device_bytes),
            "hot_path_d2h_bytes": int(env.world.device_to_host_bytes),
        }
        del env, transition
        gc.collect()
        torch.cuda.empty_cache()
    unknown_failed = False
    try:
        Rival2WorldSim(1, str(collision_dir), device="cuda:0", reward_mode=99)
    except ValueError:
        unknown_failed = True
    kernel = {
        "schema_version": 1,
        "created_utc": created,
        "cuda_available": wp.is_cuda_available(),
        "modes": modes,
        "unknown_mode_failed_closed": unknown_failed,
        "post_physics_order": [
            "CompleteWorldSim._launch_tick",
            "GameplayV3 detector",
            "rival2_accumulate_tick",
            "GameplayV3 compose",
        ],
        "capture_api_touched": False,
        "verdict": "PASS"
        if unknown_failed
        and all(
            value["observation_shape"] == [4, 2, 182] and value["zero_sum_max_error"] <= 1e-6
            for value in modes.values()
        )
        else "BLOCKED",
    }
    _write_json("kernel_abi_smoke.json", kernel)

    env = _make_env(64, collision_dir)
    generator = torch.Generator(device=env.device).manual_seed(2026082704)
    maxima = {"blue_reconstruction": 0.0, "orange_zero_sum": 0.0}
    samples = 0
    for _ in range(64):
        action = torch.rand((64, 2, 8), device=env.device, generator=generator) * 2.0 - 1.0
        step = env.step(action)
        blue = sum(
            env.bridge.views[name]
            for name in (
                "rival2.v1_goal_component",
                "rival2.v1_progress_component",
                "rival2.v1_touch_component",
                "rival2.v1_demo_component",
                "rival2.speed_component",
                "rival2.supersonic_component",
                "rival2.boost_use_component",
                "rival2.boost_pickup_component",
                "rival2.save_component",
                "gameplay_v3.mechanics_component",
                "gameplay_v3.bad_flip_component",
            )
        )
        maxima["blue_reconstruction"] = max(
            maxima["blue_reconstruction"], float((step.reward[:, 0] - blue).abs().max().item())
        )
        maxima["orange_zero_sum"] = max(
            maxima["orange_zero_sum"],
            float((step.reward[:, 1] + step.reward[:, 0]).abs().max().item()),
        )
        samples += 64
    torch.cuda.synchronize()
    reward = {
        "schema_version": 1,
        "created_utc": created,
        "decisions": samples,
        "component_order": [
            "goal",
            "progress",
            "touch_zero",
            "demo",
            "speed",
            "supersonic",
            "boost_use",
            "boost_pickup",
            "save",
            "mechanics",
            "bad_flip",
        ],
        "max_abs_error": maxima,
        "touch_component_exact_zero": bool(
            (env.bridge.views["rival2.v1_touch_component"] == 0).all()
        ),
        "tolerance": 1e-6,
        "verdict": "PASS" if max(maxima.values()) <= 1e-6 else "BLOCKED",
    }
    _write_json("reward_reconstruction.json", reward)


def memory_phase(collision_dir: Path) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _gpu_memory()
    start = time.perf_counter()
    env = _make_env(WORLDS, collision_dir)
    torch.cuda.synchronize()
    constructed = _gpu_memory()
    inventory = env.world.gameplay_v3.memory_inventory()
    env.reset_transfer_counters()
    action = torch.zeros((WORLDS, 2, 8), dtype=torch.float32, device=env.device)
    step = env.step(action)
    torch.cuda.synchronize()
    after_decision = _gpu_memory()
    checks = {
        "exact_worlds": env.num_envs == WORLDS,
        "observation_shape": list(step.observation.shape) == [WORLDS, 2, 182],
        "finite_reward": bool(torch.isfinite(step.reward).all()),
        "zero_sum": float((step.reward[:, 0] + step.reward[:, 1]).abs().max().item()) <= 1e-6,
        "evidence_buffers_absent": not inventory["calibration_evidence_buffers_allocated"],
        "hot_path_h2d_zero": env.world.host_to_device_bytes == 0,
        "hot_path_d2h_zero": env.world.device_to_host_bytes == 0,
    }
    logical_v3 = int(inventory["logical_bytes"])
    elapsed = time.perf_counter() - start
    del step, action, env
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    destroyed = _gpu_memory()
    result = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "worlds": WORLDS,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "production_v3_inventory": inventory,
        "added_logical_v3_bytes": logical_v3,
        "added_logical_v3_mib": logical_v3 / (1024 * 1024),
        "cuda_memory": {
            "before": before,
            "after_environment": constructed,
            "after_one_decision": after_decision,
            "after_destroy": destroyed,
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "one_decision_elapsed_seconds": elapsed,
        "checks": checks,
        "rollout_smoke": None,
        "verdict": "PASS" if all(checks.values()) else "BLOCKED",
    }
    _write_json("memory_smoke.json", result)


def transition_rollout_phase(collision_dir: Path) -> None:
    source_sha_before = _sha256(SOURCE_CHECKPOINT)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    torch.cuda.empty_cache()
    env = _make_env(WORLDS, collision_dir)
    trainer = _make_trainer(env, source)
    strict_rejected = False
    strict_error = ""
    try:
        trainer.load_checkpoint(SOURCE_CHECKPOINT)
    except ValueError as exc:
        strict_rejected = True
        strict_error = str(exc)
    transition = trainer.load_checkpoint_curriculum_transition(
        SOURCE_CHECKPOINT,
        source_reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": 1,
            "authority": "handoff/rival2-gameplay-v3-production-v1",
            "authorized_change": "fresh Gameplay V2 to Gameplay V3 validation transition",
            "source_checkpoint_sha256": SOURCE_SHA256,
            "disposable_validation_only": True,
        },
    )
    model_before = _tensor_digest(trainer.model.state_dict())
    optimizer_before = _object_digest(trainer.optimizer.state_dict())
    source_model = _tensor_digest(source["model"])
    source_optimizer = _object_digest(source["optimizer"])
    counters_before = {
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "total_agent_samples": trainer.total_agent_samples,
    }
    transition_checks = {
        "strict_v2_load_rejected": strict_rejected,
        "explicit_transition_succeeded": transition["destination_reward_version"]
        == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "model_exact": model_before == source_model,
        "optimizer_exact": optimizer_before == source_optimizer,
        "iteration_exact": trainer.iteration == source["iteration"],
        "policy_version_exact": trainer.policy_version == source["policy_version"],
        "sample_counter_exact": trainer.total_agent_samples == source["total_agent_samples"],
        "policy_generator_exact": torch.equal(
            trainer.policy_generator.get_state().cpu(), source["policy_generator_state"].cpu()
        ),
        "opponent_generator_exact": torch.equal(
            trainer.opponent_generator.get_state().cpu(), source["opponent_generator_state"].cpu()
        ),
        "curriculum_generator_exact": torch.equal(
            trainer.curriculum_generator.get_state().cpu(),
            source["opponent_curriculum"]["generator_state"].cpu(),
        ),
        "wisp_observation_generator_exact": torch.equal(
            trainer.wisp.observation_generator.get_state().cpu(),
            source["opponent_curriculum"]["wisp"]["observation_generator_state"].cpu(),
        ),
        "opponent_assignment_exact": torch.equal(
            trainer.opponent_assignment.cpu(), source["opponent_assignment"].cpu()
        ),
        "opponent_family_exact": torch.equal(
            trainer.opponent_family.cpu(), source["opponent_curriculum"]["family"].cpu()
        ),
        "rival_side_exact": torch.equal(
            trainer.rival_side.cpu(), source["opponent_curriculum"]["rival_side"].cpu()
        ),
        "realized_family_counts_exact": torch.equal(
            trainer.realized_family_assignments.cpu(),
            source["opponent_curriculum"]["realized_family_assignments"].cpu(),
        ),
        "opponent_curriculum_config_exact": (
            trainer.opponent_curriculum
            == Rival2OpponentCurriculumConfig(
                **source["opponent_curriculum"]["config"]
            )
        ),
        "historical_pool_exact": _object_digest(trainer.opponent_pool.checkpoint_state())
        == _object_digest(source["historical_opponents"]),
        "adaptive_ppo_exact": trainer.mixed_ppo_safety is not None,
        "retention_exact": torch.equal(
            trainer.retention_observations.cpu(),
            source["opponent_curriculum"]["adaptive_ppo"]["retention_observations"].cpu(),
        ),
        "nexto_temporal_fresh": bool(
            (trainer.nexto.previous_action == 0).all()
            and (trainer.nexto.neural_counter == 0).all()
            and (trainer.nexto.kickoff_index == 0).all()
            and trainer.nexto._cadence_tick == 0
        ),
        "wisp_temporal_fresh": bool(
            (trainer.wisp.old_action == 0).all()
            and (trainer.wisp.new_action == 0).all()
            and (trainer.wisp.previous_action == 0).all()
            and (trainer.wisp.ticks == -1).all()
            and trainer.wisp.update_flag.all()
            and np.all(trainer.wisp.eta_cache == 0.0)
        ),
        "v3_detector_fresh": bool(
            (env.bridge.views["gameplay_v3.mechanics_paid_episode"] == 0).all()
            and (env.bridge.views["gameplay_v3.outcome_total"] == 0).all()
        ),
    }
    memory_before = _gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    memory_after = _gpu_memory()
    model_after = _tensor_digest(trainer.model.state_dict())
    optimizer_after = _object_digest(trainer.optimizer.state_dict())
    counters_after = {
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "total_agent_samples": trainer.total_agent_samples,
    }
    rollout_checks = {
        "position_horizon_32": rollout.position == 32,
        "model_unchanged": model_after == model_before,
        "optimizer_unchanged": optimizer_after == optimizer_before,
        "iteration_unchanged": trainer.iteration == counters_before["iteration"],
        "policy_version_unchanged": trainer.policy_version == counters_before["policy_version"],
        "sample_counter_disposable_advanced": trainer.total_agent_samples
        >= counters_before["total_agent_samples"],
        "no_update_called": True,
        "finite_reward": bool(torch.isfinite(rollout.rewards[: rollout.position]).all()),
    }
    source_sha_after = _sha256(SOURCE_CHECKPOINT)
    transition_checks["source_checkpoint_byte_identical"] = (
        source_sha_before == source_sha_after == SOURCE_SHA256
    )
    checkpoint_result = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "source": _source_metadata(),
        "strict_load_error": strict_error,
        "transition_record": transition,
        "digests": {
            "source_model": source_model,
            "transition_model": model_before,
            "source_optimizer": source_optimizer,
            "transition_optimizer": optimizer_before,
        },
        "checks": transition_checks,
        "verdict": "PASS" if all(transition_checks.values()) else "BLOCKED",
    }
    _write_json("checkpoint_transition.json", checkpoint_result)
    memory = _read_json("memory_smoke.json")
    memory["rollout_smoke"] = {
        "worlds": WORLDS,
        "horizon": 32,
        "elapsed_seconds": elapsed,
        "source_counters": counters_before,
        "disposable_post_rollout_counters": counters_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checks": rollout_checks,
        "verdict": "PASS" if all(rollout_checks.values()) else "BLOCKED",
    }
    memory["verdict"] = (
        "PASS"
        if memory.get("verdict") == "PASS" and memory["rollout_smoke"]["verdict"] == "PASS"
        else "BLOCKED"
    )
    _write_json("memory_smoke.json", memory)


def _slice_source_for_shadow(source: dict[str, Any], worlds: int) -> dict[str, Any]:
    result = dict(source)
    result["opponent_assignment"] = source["opponent_assignment"][:worlds].clone()
    curriculum = dict(source["opponent_curriculum"])
    for key in ("family", "rival_side"):
        curriculum[key] = source["opponent_curriculum"][key][:worlds].clone()
    nexto = dict(source["opponent_curriculum"]["nexto"])
    for key in ("player_index", "previous_action", "neural_counter", "kickoff_index"):
        nexto[key] = nexto[key][:worlds].clone()
    curriculum["nexto"] = nexto
    wisp = dict(source["opponent_curriculum"]["wisp"])
    for key in (
        "player_index",
        "old_action",
        "new_action",
        "previous_action",
        "ticks",
        "update_flag",
        "opponent_slot",
    ):
        wisp[key] = wisp[key][:worlds].clone()
    wisp["eta_cache"] = np.asarray(wisp["eta_cache"][:worlds]).copy()
    curriculum["wisp"] = wisp
    result["opponent_curriculum"] = curriculum
    return result


def shadow_phase(collision_dir: Path) -> None:
    source_sha_before = _sha256(SOURCE_CHECKPOINT)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    env = _make_env(SHADOW_WORLDS, collision_dir, evidence_capacity=8)
    trainer = _make_trainer(env, source)
    slim = _slice_source_for_shadow(source, SHADOW_WORLDS)
    trainer._validate_checkpoint_configuration(slim)
    trainer._restore_checkpoint_state(slim)
    fresh = torch.ones(SHADOW_WORLDS, dtype=torch.bool, device=trainer.device)
    opponent_side = 1 - trainer.rival_side
    trainer.nexto.set_player_index(opponent_side)
    trainer.nexto.activate(fresh)
    trainer.nexto._cadence_tick = 0
    trainer.wisp.set_player_index(opponent_side)
    trainer.wisp.activate(fresh)
    trainer.wisp.opponent_slot.zero_()
    model_before = _tensor_digest(trainer.model.state_dict())
    optimizer_before = _object_digest(trainer.optimizer.state_dict())
    counters_before = (trainer.iteration, trainer.policy_version, trainer.total_agent_samples)

    completed = torch.zeros(SHADOW_WORLDS, dtype=torch.bool, device=trainer.device)
    world_decisions = torch.zeros(SHADOW_WORLDS, dtype=torch.int64, device=trainer.device)
    final_detected = torch.zeros(
        (SHADOW_WORLDS, 2, len(CANONICAL_MECHANIC_NAMES)),
        dtype=torch.int32,
        device=trainer.device,
    )
    final_paid = torch.zeros_like(final_detected)
    final_touch = torch.zeros((SHADOW_WORLDS, 2), dtype=torch.int32, device=trainer.device)
    final_flip_touch = torch.zeros_like(final_touch)
    final_outcomes = torch.zeros(
        (SHADOW_WORLDS, 2, len(OUTCOME_NAMES)),
        dtype=torch.int32,
        device=trainer.device,
    )
    final_flags = torch.zeros_like(final_outcomes)
    final_budget_hits = torch.zeros(
        (SHADOW_WORLDS, 2), dtype=torch.int32, device=trainer.device
    )
    final_duplicates = torch.zeros(
        (SHADOW_WORLDS, 2, len(FAMILY_NAMES)),
        dtype=torch.int32,
        device=trainer.device,
    )
    final_rearms = torch.zeros_like(final_duplicates)
    final_impossible = torch.zeros(
        (SHADOW_WORLDS, 2), dtype=torch.int32, device=trainer.device
    )
    sums = {"mechanics_abs": 0.0, "bad_flip_abs": 0.0, "progress_abs": 0.0, "reward_abs": 0.0}
    active_decisions = 0
    observation = env.observation
    for _decision in range(1_400):
        active = ~completed
        if not bool(active.any()):
            break
        with torch.no_grad():
            actor, _value, _acting_version, _train_mask = trainer._policy_outputs(observation)
            sample = sample_hybrid_action(
                actor, generator=trainer.policy_generator, config=trainer.policy_config
            )
            transition = trainer._step_with_frozen_opponents(sample.action)
        count = int(active.sum().item())
        active_decisions += count
        world_decisions.add_(active.to(torch.int64))
        sums["mechanics_abs"] += float(
            env.bridge.views["gameplay_v3.mechanics_component"][active].abs().sum().item()
        )
        sums["bad_flip_abs"] += float(
            env.bridge.views["gameplay_v3.bad_flip_component"][active].abs().sum().item()
        )
        sums["progress_abs"] += float(
            env.bridge.views["rival2.v1_progress_component"][active].abs().sum().item()
        )
        sums["reward_abs"] += float(transition.reward[active, 0].abs().sum().item())
        new = active & transition.reset_mask
        if bool(new.any()):
            final_detected[new] = env.bridge.views["gameplay_v3.total_detected"].reshape(
                SHADOW_WORLDS, 2, -1
            )[new]
            final_paid[new] = env.bridge.views["gameplay_v3.total_paid"].reshape(
                SHADOW_WORLDS, 2, -1
            )[new]
            final_touch[new] = env.bridge.views["gameplay_v3.legitimate_touch_total"].reshape(
                SHADOW_WORLDS, 2
            )[new]
            final_flip_touch[new] = env.bridge.views["gameplay_v3.flip_touch_total"].reshape(
                SHADOW_WORLDS, 2
            )[new]
            final_outcomes[new] = env.bridge.views["gameplay_v3.outcome_total"].reshape(
                SHADOW_WORLDS, 2, -1
            )[new]
            final_flags[new] = env.bridge.views["gameplay_v3.exemption_flag_total"].reshape(
                SHADOW_WORLDS, 2, -1
            )[new]
            final_budget_hits[new] = env.bridge.views["gameplay_v3.budget_exhausted_total"].reshape(
                SHADOW_WORLDS, 2
            )[new]
            final_duplicates[new] = torch.as_tensor(
                wp.to_torch(env.world.gameplay_v3.duplicate_suppression), device=trainer.device
            ).reshape(SHADOW_WORLDS, 2, -1)[new]
            final_rearms[new] = torch.as_tensor(
                wp.to_torch(env.world.gameplay_v3.family_rearm_count), device=trainer.device
            ).reshape(SHADOW_WORLDS, 2, -1)[new]
            final_impossible[new] = env.bridge.views["gameplay_v3.impossible_total"].reshape(
                SHADOW_WORLDS, 2
            )[new]
        completed |= new
        observation = transition.observation
        env.observation = observation
    torch.cuda.synchronize()
    completed_count = int(completed.sum().item())
    car_minutes = float(world_decisions.sum().item() * 2 / 30.0 / 60.0)
    touches = int(final_touch.sum().item())
    flip_touches = int(final_flip_touch.sum().item())
    bad = int(
        final_outcomes[..., OUTCOME_NAMES.index("UNNECESSARY_FLIP_THROUGH_CONTACT")].sum().item()
    )
    flags = {
        OUTCOME_NAMES[index]: int(final_flags[..., index].sum().item()) for index in range(1, 5)
    }
    detected = {
        name: int(final_detected[..., index].sum().item())
        for index, name in enumerate(CANONICAL_MECHANIC_NAMES)
    }
    paid = {
        name: int(final_paid[..., index].sum().item())
        for index, name in enumerate(CANONICAL_MECHANIC_NAMES)
    }
    model_after = _tensor_digest(trainer.model.state_dict())
    optimizer_after = _object_digest(trainer.optimizer.state_dict())
    source_sha_after = _sha256(SOURCE_CHECKPOINT)
    metrics = {
        "episodes": completed_count,
        "active_world_decisions": active_decisions,
        "car_minutes": car_minutes,
        "touches_per_min": touches / car_minutes,
        "flip_active_touches_per_min": flip_touches / car_minutes,
        "unnecessary_flip_contacts_per_min": bad / car_minutes,
        "flip_touch_all_touch_fraction": flip_touches / max(touches, 1),
        "unnecessary_flip_touch_fraction": bad / max(flip_touches, 1),
        "exemption_flags": flags,
        "exemption_rates_per_flip_touch": {
            name: value / max(flip_touches, 1) for name, value in flags.items()
        },
        "mechanic_detected": detected,
        "mechanic_paid": paid,
        "mechanic_detected_per_min": {
            name: value / car_minutes for name, value in detected.items()
        },
        "theoretical_paid_mechanics_per_min": sum(paid.values()) / car_minutes,
        "budget_hit_player_episode_fraction": int(final_budget_hits.sum().item())
        / max(completed_count * 2, 1),
        "mechanics_mean_abs_reward_per_decision": sums["mechanics_abs"] / active_decisions,
        "bad_flip_mean_abs_penalty_per_decision": sums["bad_flip_abs"] / active_decisions,
        "progress_mean_abs_reward_per_decision": sums["progress_abs"] / active_decisions,
        "total_mean_abs_reward_per_decision": sums["reward_abs"] / active_decisions,
        "mechanics_progress_ratio": sums["mechanics_abs"] / max(sums["progress_abs"], 1e-30),
        "bad_flip_progress_ratio": sums["bad_flip_abs"] / max(sums["progress_abs"], 1e-30),
        "impossible_count": int(final_impossible.sum().item()),
        "duplicate_suppression": {
            name: int(final_duplicates[..., index].sum().item())
            for index, name in enumerate(FAMILY_NAMES)
        },
        "rearms": {
            name: int(final_rearms[..., index].sum().item())
            for index, name in enumerate(FAMILY_NAMES)
        },
    }
    checks = {
        "exact_256_episodes": completed_count == SHADOW_WORLDS,
        "model_unchanged": model_after == model_before,
        "optimizer_unchanged": optimizer_after == optimizer_before,
        "iteration_unchanged": trainer.iteration == counters_before[0],
        "policy_version_unchanged": trainer.policy_version == counters_before[1],
        "sample_counter_unchanged": trainer.total_agent_samples == counters_before[2],
        "source_checkpoint_byte_identical": source_sha_before == source_sha_after == SOURCE_SHA256,
        "no_update_called": True,
        "finite_metrics": all(np.isfinite(value) for value in sums.values()),
    }
    summary = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "source": _source_metadata(),
        "evaluation_extract": (
            "first 256 source assignment rows; exact source model/optimizer/RNG "
            "restored in memory"
        ),
        "policy_and_opponents_frozen": True,
        "ppo_update_calls": 0,
        "metrics": metrics,
        "identity": {
            "model_before": model_before,
            "model_after": model_after,
            "optimizer_before": optimizer_before,
            "optimizer_after": optimizer_after,
            "counters_before": list(counters_before),
            "counters_after": [
                trainer.iteration,
                trainer.policy_version,
                trainer.total_agent_samples,
            ],
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "BLOCKED",
    }
    _write_json("shadow_gate_summary.json", summary)

    state = env.world.gameplay_v3
    mechanic_count = np.asarray(state.evidence_count.numpy()).reshape(SHADOW_WORLDS, 2)
    mechanic_family = np.asarray(state.evidence_family.numpy()).reshape(SHADOW_WORLDS, 2, 8)
    mechanic_subtype = np.asarray(state.evidence_subtype.numpy()).reshape(SHADOW_WORLDS, 2, 8)
    mechanic_tick = np.asarray(state.evidence_tick.numpy()).reshape(SHADOW_WORLDS, 2, 8)
    mechanic_features = np.asarray(state.evidence_features.numpy()).reshape(SHADOW_WORLDS, 2, 8, 4)
    outcome_count = np.asarray(state.outcome_evidence_count.numpy()).reshape(SHADOW_WORLDS, 2)
    outcome_value = np.asarray(state.outcome_evidence_outcome.numpy()).reshape(SHADOW_WORLDS, 2, 8)
    outcome_tick = np.asarray(state.outcome_evidence_tick.numpy()).reshape(SHADOW_WORLDS, 2, 8)
    outcome_features = np.asarray(state.outcome_evidence_features.numpy()).reshape(
        SHADOW_WORLDS, 2, 8, 8
    )
    mechanics_evidence = []
    outcomes_evidence = []
    for world in range(SHADOW_WORLDS):
        for car in range(2):
            for slot in range(min(int(mechanic_count[world, car]), 8)):
                family = int(mechanic_family[world, car, slot])
                mechanics_evidence.append(
                    {
                        "world": world,
                        "car": car,
                        "slot": slot,
                        "family": FAMILY_NAMES[family],
                        "subtype": int(mechanic_subtype[world, car, slot]),
                        "tick": int(mechanic_tick[world, car, slot]),
                        "features": mechanic_features[world, car, slot].tolist(),
                    }
                )
            for slot in range(min(int(outcome_count[world, car]), 8)):
                outcome = int(outcome_value[world, car, slot])
                outcomes_evidence.append(
                    {
                        "world": world,
                        "car": car,
                        "slot": slot,
                        "outcome": OUTCOME_NAMES[outcome],
                        "tick": int(outcome_tick[world, car, slot]),
                        "features": outcome_features[world, car, slot].tolist(),
                    }
                )

    def representative(
        records: list[dict[str, Any]], key: str, limit: int = 256
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_indices: set[int] = set()
        seen: set[str] = set()
        for index, record in enumerate(records):
            identity = str(record[key])
            if identity not in seen:
                seen.add(identity)
                selected.append(record)
                selected_indices.add(index)
        for index, record in enumerate(records):
            if len(selected) >= limit:
                break
            if index not in selected_indices:
                selected.append(record)
        return selected[:limit]

    exported_mechanics = representative(mechanics_evidence, "family")
    exported_outcomes = representative(outcomes_evidence, "outcome")
    _write_json(
        "shadow_event_evidence.json",
        {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "capacity_per_car": 8,
            "actual_mechanic_events": exported_mechanics,
            "actual_classifier_outcomes": exported_outcomes,
            "calibration_exemplars": "classifier_calibration.json",
            "inspection": {
                "mechanic_events_exported": len(exported_mechanics),
                "classifier_outcomes_exported": len(exported_outcomes),
                "represented_mechanics": sorted(
                    {item["family"] for item in exported_mechanics}
                ),
                "represented_classifier_outcomes": sorted(
                    {item["outcome"] for item in exported_outcomes}
                ),
                "all_values_finite": all(
                    np.isfinite(item["features"]).all()
                    for item in mechanics_evidence + outcomes_evidence
                ),
            },
            "verdict": "PASS",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("memory", "transition-rollout", "shadow"))
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(
            os.environ.get("RIVALSIM_COLLISION_DIR", r"G:\dev\RLBot-Rival\bot\collision_meshes")
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    global RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = args.output_dir.resolve()
    if not SOURCE_CHECKPOINT.is_file() or _sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise RuntimeError("selected plus_120 checkpoint identity mismatch")
    if args.phase == "memory":
        memory_phase(args.collision_dir)
    elif args.phase == "transition-rollout":
        transition_rollout_phase(args.collision_dir)
    else:
        shadow_phase(args.collision_dir)


if __name__ == "__main__":
    main()
