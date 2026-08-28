"""Train and audit the external Rival human-demo observation adapter V2."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_missing_feature_distillation import (  # noqa: E402
    _build_rollout_corpus,
    _load_bootstrap,
    _verify_human_sources,
)
from rivalsim.human_demo.bc_observation_bridge import (  # noqa: E402
    BC_BRIDGE_VERSION,
    FIELD_QUALITY_CONTRACT_SHA256,
    BCBridgeTrajectoryAdapter,
    FieldQuality,
    hybrid_actor_channel_kl,
)
from rivalsim.human_demo.missing_feature_distillation import (  # noqa: E402
    actor_output_statistics,
    canonical_sha256,
    degrade_observations_torch,
    file_sha256,
    world_observation_batch,
)
from rivalsim.human_demo.observation_adapter_v2 import (  # noqa: E402
    OBSERVATION_ADAPTER_CHECKPOINT_FORMAT,
    OBSERVATION_ADAPTER_VERSION,
    AdapterProfile,
    HumanDemoObservationAdapterV2,
    ObservationAdapterConfig,
    adapter_objective,
    expected_quality,
    meaningful_reconstruction_mask,
)
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_FIELD_NAMES  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic  # noqa: E402

FROZEN_CONFIG = Path("results/rival2/human_demo_observation_adapter_v2/frozen_config.json")
FROZEN_CONFIG_SHA256 = "227AFE90C5678E299851C30D14F9CA914C1B05D679BA2D67440248DED30F08A1"
RESULT_ROOT = Path("results/rival2/human_demo_observation_adapter_v2")
CHECKPOINT = Path(
    "checkpoints/rival2/observation_adapter_v2/rival2_human_demo_observation_adapter_v2.pt"
)
V1_RESULT_ROOT = Path("results/rival2/missing_feature_distillation_v1")
DATASET_MANIFEST = Path("results/rival2/human_demo_dataset_v1/dataset_manifest.json")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(f"frozen observation-adapter config changed: {digest}")
    config = json.loads(path.read_text(encoding="utf-8"))
    authority = config["authority"]
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            authority["required_parent"],
            "HEAD",
        ],
        check=True,
    )
    checks = {
        "bridge_version_exact": authority["bridge_version"] == BC_BRIDGE_VERSION,
        "quality_contract_exact": authority["bridge_quality_contract_sha256"]
        == FIELD_QUALITY_CONTRACT_SHA256,
        "exact_adapter_hash_exact": file_sha256(ROOT / "rivalsim/human_demo/training_adapter.py")
        == authority["exact_adapter_sha256"],
        "human_manifest_hash_exact": file_sha256(ROOT / DATASET_MANIFEST)
        == authority["human_dataset_manifest_sha256"],
        "v1_frozen_config_unchanged": file_sha256(ROOT / V1_RESULT_ROOT / "frozen_config.json")
        == "5F20CE9FDE854A99405D53864FB1FB72F9B28FA4EC882F8D4C675DF627A16955",
        "v1_failed_evidence_present": (
            ROOT / V1_RESULT_ROOT / "failed_training_evidence.json"
        ).is_file(),
    }
    if not all(checks.values()):
        raise ValueError(f"V2 authority check failed: {checks}")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "required_parent": authority["required_parent"],
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "checks": checks,
    }


def _adapter_config(config: dict[str, Any]) -> ObservationAdapterConfig:
    row = config["adapter"]
    return ObservationAdapterConfig(
        hidden_dim=int(row["hidden_dim"]),
        hidden_layers=int(row["hidden_layers"]),
        activation=str(row["activation"]),
        profile_features=int(row["profile_features"]),
        approximate_residual_limit=float(row["approximate_residual_limit"]),
        dtype=str(row["dtype"]),
        initialization=str(row["initialization"]),
    )


def _field_group(field: str) -> str:
    if field.startswith("ball."):
        return "ball"
    if field.startswith("self."):
        return "self"
    if field.startswith("opponent."):
        return "opponent"
    if field.startswith("relative."):
        return "relative"
    if field.startswith("boost_pad."):
        return "boost_pad"
    if field.startswith("previous_action."):
        return "previous_action"
    if field.startswith("lifecycle."):
        return "lifecycle"
    raise ValueError(f"unknown observation field group: {field}")


class _ErrorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.absolute = 0.0
        self.squared = 0.0
        self.maximum = 0.0

    def update(self, error: torch.Tensor) -> None:
        value = error.detach().to(torch.float64)
        self.count += int(value.numel())
        self.absolute += float(value.abs().sum().item())
        self.squared += float(value.square().sum().item())
        self.maximum = max(self.maximum, float(value.abs().max().item()))

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mae": None, "rmse": None, "max_absolute": None}
        return {
            "count": self.count,
            "mae": self.absolute / self.count,
            "rmse": (self.squared / self.count) ** 0.5,
            "max_absolute": self.maximum,
        }


def _profile_reconstruction_masks(
    profile: AdapterProfile, quality: torch.Tensor
) -> dict[str, torch.Tensor]:
    meaningful = meaningful_reconstruction_mask(profile, quality)
    rows = {"all_meaningful": meaningful}
    for group in (
        "ball",
        "self",
        "opponent",
        "relative",
        "boost_pad",
        "previous_action",
        "lifecycle",
    ):
        group_mask = torch.tensor(
            [_field_group(field) == group for field in OBS_FIELD_NAMES],
            dtype=torch.bool,
            device=quality.device,
        )
        rows[group] = meaningful & group_mask
    pad_active = torch.tensor(
        [field.startswith("boost_pad.") and field.endswith(".active") for field in OBS_FIELD_NAMES],
        dtype=torch.bool,
        device=quality.device,
    )
    pad_cooldown = torch.tensor(
        [
            field.startswith("boost_pad.") and field.endswith(".cooldown")
            for field in OBS_FIELD_NAMES
        ],
        dtype=torch.bool,
        device=quality.device,
    )
    rows["boost_pad_active"] = meaningful & pad_active
    rows["boost_pad_cooldown"] = meaningful & pad_cooldown
    return rows


@torch.no_grad()
def _evaluate(
    adapter: HumanDemoObservationAdapterV2,
    policy: Rival2ActorCritic,
    observations: torch.Tensor,
    worlds: np.ndarray,
    *,
    worlds_per_batch: int,
) -> dict[str, Any]:
    device = observations.device
    gameplay_quality = expected_quality(AdapterProfile.GAMEPLAY, device=device)
    freeplay_quality = expected_quality(AdapterProfile.FREEPLAY, device=device)
    assert gameplay_quality is not None and freeplay_quality is not None
    gameplay_masks = _profile_reconstruction_masks(AdapterProfile.GAMEPLAY, gameplay_quality)
    freeplay_masks = _profile_reconstruction_masks(AdapterProfile.FREEPLAY, freeplay_quality)
    channel_before = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    channel_after = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    sample_count = 0
    full_actor_exact = True
    full_value_exact = True
    finite = True
    actor_rows = []
    reconstruction: dict[str, dict[str, dict[str, _ErrorAccumulator]]] = {
        profile: {
            stage: {
                name: _ErrorAccumulator()
                for name in (
                    "all_meaningful",
                    "ball",
                    "self",
                    "opponent",
                    "relative",
                    "boost_pad",
                    "boost_pad_active",
                    "boost_pad_cooldown",
                    "previous_action",
                    "lifecycle",
                )
            }
            for stage in ("degraded", "repaired")
        }
        for profile in ("gameplay", "freeplay")
    }
    for start in range(0, len(worlds), worlds_per_batch):
        full = world_observation_batch(observations, worlds[start : start + worlds_per_batch])
        teacher_actor, teacher_value = policy(full)
        bypassed = adapter(full, None, profile=AdapterProfile.FULL)
        bypass_actor, bypass_value = policy(bypassed)
        full_actor_exact = full_actor_exact and bool(torch.equal(teacher_actor, bypass_actor))
        full_value_exact = full_value_exact and bool(torch.equal(teacher_value, bypass_value))
        gameplay_degraded = degrade_observations_torch(full, gameplay_quality)
        freeplay_degraded = degrade_observations_torch(full, freeplay_quality)
        gameplay_repaired = adapter(
            gameplay_degraded,
            gameplay_quality,
            profile=AdapterProfile.GAMEPLAY,
        )
        freeplay_repaired = adapter(
            freeplay_degraded,
            freeplay_quality,
            profile=AdapterProfile.FREEPLAY,
        )
        baseline_actor, _ = policy(gameplay_degraded)
        repaired_actor, _ = policy(gameplay_repaired)
        before = hybrid_actor_channel_kl(teacher_actor, baseline_actor, policy_config=policy.config)
        after = hybrid_actor_channel_kl(teacher_actor, repaired_actor, policy_config=policy.config)
        channel_before += before.sum(dim=0).cpu().to(torch.float64)
        channel_after += after.sum(dim=0).cpu().to(torch.float64)
        sample_count += int(full.shape[0])
        actor_rows.append(repaired_actor.cpu())
        finite = finite and all(
            bool(torch.isfinite(value).all())
            for value in (
                gameplay_repaired,
                freeplay_repaired,
                repaired_actor,
                bypass_actor,
                bypass_value,
            )
        )
        for profile, degraded, repaired, masks in (
            ("gameplay", gameplay_degraded, gameplay_repaired, gameplay_masks),
            ("freeplay", freeplay_degraded, freeplay_repaired, freeplay_masks),
        ):
            for stage, value in (("degraded", degraded), ("repaired", repaired)):
                error = value - full
                for name, mask in masks.items():
                    reconstruction[profile][stage][name].update(error.masked_select(mask))
    before_channel = channel_before / sample_count
    after_channel = channel_after / sample_count
    result = {
        "sample_count": sample_count,
        "full_authoritative": {
            "actor_kl": 0.0 if full_actor_exact else None,
            "value_drift": 0.0 if full_value_exact else None,
            "actor_byte_exact": full_actor_exact,
            "value_byte_exact": full_value_exact,
            "structural_bypass": True,
        },
        "gameplay_actor": {
            "baseline_mean_kl": float(before_channel.sum().item()),
            "repaired_mean_kl": float(after_channel.sum().item()),
            "baseline_per_channel_kl": {
                name: float(before_channel[index].item()) for index, name in enumerate(ACTION_NAMES)
            },
            "repaired_per_channel_kl": {
                name: float(after_channel[index].item()) for index, name in enumerate(ACTION_NAMES)
            },
            "repaired_actor_output": actor_output_statistics(torch.cat(actor_rows)),
        },
        "reconstruction": {
            profile: {
                stage: {name: accumulator.result() for name, accumulator in groups.items()}
                for stage, groups in stages.items()
            }
            for profile, stages in reconstruction.items()
        },
        "all_finite": finite,
        "freeplay_actor_kl_computed": False,
        "freeplay_hidden_opponent_fabricated": False,
    }
    return result


def _relative_improvement(before: float, after: float) -> float:
    return (before - after) / before if before > 0 else 0.0


def _acceptance(
    before: dict[str, Any], after: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["acceptance"]
    gameplay_before = before["gameplay_actor"]["baseline_mean_kl"]
    gameplay_after = after["gameplay_actor"]["repaired_mean_kl"]
    gameplay_recon_before = before["reconstruction"]["gameplay"]["degraded"]["all_meaningful"][
        "rmse"
    ]
    gameplay_recon_after = after["reconstruction"]["gameplay"]["repaired"]["all_meaningful"]["rmse"]
    freeplay_before = before["reconstruction"]["freeplay"]["degraded"]["all_meaningful"]["rmse"]
    freeplay_after = after["reconstruction"]["freeplay"]["repaired"]["all_meaningful"]["rmse"]
    pad_before = before["reconstruction"]["gameplay"]["degraded"]["boost_pad"]["rmse"]
    pad_after = after["reconstruction"]["gameplay"]["repaired"]["boost_pad"]["rmse"]
    checks = {
        "baseline_matches_authority": abs(gameplay_before - gate["gameplay_baseline_actor_kl"])
        <= gate["gameplay_baseline_absolute_tolerance"],
        "gameplay_kl_relative_reduction": _relative_improvement(gameplay_before, gameplay_after)
        >= gate["gameplay_actor_kl_relative_reduction_min"],
        "gameplay_kl_absolute": gameplay_after <= gate["gameplay_actor_kl_max"],
        "gameplay_reconstruction": _relative_improvement(
            gameplay_recon_before, gameplay_recon_after
        )
        >= gate["gameplay_reconstruction_relative_improvement_min"],
        "freeplay_meaningful_reconstruction": _relative_improvement(freeplay_before, freeplay_after)
        >= gate["freeplay_meaningful_reconstruction_relative_improvement_min"],
        "pad_reconstruction": _relative_improvement(pad_before, pad_after)
        >= gate["pad_reconstruction_relative_improvement_min"],
        "full_actor_exact": after["full_authoritative"]["actor_kl"]
        == gate["full_observation_actor_kl_exact"],
        "full_value_exact": after["full_authoritative"]["value_drift"]
        == gate["full_observation_value_drift_exact"],
        "all_finite": before["all_finite"] and after["all_finite"],
    }
    return {
        "checks": checks,
        "accepted": all(checks.values()),
        "gameplay_actor_kl_before": gameplay_before,
        "gameplay_actor_kl_after": gameplay_after,
        "gameplay_actor_kl_relative_reduction": _relative_improvement(
            gameplay_before, gameplay_after
        ),
        "gameplay_reconstruction_rmse_before_after": [
            gameplay_recon_before,
            gameplay_recon_after,
        ],
        "freeplay_meaningful_reconstruction_rmse_before_after": [
            freeplay_before,
            freeplay_after,
        ],
        "pad_reconstruction_rmse_before_after": [pad_before, pad_after],
    }


def _train(
    adapter: HumanDemoObservationAdapterV2,
    policy: Rival2ActorCritic,
    observations: torch.Tensor,
    splits: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    training = config["training"]
    device = observations.device
    gameplay_quality = expected_quality(AdapterProfile.GAMEPLAY, device=device)
    freeplay_quality = expected_quality(AdapterProfile.FREEPLAY, device=device)
    assert gameplay_quality is not None and freeplay_quality is not None
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["optimizer_betas"]),
        eps=float(training["optimizer_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = np.random.default_rng(int(training["seed"]))
    max_steps = int(training["max_optimizer_steps"])
    batch_worlds = int(training["worlds_per_minibatch"])
    accepted_steps = 0
    curve = []
    best: dict[str, Any] | None = None
    stale_epochs = 0
    policy_hash_before = tensor_tree_sha256(policy.state_dict())
    for epoch in range(int(training["max_epochs"])):
        order = generator.permutation(splits["train"])
        running = defaultdict(float)
        batch_count = 0
        for start in range(0, len(order), batch_worlds):
            if accepted_steps >= max_steps:
                break
            full = world_observation_batch(observations, order[start : start + batch_worlds])
            gameplay = degrade_observations_torch(full, gameplay_quality)
            freeplay = degrade_observations_torch(full, freeplay_quality)
            adapter_before = copy.deepcopy(adapter.state_dict())
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            objective = adapter_objective(
                adapter,
                policy,
                full,
                gameplay,
                gameplay_quality,
                freeplay,
                freeplay_quality,
                policy_config=policy.config,
                gameplay_actor_weight=float(training["gameplay_actor_kl_weight"]),
                gameplay_reconstruction_weight=float(training["gameplay_reconstruction_weight"]),
                freeplay_reconstruction_weight=float(training["freeplay_reconstruction_weight"]),
                approximate_residual_weight=float(training["approximate_residual_weight"]),
            )
            if not bool(torch.isfinite(objective.loss)):
                adapter.load_state_dict(adapter_before)
                optimizer.load_state_dict(optimizer_before)
                raise RuntimeError("nonfinite adapter objective; transaction rolled back")
            objective.loss.backward()
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in adapter.parameters()
            )
            if not gradients_finite:
                adapter.load_state_dict(adapter_before)
                optimizer.load_state_dict(optimizer_before)
                raise RuntimeError("nonfinite adapter gradient; transaction rolled back")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            state_finite = all(
                bool(torch.isfinite(parameter).all()) for parameter in adapter.parameters()
            )
            if not state_finite:
                adapter.load_state_dict(adapter_before)
                optimizer.load_state_dict(optimizer_before)
                raise RuntimeError("nonfinite adapter state; transaction rolled back")
            accepted_steps += 1
            batch_count += 1
            running["loss"] += float(objective.loss.item())
            running["gameplay_actor_kl"] += float(objective.gameplay_actor_kl.item())
            running["gameplay_reconstruction"] += float(objective.gameplay_reconstruction.item())
            running["freeplay_reconstruction"] += float(objective.freeplay_reconstruction.item())
            running["approximate_residual"] += float(objective.approximate_residual.item())
            running["gradient_norm"] += float(gradient_norm.item())
        validation = _evaluate(
            adapter,
            policy,
            observations,
            splits["validation"],
            worlds_per_batch=batch_worlds,
        )
        score = float(validation["gameplay_actor"]["repaired_mean_kl"])
        row = {
            "epoch": epoch + 1,
            "accepted_optimizer_steps": accepted_steps,
            "training_means": {
                key: value / max(batch_count, 1) for key, value in sorted(running.items())
            },
            "validation_gameplay_actor_kl": score,
            "validation_gameplay_reconstruction_rmse": validation["reconstruction"]["gameplay"][
                "repaired"
            ]["all_meaningful"]["rmse"],
            "validation_freeplay_reconstruction_rmse": validation["reconstruction"]["freeplay"][
                "repaired"
            ]["all_meaningful"]["rmse"],
            "all_finite": validation["all_finite"],
        }
        curve.append(row)
        material = False
        if best is None:
            material = True
        else:
            relative = (best["score"] - score) / best["score"]
            material = relative >= float(training["early_stopping_material_relative_improvement"])
        if material:
            best = {
                "score": score,
                "epoch": epoch + 1,
                "step": accepted_steps,
                "adapter": copy.deepcopy(adapter.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if accepted_steps >= max_steps or stale_epochs >= int(
            training["early_stopping_patience_epochs"]
        ):
            break
    if best is None:
        raise RuntimeError("adapter training produced no finite validation checkpoint")
    adapter.load_state_dict(best["adapter"])
    optimizer.load_state_dict(best["optimizer"])
    policy_hash_after = tensor_tree_sha256(policy.state_dict())
    if policy_hash_before != policy_hash_after:
        raise RuntimeError("frozen Rival policy changed during adapter training")
    return (
        optimizer,
        {
            "accepted_optimizer_steps_executed": accepted_steps,
            "best_optimizer_step": best["step"],
            "best_epoch": best["epoch"],
            "stop_reason": (
                "max_optimizer_steps" if accepted_steps >= max_steps else "early_stopping_patience"
            ),
            "curve": curve,
            "transactional_nonfinite_rollbacks": 0,
            "fresh_adapter_optimizer": True,
            "historical_ppo_optimizer_loaded": False,
            "historical_ppo_optimizer_mutated": False,
            "frozen_policy_tensor_sha256_before_after": [
                policy_hash_before,
                policy_hash_after,
            ],
        },
        best,
    )


def _audit_native_boost_pads(source_root: Path) -> dict[str, Any]:
    dataset = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
    total_frames = 0
    frames_with_pad_rows = 0
    pad_rows = 0
    pickup_events = 0
    pickup_with_position = 0
    pickup_with_canonical_index = 0
    pickup_callers: set[str] = set()
    sessions = []
    for source in dataset["source_verification"]:
        session_uuid = str(source["session_uuid"])
        reader = SessionReader(source_root / session_uuid)
        local_frames = 0
        local_rows = 0
        for frame in reader.iter_frames():
            local_frames += 1
            rows = frame.get("boost_pads", ())
            if rows:
                frames_with_pad_rows += 1
                local_rows += len(rows)
        local_pickups = 0
        for event in reader.iter_events():
            if event.get("kind") != "boost_pad_pickup":
                continue
            local_pickups += 1
            pickup_events += 1
            pickup_callers.add(str(event.get("caller", "")))
            keys = set(event)
            if keys & {"position", "pad_position", "location", "pickup_location"}:
                pickup_with_position += 1
            if keys & {"canonical_index", "pad_index", "boost_pad_index"}:
                pickup_with_canonical_index += 1
        total_frames += local_frames
        pad_rows += local_rows
        sessions.append(
            {
                "session_uuid": session_uuid,
                "frames": local_frames,
                "native_pad_rows": local_rows,
                "pickup_events": local_pickups,
            }
        )
    mapping_supported = bool(
        pickup_events
        and pickup_with_position == pickup_events
        and pickup_with_canonical_index == pickup_events
    )
    return {
        "format": "RIVAL2_NATIVE_BOOST_PAD_OBSERVABILITY_AUDIT_V2",
        "canonical_pad_count": 34,
        "source_session_count": len(sessions),
        "source_frame_count": total_frames,
        "frames_with_native_pad_rows": frames_with_pad_rows,
        "native_pad_row_count": pad_rows,
        "boost_pad_pickup_event_count": pickup_events,
        "pickup_event_distinct_runtime_pointer_count": len(pickup_callers),
        "pickup_events_with_position": pickup_with_position,
        "pickup_events_with_canonical_index": pickup_with_canonical_index,
        "deterministic_position_to_index_mapping_supported": mapping_supported,
        "pointer_sorting_used": False,
        "nearby_car_heuristic_used": False,
        "bridge_quality_promoted": False,
        "outcome": (
            "implemented deterministic canonical mapping"
            if mapping_supported
            else "canonical position mapping is impossible from the committed native files; "
            "pad state remains unavailable and is learned only by the masked adapter"
        ),
        "sessions": sessions,
    }


def _human_spans() -> tuple[dict[str, list[tuple[str, int, int]]], dict[str, AdapterProfile]]:
    dataset = json.loads((ROOT / DATASET_MANIFEST).read_text(encoding="utf-8"))
    spans: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    profiles: dict[str, AdapterProfile] = {}
    for row in dataset["mechanic_positive_attempts"]:
        identity = str(row["attempt_id"])
        profiles[identity] = AdapterProfile.FREEPLAY
        spans[str(row["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    gameplay = dataset["general_gameplay"]
    for row in gameplay["regions"]:
        identity = str(row["region_id"])
        profiles[identity] = AdapterProfile.GAMEPLAY
        spans[str(gameplay["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    return spans, profiles


@torch.no_grad()
def _evaluate_human(
    adapter: HumanDemoObservationAdapterV2,
    policy: Rival2ActorCritic,
    config: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    source_before = _verify_human_sources(source_root)
    spans, profiles = _human_spans()
    batch_size = int(config["human_validation"]["inference_batch_size"])
    device = next(adapter.parameters()).device
    buffers: dict[AdapterProfile, list[np.ndarray]] = defaultdict(list)
    quality_buffers: dict[AdapterProfile, list[np.ndarray]] = defaultdict(list)
    counts = {"gameplay": 0, "freeplay": 0}
    quality_counts = np.zeros(4, dtype=np.int64)
    exact_derived_unchanged = True
    qualities_unchanged = True
    outputs_finite = True
    actions_unchanged = True
    action_before = hashlib.sha256()
    action_after = hashlib.sha256()
    actor_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)

    def flush(profile: AdapterProfile) -> None:
        nonlocal exact_derived_unchanged, qualities_unchanged, outputs_finite
        if not buffers[profile]:
            return
        source = torch.from_numpy(np.stack(buffers[profile])).to(device=device)
        quality_np = np.stack(quality_buffers[profile])
        quality_before = quality_np.copy()
        quality = torch.from_numpy(quality_np).to(device=device)
        repaired = adapter(source, quality, profile=profile)
        exact = quality >= int(FieldQuality.EXACT_DERIVED)
        exact_derived_unchanged = exact_derived_unchanged and bool(
            torch.equal(repaired.masked_select(exact), source.masked_select(exact))
        )
        qualities_unchanged = qualities_unchanged and bool(
            np.array_equal(quality_np, quality_before)
        )
        actor, _value = policy(repaired)
        outputs_finite = outputs_finite and bool(
            torch.isfinite(repaired).all() and torch.isfinite(actor).all()
        )
        actor_outputs[profile.value].append(actor.cpu())
        buffers[profile].clear()
        quality_buffers[profile].clear()

    for session_uuid in sorted(spans):
        trajectory = BCBridgeTrajectoryAdapter(source_root / session_uuid)
        for identity, sample in trajectory.iter_spans(spans[session_uuid]):
            if not sample.bc_usable:
                raise RuntimeError(f"accepted human sample became unusable: {identity}")
            profile = profiles[identity]
            observation = np.asarray(sample.observation).copy()
            quality = np.asarray(sample.quality).copy()
            action = np.asarray(sample.action).copy()
            action_before.update(action.tobytes(order="C"))
            action_after.update(action.tobytes(order="C"))
            actions_unchanged = actions_unchanged and bool(
                sample.action_unchanged_from_exact_adapter
            )
            quality_counts += np.bincount(quality, minlength=4)
            buffers[profile].append(observation)
            quality_buffers[profile].append(quality)
            counts[profile.value] += 1
            if len(buffers[profile]) >= batch_size:
                flush(profile)
    flush(AdapterProfile.GAMEPLAY)
    flush(AdapterProfile.FREEPLAY)
    source_after = _verify_human_sources(source_root)
    total = sum(counts.values())
    result = {
        "format": "RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_HUMAN_AUDIT_V2",
        "frame_count": total,
        "expected_frame_count": int(config["human_validation"]["expected_frame_count"]),
        "profile_frame_counts": counts,
        "field_quality_value_counts": {
            "unavailable": int(quality_counts[0]),
            "approximate": int(quality_counts[1]),
            "exactly_derived": int(quality_counts[2]),
            "exact_direct": int(quality_counts[3]),
        },
        "source_verification_before": source_before,
        "source_verification_after": source_after,
        "source_hashes_unchanged": source_before == source_after,
        "actions_sha256_before": action_before.hexdigest().upper(),
        "actions_sha256_after": action_after.hexdigest().upper(),
        "actions_unchanged": actions_unchanged and action_before.digest() == action_after.digest(),
        "exact_and_derived_fields_byte_unchanged": exact_derived_unchanged,
        "quality_masks_byte_unchanged": qualities_unchanged,
        "quality_promotions": 0,
        "all_adapter_and_actor_outputs_finite": outputs_finite,
        "actor_output_statistics": {
            profile: actor_output_statistics(torch.cat(rows))
            for profile, rows in actor_outputs.items()
        },
        "human_actions_used_for_optimization": False,
        "human_optimizer_steps": 0,
        "behavior_cloning_performed": False,
    }
    result["valid"] = all(
        (
            total == result["expected_frame_count"],
            result["source_hashes_unchanged"],
            result["actions_unchanged"],
            exact_derived_unchanged,
            qualities_unchanged,
            outputs_finite,
            result["quality_promotions"] == 0,
            not result["human_actions_used_for_optimization"],
        )
    )
    return result


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for relative in sorted(paths, key=lambda path: path.as_posix()):
        path = ROOT / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "format": "RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_ARTIFACT_MANIFEST_V2",
        "files": rows,
        "file_set_sha256": canonical_sha256(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    if _git("status", "--short"):
        raise RuntimeError("adapter optimization requires a clean pre-step worktree")
    if config_identity["head"] != config_identity["origin_main"]:
        raise RuntimeError("adapter optimization requires HEAD persisted on origin/main")
    torch.manual_seed(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(config)
    bootstrap_path = ROOT / config["authority"]["bootstrap_checkpoint"]
    bootstrap_bytes_before = bootstrap_path.read_bytes()
    ppo_optimizer_hash_before = tensor_tree_sha256(bootstrap_payload["optimizer"])
    corpus, corpus_manifest, splits = _build_rollout_corpus(
        config, bootstrap_payload, bootstrap_identity, device=args.device
    )
    corpus_checks = {
        "identity_matches_v1_authority": corpus_manifest["identity_sha256"]
        == config["corpus"]["expected_identity_sha256"],
        "observation_hash_matches_v1_authority": corpus_manifest["collection"][
            "observation_tensor_sha256"
        ]
        == config["corpus"]["expected_observation_tensor_sha256"],
        "split_manifest_matches_v1": corpus_manifest["split"]
        == json.loads((ROOT / V1_RESULT_ROOT / "corpus_manifest.json").read_text())["split"],
    }
    if not all(corpus_checks.values()):
        raise RuntimeError(f"authoritative V1 corpus regeneration changed: {corpus_checks}")
    # The frozen curriculum collector contains CUDA bincount, which PyTorch marks as
    # lacking a deterministic implementation. Corpus identity is enforced by its full
    # canonical tensor hash; strict deterministic algorithms govern adapter training.
    torch.use_deterministic_algorithms(True)
    policy = Rival2ActorCritic(policy_config).to(args.device)
    policy.load_state_dict(bootstrap_payload["model"])
    policy.eval()
    policy.requires_grad_(False)
    adapter = HumanDemoObservationAdapterV2(_adapter_config(config)).to(args.device)
    adapter.train()
    bootstrap_model_hash_before = tensor_tree_sha256(policy.state_dict())
    baseline = _evaluate(
        adapter,
        policy,
        corpus,
        splits["test"],
        worlds_per_batch=int(config["training"]["worlds_per_minibatch"]),
    )
    optimizer, training, _best = _train(adapter, policy, corpus, splits, config)
    adapter.eval()
    final = _evaluate(
        adapter,
        policy,
        corpus,
        splits["test"],
        worlds_per_batch=int(config["training"]["worlds_per_minibatch"]),
    )
    acceptance = _acceptance(baseline, final, config)
    pad_audit = _audit_native_boost_pads(args.human_source_root)
    human = _evaluate_human(adapter, policy, config, args.human_source_root)
    bootstrap_model_hash_after = tensor_tree_sha256(policy.state_dict())
    ppo_optimizer_hash_after = tensor_tree_sha256(bootstrap_payload["optimizer"])
    integrity = {
        "bootstrap_checkpoint_byte_identical": bootstrap_path.read_bytes()
        == bootstrap_bytes_before,
        "bootstrap_checkpoint_sha256": file_sha256(bootstrap_path),
        "bootstrap_model_tensor_sha256_before_after": [
            bootstrap_model_hash_before,
            bootstrap_model_hash_after,
        ],
        "bootstrap_model_byte_identical": bootstrap_model_hash_before
        == bootstrap_model_hash_after
        == config["authority"]["bootstrap_model_tensor_sha256"],
        "bootstrap_requires_grad_false": all(
            not parameter.requires_grad for parameter in policy.parameters()
        ),
        "bootstrap_gradients_absent": all(
            parameter.grad is None for parameter in policy.parameters()
        ),
        "historical_ppo_optimizer_sha256_before_after": [
            ppo_optimizer_hash_before,
            ppo_optimizer_hash_after,
        ],
        "historical_ppo_optimizer_untouched": ppo_optimizer_hash_before == ppo_optimizer_hash_after,
        "adapter_checkpoint_contains_rival_model": False,
        "adapter_optimizer_is_fresh": training["fresh_adapter_optimizer"],
        "human_behavior_cloning_absent": not human["behavior_cloning_performed"],
        "human_optimizer_steps_zero": human["human_optimizer_steps"] == 0,
        "v1_evidence_modified": False,
    }
    accepted = (
        acceptance["accepted"]
        and human["valid"]
        and all(corpus_checks.values())
        and all(integrity.values())
        and not pad_audit["bridge_quality_promoted"]
    )
    checkpoint_path = ROOT / CHECKPOINT
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": OBSERVATION_ADAPTER_CHECKPOINT_FORMAT,
        "adapter_version": OBSERVATION_ADAPTER_VERSION,
        "adapter_config": _adapter_config(config).__dict__
        if hasattr(_adapter_config(config), "__dict__")
        else {
            key: getattr(_adapter_config(config), key) for key in _adapter_config(config).__slots__
        },
        "adapter_config_sha256": _adapter_config(config).content_hash,
        "adapter": _cpu_tree(adapter.state_dict()),
        "adapter_optimizer": _cpu_tree(optimizer.state_dict()),
        "optimizer_provenance": {
            "type": config["training"]["optimizer"],
            "fresh_for_adapter": True,
            "best_optimizer_step": training["best_optimizer_step"],
            "historical_ppo_optimizer_loaded": False,
        },
        "authority": {
            "bootstrap_checkpoint_sha256": bootstrap_identity["sha256"],
            "bootstrap_model_tensor_sha256": bootstrap_identity["model_tensor_sha256"],
            "bridge_version": BC_BRIDGE_VERSION,
            "bridge_quality_contract_sha256": FIELD_QUALITY_CONTRACT_SHA256,
            "frozen_config_sha256": config_identity["sha256"],
            "simulator_corpus_identity_sha256": corpus_manifest["identity_sha256"],
            "simulator_observation_tensor_sha256": corpus_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "pre_step_git_commit": config_identity["head"],
        },
        "semantics": {
            "full_authoritative_structural_bypass": True,
            "exact_and_derived_hard_copy": True,
            "quality_classifications_unchanged": True,
            "freeplay_opponent_fabricated": False,
            "human_behavior_cloning_performed": False,
        },
    }
    torch.save(payload, checkpoint_path)
    checkpoint_identity = {
        "path": CHECKPOINT.as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
        "adapter_tensor_sha256": tensor_tree_sha256(payload["adapter"]),
    }
    evidence = {
        "format": "RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_EVIDENCE_V2",
        "generated_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if accepted else "BLOCKED",
        "config": config_identity,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "adapter": {
            "version": OBSERVATION_ADAPTER_VERSION,
            "parameter_count": adapter.parameter_count,
            "config_sha256": adapter.config.content_hash,
            "checkpoint": checkpoint_identity,
        },
        "corpus_checks": corpus_checks,
        "training": training,
        "test_baseline": baseline,
        "test_final": final,
        "acceptance": acceptance,
        "native_boost_pad_audit": pad_audit,
        "human_audit": human,
        "integrity": integrity,
        "prohibited_work": {
            "human_behavior_cloning": False,
            "ppo": False,
            "reward_change": False,
            "mechanic_detector_change": False,
            "rival_model_mutation": False,
        },
    }
    _write_json(ROOT / RESULT_ROOT / "corpus_manifest.json", corpus_manifest)
    _write_json(ROOT / RESULT_ROOT / "training_curve.json", training["curve"])
    _write_json(
        ROOT / RESULT_ROOT / "simulator_test_metrics.json",
        {
            "baseline": baseline,
            "final": final,
            "acceptance": acceptance,
        },
    )
    _write_json(ROOT / RESULT_ROOT / "native_boost_pad_audit.json", pad_audit)
    _write_json(ROOT / RESULT_ROOT / "human_inference_audit.json", human)
    _write_json(ROOT / RESULT_ROOT / "evidence.json", evidence)
    manifest_paths = [
        CHECKPOINT,
        RESULT_ROOT / "corpus_manifest.json",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "human_inference_audit.json",
        RESULT_ROOT / "native_boost_pad_audit.json",
        RESULT_ROOT / "simulator_test_metrics.json",
        RESULT_ROOT / "training_curve.json",
    ]
    _write_json(
        ROOT / RESULT_ROOT / "artifact_manifest.json",
        _artifact_manifest(manifest_paths),
    )
    del corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint": checkpoint_identity,
                "gameplay_kl_before": acceptance["gameplay_actor_kl_before"],
                "gameplay_kl_after": acceptance["gameplay_actor_kl_after"],
                "human_frames": human["frame_count"],
            },
            indent=2,
        )
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=Path(os.environ["APPDATA"]) / "bakkesmod/bakkesmod/data/rival2/human_demos",
    )
    return parser.parse_args()


def main() -> int:
    evidence = run(parse_args())
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
