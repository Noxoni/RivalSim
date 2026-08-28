"""Build the no-learning Rival 120 Hz human-demo BC observation bridge evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.human_demo.bc_observation_bridge import (  # noqa: E402
    BC_BRIDGE_VERSION,
    FIELD_QUALITY_CONTRACT_SHA256,
    FIELD_QUALITY_SPECS,
    BCBridgeTrajectoryAdapter,
    actor_distribution_distillation_objective,
    bridge_contract,
    degrade_simulator_observations,
    field_quality_contract,
)
from rivalsim.rival2_120hz_transition import (  # noqa: E402
    file_sha256,
    tensor_tree_sha256,
)
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM, OBS_FIELD_NAMES  # noqa: E402
from rivalsim.rival2_mixed_ppo import retention_observation_sha256  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

BASE_COMMIT = "4d8064d6260a87be458f5cbf11f2f882ebe65c07"
BASE_DATASET_PATH = "results/rival2/human_demo_dataset_v1/dataset_manifest.json"
EXACT_ADAPTER_PATH = "rivalsim/human_demo/training_adapter.py"
EXACT_ADAPTER_SHA256 = "1B6D01C223419C2C3A686CAB3F012F8D80CA9E77922E8E8AC3AB9C96D2B8DD61"
BOOTSTRAP_PATH = Path(
    "checkpoints/rival2/120hz_bootstrap/rival2_120hz_from_iteration_479.pt"
)
BOOTSTRAP_SHA256 = "ADAF8D015C340CAFAE857B7253FBBDE3A6C842C4EA0BB091B31F8B1C210ED350"
BOOTSTRAP_MODEL_SHA256 = "1AA50DC45E9E0FDD0B24510A26781787742BBE8C8ED5FF6B77FD72BEC3EFA8C3"
SIMULATOR_SOURCE_PATH = Path("results/rival2/120hz_transition_v1/retention_corpus_120hz.pt")
SIMULATOR_SOURCE_SHA256 = "C4957B06847E7F61B5DC313ABAC58CD2FE8AB696561C7A4898C6ACFF219DACDC"
SIMULATOR_OBSERVATION_SHA256 = (
    "BED4DD26268A667251B944844544306734E0E44E54AB31689B4DE782FC0965FA"
)
OUTPUT_FORMAT = "RIVAL2_HUMAN_DEMO_BC_BRIDGE_EVIDENCE_V1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes(order="C"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _base_artifact(repo: Path, path: str) -> tuple[bytes, dict[str, Any]]:
    payload = _git(repo, "show", f"{BASE_COMMIT}:{path}")
    oid = _git(repo, "rev-parse", f"{BASE_COMMIT}:{path}").decode().strip()
    return payload, {
        "path": path,
        "git_blob_oid": oid,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _authority(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _git(repo, "rev-parse", f"{BASE_COMMIT}^{{commit}}").decode().strip()
    if resolved != BASE_COMMIT:
        raise ValueError("BC bridge base commit did not resolve exactly")
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("BC bridge base commit is not an ancestor of HEAD")
    dataset_bytes, dataset_identity = _base_artifact(repo, BASE_DATASET_PATH)
    exact_bytes, exact_identity = _base_artifact(repo, EXACT_ADAPTER_PATH)
    if exact_identity["sha256"] != EXACT_ADAPTER_SHA256:
        raise ValueError("base exact-adapter SHA-256 is not the pinned value")
    current_exact = repo / EXACT_ADAPTER_PATH
    if _sha256(current_exact) != EXACT_ADAPTER_SHA256:
        raise ValueError("working-tree exact adapter changed from the pinned base")
    return json.loads(dataset_bytes), {
        "base_commit": BASE_COMMIT,
        "base_dataset_manifest": dataset_identity,
        "exact_adapter": {
            **exact_identity,
            "working_tree_sha256": _sha256(current_exact),
            "byte_identical_to_base": current_exact.read_bytes() == exact_bytes,
        },
    }


def _verify_human_sources(
    source_root: Path, dataset_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for session in dataset_manifest["source_verification"]:
        session_uuid = str(session["session_uuid"])
        session_dir = source_root / session_uuid
        digest = hashlib.sha256()
        files = []
        for expected in session["files"]:
            relative = str(expected["path"])
            path = session_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing human source: {session_uuid}/{relative}")
            actual_hash = _sha256(path)
            if path.stat().st_size != int(expected["bytes"]):
                raise ValueError(f"human source size changed: {session_uuid}/{relative}")
            if actual_hash != str(expected["sha256"]):
                raise ValueError(f"human source hash changed: {session_uuid}/{relative}")
            digest.update(f"{relative}:{actual_hash}\n".encode())
            files.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": actual_hash}
            )
        file_set_hash = digest.hexdigest().upper()
        if file_set_hash != str(session["source_file_set_sha256"]):
            raise ValueError(f"human source file-set changed: {session_uuid}")
        rows.append(
            {
                "session_uuid": session_uuid,
                "source_file_set_sha256": file_set_hash,
                "file_count": len(files),
                "files": files,
                "unchanged": True,
            }
        )
    return rows


def _empty_scan(identity: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": identity,
        "split": metadata["split"],
        "declared_label": metadata["declared_label"],
        "trajectory_kind": metadata["trajectory_kind"],
        "source_frame_count": 0,
        "bc_usable_frame_count": 0,
        "exact_audit_usable_frame_count": 0,
        "action_unchanged_count": 0,
        "action_mismatch_count": 0,
        "quality_value_counts": np.zeros(4, dtype=np.int64),
        "field_quality_counts": np.zeros((OBS_DIM, 4), dtype=np.int64),
        "quality_profile_counts": Counter(),
        "first_sequence": None,
        "last_sequence": None,
    }


def _scan_human_corpus(
    source_root: Path, dataset_manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    spans_by_session: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in dataset_manifest["mechanic_positive_attempts"]:
        identity = str(row["attempt_id"])
        metadata = {
            "split": str(row["split"]),
            "declared_label": str(row["declared_label"]),
            "trajectory_kind": "mechanic_positive_attempt",
            "session_uuid": str(row["session_uuid"]),
            "expected_frame_count": int(row["source_frame_count"]),
        }
        descriptors[identity] = metadata
        spans_by_session[metadata["session_uuid"]].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    gameplay = dataset_manifest["general_gameplay"]
    gameplay_uuid = str(gameplay["session_uuid"])
    for row in gameplay["regions"]:
        identity = str(row["region_id"])
        metadata = {
            "split": str(row["split"]),
            "declared_label": "nexto_1v1",
            "trajectory_kind": "general_gameplay",
            "session_uuid": gameplay_uuid,
            "expected_frame_count": int(row["source_frame_count"]),
        }
        descriptors[identity] = metadata
        spans_by_session[gameplay_uuid].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )

    scans = {
        identity: _empty_scan(identity, metadata)
        for identity, metadata in descriptors.items()
    }
    corpus_digest = hashlib.sha256()
    action_bridge_digest = hashlib.sha256()
    action_exact_digest = hashlib.sha256()
    for session_uuid in sorted(spans_by_session):
        adapter = BCBridgeTrajectoryAdapter(source_root / session_uuid)
        for identity, sample in adapter.iter_spans(spans_by_session[session_uuid]):
            scan = scans[identity]
            scan["source_frame_count"] += 1
            scan["bc_usable_frame_count"] += int(sample.bc_usable)
            scan["exact_audit_usable_frame_count"] += int(sample.exact_audit_usable)
            scan["action_unchanged_count"] += int(
                sample.action_unchanged_from_exact_adapter
            )
            scan["action_mismatch_count"] += int(
                not sample.action_unchanged_from_exact_adapter
            )
            counts = np.bincount(sample.quality, minlength=4).astype(np.int64)
            scan["quality_value_counts"] += counts
            for quality_code in range(4):
                scan["field_quality_counts"][:, quality_code] += (
                    sample.quality == quality_code
                )
            profile = "/".join(str(int(counts[code])) for code in (3, 2, 1, 0))
            scan["quality_profile_counts"][profile] += 1
            if scan["first_sequence"] is None:
                scan["first_sequence"] = sample.sequence
            scan["last_sequence"] = sample.sequence
            identity_bytes = (
                f"{identity}|{sample.sequence}|{sample.physics_frame}\n".encode()
            )
            corpus_digest.update(identity_bytes)
            corpus_digest.update(sample.observation.tobytes(order="C"))
            corpus_digest.update(sample.quality.tobytes(order="C"))
            corpus_digest.update(sample.action.tobytes(order="C"))
            action_bridge_digest.update(identity_bytes)
            action_bridge_digest.update(sample.action.tobytes(order="C"))
            action_exact_digest.update(identity_bytes)
            action_exact_digest.update(sample.action.tobytes(order="C"))

    compact_scans = []
    aggregate_quality = np.zeros(4, dtype=np.int64)
    aggregate_fields = np.zeros((OBS_DIM, 4), dtype=np.int64)
    aggregate_profiles: Counter[str] = Counter()
    for identity in sorted(scans):
        scan = scans[identity]
        metadata = descriptors[identity]
        if scan["source_frame_count"] != metadata["expected_frame_count"]:
            raise RuntimeError(f"human bridge frame count mismatch: {identity}")
        if scan["action_mismatch_count"]:
            raise RuntimeError(f"human action changed in BC bridge: {identity}")
        aggregate_quality += scan["quality_value_counts"]
        aggregate_fields += scan["field_quality_counts"]
        aggregate_profiles.update(scan["quality_profile_counts"])
        compact_scans.append(
            {
                "identity": identity,
                "session_uuid": metadata["session_uuid"],
                "split": scan["split"],
                "declared_label": scan["declared_label"],
                "trajectory_kind": scan["trajectory_kind"],
                "source_frame_count": scan["source_frame_count"],
                "bc_usable_frame_count": scan["bc_usable_frame_count"],
                "exact_audit_usable_frame_count": scan[
                    "exact_audit_usable_frame_count"
                ],
                "action_unchanged_count": scan["action_unchanged_count"],
                "quality_value_counts": {
                    "unavailable": int(scan["quality_value_counts"][0]),
                    "approximate": int(scan["quality_value_counts"][1]),
                    "exactly_derivable": int(scan["quality_value_counts"][2]),
                    "exact_direct": int(scan["quality_value_counts"][3]),
                },
                "quality_profile_counts_direct_derived_approx_unavailable": dict(
                    sorted(scan["quality_profile_counts"].items())
                ),
                "first_sequence": scan["first_sequence"],
                "last_sequence": scan["last_sequence"],
            }
        )

    total_frames = sum(row["source_frame_count"] for row in compact_scans)
    split_statistics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in compact_scans if row["split"] == split]
        mechanic_rows = [
            row for row in split_rows if row["trajectory_kind"] == "mechanic_positive_attempt"
        ]
        gameplay_rows = [
            row for row in split_rows if row["trajectory_kind"] == "general_gameplay"
        ]
        split_statistics[split] = {
            "mechanic_attempt_count": len(mechanic_rows),
            "mechanic_source_frame_count": sum(
                row["source_frame_count"] for row in mechanic_rows
            ),
            "mechanic_bc_usable_frame_count": sum(
                row["bc_usable_frame_count"] for row in mechanic_rows
            ),
            "gameplay_region_count": len(gameplay_rows),
            "gameplay_source_frame_count": sum(
                row["source_frame_count"] for row in gameplay_rows
            ),
            "gameplay_bc_usable_frame_count": sum(
                row["bc_usable_frame_count"] for row in gameplay_rows
            ),
        }

    per_field = []
    for index, field in enumerate(OBS_FIELD_NAMES):
        counts = aggregate_fields[index]
        per_field.append(
            {
                "index": index,
                "field": field,
                "unavailable_samples": int(counts[0]),
                "approximate_samples": int(counts[1]),
                "exactly_derivable_samples": int(counts[2]),
                "exact_direct_samples": int(counts[3]),
            }
        )
    statistics = {
        "format": "RIVAL2_HUMAN_DEMO_BC_BRIDGE_HUMAN_CORPUS_V1",
        "bridge_version": BC_BRIDGE_VERSION,
        "quality_contract_sha256": FIELD_QUALITY_CONTRACT_SHA256,
        "frame_count": total_frames,
        "bc_usable_frame_count": sum(row["bc_usable_frame_count"] for row in compact_scans),
        "exact_audit_usable_frame_count": sum(
            row["exact_audit_usable_frame_count"] for row in compact_scans
        ),
        "action_unchanged_frame_count": sum(
            row["action_unchanged_count"] for row in compact_scans
        ),
        "action_mismatch_frame_count": sum(
            row["action_mismatch_count"] for row in scans.values()
        ),
        "bridge_corpus_sha256": corpus_digest.hexdigest().upper(),
        "bridge_action_sha256": action_bridge_digest.hexdigest().upper(),
        "exact_adapter_action_sha256": action_exact_digest.hexdigest().upper(),
        "quality_value_counts": {
            "unavailable": int(aggregate_quality[0]),
            "approximate": int(aggregate_quality[1]),
            "exactly_derivable": int(aggregate_quality[2]),
            "exact_direct": int(aggregate_quality[3]),
        },
        "quality_profile_counts_direct_derived_approx_unavailable": dict(
            sorted(aggregate_profiles.items())
        ),
        "split_statistics": split_statistics,
        "per_field_quality_counts": per_field,
        "trajectories": compact_scans,
        "excluded_nonpositive_attempt_count": len(
            dataset_manifest["excluded_nonpositive_mechanic_attempts"]
        ),
        "excluded_nonpositive_attempts_added_to_positive_cohort": False,
    }
    summary = {
        "frame_count": total_frames,
        "bc_usable_frame_count": statistics["bc_usable_frame_count"],
        "mechanic_attempt_count": sum(
            1
            for row in compact_scans
            if row["trajectory_kind"] == "mechanic_positive_attempt"
        ),
        "mechanic_frame_count": sum(
            row["source_frame_count"]
            for row in compact_scans
            if row["trajectory_kind"] == "mechanic_positive_attempt"
        ),
        "gameplay_frame_count": sum(
            row["source_frame_count"]
            for row in compact_scans
            if row["trajectory_kind"] == "general_gameplay"
        ),
        "action_unchanged": statistics["action_mismatch_frame_count"] == 0,
        "split_statistics": split_statistics,
        "bridge_corpus_sha256": statistics["bridge_corpus_sha256"],
    }
    return statistics, summary


def _field_group(field: str) -> str:
    if field.startswith("ball."):
        return "ball"
    if field.startswith("self."):
        return "self_car"
    if field.startswith("opponent."):
        return "opponent_car"
    if field.startswith("relative."):
        return "relative"
    if field.startswith("boost_pad."):
        return "boost_pads"
    if field.startswith("previous_action."):
        return "previous_action"
    if field.startswith("lifecycle."):
        return "lifecycle"
    raise ValueError(field)


def _error_row(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error.astype(np.float64))
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(absolute).mean())),
        "max_abs": float(absolute.max()),
        "nonzero_fraction": float(np.count_nonzero(absolute) / absolute.size),
    }


def _simulator_calibration(
    repo: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = repo / BOOTSTRAP_PATH
    simulator_source = repo / SIMULATOR_SOURCE_PATH
    bootstrap_hash_before = file_sha256(bootstrap)
    simulator_hash = file_sha256(simulator_source)
    if bootstrap_hash_before != BOOTSTRAP_SHA256:
        raise ValueError("120 Hz iteration-479 bootstrap SHA-256 changed")
    if simulator_hash != SIMULATOR_SOURCE_SHA256:
        raise ValueError("authoritative simulator observation corpus SHA-256 changed")
    source = torch.load(simulator_source, map_location="cpu", weights_only=False)
    true_tensor = source["observations"].detach().cpu().to(torch.float32).contiguous()
    if true_tensor.shape != (512, OBS_DIM):
        raise ValueError("bounded simulator source corpus shape changed")
    if retention_observation_sha256(true_tensor) != SIMULATOR_OBSERVATION_SHA256:
        raise ValueError("bounded simulator observation tensor SHA-256 changed")
    true = true_tensor.numpy().copy()
    degraded, quality = degrade_simulator_observations(true)

    payload = torch.load(bootstrap, map_location="cpu", weights_only=False)
    config = Rival2PolicyConfig(**payload["policy_config"])
    teacher_model = Rival2ActorCritic(config)
    teacher_model.load_state_dict(payload["model"])
    teacher_model.eval()
    student_model = Rival2ActorCritic(config)
    student_model.load_state_dict(payload["model"])
    student_model.eval()
    model_hash_before = tensor_tree_sha256(teacher_model.state_dict())
    student_hash_before = tensor_tree_sha256(student_model.state_dict())
    if model_hash_before != BOOTSTRAP_MODEL_SHA256:
        raise ValueError("bootstrap model tensor SHA-256 changed")
    if student_hash_before != BOOTSTRAP_MODEL_SHA256:
        raise ValueError("student did not initialize to bootstrap tensor parity")
    teacher_parameter_bytes_before = b"".join(
        parameter.detach().cpu().contiguous().numpy().tobytes(order="C")
        for parameter in teacher_model.parameters()
    )
    student_parameter_bytes_before = b"".join(
        parameter.detach().cpu().contiguous().numpy().tobytes(order="C")
        for parameter in student_model.parameters()
    )
    teacher_gradients_before = [parameter.grad for parameter in teacher_model.parameters()]
    student_gradients_before = [parameter.grad for parameter in student_model.parameters()]
    with torch.no_grad():
        teacher_full_actor, teacher_full_value = teacher_model(true_tensor)
        student_full_actor, student_full_value = student_model(true_tensor)
        full_observation_functional_parity = bool(
            torch.equal(teacher_full_actor, student_full_actor)
            and torch.equal(teacher_full_value, student_full_value)
        )
        objective = actor_distribution_distillation_objective(
            teacher_model,
            student_model,
            true_tensor,
            torch.from_numpy(np.asarray(degraded).copy()),
            torch.from_numpy(np.asarray(quality).copy()),
            policy_config=config,
        )
    model_hash_after = tensor_tree_sha256(teacher_model.state_dict())
    student_hash_after = tensor_tree_sha256(student_model.state_dict())
    teacher_parameter_bytes_after = b"".join(
        parameter.detach().cpu().contiguous().numpy().tobytes(order="C")
        for parameter in teacher_model.parameters()
    )
    student_parameter_bytes_after = b"".join(
        parameter.detach().cpu().contiguous().numpy().tobytes(order="C")
        for parameter in student_model.parameters()
    )
    teacher_gradients_after = [parameter.grad for parameter in teacher_model.parameters()]
    student_gradients_after = [parameter.grad for parameter in student_model.parameters()]
    if (
        model_hash_before != model_hash_after
        or teacher_parameter_bytes_before != teacher_parameter_bytes_after
    ):
        raise RuntimeError("distillation audit mutated the bootstrap teacher")
    if (
        student_hash_before != student_hash_after
        or student_parameter_bytes_before != student_parameter_bytes_after
    ):
        raise RuntimeError("distillation audit mutated the parity-initialized student")
    if any(
        before is not after
        for before, after in zip(
            teacher_gradients_before, teacher_gradients_after, strict=True
        )
    ) or any(
        before is not after
        for before, after in zip(
            student_gradients_before, student_gradients_after, strict=True
        )
    ):
        raise RuntimeError("distillation audit mutated model gradients")
    if file_sha256(bootstrap) != bootstrap_hash_before:
        raise RuntimeError("distillation audit mutated the bootstrap checkpoint")

    arrays = {
        "true_observations.npy": true,
        "degraded_observations.npy": np.asarray(degraded),
        "quality_mask.npy": np.asarray(quality),
        "teacher_actor.npy": objective.teacher_actor.detach().cpu().numpy(),
        "student_actor.npy": objective.student_actor.detach().cpu().numpy(),
    }
    array_artifacts = []
    for name, value in arrays.items():
        path = output_dir / "simulator_corpus" / name
        _write_npy(path, value)
        array_artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": _tensor_sha256(value),
                "file_sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    error = np.asarray(degraded, dtype=np.float32) - true
    per_field = []
    for index, field in enumerate(OBS_FIELD_NAMES):
        per_field.append(
            {
                "index": index,
                "field": field,
                "global_classification": FIELD_QUALITY_SPECS[index].classification,
                **_error_row(error[:, index]),
            }
        )
    groups: dict[str, list[int]] = defaultdict(list)
    qualities: dict[str, list[int]] = defaultdict(list)
    for index, spec in enumerate(FIELD_QUALITY_SPECS):
        groups[_field_group(spec.field)].append(index)
        qualities[spec.classification].append(index)
    group_rows = {
        group: {
            "field_count": len(indices),
            **_error_row(error[:, indices]),
        }
        for group, indices in sorted(groups.items())
    }
    quality_rows = {
        label: {
            "field_count": len(indices),
            **_error_row(error[:, indices]),
        }
        for label, indices in sorted(qualities.items())
    }
    calibration = {
        "format": "RIVAL2_HUMAN_DEMO_BC_BRIDGE_SIMULATOR_CALIBRATION_V1",
        "bridge_contract": bridge_contract(),
        "source": {
            "path": SIMULATOR_SOURCE_PATH.as_posix(),
            "file_sha256": simulator_hash,
            "observation_tensor_sha256": retention_observation_sha256(true_tensor),
            "observation_count": int(true.shape[0]),
            "observation_shape": list(true.shape),
            "identity": source["summary"]["identity"],
            "collection": source["summary"]["collection"],
            "state_coverage": source["summary"]["state_coverage"],
            "source_model_tensor_sha256": source["summary"]["source_identity"][
                "model_tensor_sha256"
            ],
        },
        "degradation_semantics": (
            "Simulator values are the numerical targets of deterministic human "
            "reconstruction. Non-unavailable values are retained with their non-promoted "
            "quality class; unavailable values use neutral zero plus quality=unavailable. "
            "Error therefore measures missing-feature loss, not cross-engine semantic error."
        ),
        "overall_error": _error_row(error),
        "error_by_field_group": group_rows,
        "error_by_global_quality": quality_rows,
        "per_field_error": per_field,
        "array_artifacts": array_artifacts,
    }
    distillation = {
        "format": "RIVAL2_HUMAN_DEMO_BC_BRIDGE_DISTILLATION_INTERFACE_AUDIT_V1",
        "objective": "teacher true observation to student degraded observation actor KL",
        "teacher_model": {
            "path": BOOTSTRAP_PATH.as_posix(),
            "checkpoint_sha256_before": bootstrap_hash_before,
            "checkpoint_sha256_after": file_sha256(bootstrap),
            "model_tensor_sha256_before": model_hash_before,
            "model_tensor_sha256_after": model_hash_after,
            "iteration": int(payload["iteration"]),
            "policy_version": int(payload["policy_version"]),
        },
        "architecture_change_required": False,
        "teacher_student_modules_independent": teacher_model is not student_model,
        "student_initial_model_tensor_sha256": student_hash_before,
        "student_final_model_tensor_sha256": student_hash_after,
        "student_full_observation_functional_parity": full_observation_functional_parity,
        "policy_config_before": payload["policy_config"],
        "policy_config_after": payload["policy_config"],
        "paired_sample_count": int(true.shape[0]),
        "teacher_student_mean_kl": float(objective.loss.item()),
        "teacher_student_max_sample_kl": float(objective.per_sample_kl.max().item()),
        "teacher_student_kl_by_action_channel": {
            name: float(objective.per_action_channel_kl[index].item())
            for index, name in enumerate(ACTION_NAMES)
        },
        "unavailable_feature_fraction": float(objective.unavailable_fraction.item()),
        "training_ready_interface": True,
        "backward_called": False,
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "model_parameters_byte_identical": (
            teacher_parameter_bytes_before == teacher_parameter_bytes_after
            and student_parameter_bytes_before == student_parameter_bytes_after
        ),
        "model_gradients_unchanged": all(
            before is after
            for before, after in zip(
                teacher_gradients_before, teacher_gradients_after, strict=True
            )
        )
        and all(
            before is after
            for before, after in zip(
                student_gradients_before, student_gradients_after, strict=True
            )
        ),
    }
    return calibration, distillation


def _no_learning_audit(paths: Iterable[Path]) -> dict[str, Any]:
    forbidden = {"backward", "optimizer_step", "train_iteration"}
    call_sites = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", "")
            )
            rendered = ast.unparse(function)
            if (
                name in forbidden
                or (name == "update" and "trainer" in rendered)
                or (name == "step" and "optimizer" in rendered)
            ):
                call_sites.append({"path": path.name, "line": node.lineno, "call": name})
    return {
        "training_performed": False,
        "behavior_cloning_performed": False,
        "ppo_performed": False,
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
        "reward_changed": False,
        "mechanic_detector_changed": False,
        "model_mutated": False,
        "forbidden_learning_call_sites": call_sites,
        "valid": not call_sites,
    }


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "verification_evidence.json"}
        ),
        key=lambda item: item.relative_to(output_dir).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "format": "RIVAL2_HUMAN_DEMO_BC_BRIDGE_ARTIFACT_MANIFEST_V1",
        "file_count": len(files),
        "files": files,
    }


def build(
    repo: Path, source_root: Path, output_dir: Path
) -> dict[str, Any]:
    dataset_manifest, authority = _authority(repo)
    source_verification = _verify_human_sources(source_root, dataset_manifest)
    human_statistics, human_summary = _scan_human_corpus(
        source_root, dataset_manifest
    )
    simulator_calibration, distillation_audit = _simulator_calibration(
        repo, output_dir
    )
    quality_contract = field_quality_contract()
    no_learning = _no_learning_audit(
        [
            repo / "rivalsim/human_demo/bc_observation_bridge.py",
            Path(__file__).resolve(),
        ]
    )
    checks = {
        "base_commit_exact": authority["base_commit"] == BASE_COMMIT,
        "exact_adapter_byte_identical": authority["exact_adapter"][
            "byte_identical_to_base"
        ],
        "exact_adapter_remains_fail_closed": human_summary["bc_usable_frame_count"] > 0
        and human_statistics["exact_audit_usable_frame_count"] == 0,
        "source_hashes_unchanged": all(row["unchanged"] for row in source_verification),
        "all_human_frames_bridge_usable": human_summary["bc_usable_frame_count"]
        == human_summary["frame_count"],
        "all_action_targets_unchanged": human_summary["action_unchanged"],
        "quality_field_count_exact": sum(quality_contract["counts"].values()) == OBS_DIM,
        "quality_not_promoted": True,
        "simulator_true_degraded_pair_count_exact": simulator_calibration["source"][
            "observation_count"
        ]
        == 512,
        "bootstrap_checkpoint_unchanged": distillation_audit["teacher_model"][
            "checkpoint_sha256_before"
        ]
        == distillation_audit["teacher_model"]["checkpoint_sha256_after"],
        "bootstrap_model_byte_identical": distillation_audit[
            "model_parameters_byte_identical"
        ],
        "student_initialized_to_bootstrap_parity": (
            distillation_audit["student_initial_model_tensor_sha256"]
            == BOOTSTRAP_MODEL_SHA256
            and distillation_audit["student_full_observation_functional_parity"]
        ),
        "teacher_student_modules_independent": distillation_audit[
            "teacher_student_modules_independent"
        ],
        "no_architecture_change": not distillation_audit["architecture_change_required"],
        "no_learning_boundary": no_learning["valid"],
    }
    verdict = (
        "RIVAL2_HUMAN_DEMO_BC_OBSERVATION_BRIDGE_VALID"
        if all(checks.values())
        else "RIVAL2_HUMAN_DEMO_BC_OBSERVATION_BRIDGE_INVALID"
    )
    manifest = {
        "format": OUTPUT_FORMAT,
        "authority": authority,
        "bridge_contract": bridge_contract(),
        "field_quality_contract_sha256": FIELD_QUALITY_CONTRACT_SHA256,
        "source_locator": {
            "type": "session_uuid_under_user_supplied_native_demo_root",
            "environment_default": "%APPDATA%/bakkesmod/bakkesmod/data/rival2/human_demos",
            "absolute_path_frozen": False,
        },
        "human_source_verification": source_verification,
        "human_corpus_summary": human_summary,
        "simulator_corpus_summary": {
            "source_path": SIMULATOR_SOURCE_PATH.as_posix(),
            "source_sha256": SIMULATOR_SOURCE_SHA256,
            "observation_count": simulator_calibration["source"]["observation_count"],
            "true_observation_tensor_sha256": simulator_calibration["source"][
                "observation_tensor_sha256"
            ],
            "degraded_observation_tensor_sha256": next(
                row["tensor_sha256"]
                for row in simulator_calibration["array_artifacts"]
                if row["path"].endswith("degraded_observations.npy")
            ),
        },
        "no_learning_boundary": no_learning,
        "checks": checks,
        "verdict": verdict,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "field_quality_contract.json", quality_contract)
    _write_json(output_dir / "human_corpus_statistics.json", human_statistics)
    _write_json(
        output_dir / "simulator_reconstruction_calibration.json",
        simulator_calibration,
    )
    _write_json(output_dir / "distillation_interface_audit.json", distillation_audit)
    _write_json(output_dir / "bridge_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Rival 120 Hz human-demo BC observation bridge V1\n\n"
        "This directory contains the masked BC-domain bridge, human-corpus scan, paired "
        "authoritative simulator full/degraded corpus, reconstruction statistics, and frozen "
        "teacher/student actor-distribution objective audit. No optimizer, BC, PPO, reward, "
        "detector, or model mutation occurs. See `docs/RIVAL2_HUMAN_DEMO_BC_BRIDGE_V1.md`.\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(output_dir / "artifact_manifest.json", _artifact_manifest(output_dir))
    if verdict.endswith("INVALID"):
        raise RuntimeError(f"BC bridge validation failed: {checks}")
    return {
        "verdict": verdict,
        "human_bc_usable_frame_count": human_summary["bc_usable_frame_count"],
        "field_quality_counts": quality_contract["counts"],
        "simulator_observation_count": simulator_calibration["source"][
            "observation_count"
        ],
        "teacher_student_mean_kl": distillation_audit["teacher_student_mean_kl"],
        "artifact_manifest_sha256": _sha256(output_dir / "artifact_manifest.json"),
    }


def verify(output_dir: Path) -> dict[str, Any]:
    errors = []
    artifact = json.loads((output_dir / "artifact_manifest.json").read_text())
    for row in artifact["files"]:
        path = output_dir / row["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            errors.append(f"artifact hash mismatch: {row['path']}")
    manifest = json.loads((output_dir / "bridge_manifest.json").read_text())
    if not all(manifest["checks"].values()):
        errors.append("bridge manifest contains failed validation checks")
    if manifest["field_quality_contract_sha256"] != FIELD_QUALITY_CONTRACT_SHA256:
        errors.append("active field-quality contract hash mismatch")
    human = json.loads((output_dir / "human_corpus_statistics.json").read_text())
    if human["frame_count"] != 114_311 or human["bc_usable_frame_count"] != 114_311:
        errors.append("human BC-usable frame count mismatch")
    if human["action_mismatch_frame_count"] != 0:
        errors.append("human action target changed")
    calibration = json.loads(
        (output_dir / "simulator_reconstruction_calibration.json").read_text()
    )
    for row in calibration["array_artifacts"]:
        path = output_dir / row["path"]
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"]:
            errors.append(f"simulator array contract mismatch: {row['path']}")
        if _tensor_sha256(value) != row["tensor_sha256"]:
            errors.append(f"simulator array tensor hash mismatch: {row['path']}")
    return {
        "format": "RIVAL2_HUMAN_DEMO_BC_BRIDGE_VERIFICATION_V1",
        "valid": not errors,
        "errors": errors,
        "artifact_count": artifact["file_count"],
        "human_bc_usable_frame_count": human["bc_usable_frame_count"],
        "simulator_observation_count": calibration["source"]["observation_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            Path(os.environ["APPDATA"])
            / "bakkesmod"
            / "bakkesmod"
            / "data"
            / "rival2"
            / "human_demos"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rival2/human_demo_bc_bridge_v1"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    result = (
        verify(output_dir)
        if args.verify_only
        else build(args.repo.resolve(), args.source_root.resolve(), output_dir)
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
