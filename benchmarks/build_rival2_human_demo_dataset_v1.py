"""Build the frozen read-only Rival 120 Hz human-demonstration dataset manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from rivalsim.human_demo import SessionReader
from rivalsim.human_demo.training_adapter import (
    ADAPTER_FORMAT,
    GAMEPLAY_SPLIT_SEED,
    HARD_BOUNDARY_KINDS,
    MECHANIC_SPLIT_SEED,
    ReadOnlyTrajectoryAdapter,
    contract_identity,
    split_gameplay_regions,
    split_mechanic_candidates,
)

AUTHORITY_COMMIT = "08a55e8326940271689847b316fa096a7fed3c71"
REVIEW_ROOT = "results/rival2/human_demo_review_v2"
DATASET_FORMAT = "RIVAL2_HUMAN_DEMO_DATASET_MANIFEST_V1"
CORE_AUTHORITY_PATHS = (
    f"{REVIEW_ROOT}/index.json",
    f"{REVIEW_ROOT}/source_inventory.json",
    f"{REVIEW_ROOT}/behavior_cloning_candidates.json",
    f"{REVIEW_ROOT}/mechanic_assessments.json",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _authority_bytes(repo: Path, path: str) -> bytes:
    return _git(repo, "show", f"{AUTHORITY_COMMIT}:{path}")


def _authority_json(repo: Path, path: str) -> dict[str, Any]:
    return json.loads(_authority_bytes(repo, path))


def _authority_identity(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", f"{AUTHORITY_COMMIT}^{{commit}}").decode().strip()
    if commit != AUTHORITY_COMMIT:
        raise ValueError("authority commit did not resolve exactly")
    attempt_paths = sorted(
        path
        for path in _git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            AUTHORITY_COMMIT,
            f"{REVIEW_ROOT}/attempts",
        )
        .decode()
        .splitlines()
        if path.endswith(".jsonl")
    )
    if not attempt_paths:
        raise ValueError("authority review contains no per-session attempt artifacts")
    rows = []
    for path in (*CORE_AUTHORITY_PATHS, *attempt_paths):
        payload = _authority_bytes(repo, path)
        oid = _git(repo, "rev-parse", f"{AUTHORITY_COMMIT}:{path}").decode().strip()
        rows.append(
            {
                "path": path,
                "git_blob_oid": oid,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return {
        "commit": commit,
        "review_format": "RIVALRL_HUMAN_DEMO_REVIEW_V2",
        "artifacts": rows,
    }


def _verify_source_session(
    source_root: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    session_uuid = str(inventory["session_uuid"])
    session_dir = source_root / session_uuid
    digest = hashlib.sha256()
    files = []
    for expected in inventory["files"]:
        relative = str(expected["path"])
        path = session_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing native source artifact: {session_uuid}/{relative}")
        actual_hash = _sha256(path)
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"native source byte-size mismatch: {session_uuid}/{relative}")
        if actual_hash != str(expected["sha256"]):
            raise ValueError(f"native source SHA-256 mismatch: {session_uuid}/{relative}")
        digest.update(f"{relative}:{actual_hash}\n".encode())
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
            }
        )
    file_set_hash = digest.hexdigest().upper()
    if file_set_hash != str(inventory["source_file_set_sha256"]):
        raise ValueError(f"native source file-set mismatch: {session_uuid}")
    validation = SessionReader(session_dir).validate().as_dict()
    if not bool(validation["container_valid"]):
        raise ValueError(f"native source container validation failed: {session_uuid}")
    if not bool(validation["manifest_hashes_valid"]):
        raise ValueError(f"native source manifest hash validation failed: {session_uuid}")
    return {
        "session_uuid": session_uuid,
        "source_file_set_sha256": file_set_hash,
        "file_count": len(files),
        "files": files,
        "container_valid": bool(validation["container_valid"]),
        "manifest_hashes_valid": bool(validation["manifest_hashes_valid"]),
        "frame_count": int(validation["frame_count"]),
    }


def _authority_attempts(repo: Path, session_uuid: str) -> list[dict[str, Any]]:
    path = f"{REVIEW_ROOT}/attempts/{session_uuid}.jsonl"
    return [
        json.loads(line)
        for line in _authority_bytes(repo, path).decode().splitlines()
        if line.strip()
    ]


def _negative_attempt_references(
    repo: Path, review_index: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for session in review_index["sessions"]:
        if session["classification"] != "freeplay_mechanic_practice":
            continue
        for attempt in _authority_attempts(repo, str(session["session_uuid"])):
            assessment = attempt["mechanic_assessment"]
            if assessment["verdict"] == "success":
                continue
            segmentation = attempt["segmentation"]
            rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "session_uuid": attempt["session_uuid"],
                    "declared_label": attempt["declared_label"],
                    "verdict": assessment["verdict"],
                    "start_sequence": segmentation["start_sequence"],
                    "end_sequence": segmentation["end_sequence"],
                    "start_physics_frame": segmentation["start_physics_frame"],
                    "end_physics_frame": segmentation["end_physics_frame"],
                    "initial_bc_positive_cohort": False,
                    "retained_for": ["future_evaluation", "future_detector_work"],
                }
            )
    return sorted(rows, key=lambda row: str(row["attempt_id"]))


def _gameplay_regions(
    frames: Sequence[dict[str, int]], events: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not frames:
        raise ValueError("gameplay recording contains no frames")
    physics = [int(frame["physics_frame"]) for frame in frames]
    boundaries: dict[int, set[str]] = {0: {"recording_start"}, len(frames): {"recording_end"}}
    for index in range(1, len(frames)):
        if (
            int(frames[index]["sequence"]) != int(frames[index - 1]["sequence"]) + 1
            or physics[index] != physics[index - 1] + 1
        ):
            boundaries.setdefault(index, set()).add("native_frame_discontinuity")
    for event in events:
        kind = str(event.get("kind", ""))
        event_physics = int(event.get("physics_frame", -1))
        if kind not in HARD_BOUNDARY_KINDS or not physics[0] <= event_physics <= physics[-1]:
            continue
        index = bisect_left(physics, event_physics)
        if 0 <= index < len(frames):
            boundaries.setdefault(index, set()).add(kind)
    ordered = sorted(boundaries)
    regions = []
    for region_number, (start, stop) in enumerate(pairwise(ordered), 1):
        if stop <= start:
            continue
        first, last = frames[start], frames[stop - 1]
        regions.append(
            {
                "region_id": f"gameplay-region-{region_number:04d}",
                "start_sequence": int(first["sequence"]),
                "end_sequence": int(last["sequence"]),
                "start_physics_frame": int(first["physics_frame"]),
                "end_physics_frame": int(last["physics_frame"]),
                "source_frame_count": stop - start,
                "boundary_before": sorted(boundaries[start]),
                "first_frame_previous_action_requires_external_predecessor": True,
            }
        )
    return regions


def bisect_left(values: Sequence[int], target: int) -> int:
    """Small local wrapper keeps the manifest builder dependency surface explicit."""

    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def _scan_spans(
    source_root: Path,
    session_uuid: str,
    spans: Sequence[dict[str, Any]],
    identity_key: str,
) -> dict[str, dict[str, Any]]:
    adapter = ReadOnlyTrajectoryAdapter(source_root / session_uuid)
    requested = [
        (
            str(row[identity_key]),
            int(row["start_sequence"]),
            int(row["end_sequence"]),
        )
        for row in spans
    ]
    results: dict[str, dict[str, Any]] = {
        identity: {
            "source_frame_count": 0,
            "exact_action_target_count": 0,
            "exact_previous_action_count": 0,
            "boundary_previous_action_exclusion_count": 0,
            "exact_observation_usable_frame_count": 0,
            "blocked_field_counts": Counter(),
            "blocker_reason_counts": Counter(),
            "first_sequence": None,
            "last_sequence": None,
        }
        for identity, _start, _end in requested
    }
    for identity, sample in adapter.iter_spans(requested):
        result = results[identity]
        result["source_frame_count"] += 1
        result["exact_action_target_count"] += 1
        result["exact_previous_action_count"] += int(
            sample.previous_action_source_sequence is not None
        )
        result["boundary_previous_action_exclusion_count"] += int(
            sample.previous_action_source_sequence is None
        )
        result["exact_observation_usable_frame_count"] += int(sample.usable)
        result["blocked_field_counts"].update(sample.blocked_fields)
        result["blocker_reason_counts"].update(sample.blocker_reasons)
        if result["first_sequence"] is None:
            result["first_sequence"] = sample.sequence
        result["last_sequence"] = sample.sequence
    for result in results.values():
        result["blocked_field_counts"] = dict(sorted(result["blocked_field_counts"].items()))
        result["blocker_reason_counts"] = dict(
            sorted(result["blocker_reason_counts"].items())
        )
    return results


def _split_statistics(
    mechanic_attempts: Sequence[dict[str, Any]], gameplay_regions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    rows = {}
    for split in ("train", "validation", "test"):
        mechanic = [row for row in mechanic_attempts if row["split"] == split]
        gameplay = [row for row in gameplay_regions if row["split"] == split]
        rows[split] = {
            "mechanic_attempt_count": len(mechanic),
            "mechanic_source_frame_count": sum(
                int(row["source_frame_count"]) for row in mechanic
            ),
            "mechanic_exact_observation_usable_frame_count": sum(
                int(row["adapter_scan"]["exact_observation_usable_frame_count"])
                for row in mechanic
            ),
            "gameplay_region_count": len(gameplay),
            "gameplay_source_frame_count": sum(
                int(row["source_frame_count"]) for row in gameplay
            ),
            "gameplay_exact_observation_usable_frame_count": sum(
                int(row["adapter_scan"]["exact_observation_usable_frame_count"])
                for row in gameplay
            ),
        }
    return rows


def _sampling_metadata(mechanic_attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    splits = {}
    for split in ("train", "validation", "test"):
        labels = defaultdict(list)
        for row in mechanic_attempts:
            if row["split"] == split:
                labels[str(row["declared_label"])].append(row)
        splits[split] = {
            "labels": {
                label: {
                    "attempt_ids": sorted(str(row["attempt_id"]) for row in rows),
                    "attempt_count": len(rows),
                    "source_frame_count": sum(int(row["source_frame_count"]) for row in rows),
                    "exact_observation_usable_frame_count": sum(
                        int(row["adapter_scan"]["exact_observation_usable_frame_count"])
                        for row in rows
                    ),
                }
                for label, rows in sorted(labels.items())
            }
        }
    return {
        "format": "RIVAL2_HUMAN_DEMO_BALANCED_SAMPLING_METADATA_V1",
        "splits": splits,
        "interface": {
            "natural_frame_sampling": {
                "hierarchy": ["split", "trajectory", "frame"],
                "coefficients": None,
            },
            "balanced_mechanic_sampling": {
                "hierarchy": ["split", "mechanic_label", "attempt", "frame"],
                "selection": [
                    "choose an available mechanic label",
                    "choose a whole attempt within that label",
                    "choose or iterate a frame within that attempt",
                ],
                "coefficients": None,
                "aggressive_oversampling_selected": False,
            },
        },
        "policy": (
            "Counts and label/attempt buckets are frozen. A later trainer may choose a "
            "sampling policy, but this task selects no oversampling coefficients."
        ),
    }


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
            if name in forbidden or (name == "step" and "optimizer" in ast.unparse(function)):
                call_sites.append({"path": path.name, "line": node.lineno, "call": name})
    return {
        "training_performed": False,
        "behavior_cloning_performed": False,
        "ppo_performed": False,
        "model_or_optimizer_mutated": False,
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
        key=lambda path: path.relative_to(output_dir).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "format": "RIVAL2_HUMAN_DEMO_DATASET_ARTIFACT_MANIFEST_V1",
        "files": files,
        "file_count": len(files),
    }


def build(repo: Path, source_root: Path, output_dir: Path) -> dict[str, Any]:
    authority = _authority_identity(repo)
    review_index = _authority_json(repo, f"{REVIEW_ROOT}/index.json")
    inventory_document = _authority_json(repo, f"{REVIEW_ROOT}/source_inventory.json")
    candidate_document = _authority_json(
        repo, f"{REVIEW_ROOT}/behavior_cloning_candidates.json"
    )
    if review_index["behavior_cloning_candidate_count"] != 110:
        raise ValueError("authority review does not contain exactly 110 BC candidates")
    if candidate_document["candidate_count"] != 110:
        raise ValueError("authority candidate index does not contain exactly 110 candidates")

    inventory_by_uuid = {
        str(row["session_uuid"]): row for row in inventory_document["sessions"]
    }
    source_verification = [
        _verify_source_session(source_root, inventory_by_uuid[str(session["session_uuid"])])
        for session in review_index["sessions"]
    ]
    source_by_uuid = {row["session_uuid"]: row for row in source_verification}

    mechanic_attempts = split_mechanic_candidates(candidate_document["candidates"])
    for row in mechanic_attempts:
        row["source_frame_count"] = int(row["end_sequence"]) - int(row["start_sequence"]) + 1
        row["trajectory_kind"] = "mechanic_positive_attempt"
        row["initial_bc_positive_cohort"] = True
        row["source_contract"] = {
            "session_uuid": row["session_uuid"],
            "source_file_set_sha256": source_by_uuid[str(row["session_uuid"])][
                "source_file_set_sha256"
            ],
        }
    mechanics_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mechanic_attempts:
        mechanics_by_session[str(row["session_uuid"])].append(row)
    for session_uuid, rows in mechanics_by_session.items():
        scans = _scan_spans(source_root, session_uuid, rows, "attempt_id")
        for row in rows:
            row["adapter_scan"] = scans[str(row["attempt_id"])]

    gameplay_sessions = [
        row for row in review_index["sessions"] if row["classification"] == "gameplay"
    ]
    if len(gameplay_sessions) != 1:
        raise ValueError("authority review must contain exactly one gameplay session")
    gameplay_session = gameplay_sessions[0]
    gameplay_uuid = str(gameplay_session["session_uuid"])
    reader = SessionReader(source_root / gameplay_uuid)
    compact_frames = [
        {"sequence": int(frame["sequence"]), "physics_frame": int(frame["physics_frame"])}
        for frame in reader.iter_frames()
    ]
    first_physics = int(compact_frames[0]["physics_frame"])
    last_physics = int(compact_frames[-1]["physics_frame"])
    paired_events = [
        event
        for event in reader.iter_events()
        if first_physics <= int(event.get("physics_frame", -1)) <= last_physics
    ]
    gameplay_regions = split_gameplay_regions(
        _gameplay_regions(compact_frames, paired_events),
        session_uuid=gameplay_uuid,
    )
    gameplay_scans = _scan_spans(
        source_root, gameplay_uuid, gameplay_regions, "region_id"
    )
    for row in gameplay_regions:
        row["session_uuid"] = gameplay_uuid
        row["trajectory_kind"] = "general_gameplay"
        row["declared_label"] = "nexto_1v1"
        row["source_file_set_sha256"] = source_by_uuid[gameplay_uuid][
            "source_file_set_sha256"
        ]
        row["adapter_scan"] = gameplay_scans[str(row["region_id"])]

    negative_references = _negative_attempt_references(repo, review_index)
    split_statistics = _split_statistics(mechanic_attempts, gameplay_regions)
    sampling = _sampling_metadata(mechanic_attempts)
    blocker_fields = Counter()
    blocker_reasons = Counter()
    for row in [*mechanic_attempts, *gameplay_regions]:
        blocker_fields.update(row["adapter_scan"]["blocked_field_counts"])
        blocker_reasons.update(row["adapter_scan"]["blocker_reason_counts"])
    exact_usable = sum(
        int(row["adapter_scan"]["exact_observation_usable_frame_count"])
        for row in [*mechanic_attempts, *gameplay_regions]
    )

    manifest = {
        "format": DATASET_FORMAT,
        "authority": authority,
        "contracts": contract_identity(),
        "source_locator": {
            "type": "session_uuid_under_user_supplied_native_demo_root",
            "environment_default": "%APPDATA%/bakkesmod/bakkesmod/data/rival2/human_demos",
            "absolute_path_frozen": False,
        },
        "split_policy": {
            "mechanic_seed": MECHANIC_SPLIT_SEED,
            "mechanic_unit": "whole_attempt",
            "mechanic_strategy": (
                "per-label deterministic SHA-256 order; labels with >=3 successes reserve "
                "one validation and one test attempt; labels with 2 reserve validation; "
                "labels with >=10 use rounded 10% validation and test"
            ),
            "gameplay_seed": GAMEPLAY_SPLIT_SEED,
            "gameplay_unit": "whole_reset_rebind_episode_region",
            "gameplay_strategy": "deterministic SHA-256 region order with 10% validation/test",
            "neighboring_frame_leakage": False,
        },
        "mechanic_positive_attempts": mechanic_attempts,
        "general_gameplay": {
            "session_uuid": gameplay_uuid,
            "declared_label": "nexto_1v1",
            "source_frame_count": len(compact_frames),
            "regions": gameplay_regions,
        },
        "excluded_nonpositive_mechanic_attempts": negative_references,
        "split_statistics": split_statistics,
        "source_verification": source_verification,
        "authority_boundary": {
            "training_performed": False,
            "behavior_cloning_performed": False,
            "ppo_performed": False,
            "optimizer_step_performed": False,
            "reward_changed": False,
            "mechanic_detector_defined": False,
            "model_mutated": False,
        },
    }
    audit = {
        "format": "RIVAL2_HUMAN_DEMO_ADAPTER_AUDIT_V1",
        "adapter_format": ADAPTER_FORMAT,
        "dataset_format": DATASET_FORMAT,
        "contract_identity": contract_identity(),
        "source_session_count": len(source_verification),
        "mechanic_positive_attempt_count": len(mechanic_attempts),
        "mechanic_positive_source_frame_count": sum(
            int(row["source_frame_count"]) for row in mechanic_attempts
        ),
        "gameplay_source_frame_count": len(compact_frames),
        "exact_observation_usable_frame_count": exact_usable,
        "exact_action_target_frame_count": sum(
            int(row["adapter_scan"]["exact_action_target_count"])
            for row in [*mechanic_attempts, *gameplay_regions]
        ),
        "blocked_field_counts": dict(sorted(blocker_fields.items())),
        "blocker_reason_counts": dict(sorted(blocker_reasons.items())),
        "validation_verdict": (
            "BLOCKED_EXACT_OBSERVATION_SOURCE_INCOMPLETE"
            if exact_usable == 0
            else "RIVAL2_HUMAN_DEMO_ADAPTER_VALID"
        ),
        "fail_closed": True,
        "unavailable_values_filled": False,
        "observation_emitted_when_incomplete": False,
        "action_temporal_reduction": None,
        "no_learning_audit": _no_learning_audit(
            [
                repo / "rivalsim/human_demo/training_adapter.py",
                Path(__file__).resolve(),
            ]
        ),
    }
    statistics = {
        "format": "RIVAL2_HUMAN_DEMO_DATASET_STATISTICS_V1",
        "split_statistics": split_statistics,
        "mechanic_verdict_reference_counts": dict(
            sorted(Counter(row["verdict"] for row in negative_references).items())
        ),
        "mechanic_label_counts": sampling["splits"],
        "gameplay_region_count": len(gameplay_regions),
        "gameplay_source_frame_count": len(compact_frames),
        "gameplay_exact_observation_usable_frame_count": sum(
            int(row["adapter_scan"]["exact_observation_usable_frame_count"])
            for row in gameplay_regions
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dataset_manifest.json", manifest)
    _write_json(output_dir / "adapter_audit.json", audit)
    _write_json(output_dir / "statistics.json", statistics)
    _write_json(output_dir / "sampling_metadata.json", sampling)
    (output_dir / "README.md").write_text(
        "# Rival 120 Hz human-demo dataset V1\n\n"
        "This directory freezes source references, whole-attempt and lifecycle-region splits, "
        "read-only adapter exactness evidence, and balanced-sampling metadata. No training was "
        "performed. `adapter_audit.json` is fail-closed: the committed native recorder does not "
        "contain enough state to emit any exact 182-field observation.\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(output_dir / "artifact_manifest.json", _artifact_manifest(output_dir))
    return {
        "dataset_manifest_sha256": _sha256(output_dir / "dataset_manifest.json"),
        "artifact_manifest_sha256": _sha256(output_dir / "artifact_manifest.json"),
        "mechanic_attempt_count": len(mechanic_attempts),
        "mechanic_source_frame_count": sum(
            int(row["source_frame_count"]) for row in mechanic_attempts
        ),
        "gameplay_source_frame_count": len(compact_frames),
        "exact_observation_usable_frame_count": exact_usable,
        "validation_verdict": audit["validation_verdict"],
    }


def verify(output_dir: Path) -> dict[str, Any]:
    errors = []
    artifact = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    for row in artifact["files"]:
        path = output_dir / row["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            errors.append(f"artifact hash mismatch: {row['path']}")
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    attempts = manifest["mechanic_positive_attempts"]
    attempt_ids = [row["attempt_id"] for row in attempts]
    if len(attempt_ids) != 110 or len(set(attempt_ids)) != 110:
        errors.append("mechanic candidate identity count mismatch")
    for row in attempts:
        if row["adapter_scan"]["source_frame_count"] != row["source_frame_count"]:
            errors.append(f"mechanic source frame count mismatch: {row['attempt_id']}")
        if row["adapter_scan"]["exact_action_target_count"] != row["source_frame_count"]:
            errors.append(f"mechanic action count mismatch: {row['attempt_id']}")
        scan = row["adapter_scan"]
        if (
            int(scan["exact_previous_action_count"])
            + int(scan["boundary_previous_action_exclusion_count"])
            != int(row["source_frame_count"])
        ):
            errors.append(f"mechanic previous-action accounting mismatch: {row['attempt_id']}")
    attempts_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_session[str(row["session_uuid"])].append(row)
    for session_attempts in attempts_by_session.values():
        ordered_attempts = sorted(
            session_attempts, key=lambda row: int(row["start_sequence"])
        )
        for previous, current in pairwise(ordered_attempts):
            if int(current["start_sequence"]) <= int(previous["end_sequence"]):
                errors.append(
                    "mechanic source spans overlap: "
                    f"{previous['attempt_id']} and {current['attempt_id']}"
                )
    gameplay = manifest["general_gameplay"]
    regions = gameplay["regions"]
    if sum(int(row["source_frame_count"]) for row in regions) != int(
        gameplay["source_frame_count"]
    ):
        errors.append("gameplay regions do not cover the complete trajectory")
    for previous, current in pairwise(regions):
        if int(current["start_sequence"]) != int(previous["end_sequence"]) + 1:
            errors.append("gameplay regions are not an exact sequence partition")
        if previous["split"] != current["split"] and not current["boundary_before"]:
            errors.append("neighboring gameplay frames leak across splits")
    for row in regions:
        scan = row["adapter_scan"]
        if int(scan["exact_action_target_count"]) != int(row["source_frame_count"]):
            errors.append(f"gameplay action count mismatch: {row['region_id']}")
        if (
            int(scan["exact_previous_action_count"])
            + int(scan["boundary_previous_action_exclusion_count"])
            != int(row["source_frame_count"])
        ):
            errors.append(f"gameplay previous-action accounting mismatch: {row['region_id']}")
    excluded = manifest["excluded_nonpositive_mechanic_attempts"]
    if len(excluded) != 85 or any(row["initial_bc_positive_cohort"] for row in excluded):
        errors.append("nonpositive mechanic reference count mismatch")
    if manifest["contracts"] != contract_identity():
        errors.append("active contract identity mismatch")
    if any(
        not row["container_valid"] or not row["manifest_hashes_valid"]
        for row in manifest["source_verification"]
    ):
        errors.append("native source validation failed")
    if any(manifest["authority_boundary"].values()):
        errors.append("no-learning authority boundary violated")
    return {
        "format": "RIVAL2_HUMAN_DEMO_DATASET_VERIFICATION_V1",
        "valid": not errors,
        "errors": errors,
        "artifact_count": int(artifact["file_count"]),
        "mechanic_attempt_count": len(attempts),
        "gameplay_region_count": len(regions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
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
        default=Path("results/rival2/human_demo_dataset_v1"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        result = verify(args.output_dir.resolve())
    else:
        result = build(args.repo.resolve(), args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
