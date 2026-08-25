"""Content-addressed isolated native authority for v0.3 Phase B."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.arena import ArenaGeometry
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    CarBallBatchOracleFrame,
)
from rivalsim.v03_corpus import (
    V03_GENERATOR_SCHEMA_VERSION,
    V03_GENERATOR_SEED,
    V03_HARD_HORIZONS,
    CarBallCase,
    phase_b_corpus_sha256,
    phase_b_generator_config,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_ROCKETSIM_EXTENSION_SHA256,
    V03_CACHE_FORMAT_VERSION,
    canonical_json_bytes,
    phase_cache_dir,
    sha256_bytes,
    sha256_file,
)

PHASE_B_CACHE_CHUNK_SIZE = 128
PHASE_B_CAPTURE_TICKS = tuple(range(1, max(V03_HARD_HORIZONS) + 1))
PHASE_B_FIELDS = (
    "car_pos",
    "car_vel",
    "car_matrix",
    "car_ang_vel",
    "car_boost",
    "car_handbrake",
    "car_on_ground",
    "car_wheel_contacts",
    "car_world_contact",
    "car_world_contact_normal",
    "ball_pos",
    "ball_vel",
    "ball_matrix",
    "ball_ang_vel",
    "ball_last_hit_car_id",
    "pair_hit_valid",
    "pair_hit_tick",
    "pair_extra_hit_vel",
    "pair_relative_pos_on_ball",
)
PHASE_B_FRAME_FIELDS = (
    *(f"initial_{field}" for field in PHASE_B_FIELDS),
    *PHASE_B_FIELDS,
)
PHASE_B_AUTHORITY_SETTINGS: dict[str, Any] = {
    "protocol_version": 1,
    "phase": "B_car_ball",
    "game_mode": "SOCCAR",
    "tick_rate_hz": 120.0,
    "arena_config": {"no_ball_rot": False},
    "collision_switches": {
        "car_car_collision": False,
        "car_ball_collision": True,
    },
    "isolation": "one fresh one-Octane Soccar arena per corpus case",
    "car": {
        "count": 1,
        "config": "CarConfig.OCTANE",
        "team": "BLUE",
        "controls": "zero for all captured ticks",
    },
    "ball": {"shape": "standard Soccar btSphereShape", "rotation": "enabled"},
    "initial_state_custody": {
        "source_state": "exact frozen corpus record",
        "native_readback": "complete car, ball, contact semantic state immediately after SetState",
    },
    "step_order": (
        "immediate readback, then arena.step(1) and complete frame readback "
        "for ticks 1..12"
    ),
    "captured_ticks": list(PHASE_B_CAPTURE_TICKS),
    "captured_frame_fields": list(PHASE_B_FIELDS),
}

_FIELD_SCHEMA: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "car_pos": (np.dtype(np.float32), (3,)),
    "car_vel": (np.dtype(np.float32), (3,)),
    "car_matrix": (np.dtype(np.float32), (3, 3)),
    "car_ang_vel": (np.dtype(np.float32), (3,)),
    "car_boost": (np.dtype(np.float32), ()),
    "car_handbrake": (np.dtype(np.float32), ()),
    "car_on_ground": (np.dtype(np.bool_), ()),
    "car_wheel_contacts": (np.dtype(np.bool_), (4,)),
    "car_world_contact": (np.dtype(np.bool_), ()),
    "car_world_contact_normal": (np.dtype(np.float32), (3,)),
    "ball_pos": (np.dtype(np.float32), (3,)),
    "ball_vel": (np.dtype(np.float32), (3,)),
    "ball_matrix": (np.dtype(np.float32), (3, 3)),
    "ball_ang_vel": (np.dtype(np.float32), (3,)),
    "ball_last_hit_car_id": (np.dtype(np.uint32), ()),
    "pair_hit_valid": (np.dtype(np.bool_), ()),
    "pair_hit_tick": (np.dtype(np.uint64), ()),
    "pair_extra_hit_vel": (np.dtype(np.float32), (3,)),
    "pair_relative_pos_on_ball": (np.dtype(np.float32), (3,)),
}


def build_phase_b_identity(
    geometry: ArenaGeometry, cases: tuple[CarBallCase, ...]
) -> dict[str, Any]:
    config = phase_b_generator_config()
    package_root = Path(__file__).parents[1]
    corpus_path = Path(__file__).with_name("v03_corpus.py")
    oracle_path = Path(__file__).with_name("reference") / "rocketsim_oracle.py"
    builder_path = package_root / "benchmarks" / "build_v03_phase_b_cache.py"
    inputs = {
        "rocketsim": {
            "primary_repository": "https://github.com/ZealanL/RocketSim.git",
            "primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "binding_repository": "https://github.com/mtheall/RocketSim.git",
            "binding_commit": ROCKETSIM_BINDING_COMMIT,
            "binding_package_version": ROCKETSIM_BINDING_VERSION,
            "installed_extension_sha256": EXPECTED_ROCKETSIM_EXTENSION_SHA256,
        },
        "collision_assets": {
            "format": "RocketSim CMF",
            "combined_content_sha256": geometry.content_sha256,
            "files": [
                {
                    "file": mesh.path.name,
                    "size_bytes": mesh.path.stat().st_size,
                    "sha256": mesh.sha256,
                }
                for mesh in geometry.meshes
            ],
        },
        "corpus": {
            "generator_source_path": "rivalsim/v03_corpus.py",
            "generator_source_sha256": sha256_file(corpus_path),
            "generator_schema_version": V03_GENERATOR_SCHEMA_VERSION,
            "generator_config": config,
            "generator_config_sha256": sha256_bytes(canonical_json_bytes(config)),
            "seed": V03_GENERATOR_SEED,
            "case_count": len(cases),
            "corpus_sha256": phase_b_corpus_sha256(cases),
        },
        "authority_settings": PHASE_B_AUTHORITY_SETTINGS,
        "authority_settings_sha256": sha256_bytes(
            canonical_json_bytes(PHASE_B_AUTHORITY_SETTINGS)
        ),
        "authority_tooling": {
            "rivalsim/reference/rocketsim_oracle.py": sha256_file(oracle_path),
            "benchmarks/build_v03_phase_b_cache.py": sha256_file(builder_path),
        },
    }
    return {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "B_car_ball",
        "authority_identity_sha256": sha256_bytes(canonical_json_bytes(inputs)),
        "identity_inputs": inputs,
    }


def freeze_phase_b_corpus(
    cache_dir: Path, identity: dict[str, Any], cases: tuple[CarBallCase, ...]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    if identity_path.exists() and _read_json(identity_path) != identity:
        raise RuntimeError(f"Phase B authority identity mismatch: {identity_path}")
    if not identity_path.exists():
        _write_json_atomic(identity_path, identity)
    payload = canonical_json_bytes(
        {
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "corpus_sha256": phase_b_corpus_sha256(cases),
            "case_count": len(cases),
            "cases": [_case_record(case) for case in cases],
        }
    )
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    corpus_path = cache_dir / "corpus.json.gz"
    expected_hash = sha256_bytes(compressed)
    if corpus_path.exists() and sha256_file(corpus_path) != expected_hash:
        raise RuntimeError(f"frozen Phase B corpus differs: {corpus_path}")
    if not corpus_path.exists():
        _write_bytes_atomic(corpus_path, compressed)
    frozen = {
        "path": corpus_path.name,
        "size_bytes": corpus_path.stat().st_size,
        "sha256": expected_hash,
        "case_count": len(cases),
        "corpus_sha256": phase_b_corpus_sha256(cases),
    }
    _write_json_atomic(
        cache_dir / "frozen.json",
        {
            "status": "CORPUS_FROZEN_NATIVE_NOT_YET_COMPLETE",
            "phase": "B_car_ball",
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "frozen_corpus": frozen,
        },
    )
    return frozen


def phase_b_frame_arrays(
    initial: CarBallBatchOracleFrame, frames: list[CarBallBatchOracleFrame]
) -> dict[str, np.ndarray]:
    if len(frames) != len(PHASE_B_CAPTURE_TICKS):
        raise ValueError("Phase B frame sequence must contain all ticks 1..12")
    arrays: dict[str, np.ndarray] = {}
    for field in PHASE_B_FIELDS:
        dtype, _tail = _FIELD_SCHEMA[field]
        arrays[field] = np.ascontiguousarray(
            np.stack([getattr(frame, field) for frame in frames], axis=1), dtype=dtype
        )
        arrays[f"initial_{field}"] = np.ascontiguousarray(
            getattr(initial, field), dtype=dtype
        )
    return arrays


def phase_b_chunk_paths(cache_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"authority-{start:06d}-{stop:06d}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def write_phase_b_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_b_chunk_paths(cache_dir, start, stop)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, npz_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "phase": "B_car_ball",
        "start": start,
        "stop": stop,
        "case_ids": list(case_ids),
        "captured_ticks": list(PHASE_B_CAPTURE_TICKS),
        "frame_fields": list(PHASE_B_FRAME_FIELDS),
        "npz": {
            "path": npz_path.name,
            "size_bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
        },
    }
    _write_json_atomic(meta_path, metadata)
    validate_phase_b_chunk(cache_dir, identity, start, case_ids)
    return metadata


def validate_phase_b_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_b_chunk_paths(cache_dir, start, stop)
    if not npz_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing Phase B authority chunk {start}:{stop}")
    metadata = _read_json(meta_path)
    if metadata.get("authority_identity_sha256") != identity["authority_identity_sha256"]:
        raise RuntimeError(f"authority identity mismatch in {meta_path}")
    if metadata.get("case_ids") != list(case_ids):
        raise RuntimeError(f"case identity mismatch in {meta_path}")
    if metadata["npz"]["sha256"] != sha256_file(npz_path):
        raise RuntimeError(f"authority payload hash mismatch in {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(PHASE_B_FRAME_FIELDS):
            raise RuntimeError(f"Phase B frame fields mismatch in {npz_path}")
        count = len(case_ids)
        for stored_field in PHASE_B_FRAME_FIELDS:
            initial = stored_field.startswith("initial_")
            field = stored_field.removeprefix("initial_")
            dtype, tail = _FIELD_SCHEMA[field]
            expected = (count, *tail) if initial else (count, len(PHASE_B_CAPTURE_TICKS), *tail)
            if arrays[stored_field].shape != expected:
                raise RuntimeError(f"invalid {stored_field} shape in {npz_path}")
            if arrays[stored_field].dtype != dtype:
                raise RuntimeError(f"invalid {stored_field} dtype in {npz_path}")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(arrays[stored_field]).all():
                raise RuntimeError(f"invalid {stored_field} values in {npz_path}")
    return metadata


def finalize_phase_b_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[CarBallCase, ...],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    chunks = []
    for start in range(0, len(cases), PHASE_B_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_B_CACHE_CHUNK_SIZE, len(cases))
        chunks.append(
            validate_phase_b_chunk(
                cache_dir,
                identity,
                start,
                tuple(case.case_id for case in cases[start:stop]),
            )
        )
    manifest = {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "B_car_ball",
        "status": "COMPLETE_NATIVE_AUTHORITY",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "frozen_corpus": frozen,
        "case_count": len(cases),
        "captured_ticks": list(PHASE_B_CAPTURE_TICKS),
        "frame_count": len(cases) * len(PHASE_B_CAPTURE_TICKS),
        "initial_readback_count": len(cases),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    manifest_path = cache_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        cache_dir / "complete.json",
        {
            "status": manifest["status"],
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest


def verify_phase_b_cache(
    cache_root: Path, identity: dict[str, Any], cases: tuple[CarBallCase, ...]
) -> dict[str, Any]:
    cache_dir = phase_cache_dir(cache_root, identity)
    if _read_json(cache_dir / "identity.json") != identity:
        raise RuntimeError("Phase B authority identity mismatch")
    manifest = _read_json(cache_dir / "manifest.json")
    complete = _read_json(cache_dir / "complete.json")
    if manifest.get("status") != "COMPLETE_NATIVE_AUTHORITY":
        raise RuntimeError("Phase B authority cache is incomplete")
    if complete.get("manifest_sha256") != sha256_file(cache_dir / "manifest.json"):
        raise RuntimeError("Phase B authority manifest hash mismatch")
    frozen = cache_dir / manifest["frozen_corpus"]["path"]
    if sha256_file(frozen) != manifest["frozen_corpus"]["sha256"]:
        raise RuntimeError("Phase B frozen corpus hash mismatch")
    for start in range(0, len(cases), PHASE_B_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_B_CACHE_CHUNK_SIZE, len(cases))
        validate_phase_b_chunk(
            cache_dir,
            identity,
            start,
            tuple(case.case_id for case in cases[start:stop]),
        )
    return manifest


def load_phase_b_frames(
    cache_root: Path,
    identity: dict[str, Any],
    cases: tuple[CarBallCase, ...],
    indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Load frozen Phase B truth; missing data is always a hard error."""

    verify_phase_b_cache(cache_root, identity, cases)
    cache_dir = phase_cache_dir(cache_root, identity)
    staged: dict[str, list[np.ndarray | None]] = {
        field: [None] * len(indices) for field in PHASE_B_FRAME_FIELDS
    }
    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, corpus_index in enumerate(indices):
        if not 0 <= corpus_index < len(cases):
            raise IndexError("Phase B authority index outside the frozen corpus")
        start = (corpus_index // PHASE_B_CACHE_CHUNK_SIZE) * PHASE_B_CACHE_CHUNK_SIZE
        by_chunk.setdefault(start, []).append((output_index, corpus_index - start))
    for start, requests in by_chunk.items():
        stop = min(start + PHASE_B_CACHE_CHUNK_SIZE, len(cases))
        npz_path, _ = phase_b_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as arrays:
            for output_index, local_index in requests:
                for field in PHASE_B_FRAME_FIELDS:
                    staged[field][output_index] = np.asarray(arrays[field][local_index])
    return {
        field: np.ascontiguousarray(np.stack(values, axis=0))
        for field, values in staged.items()
    }


def _case_record(case: CarBallCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "contact_region": case.contact_region,
        "feature_index": case.feature_index,
        "motion_mode": case.motion_mode,
        "orientation_mode": case.orientation_mode,
        "static_context": case.static_context,
        "overlap_uu": case.overlap_uu,
        "car_on_ground": case.car_on_ground,
        "car_position": case.car_position.astype(float).tolist(),
        "car_velocity": case.car_velocity.astype(float).tolist(),
        "car_quaternion": case.car_quaternion.astype(float).tolist(),
        "car_angular_velocity": case.car_angular_velocity.astype(float).tolist(),
        "ball_position": case.ball_position.astype(float).tolist(),
        "ball_velocity": case.ball_velocity.astype(float).tolist(),
        "ball_quaternion": case.ball_quaternion.astype(float).tolist(),
        "ball_angular_velocity": case.ball_angular_velocity.astype(float).tolist(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_bytes_atomic(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
