"""Content-addressed native oracle cache for RivalSim v0.3 phases."""

from __future__ import annotations

import gzip
import hashlib
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
    BallWorldBatchOracleFrame,
)
from rivalsim.v03_corpus import (
    V03_GENERATOR_SCHEMA_VERSION,
    V03_GENERATOR_SEED,
    V03_HARD_HORIZONS,
    BallWorldCase,
    phase_a_corpus_sha256,
    phase_a_generator_config,
)

V03_CACHE_FORMAT_VERSION = 2
V03_CACHE_CAPTURE_TICKS = tuple(range(1, max(V03_HARD_HORIZONS) + 1))
V03_CACHE_CHUNK_SIZE = 256
EXPECTED_ROCKETSIM_EXTENSION_SHA256 = (
    "E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0"
)
EXPECTED_SOCCAR_CMF_SHA256 = (
    "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
)

PHASE_A_TICK_FIELDS = ("ball_pos", "ball_vel", "ball_matrix", "ball_ang_vel")
PHASE_A_INITIAL_FIELDS = tuple(f"initial_{field}" for field in PHASE_A_TICK_FIELDS)
PHASE_A_FRAME_FIELDS = (*PHASE_A_INITIAL_FIELDS, *PHASE_A_TICK_FIELDS)
PHASE_A_AUTHORITY_SETTINGS: dict[str, Any] = {
    "protocol_version": 1,
    "phase": "A_ball_world",
    "game_mode": "SOCCAR",
    "tick_rate_hz": 120.0,
    "arena_config": {"no_ball_rot": False},
    "collision_switches": {
        "car_car_collision": False,
        "car_ball_collision": False,
    },
    "isolation": "one fresh ball-only Soccar arena per corpus case",
    "cars": "none",
    "ball": {
        "shape": "RocketSim Soccar btSphereShape",
        "state_source": "generated BallState set once before tick 1",
    },
    "initial_state_custody": {
        "source_state": "exact frozen corpus record",
        "native_readback": "complete BallState immediately after SetState and before tick 1",
    },
    "step_order": (
        "SetState, immediate native readback, then arena.step(1) and complete "
        "BallState readback for every tick"
    ),
    "captured_ticks": list(V03_CACHE_CAPTURE_TICKS),
    "initial_readback_fields": list(PHASE_A_INITIAL_FIELDS),
    "captured_frame_fields": list(PHASE_A_TICK_FIELDS),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_phase_a_identity(
    geometry: ArenaGeometry, cases: tuple[BallWorldCase, ...]
) -> dict[str, Any]:
    config = phase_a_generator_config()
    source_path = Path(__file__).with_name("v03_corpus.py")
    oracle_path = Path(__file__).with_name("reference") / "rocketsim_oracle.py"
    builder_path = Path(__file__).parents[1] / "benchmarks" / "build_v03_phase_a_cache.py"
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
            "generator_source_sha256": sha256_file(source_path),
            "generator_schema_version": V03_GENERATOR_SCHEMA_VERSION,
            "generator_config": config,
            "generator_config_sha256": sha256_bytes(canonical_json_bytes(config)),
            "seed": V03_GENERATOR_SEED,
            "case_count": len(cases),
            "corpus_sha256": phase_a_corpus_sha256(cases),
        },
        "authority_settings": PHASE_A_AUTHORITY_SETTINGS,
        "authority_settings_sha256": sha256_bytes(
            canonical_json_bytes(PHASE_A_AUTHORITY_SETTINGS)
        ),
        "authority_tooling": {
            "rivalsim/reference/rocketsim_oracle.py": sha256_file(oracle_path),
            "benchmarks/build_v03_phase_a_cache.py": sha256_file(builder_path),
        },
    }
    return {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "A_ball_world",
        "authority_identity_sha256": sha256_bytes(canonical_json_bytes(inputs)),
        "identity_inputs": inputs,
    }


def phase_cache_dir(cache_root: Path, identity: dict[str, Any]) -> Path:
    return cache_root.resolve() / str(identity["authority_identity_sha256"])


def validate_installed_rocketsim_extension() -> dict[str, str]:
    import RocketSim

    path = Path(RocketSim.__file__).resolve()
    resolved = sha256_file(path)
    if resolved != EXPECTED_ROCKETSIM_EXTENSION_SHA256:
        raise RuntimeError(
            f"installed RocketSim extension hash mismatch: {resolved}, "
            f"expected {EXPECTED_ROCKETSIM_EXTENSION_SHA256}"
        )
    return {"path": str(path), "sha256": resolved}


def freeze_phase_a_corpus(
    cache_dir: Path, identity: dict[str, Any], cases: tuple[BallWorldCase, ...]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    if identity_path.exists():
        if _read_json(identity_path) != identity:
            raise RuntimeError(f"authority identity mismatch: {identity_path}")
    else:
        _write_json_atomic(identity_path, identity)

    corpus_path = cache_dir / "corpus.json.gz"
    payload = canonical_json_bytes(
        {
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "corpus_sha256": phase_a_corpus_sha256(cases),
            "case_count": len(cases),
            "cases": [_case_record(case) for case in cases],
        }
    )
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    expected_hash = sha256_bytes(compressed)
    if corpus_path.exists():
        if sha256_file(corpus_path) != expected_hash:
            raise RuntimeError(f"frozen Phase A corpus differs: {corpus_path}")
    else:
        _write_bytes_atomic(corpus_path, compressed)
    frozen = {
        "path": corpus_path.name,
        "size_bytes": corpus_path.stat().st_size,
        "sha256": expected_hash,
        "case_count": len(cases),
        "corpus_sha256": phase_a_corpus_sha256(cases),
    }
    _write_json_atomic(
        cache_dir / "frozen.json",
        {
            "status": "CORPUS_FROZEN_NATIVE_NOT_YET_COMPLETE",
            "phase": "A_ball_world",
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "frozen_corpus": frozen,
        },
    )
    return frozen


def phase_a_chunk_paths(cache_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"authority-{start:06d}-{stop:06d}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def phase_a_frame_arrays(
    initial_frame: BallWorldBatchOracleFrame,
    frames: list[BallWorldBatchOracleFrame],
) -> dict[str, np.ndarray]:
    if len(frames) != len(V03_CACHE_CAPTURE_TICKS):
        raise ValueError("native frame sequence must contain every cached tick")
    arrays = {
        field: np.ascontiguousarray(
            np.stack([getattr(frame, field) for frame in frames], axis=1)
        )
        for field in PHASE_A_TICK_FIELDS
    }
    arrays.update(
        {
            f"initial_{field}": np.ascontiguousarray(
                getattr(initial_frame, field), dtype=np.float32
            )
            for field in PHASE_A_TICK_FIELDS
        }
    )
    return arrays


def write_phase_a_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_a_chunk_paths(cache_dir, start, stop)
    with tempfile.NamedTemporaryFile(
        prefix=npz_path.name,
        suffix=".tmp",
        dir=cache_dir,
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
    try:
        with temp_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp_path, npz_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    metadata = {
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "phase": "A_ball_world",
        "start": start,
        "stop": stop,
        "case_ids": list(case_ids),
        "captured_ticks": list(V03_CACHE_CAPTURE_TICKS),
        "initial_readback_fields": list(PHASE_A_INITIAL_FIELDS),
        "frame_fields": list(PHASE_A_FRAME_FIELDS),
        "npz": {
            "path": npz_path.name,
            "size_bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
        },
    }
    _write_json_atomic(meta_path, metadata)
    validate_phase_a_chunk(cache_dir, identity, start, case_ids)
    return metadata


def validate_phase_a_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_a_chunk_paths(cache_dir, start, stop)
    if not npz_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing Phase A authority chunk {start}:{stop}")
    metadata = _read_json(meta_path)
    if metadata.get("authority_identity_sha256") != identity["authority_identity_sha256"]:
        raise RuntimeError(f"authority identity mismatch in {meta_path}")
    if metadata.get("case_ids") != list(case_ids):
        raise RuntimeError(f"case identity mismatch in {meta_path}")
    if sha256_file(npz_path) != metadata["npz"]["sha256"]:
        raise RuntimeError(f"authority payload hash mismatch in {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(PHASE_A_FRAME_FIELDS):
            raise RuntimeError(f"authority frame fields mismatch in {npz_path}")
        count = len(case_ids)
        for field in PHASE_A_FRAME_FIELDS:
            expected_tail = (3, 3) if field.endswith("matrix") else (3,)
            expected_shape = (
                (count, *expected_tail)
                if field.startswith("initial_")
                else (count, len(V03_CACHE_CAPTURE_TICKS), *expected_tail)
            )
            if arrays[field].shape != expected_shape:
                raise RuntimeError(f"invalid {field} shape in {npz_path}")
            if arrays[field].dtype != np.float32 or not np.isfinite(arrays[field]).all():
                raise RuntimeError(f"invalid {field} values in {npz_path}")
    return metadata


def finalize_phase_a_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[BallWorldCase, ...],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(cases), V03_CACHE_CHUNK_SIZE):
        stop = min(start + V03_CACHE_CHUNK_SIZE, len(cases))
        case_ids = tuple(case.case_id for case in cases[start:stop])
        chunks.append(validate_phase_a_chunk(cache_dir, identity, start, case_ids))
    manifest = {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "A_ball_world",
        "status": "COMPLETE_NATIVE_AUTHORITY",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "frozen_corpus": frozen,
        "case_count": len(cases),
        "captured_ticks": list(V03_CACHE_CAPTURE_TICKS),
        "frame_count": len(cases) * len(V03_CACHE_CAPTURE_TICKS),
        "initial_readback_count": len(cases),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    _write_json_atomic(cache_dir / "manifest.json", manifest)
    complete = {
        "status": manifest["status"],
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
    }
    _write_json_atomic(cache_dir / "complete.json", complete)
    return manifest


def verify_phase_a_cache(
    cache_root: Path, identity: dict[str, Any], cases: tuple[BallWorldCase, ...]
) -> dict[str, Any]:
    cache_dir = phase_cache_dir(cache_root, identity)
    if _read_json(cache_dir / "identity.json") != identity:
        raise RuntimeError("Phase A authority identity mismatch")
    manifest = _read_json(cache_dir / "manifest.json")
    complete = _read_json(cache_dir / "complete.json")
    if manifest.get("status") != "COMPLETE_NATIVE_AUTHORITY":
        raise RuntimeError("Phase A authority cache is incomplete")
    if complete.get("manifest_sha256") != sha256_file(cache_dir / "manifest.json"):
        raise RuntimeError("Phase A authority manifest hash mismatch")
    frozen_path = cache_dir / manifest["frozen_corpus"]["path"]
    if sha256_file(frozen_path) != manifest["frozen_corpus"]["sha256"]:
        raise RuntimeError("Phase A frozen corpus hash mismatch")
    for start in range(0, len(cases), V03_CACHE_CHUNK_SIZE):
        stop = min(start + V03_CACHE_CHUNK_SIZE, len(cases))
        validate_phase_a_chunk(
            cache_dir,
            identity,
            start,
            tuple(case.case_id for case in cases[start:stop]),
        )
    return manifest


def load_phase_a_frames(
    cache_root: Path,
    identity: dict[str, Any],
    cases: tuple[BallWorldCase, ...],
    indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Load selected frames with no native fallback."""

    verify_phase_a_cache(cache_root, identity, cases)
    cache_dir = phase_cache_dir(cache_root, identity)
    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, corpus_index in enumerate(indices):
        if not 0 <= corpus_index < len(cases):
            raise IndexError("Phase A cached authority index outside frozen corpus")
        start = (corpus_index // V03_CACHE_CHUNK_SIZE) * V03_CACHE_CHUNK_SIZE
        by_chunk.setdefault(start, []).append((output_index, corpus_index - start))
    staged: dict[str, list[np.ndarray | None]] = {
        field: [None] * len(indices) for field in PHASE_A_FRAME_FIELDS
    }
    for start, requests in by_chunk.items():
        stop = min(start + V03_CACHE_CHUNK_SIZE, len(cases))
        npz_path, _meta_path = phase_a_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as arrays:
            for output_index, local_index in requests:
                for field in PHASE_A_FRAME_FIELDS:
                    staged[field][output_index] = np.asarray(
                        arrays[field][local_index], dtype=np.float32
                    )
    return {
        field: np.ascontiguousarray(np.stack(values, axis=0))
        for field, values in staged.items()
    }


def _case_record(case: BallWorldCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "family": case.family,
        "mode": case.mode,
        "mesh_index": case.mesh_index,
        "mesh_file": case.mesh_file,
        "target_face": case.target_face,
        "target_neighbor_face": case.target_neighbor_face,
        "target_edge": case.target_edge,
        "edge_class": case.edge_class,
        "analytic_plane": case.analytic_plane,
        "region_labels": list(case.region_labels),
        "target_point": case.target_point.astype(float).tolist(),
        "target_normal": case.target_normal.astype(float).tolist(),
        "position": case.position.astype(float).tolist(),
        "velocity": case.velocity.astype(float).tolist(),
        "quaternion": case.quaternion.astype(float).tolist(),
        "angular_velocity": case.angular_velocity.astype(float).tolist(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_bytes_atomic(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temp_path = Path(stream.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
