"""Content-addressed relational native authority for v0.3 Phase D."""

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
from rivalsim.reference.v03_phase_c_oracle import MAX_CAR_BUMP_EVENTS_PER_TICK
from rivalsim.reference.v03_phase_d_oracle import (
    PHASE_D_NATIVE_BRANCHES,
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    IntegratedBatchOracleFrame,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_ROCKETSIM_EXTENSION_SHA256,
    V03_CACHE_FORMAT_VERSION,
    canonical_json_bytes,
    phase_cache_dir,
    sha256_bytes,
    sha256_file,
)
from rivalsim.v03_phase_d_corpus import (
    PHASE_D_GENERATOR_SCHEMA_VERSION,
    PHASE_D_GENERATOR_SEED,
    PHASE_D_HARD_HORIZONS,
    IntegratedCase,
    phase_d_case_record,
    phase_d_corpus_sha256,
    phase_d_generator_config,
)

PHASE_D_CACHE_CHUNK_SIZE = 32
PHASE_D_CAPTURE_TICKS = tuple(range(1, max(PHASE_D_HARD_HORIZONS) + 1))
EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256 = (
    "A92F6680284A7149843AFC1041C70DB574ED4CAEA5176F3EF0F1E0E216763807"
)
PHASE_D_FIELDS = (
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
    "car_is_supersonic",
    "car_supersonic_time",
    "car_contact_id",
    "car_contact_cooldown",
    "car_is_demoed",
    "car_demo_respawn_timer",
    "car_ball_hit_valid",
    "car_ball_hit_tick",
    "car_ball_extra_hit_vel",
    "car_ball_relative_pos",
    "ball_pos",
    "ball_vel",
    "ball_matrix",
    "ball_ang_vel",
    "ball_last_hit_car_id",
    "bump_event_count",
    "bump_event_bumper",
    "bump_event_victim",
    "bump_event_is_demo",
)
PHASE_D_FRAME_FIELDS = (
    *(f"initial_{field}" for field in PHASE_D_FIELDS),
    *PHASE_D_FIELDS,
)

PHASE_D_AUTHORITY_SETTINGS: dict[str, Any] = {
    "protocol_version": 1,
    "relation_schema": "complete_native_branch_trajectory_v1",
    "phase": "D_integrated",
    "game_mode": "SOCCAR",
    "tick_rate_hz": 120.0,
    "arena_config": {"no_ball_rot": False},
    "collision_switches": {"car_car_collision": True, "car_ball_collision": True},
    "world": "exactly two Octanes, one standard ball, immutable static Soccar",
    "native_multi_outcome_relation": {
        "branches": list(PHASE_D_NATIVE_BRANCHES),
        "branch_cause": "source Arena::_cars iteration order established by membership lifecycle",
        "collection": (
            "prospective fresh-arena construction until the read-only logical-ID "
            "diagnostic observes the requested source-valid order; physical state "
            "and controls are applied only after retaining that arena"
        ),
        "coherence": "every field and tick comes from one complete labeled branch",
        "acceptance": "one complete RivalSim trajectory must match one complete native branch",
        "metric_mixing": False,
        "best_match_runtime_selection": False,
    },
    "order_diagnostic": {
        "returns": "logical car IDs only",
        "native_pointers_exposed": False,
        "allocator_addresses_exposed": False,
        "behavior_mutation": False,
        "patch": "tools/rocketsim_phase_c_order_probe.patch",
    },
    "cars": {
        "count": 2,
        "configs": ["CarConfig.OCTANE", "CarConfig.OCTANE"],
        "teams": ["BLUE", "ORANGE"],
        "controls": "frozen per-case/per-car/per-tick controls",
    },
    "ball": {"shape": "standard Soccar sphere", "rotation_enabled": True},
    "bump_demo": {
        "demo_mode": "NORMAL",
        "enable_team_demos": False,
        "callback": "ordered logical bumper/victim/is_demo stream",
        "max_events_per_tick": MAX_CAR_BUMP_EVENTS_PER_TICK,
        "removal_respawn": "cached authority readback but outside v0.3 runtime scope",
    },
    "captured_ticks": list(PHASE_D_CAPTURE_TICKS),
    "captured_frame_fields": list(PHASE_D_FIELDS),
}

_FIELD_SCHEMA: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "car_pos": (np.dtype(np.float32), (2, 3)),
    "car_vel": (np.dtype(np.float32), (2, 3)),
    "car_matrix": (np.dtype(np.float32), (2, 3, 3)),
    "car_ang_vel": (np.dtype(np.float32), (2, 3)),
    "car_boost": (np.dtype(np.float32), (2,)),
    "car_handbrake": (np.dtype(np.float32), (2,)),
    "car_on_ground": (np.dtype(np.bool_), (2,)),
    "car_wheel_contacts": (np.dtype(np.bool_), (2, 4)),
    "car_world_contact": (np.dtype(np.bool_), (2,)),
    "car_world_contact_normal": (np.dtype(np.float32), (2, 3)),
    "car_is_supersonic": (np.dtype(np.bool_), (2,)),
    "car_supersonic_time": (np.dtype(np.float32), (2,)),
    "car_contact_id": (np.dtype(np.uint32), (2,)),
    "car_contact_cooldown": (np.dtype(np.float32), (2,)),
    "car_is_demoed": (np.dtype(np.bool_), (2,)),
    "car_demo_respawn_timer": (np.dtype(np.float32), (2,)),
    "car_ball_hit_valid": (np.dtype(np.bool_), (2,)),
    "car_ball_hit_tick": (np.dtype(np.uint64), (2,)),
    "car_ball_extra_hit_vel": (np.dtype(np.float32), (2, 3)),
    "car_ball_relative_pos": (np.dtype(np.float32), (2, 3)),
    "ball_pos": (np.dtype(np.float32), (3,)),
    "ball_vel": (np.dtype(np.float32), (3,)),
    "ball_matrix": (np.dtype(np.float32), (3, 3)),
    "ball_ang_vel": (np.dtype(np.float32), (3,)),
    "ball_last_hit_car_id": (np.dtype(np.uint32), ()),
    "bump_event_count": (np.dtype(np.int32), ()),
    "bump_event_bumper": (np.dtype(np.int32), (MAX_CAR_BUMP_EVENTS_PER_TICK,)),
    "bump_event_victim": (np.dtype(np.int32), (MAX_CAR_BUMP_EVENTS_PER_TICK,)),
    "bump_event_is_demo": (np.dtype(np.bool_), (MAX_CAR_BUMP_EVENTS_PER_TICK,)),
}


def build_phase_d_identity(
    geometry: ArenaGeometry, cases: tuple[IntegratedCase, ...]
) -> dict[str, Any]:
    package_root = Path(__file__).parents[1]
    paths = {
        "corpus": Path(__file__).with_name("v03_phase_d_corpus.py"),
        "oracle": Path(__file__).with_name("reference") / "v03_phase_d_oracle.py",
        "builder": package_root / "benchmarks" / "build_v03_phase_d_cache.py",
        "patch": package_root / "tools" / "rocketsim_phase_c_order_probe.patch",
    }
    config = phase_d_generator_config()
    inputs = {
        "rocketsim": {
            "primary_repository": "https://github.com/ZealanL/RocketSim.git",
            "primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "binding_repository": "https://github.com/mtheall/RocketSim.git",
            "binding_commit": ROCKETSIM_BINDING_COMMIT,
            "binding_package_version": ROCKETSIM_BINDING_VERSION,
            "installed_extension_sha256": EXPECTED_ROCKETSIM_EXTENSION_SHA256,
            "logical_order_diagnostic": {
                "patch_sha256": sha256_file(paths["patch"]),
                "expected_extension_sha256": EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256,
                "returns": "logical car IDs only",
            },
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
            "generator_source_path": "rivalsim/v03_phase_d_corpus.py",
            "generator_source_sha256": sha256_file(paths["corpus"]),
            "generator_schema_version": PHASE_D_GENERATOR_SCHEMA_VERSION,
            "generator_config": config,
            "generator_config_sha256": sha256_bytes(canonical_json_bytes(config)),
            "seed": PHASE_D_GENERATOR_SEED,
            "case_count": len(cases),
            "corpus_sha256": phase_d_corpus_sha256(cases),
        },
        "authority_settings": PHASE_D_AUTHORITY_SETTINGS,
        "authority_settings_sha256": sha256_bytes(
            canonical_json_bytes(PHASE_D_AUTHORITY_SETTINGS)
        ),
        "authority_tooling": {
            "rivalsim/reference/v03_phase_d_oracle.py": sha256_file(paths["oracle"]),
            "benchmarks/build_v03_phase_d_cache.py": sha256_file(paths["builder"]),
            "tools/rocketsim_phase_c_order_probe.patch": sha256_file(paths["patch"]),
        },
    }
    return {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "D_integrated",
        "authority_identity_sha256": sha256_bytes(canonical_json_bytes(inputs)),
        "identity_inputs": inputs,
    }


def phase_d_frame_arrays(
    initial: list[IntegratedBatchOracleFrame],
    frames: list[list[IntegratedBatchOracleFrame]],
) -> dict[str, np.ndarray]:
    if len(initial) != len(PHASE_D_NATIVE_BRANCHES) or len(frames) != len(
        PHASE_D_NATIVE_BRANCHES
    ):
        raise ValueError("Phase D authority must contain both native branches")
    if any(len(branch) != len(PHASE_D_CAPTURE_TICKS) for branch in frames):
        raise ValueError("each Phase D branch must contain ticks 1..12")
    arrays: dict[str, np.ndarray] = {}
    for field in PHASE_D_FIELDS:
        dtype, _tail = _FIELD_SCHEMA[field]
        arrays[field] = np.ascontiguousarray(
            np.stack(
                [
                    np.stack([getattr(frame, field) for frame in branch], axis=1)
                    for branch in frames
                ],
                axis=1,
            ),
            dtype=dtype,
        )
        arrays[f"initial_{field}"] = np.ascontiguousarray(
            np.stack([getattr(frame, field) for frame in initial], axis=1), dtype=dtype
        )
    return arrays


def freeze_phase_d_corpus(
    cache_dir: Path, identity: dict[str, Any], cases: tuple[IntegratedCase, ...]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    if identity_path.exists() and _read_json(identity_path) != identity:
        raise RuntimeError(f"Phase D authority identity mismatch: {identity_path}")
    if not identity_path.exists():
        _write_json_atomic(identity_path, identity)
    payload = canonical_json_bytes(
        {
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "corpus_sha256": phase_d_corpus_sha256(cases),
            "case_count": len(cases),
            "cases": [phase_d_case_record(case) for case in cases],
        }
    )
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    corpus_path = cache_dir / "corpus.json.gz"
    expected_hash = sha256_bytes(compressed)
    if corpus_path.exists() and sha256_file(corpus_path) != expected_hash:
        raise RuntimeError(f"frozen Phase D corpus differs: {corpus_path}")
    if not corpus_path.exists():
        _write_bytes_atomic(corpus_path, compressed)
    frozen = {
        "path": corpus_path.name,
        "size_bytes": corpus_path.stat().st_size,
        "sha256": expected_hash,
        "case_count": len(cases),
        "corpus_sha256": phase_d_corpus_sha256(cases),
    }
    _write_json_atomic(
        cache_dir / "frozen.json",
        {
            "status": "CORPUS_FROZEN_NATIVE_NOT_YET_COMPLETE",
            "phase": "D_integrated",
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "frozen_corpus": frozen,
        },
    )
    return frozen


def phase_d_chunk_paths(cache_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"authority-{start:06d}-{stop:06d}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def write_phase_d_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
    arrays: dict[str, np.ndarray],
    branch_construction: dict[str, Any],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_d_chunk_paths(cache_dir, start, stop)
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
        "phase": "D_integrated",
        "start": start,
        "stop": stop,
        "case_ids": list(case_ids),
        "captured_ticks": list(PHASE_D_CAPTURE_TICKS),
        "native_branches": list(PHASE_D_NATIVE_BRANCHES),
        "branch_construction": branch_construction,
        "frame_fields": list(PHASE_D_FRAME_FIELDS),
        "npz": {
            "path": npz_path.name,
            "size_bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
        },
    }
    _write_json_atomic(meta_path, metadata)
    validate_phase_d_chunk(cache_dir, identity, start, case_ids)
    return metadata


def validate_phase_d_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_d_chunk_paths(cache_dir, start, stop)
    if not npz_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing Phase D authority chunk {start}:{stop}")
    metadata = _read_json(meta_path)
    if metadata.get("authority_identity_sha256") != identity["authority_identity_sha256"]:
        raise RuntimeError(f"authority identity mismatch in {meta_path}")
    if metadata.get("case_ids") != list(case_ids):
        raise RuntimeError(f"case identity mismatch in {meta_path}")
    if metadata.get("native_branches") != list(PHASE_D_NATIVE_BRANCHES):
        raise RuntimeError(f"native branch schema mismatch in {meta_path}")
    for branch_index, branch in enumerate(PHASE_D_NATIVE_BRANCHES):
        record = metadata.get("branch_construction", {}).get(branch, {})
        if record.get("logical_first_car") != branch_index:
            raise RuntimeError(f"native branch label mismatch in {meta_path}: {branch}")
        if record.get("all_observed_as_requested") is not True:
            raise RuntimeError(f"native branch observation incomplete in {meta_path}: {branch}")
    if metadata["npz"]["sha256"] != sha256_file(npz_path):
        raise RuntimeError(f"authority payload hash mismatch in {npz_path}")
    count = len(case_ids)
    branches = len(PHASE_D_NATIVE_BRANCHES)
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(PHASE_D_FRAME_FIELDS):
            raise RuntimeError(f"Phase D frame fields mismatch in {npz_path}")
        for stored in PHASE_D_FRAME_FIELDS:
            initial = stored.startswith("initial_")
            field = stored.removeprefix("initial_")
            dtype, tail = _FIELD_SCHEMA[field]
            expected = (
                (count, branches, *tail)
                if initial
                else (count, branches, len(PHASE_D_CAPTURE_TICKS), *tail)
            )
            if arrays[stored].shape != expected or arrays[stored].dtype != dtype:
                raise RuntimeError(f"invalid {stored} schema in {npz_path}")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(arrays[stored]).all():
                raise RuntimeError(f"non-finite {stored} values in {npz_path}")
    return metadata


def finalize_phase_d_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[IntegratedCase, ...],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    chunks = []
    for start in range(0, len(cases), PHASE_D_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_D_CACHE_CHUNK_SIZE, len(cases))
        chunks.append(
            validate_phase_d_chunk(
                cache_dir, identity, start, tuple(case.case_id for case in cases[start:stop])
            )
        )
    manifest = {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "D_integrated",
        "status": "COMPLETE_NATIVE_AUTHORITY",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "frozen_corpus": frozen,
        "case_count": len(cases),
        "captured_ticks": list(PHASE_D_CAPTURE_TICKS),
        "native_branches": list(PHASE_D_NATIVE_BRANCHES),
        "frame_count": len(cases) * len(PHASE_D_NATIVE_BRANCHES) * len(PHASE_D_CAPTURE_TICKS),
        "initial_readback_count": len(cases) * len(PHASE_D_NATIVE_BRANCHES),
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


def verify_phase_d_cache(
    cache_root: Path, identity: dict[str, Any], cases: tuple[IntegratedCase, ...]
) -> dict[str, Any]:
    cache_dir = phase_cache_dir(cache_root, identity)
    if _read_json(cache_dir / "identity.json") != identity:
        raise RuntimeError("Phase D authority identity mismatch")
    manifest_path = cache_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    complete = _read_json(cache_dir / "complete.json")
    if manifest.get("status") != "COMPLETE_NATIVE_AUTHORITY":
        raise RuntimeError("Phase D authority cache is incomplete")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Phase D authority manifest hash mismatch")
    frozen = cache_dir / manifest["frozen_corpus"]["path"]
    if sha256_file(frozen) != manifest["frozen_corpus"]["sha256"]:
        raise RuntimeError("Phase D frozen corpus hash mismatch")
    for start in range(0, len(cases), PHASE_D_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_D_CACHE_CHUNK_SIZE, len(cases))
        validate_phase_d_chunk(
            cache_dir, identity, start, tuple(case.case_id for case in cases[start:stop])
        )
    return manifest


def load_frozen_phase_d_identity(
    cache_root: Path,
    authority_identity_sha256: str,
    geometry: ArenaGeometry,
    cases: tuple[IntegratedCase, ...],
) -> dict[str, Any]:
    if not authority_identity_sha256 or any(
        character not in "0123456789ABCDEF" for character in authority_identity_sha256
    ):
        raise ValueError("Phase D authority identity must be uppercase hexadecimal")
    identity = _read_json(cache_root.resolve() / authority_identity_sha256 / "identity.json")
    if identity.get("authority_identity_sha256") != authority_identity_sha256:
        raise RuntimeError("Phase D authority directory/identity mismatch")
    current = build_phase_d_identity(geometry, cases)
    stored = identity.get("identity_inputs", {})
    for key in (
        "rocketsim",
        "collision_assets",
        "corpus",
        "authority_settings",
        "authority_settings_sha256",
    ):
        if stored.get(key) != current["identity_inputs"][key]:
            raise RuntimeError(f"Phase D frozen authority input mismatch: {key}")
    return identity


def load_phase_d_frames(
    cache_root: Path,
    identity: dict[str, Any],
    cases: tuple[IntegratedCase, ...],
    indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    verify_phase_d_cache(cache_root, identity, cases)
    cache_dir = phase_cache_dir(cache_root, identity)
    staged: dict[str, list[np.ndarray | None]] = {
        field: [None] * len(indices) for field in PHASE_D_FRAME_FIELDS
    }
    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, corpus_index in enumerate(indices):
        if not 0 <= corpus_index < len(cases):
            raise IndexError("Phase D authority index outside the frozen corpus")
        start = (corpus_index // PHASE_D_CACHE_CHUNK_SIZE) * PHASE_D_CACHE_CHUNK_SIZE
        by_chunk.setdefault(start, []).append((output_index, corpus_index - start))
    for start, requests in by_chunk.items():
        stop = min(start + PHASE_D_CACHE_CHUNK_SIZE, len(cases))
        npz_path, _meta = phase_d_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as arrays:
            for output_index, local_index in requests:
                for field in PHASE_D_FRAME_FIELDS:
                    staged[field][output_index] = np.asarray(arrays[field][local_index])
    return {
        field: np.ascontiguousarray(np.stack(values, axis=0))
        for field, values in staged.items()
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


__all__ = [
    "EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256",
    "PHASE_D_AUTHORITY_SETTINGS",
    "PHASE_D_CACHE_CHUNK_SIZE",
    "PHASE_D_CAPTURE_TICKS",
    "PHASE_D_FIELDS",
    "PHASE_D_FRAME_FIELDS",
    "PHASE_D_NATIVE_BRANCHES",
    "build_phase_d_identity",
    "finalize_phase_d_cache",
    "freeze_phase_d_corpus",
    "load_frozen_phase_d_identity",
    "load_phase_d_frames",
    "phase_d_chunk_paths",
    "phase_d_frame_arrays",
    "validate_phase_d_chunk",
    "verify_phase_d_cache",
    "write_phase_d_chunk",
]
