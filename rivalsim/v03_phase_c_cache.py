"""Content-addressed isolated native authority for v0.3 Phase C."""

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
from rivalsim.reference.v03_phase_c_oracle import (
    MAX_CAR_BUMP_EVENTS_PER_TICK,
    PHASE_C_NATIVE_BRANCHES,
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    CarCarBatchOracleFrame,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_ROCKETSIM_EXTENSION_SHA256,
    V03_CACHE_FORMAT_VERSION,
    canonical_json_bytes,
    phase_cache_dir,
    sha256_bytes,
    sha256_file,
)
from rivalsim.v03_phase_c_corpus import (
    PHASE_C_GENERATOR_SCHEMA_VERSION,
    PHASE_C_GENERATOR_SEED,
    PHASE_C_HARD_HORIZONS,
    CarCarCase,
    case_record,
    phase_c_corpus_sha256,
    phase_c_generator_config,
)

PHASE_C_CACHE_CHUNK_SIZE = 128
PHASE_C_CAPTURE_TICKS = tuple(range(1, max(PHASE_C_HARD_HORIZONS) + 1))
EXPECTED_PHASE_C_ORDER_DIAGNOSTIC_SHA256 = (
    "A92F6680284A7149843AFC1041C70DB574ED4CAEA5176F3EF0F1E0E216763807"
)
PHASE_C_FIELDS = (
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
    "bump_event_count",
    "bump_event_bumper",
    "bump_event_victim",
    "bump_event_is_demo",
)
PHASE_C_FRAME_FIELDS = (
    *(f"initial_{field}" for field in PHASE_C_FIELDS),
    *PHASE_C_FIELDS,
)
PHASE_C_AUTHORITY_SETTINGS: dict[str, Any] = {
    "protocol_version": 2,
    "relation_schema": "complete_native_branch_trajectory_v1",
    "phase": "C_car_car",
    "game_mode": "SOCCAR",
    "tick_rate_hz": 120.0,
    "arena_config": {"no_ball_rot": False},
    "collision_switches": {
        "car_car_collision": True,
        "car_ball_collision": False,
    },
    "isolation": (
        "each corpus case/logical branch is captured in its own fresh two-Octane "
        "Soccar arena and released before the next case"
    ),
    "native_multi_outcome_relation": {
        "branches": list(PHASE_C_NATIVE_BRANCHES),
        "branch_cause": (
            "source Arena::_cars iteration order established by construction/membership"
        ),
        "collection": (
            "prospective fresh-arena construction until the read-only logical-ID "
            "diagnostic observes the requested source-valid order; physical state is "
            "set only after the branch arena is retained"
        ),
        "coherence": (
            "all fields and ticks for a comparison come from one complete labeled branch"
        ),
        "acceptance": (
            "a case passes when one complete RivalSim trajectory matches one complete "
            "native-valid branch; metric-by-metric branch mixing is forbidden"
        ),
        "runtime_selection": (
            "none; authority comparison never selects a branch inside RivalSim"
        ),
    },
    "order_diagnostic": {
        "output": "logical car IDs in exact Arena::_cars order",
        "native_pointers_exposed": False,
        "allocator_addresses_exposed": False,
        "behavior_mutation": False,
        "patch": "tools/rocketsim_phase_c_order_probe.patch",
    },
    "cars": {
        "count": 2,
        "configs": ["CarConfig.OCTANE", "CarConfig.OCTANE"],
        "teams": ["BLUE", "ORANGE"],
        "controls": "zero for all captured ticks",
    },
    "bump_demo": {
        "demo_mode": "NORMAL",
        "enable_team_demos": False,
        "callback": "complete ordered bumper/victim/is_demo event stream",
        "max_events_per_tick": MAX_CAR_BUMP_EVENTS_PER_TICK,
        "removal_respawn_rules": (
            "native readback cached but outside RivalSim v0.3 implementation scope"
        ),
    },
    "ball": "parked outside interaction; car-ball collision disabled",
    "initial_state_custody": {
        "source_state": "exact frozen corpus record",
        "native_readback": (
            "complete two-car and bump/demo semantic state immediately after SetState"
        ),
    },
    "step_order": (
        "immediate readback, then arena.step(1) and complete frame readback for ticks 1..12"
    ),
    "captured_ticks": list(PHASE_C_CAPTURE_TICKS),
    "captured_frame_fields": list(PHASE_C_FIELDS),
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
    "bump_event_count": (np.dtype(np.int32), ()),
    "bump_event_bumper": (
        np.dtype(np.int32),
        (MAX_CAR_BUMP_EVENTS_PER_TICK,),
    ),
    "bump_event_victim": (
        np.dtype(np.int32),
        (MAX_CAR_BUMP_EVENTS_PER_TICK,),
    ),
    "bump_event_is_demo": (
        np.dtype(np.bool_),
        (MAX_CAR_BUMP_EVENTS_PER_TICK,),
    ),
}


def build_phase_c_identity(
    geometry: ArenaGeometry, cases: tuple[CarCarCase, ...]
) -> dict[str, Any]:
    package_root = Path(__file__).parents[1]
    corpus_path = Path(__file__).with_name("v03_phase_c_corpus.py")
    oracle_path = Path(__file__).with_name("reference") / "v03_phase_c_oracle.py"
    builder_path = package_root / "benchmarks" / "build_v03_phase_c_cache.py"
    diagnostic_patch_path = package_root / "tools" / "rocketsim_phase_c_order_probe.patch"
    config = phase_c_generator_config()
    inputs = {
        "rocketsim": {
            "primary_repository": "https://github.com/ZealanL/RocketSim.git",
            "primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "binding_repository": "https://github.com/mtheall/RocketSim.git",
            "binding_commit": ROCKETSIM_BINDING_COMMIT,
            "binding_package_version": ROCKETSIM_BINDING_VERSION,
            "installed_extension_sha256": EXPECTED_ROCKETSIM_EXTENSION_SHA256,
            "logical_order_diagnostic": {
                "patch_sha256": sha256_file(diagnostic_patch_path),
                "expected_extension_sha256": (
                    EXPECTED_PHASE_C_ORDER_DIAGNOSTIC_SHA256
                ),
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
            "generator_source_path": "rivalsim/v03_phase_c_corpus.py",
            "generator_source_sha256": sha256_file(corpus_path),
            "generator_schema_version": PHASE_C_GENERATOR_SCHEMA_VERSION,
            "generator_config": config,
            "generator_config_sha256": sha256_bytes(canonical_json_bytes(config)),
            "seed": PHASE_C_GENERATOR_SEED,
            "case_count": len(cases),
            "corpus_sha256": phase_c_corpus_sha256(cases),
        },
        "authority_settings": PHASE_C_AUTHORITY_SETTINGS,
        "authority_settings_sha256": sha256_bytes(
            canonical_json_bytes(PHASE_C_AUTHORITY_SETTINGS)
        ),
        "authority_tooling": {
            "rivalsim/reference/v03_phase_c_oracle.py": sha256_file(oracle_path),
            "benchmarks/build_v03_phase_c_cache.py": sha256_file(builder_path),
            "tools/rocketsim_phase_c_order_probe.patch": sha256_file(
                diagnostic_patch_path
            ),
        },
    }
    return {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "C_car_car",
        "authority_identity_sha256": sha256_bytes(canonical_json_bytes(inputs)),
        "identity_inputs": inputs,
    }


def phase_c_frame_arrays(
    initial: list[CarCarBatchOracleFrame],
    frames: list[list[CarCarBatchOracleFrame]],
) -> dict[str, np.ndarray]:
    if len(initial) != len(PHASE_C_NATIVE_BRANCHES) or len(frames) != len(
        PHASE_C_NATIVE_BRANCHES
    ):
        raise ValueError("Phase C authority must contain every native-valid branch")
    if any(len(branch_frames) != len(PHASE_C_CAPTURE_TICKS) for branch_frames in frames):
        raise ValueError("each Phase C branch must contain ticks 1..12")
    arrays: dict[str, np.ndarray] = {}
    for field in PHASE_C_FIELDS:
        dtype, _tail = _FIELD_SCHEMA[field]
        arrays[field] = np.ascontiguousarray(
            np.stack(
                [
                    np.stack(
                        [getattr(frame, field) for frame in branch_frames], axis=1
                    )
                    for branch_frames in frames
                ],
                axis=1,
            ),
            dtype=dtype,
        )
        arrays[f"initial_{field}"] = np.ascontiguousarray(
            np.stack([getattr(frame, field) for frame in initial], axis=1),
            dtype=dtype,
        )
    return arrays


def freeze_phase_c_corpus(
    cache_dir: Path, identity: dict[str, Any], cases: tuple[CarCarCase, ...]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    if identity_path.exists() and _read_json(identity_path) != identity:
        raise RuntimeError(f"Phase C authority identity mismatch: {identity_path}")
    if not identity_path.exists():
        _write_json_atomic(identity_path, identity)
    payload = canonical_json_bytes(
        {
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "corpus_sha256": phase_c_corpus_sha256(cases),
            "case_count": len(cases),
            "cases": [case_record(case) for case in cases],
        }
    )
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    corpus_path = cache_dir / "corpus.json.gz"
    expected_hash = sha256_bytes(compressed)
    if corpus_path.exists() and sha256_file(corpus_path) != expected_hash:
        raise RuntimeError(f"frozen Phase C corpus differs: {corpus_path}")
    if not corpus_path.exists():
        _write_bytes_atomic(corpus_path, compressed)
    frozen = {
        "path": corpus_path.name,
        "size_bytes": corpus_path.stat().st_size,
        "sha256": expected_hash,
        "case_count": len(cases),
        "corpus_sha256": phase_c_corpus_sha256(cases),
    }
    _write_json_atomic(
        cache_dir / "frozen.json",
        {
            "status": "CORPUS_FROZEN_NATIVE_NOT_YET_COMPLETE",
            "phase": "C_car_car",
            "authority_identity_sha256": identity["authority_identity_sha256"],
            "frozen_corpus": frozen,
        },
    )
    return frozen


def phase_c_chunk_paths(cache_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"authority-{start:06d}-{stop:06d}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def write_phase_c_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
    arrays: dict[str, np.ndarray],
    branch_construction: dict[str, Any],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_c_chunk_paths(cache_dir, start, stop)
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
        "phase": "C_car_car",
        "start": start,
        "stop": stop,
        "case_ids": list(case_ids),
        "captured_ticks": list(PHASE_C_CAPTURE_TICKS),
        "native_branches": list(PHASE_C_NATIVE_BRANCHES),
        "branch_construction": branch_construction,
        "frame_fields": list(PHASE_C_FRAME_FIELDS),
        "npz": {
            "path": npz_path.name,
            "size_bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
        },
    }
    _write_json_atomic(meta_path, metadata)
    validate_phase_c_chunk(cache_dir, identity, start, case_ids)
    return metadata


def validate_phase_c_chunk(
    cache_dir: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    stop = start + len(case_ids)
    npz_path, meta_path = phase_c_chunk_paths(cache_dir, start, stop)
    if not npz_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing Phase C authority chunk {start}:{stop}")
    metadata = _read_json(meta_path)
    if metadata.get("authority_identity_sha256") != identity["authority_identity_sha256"]:
        raise RuntimeError(f"authority identity mismatch in {meta_path}")
    if metadata.get("case_ids") != list(case_ids):
        raise RuntimeError(f"case identity mismatch in {meta_path}")
    if metadata.get("native_branches") != list(PHASE_C_NATIVE_BRANCHES):
        raise RuntimeError(f"native branch schema mismatch in {meta_path}")
    branch_construction = metadata.get("branch_construction", {})
    for branch_index, branch in enumerate(PHASE_C_NATIVE_BRANCHES):
        branch_record = branch_construction.get(branch, {})
        if branch_record.get("logical_first_car") != branch_index:
            raise RuntimeError(f"native branch label mismatch in {meta_path}: {branch}")
        if branch_record.get("all_observed_as_requested") is not True:
            raise RuntimeError(f"native branch observation incomplete in {meta_path}: {branch}")
    if metadata["npz"]["sha256"] != sha256_file(npz_path):
        raise RuntimeError(f"authority payload hash mismatch in {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(PHASE_C_FRAME_FIELDS):
            raise RuntimeError(f"Phase C frame fields mismatch in {npz_path}")
        count = len(case_ids)
        branch_count = len(PHASE_C_NATIVE_BRANCHES)
        for stored_field in PHASE_C_FRAME_FIELDS:
            initial = stored_field.startswith("initial_")
            field = stored_field.removeprefix("initial_")
            dtype, tail = _FIELD_SCHEMA[field]
            expected = (
                (count, branch_count, *tail)
                if initial
                else (
                    count,
                    branch_count,
                    len(PHASE_C_CAPTURE_TICKS),
                    *tail,
                )
            )
            if arrays[stored_field].shape != expected:
                raise RuntimeError(f"invalid {stored_field} shape in {npz_path}")
            if arrays[stored_field].dtype != dtype:
                raise RuntimeError(f"invalid {stored_field} dtype in {npz_path}")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(
                arrays[stored_field]
            ).all():
                raise RuntimeError(f"invalid {stored_field} values in {npz_path}")
    return metadata


def finalize_phase_c_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[CarCarCase, ...],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    chunks = []
    for start in range(0, len(cases), PHASE_C_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_C_CACHE_CHUNK_SIZE, len(cases))
        chunks.append(
            validate_phase_c_chunk(
                cache_dir,
                identity,
                start,
                tuple(case.case_id for case in cases[start:stop]),
            )
        )
    manifest = {
        "cache_format_version": V03_CACHE_FORMAT_VERSION,
        "milestone": "v0.3",
        "phase": "C_car_car",
        "status": "COMPLETE_NATIVE_AUTHORITY",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "frozen_corpus": frozen,
        "case_count": len(cases),
        "captured_ticks": list(PHASE_C_CAPTURE_TICKS),
        "native_branches": list(PHASE_C_NATIVE_BRANCHES),
        "frame_count": (
            len(cases) * len(PHASE_C_NATIVE_BRANCHES) * len(PHASE_C_CAPTURE_TICKS)
        ),
        "initial_readback_count": len(cases) * len(PHASE_C_NATIVE_BRANCHES),
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


def verify_phase_c_cache(
    cache_root: Path, identity: dict[str, Any], cases: tuple[CarCarCase, ...]
) -> dict[str, Any]:
    cache_dir = phase_cache_dir(cache_root, identity)
    if _read_json(cache_dir / "identity.json") != identity:
        raise RuntimeError("Phase C authority identity mismatch")
    manifest = _read_json(cache_dir / "manifest.json")
    complete = _read_json(cache_dir / "complete.json")
    if manifest.get("status") != "COMPLETE_NATIVE_AUTHORITY":
        raise RuntimeError("Phase C authority cache is incomplete")
    if complete.get("manifest_sha256") != sha256_file(cache_dir / "manifest.json"):
        raise RuntimeError("Phase C authority manifest hash mismatch")
    frozen = cache_dir / manifest["frozen_corpus"]["path"]
    if sha256_file(frozen) != manifest["frozen_corpus"]["sha256"]:
        raise RuntimeError("Phase C frozen corpus hash mismatch")
    for start in range(0, len(cases), PHASE_C_CACHE_CHUNK_SIZE):
        stop = min(start + PHASE_C_CACHE_CHUNK_SIZE, len(cases))
        validate_phase_c_chunk(
            cache_dir,
            identity,
            start,
            tuple(case.case_id for case in cases[start:stop]),
        )
    return manifest


def load_frozen_phase_c_identity(
    cache_root: Path,
    authority_identity_sha256: str,
    geometry: ArenaGeometry,
    cases: tuple[CarCarCase, ...],
) -> dict[str, Any]:
    """Select an already-frozen authority by hash and revalidate its inputs.

    Cache generation tooling can gain verification and resume features after
    an authority is frozen.  Those changes must not make the immutable native
    payload unreachable.  The invalidation inputs named by the v0.3 protocol
    remain hard gates: RocketSim, collision assets, corpus source/config/seed,
    and authority settings all have to match the current lane exactly.
    """

    if not authority_identity_sha256 or any(
        character not in "0123456789ABCDEF" for character in authority_identity_sha256
    ):
        raise ValueError("Phase C authority identity must be uppercase hexadecimal")
    cache_dir = cache_root.resolve() / authority_identity_sha256
    identity = _read_json(cache_dir / "identity.json")
    if identity.get("authority_identity_sha256") != authority_identity_sha256:
        raise RuntimeError("Phase C authority directory/identity mismatch")

    current = build_phase_c_identity(geometry, cases)
    stored_inputs = identity.get("identity_inputs", {})
    current_inputs = current["identity_inputs"]
    for key in (
        "rocketsim",
        "collision_assets",
        "corpus",
        "authority_settings",
        "authority_settings_sha256",
    ):
        if stored_inputs.get(key) != current_inputs[key]:
            raise RuntimeError(f"Phase C frozen authority input mismatch: {key}")
    return identity


def load_phase_c_frames(
    cache_root: Path,
    identity: dict[str, Any],
    cases: tuple[CarCarCase, ...],
    indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Load frozen Phase C truth; missing data is always a hard error."""

    verify_phase_c_cache(cache_root, identity, cases)
    cache_dir = phase_cache_dir(cache_root, identity)
    staged: dict[str, list[np.ndarray | None]] = {
        field: [None] * len(indices) for field in PHASE_C_FRAME_FIELDS
    }
    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, corpus_index in enumerate(indices):
        if not 0 <= corpus_index < len(cases):
            raise IndexError("Phase C authority index outside the frozen corpus")
        start = (corpus_index // PHASE_C_CACHE_CHUNK_SIZE) * PHASE_C_CACHE_CHUNK_SIZE
        by_chunk.setdefault(start, []).append((output_index, corpus_index - start))
    for start, requests in by_chunk.items():
        stop = min(start + PHASE_C_CACHE_CHUNK_SIZE, len(cases))
        npz_path, _ = phase_c_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as arrays:
            for output_index, local_index in requests:
                for field in PHASE_C_FRAME_FIELDS:
                    staged[field][output_index] = np.asarray(
                        arrays[field][local_index]
                    )
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
