"""Content-addressed RocketSim authority cache for the frozen v0.2.2 corpus.

The cache is deliberately independent of RivalSim's GPU implementation.  Its
semantic identity changes only when the pinned RocketSim authority, collision
assets, generated corpus, or authority protocol changes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.arena import ArenaGeometry
from rivalsim.dfh_breadth import (
    GENERATOR_SCHEMA_VERSION,
    GENERATOR_SEED,
    LOCAL_HORIZONS,
    BreadthCase,
    corpus_sha256,
    generator_config,
)
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    StaticWorldBatchOracleFrame,
)
from rivalsim.state import StateSnapshot

CACHE_FORMAT_VERSION = 1
CACHE_CHUNK_SIZE = 256
CACHE_CAPTURE_TICKS = tuple(range(1, max(LOCAL_HORIZONS) + 1))
EXPECTED_ROCKETSIM_EXTENSION_SHA256 = (
    "E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0"
)
IDENTITY_FILENAME = "identity.json"
CORPUS_FILENAME = "corpus.json.gz"
MANIFEST_FILENAME = "manifest.json"
COMPLETE_FILENAME = "complete.json"

FRAME_FIELDS = (
    "car_pos",
    "car_vel",
    "car_matrix",
    "car_ang_vel",
    "boost",
    "handbrake_value",
    "on_ground",
    "wheel_contacts",
    "has_world_contact",
    "world_contact_normal",
)

AUTHORITY_SETTINGS: dict[str, Any] = {
    "protocol_version": 1,
    "game_mode": "SOCCAR",
    "tick_rate_hz": 120.0,
    "arena_config": {"no_ball_rot": False},
    "collision_switches": {
        "car_car_collision": False,
        "car_ball_collision": False,
    },
    "isolation": "one fresh arena and one authority car per corpus case",
    "authority_car": {"team": "BLUE", "config": "OCTANE"},
    "ball_initial_position_uu": [0.0, 0.0, 1500.0],
    "controls": "case controls set once and held constant through tick 12",
    "initial_state": (
        "generated state is set through RocketSim CarState; the exact immediate "
        "RocketSim readback initializes the GPU comparison"
    ),
    "step_order": "arena.step(1), then complete CarState readback",
    "captured_ticks": list(CACHE_CAPTURE_TICKS),
    "captured_frame_fields": list(FRAME_FIELDS),
    "contact_semantics": {
        "chassis": "CarState.has_world_contact",
        "wheels": "CarState.wheels_with_contact[0:4]",
        "normal": "CarState.world_contact_normal",
    },
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


def build_authority_identity(
    geometry: ArenaGeometry,
    cases: tuple[BreadthCase, ...],
) -> dict[str, Any]:
    """Build the exact semantic identity requested for the native authority."""

    generator_source = Path(__file__).with_name("dfh_breadth.py")
    config = generator_config()
    settings = AUTHORITY_SETTINGS
    identity_inputs = {
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
            "generator_source_path": "rivalsim/dfh_breadth.py",
            "generator_source_sha256": sha256_file(generator_source),
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "generator_config": config,
            "generator_config_sha256": sha256_bytes(canonical_json_bytes(config)),
            "seed": GENERATOR_SEED,
            "case_count": len(cases),
            "corpus_sha256": corpus_sha256(cases),
        },
        "authority_settings": settings,
        "authority_settings_sha256": sha256_bytes(canonical_json_bytes(settings)),
    }
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "milestone": "v0.2.2",
        "authority_identity_sha256": sha256_bytes(canonical_json_bytes(identity_inputs)),
        "identity_inputs": identity_inputs,
    }


def authority_cache_dir(cache_root: Path, identity: dict[str, Any]) -> Path:
    return cache_root.resolve() / str(identity["authority_identity_sha256"])


def validate_installed_rocketsim_extension() -> dict[str, str]:
    """Refuse native generation from a binary outside the frozen authority."""

    import RocketSim

    path = Path(RocketSim.__file__).resolve()
    resolved = sha256_file(path)
    if resolved != EXPECTED_ROCKETSIM_EXTENSION_SHA256:
        raise RuntimeError(
            f"installed RocketSim extension hash mismatch: {resolved}, "
            f"expected {EXPECTED_ROCKETSIM_EXTENSION_SHA256}"
        )
    return {"path": str(path), "sha256": resolved}


def validate_or_write_identity(cache_dir: Path, identity: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / IDENTITY_FILENAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(f"authority cache identity mismatch: {path.resolve()}")
        return
    _write_json_atomic(path, identity)


def write_frozen_corpus(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[BreadthCase, ...],
) -> dict[str, Any]:
    """Persist a deterministic, complete description of every generated case."""

    path = cache_dir / CORPUS_FILENAME
    payload = {
        "schema_version": CACHE_FORMAT_VERSION,
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "corpus_sha256": identity["identity_inputs"]["corpus"]["corpus_sha256"],
        "case_count": len(cases),
        "cases": [_case_payload(index, case) for index, case in enumerate(cases)],
    }
    rewrite = not path.exists()
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rewrite = json.load(stream) != payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            rewrite = True
    if rewrite:
        _write_deterministic_gzip_json(path, payload)
    return {
        "path": CORPUS_FILENAME,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def authority_chunk_path(cache_dir: Path, start: int, stop: int) -> Path:
    return cache_dir / "chunks" / f"authority-{start:06d}-{stop:06d}.npz"


def write_authority_chunk(
    path: Path,
    identity: dict[str, Any],
    start: int,
    case_ids: tuple[str, ...],
    initial: StateSnapshot,
    frames: dict[str, np.ndarray],
) -> None:
    stop = start + len(case_ids)
    _validate_authority_arrays(initial, frames, len(case_ids))
    arrays: dict[str, np.ndarray] = {
        "cache_format_version": np.asarray(CACHE_FORMAT_VERSION, dtype=np.int32),
        "authority_identity_sha256": np.asarray(identity["authority_identity_sha256"]),
        "start": np.asarray(start, dtype=np.int32),
        "stop": np.asarray(stop, dtype=np.int32),
        "case_ids": np.asarray(case_ids),
        "captured_ticks": np.asarray(CACHE_CAPTURE_TICKS, dtype=np.int32),
    }
    for field in fields(StateSnapshot):
        arrays[f"initial__{field.name}"] = np.ascontiguousarray(getattr(initial, field.name))
    for name in FRAME_FIELDS:
        arrays[f"frame__{name}"] = np.ascontiguousarray(frames[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def validate_authority_chunk(
    path: Path,
    identity: dict[str, Any],
    start: int,
    expected_case_ids: tuple[str, ...],
) -> None:
    stop = start + len(expected_case_ids)
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["cache_format_version"]) != CACHE_FORMAT_VERSION:
                raise RuntimeError("cache format version mismatch")
            if str(data["authority_identity_sha256"]) != str(identity["authority_identity_sha256"]):
                raise RuntimeError("authority identity mismatch")
            if int(data["start"]) != start or int(data["stop"]) != stop:
                raise RuntimeError("chunk range mismatch")
            if tuple(str(value) for value in data["case_ids"].tolist()) != (expected_case_ids):
                raise RuntimeError("chunk case order mismatch")
            if tuple(int(value) for value in data["captured_ticks"].tolist()) != (
                CACHE_CAPTURE_TICKS
            ):
                raise RuntimeError("captured tick mismatch")
            count = int(np.asarray(data["initial__car_pos"]).shape[0])
            initial = StateSnapshot.empty(count)
            for field in fields(StateSnapshot):
                key = f"initial__{field.name}"
                if key in data:
                    getattr(initial, field.name)[...] = np.asarray(data[key])
            frames = {name: np.asarray(data[f"frame__{name}"]) for name in FRAME_FIELDS}
            _validate_authority_arrays(initial, frames, len(expected_case_ids))
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"invalid authority cache chunk: {path.resolve()}") from error


def finalize_authority_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[BreadthCase, ...],
    corpus_artifact: dict[str, Any],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(cases), CACHE_CHUNK_SIZE):
        stop = min(start + CACHE_CHUNK_SIZE, len(cases))
        case_ids = tuple(case.case_id for case in cases[start:stop])
        path = authority_chunk_path(cache_dir, start, stop)
        if not path.exists():
            raise RuntimeError(f"missing authority cache chunk: {path.resolve()}")
        validate_authority_chunk(path, identity, start, case_ids)
        chunks.append(
            {
                "path": path.relative_to(cache_dir).as_posix(),
                "start": start,
                "stop": stop,
                "case_count": stop - start,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if sha256_file(cache_dir / CORPUS_FILENAME) != corpus_artifact["sha256"]:
        raise RuntimeError("frozen corpus changed while finalizing cache")
    manifest = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "milestone": "v0.2.2",
        "purpose": "RocketSim oracle data generation; not a RivalSim acceptance run",
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "case_count": len(cases),
        "captured_ticks": list(CACHE_CAPTURE_TICKS),
        "frame_count": len(cases) * len(CACHE_CAPTURE_TICKS),
        "chunk_size": CACHE_CHUNK_SIZE,
        "chunk_count": len(chunks),
        "frozen_corpus": corpus_artifact,
        "chunks": chunks,
    }
    manifest_path = cache_dir / MANIFEST_FILENAME
    _write_json_atomic(manifest_path, manifest)
    marker = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": "complete",
    }
    _write_json_atomic(cache_dir / COMPLETE_FILENAME, marker)
    return manifest


@dataclass(slots=True)
class CachedAuthorityBatch:
    authoritative_snapshot: StateSnapshot
    frames: dict[str, np.ndarray]

    def frame(self, tick: int) -> StaticWorldBatchOracleFrame:
        if tick not in CACHE_CAPTURE_TICKS:
            raise ValueError(f"tick {tick} is not present in the authority cache")
        index = tick - CACHE_CAPTURE_TICKS[0]
        return StaticWorldBatchOracleFrame(
            **{name: self.frames[name][index] for name in FRAME_FIELDS}
        )


class RocketSimAuthorityCache:
    """Strict reader for a complete content-addressed authority cache."""

    def __init__(
        self,
        cache_root: Path,
        identity: dict[str, Any],
        cases: tuple[BreadthCase, ...],
    ):
        self.identity = identity
        self.cases = cases
        self.cache_dir = authority_cache_dir(cache_root, identity)
        self.manifest = _load_complete_manifest(self.cache_dir, identity, cases)

    @property
    def identity_sha256(self) -> str:
        return str(self.identity["authority_identity_sha256"])

    def load(self, corpus_indices: tuple[int, ...]) -> CachedAuthorityBatch:
        if len(set(corpus_indices)) != len(corpus_indices):
            raise ValueError("cached authority indices must be unique")
        if any(index < 0 or index >= len(self.cases) for index in corpus_indices):
            raise IndexError("cached authority index outside frozen corpus")
        count = len(corpus_indices)
        initial = StateSnapshot.empty(count)
        frames_out = _empty_frame_arrays(count)
        destination = {corpus_index: index for index, corpus_index in enumerate(corpus_indices)}
        loaded: set[int] = set()
        for entry in self.manifest["chunks"]:
            start = int(entry["start"])
            stop = int(entry["stop"])
            selected = [index for index in corpus_indices if start <= index < stop]
            if not selected:
                continue
            path = self.cache_dir / str(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"authority cache chunk hash mismatch: {path.resolve()}")
            expected_ids = tuple(case.case_id for case in self.cases[start:stop])
            validate_authority_chunk(path, self.identity, start, expected_ids)
            with np.load(path, allow_pickle=False) as data:
                for corpus_index in selected:
                    source = corpus_index - start
                    target = destination[corpus_index]
                    for field in fields(StateSnapshot):
                        key = f"initial__{field.name}"
                        if key in data:
                            getattr(initial, field.name)[target] = data[key][source]
                    for name in FRAME_FIELDS:
                        frames_out[name][:, target] = data[f"frame__{name}"][:, source]
                    loaded.add(corpus_index)
        if loaded != set(corpus_indices):
            missing = sorted(set(corpus_indices) - loaded)
            raise RuntimeError(f"authority cache did not load corpus indices: {missing[:10]}")
        # Cache format v1 stored the binding's immediate post-SetState readback.
        # Position/velocity were already round-tripped through Bullet units and
        # rotation was available only as a matrix. Reconstruct the exact frozen
        # source fields that produced the rigid state RocketSim actually steps;
        # the case payload is part of the authority identity and cache manifest.
        for corpus_index, target in destination.items():
            case = self.cases[corpus_index]
            initial.car_pos[target, 0] = case.position
            initial.car_vel[target, 0] = case.velocity
            initial.car_quat[target, 0] = case.quaternion
            initial.car_ang_vel[target, 0] = case.angular_velocity
        initial.validate()
        return CachedAuthorityBatch(initial, frames_out)


def verify_complete_authority_cache(
    cache_root: Path,
    identity: dict[str, Any],
    cases: tuple[BreadthCase, ...],
) -> dict[str, Any]:
    """Verify every artifact and array in an already completed authority cache."""

    cache = RocketSimAuthorityCache(cache_root, identity, cases)
    for entry in cache.manifest["chunks"]:
        start = int(entry["start"])
        stop = int(entry["stop"])
        path = cache.cache_dir / str(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"authority cache chunk hash mismatch: {path.resolve()}")
        validate_authority_chunk(
            path,
            identity,
            start,
            tuple(case.case_id for case in cases[start:stop]),
        )
    return cache.manifest


def frame_arrays_from_sequence(
    frames: list[StaticWorldBatchOracleFrame],
) -> dict[str, np.ndarray]:
    if len(frames) != len(CACHE_CAPTURE_TICKS):
        raise ValueError("authority frame sequence must contain all captured ticks")
    return {
        name: np.ascontiguousarray(np.stack([getattr(frame, name) for frame in frames], axis=0))
        for name in FRAME_FIELDS
    }


def _load_complete_manifest(
    cache_dir: Path,
    identity: dict[str, Any],
    cases: tuple[BreadthCase, ...],
) -> dict[str, Any]:
    identity_path = cache_dir / IDENTITY_FILENAME
    manifest_path = cache_dir / MANIFEST_FILENAME
    marker_path = cache_dir / COMPLETE_FILENAME
    if not identity_path.exists() or not manifest_path.exists() or not marker_path.exists():
        raise RuntimeError(f"complete RocketSim authority cache not found: {cache_dir.resolve()}")
    if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise RuntimeError("authority cache identity file is stale")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("authority cache completion marker does not match manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("authority_identity_sha256") != identity["authority_identity_sha256"]
        or manifest.get("case_count") != len(cases)
        or manifest.get("captured_ticks") != list(CACHE_CAPTURE_TICKS)
    ):
        raise RuntimeError("authority cache manifest is stale or incomplete")
    corpus = cache_dir / str(manifest["frozen_corpus"]["path"])
    if sha256_file(corpus) != manifest["frozen_corpus"]["sha256"]:
        raise RuntimeError("frozen corpus artifact hash mismatch")
    expected_start = 0
    for entry in manifest["chunks"]:
        if int(entry["start"]) != expected_start:
            raise RuntimeError("authority cache manifest has a range gap")
        expected_start = int(entry["stop"])
    if expected_start != len(cases):
        raise RuntimeError("authority cache manifest does not cover the complete corpus")
    return manifest


def _validate_authority_arrays(
    initial: StateSnapshot,
    frames_in: dict[str, np.ndarray],
    count: int,
) -> None:
    initial.validate()
    if initial.num_envs != count:
        raise ValueError("authority initial snapshot count mismatch")
    expected = {
        "car_pos": ((len(CACHE_CAPTURE_TICKS), count, 3), np.float32),
        "car_vel": ((len(CACHE_CAPTURE_TICKS), count, 3), np.float32),
        "car_matrix": ((len(CACHE_CAPTURE_TICKS), count, 3, 3), np.float32),
        "car_ang_vel": ((len(CACHE_CAPTURE_TICKS), count, 3), np.float32),
        "boost": ((len(CACHE_CAPTURE_TICKS), count), np.float32),
        "handbrake_value": ((len(CACHE_CAPTURE_TICKS), count), np.float32),
        "on_ground": ((len(CACHE_CAPTURE_TICKS), count), np.bool_),
        "wheel_contacts": ((len(CACHE_CAPTURE_TICKS), count, 4), np.bool_),
        "has_world_contact": ((len(CACHE_CAPTURE_TICKS), count), np.bool_),
        "world_contact_normal": (
            (len(CACHE_CAPTURE_TICKS), count, 3),
            np.float32,
        ),
    }
    if set(frames_in) != set(FRAME_FIELDS):
        raise ValueError("authority frame field set mismatch")
    for name, (shape, dtype) in expected.items():
        value = frames_in[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"invalid cached {name}: {value.shape}/{value.dtype}, expected {shape}/{dtype}"
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"non-finite cached authority frame: {name}")


def _empty_frame_arrays(count: int) -> dict[str, np.ndarray]:
    ticks = len(CACHE_CAPTURE_TICKS)
    return {
        "car_pos": np.empty((ticks, count, 3), dtype=np.float32),
        "car_vel": np.empty((ticks, count, 3), dtype=np.float32),
        "car_matrix": np.empty((ticks, count, 3, 3), dtype=np.float32),
        "car_ang_vel": np.empty((ticks, count, 3), dtype=np.float32),
        "boost": np.empty((ticks, count), dtype=np.float32),
        "handbrake_value": np.empty((ticks, count), dtype=np.float32),
        "on_ground": np.empty((ticks, count), dtype=np.bool_),
        "wheel_contacts": np.empty((ticks, count, 4), dtype=np.bool_),
        "has_world_contact": np.empty((ticks, count), dtype=np.bool_),
        "world_contact_normal": np.empty((ticks, count, 3), dtype=np.float32),
    }


def _case_payload(index: int, case: BreadthCase) -> dict[str, Any]:
    def vector(value: np.ndarray | None) -> list[float] | None:
        return None if value is None else np.asarray(value, dtype=np.float32).astype(float).tolist()

    return {
        "corpus_index": index,
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "family": case.family,
        "mode": case.mode,
        "contact_path": case.contact_path,
        "mesh_index": case.mesh_index,
        "mesh_file": case.mesh_file,
        "target_face": case.target_face,
        "target_neighbor_face": case.target_neighbor_face,
        "target_edge": case.target_edge,
        "edge_class": case.edge_class,
        "analytic_plane": case.analytic_plane,
        "expected_plane_face": case.expected_plane_face,
        "target_point": vector(case.target_point),
        "target_normal": vector(case.target_normal),
        "edge_start": vector(case.edge_start),
        "edge_end": vector(case.edge_end),
        "region_labels": list(case.region_labels),
        "position": vector(case.position),
        "quaternion": vector(case.quaternion),
        "velocity": vector(case.velocity),
        "angular_velocity": vector(case.angular_velocity),
        "controls": list(case.controls),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(payload, indent=2).encode("utf-8") + b"\n")
    temporary.replace(path)


def _write_deterministic_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text,
    ):
        json.dump(
            payload,
            text,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        text.write("\n")
    temporary.replace(path)
