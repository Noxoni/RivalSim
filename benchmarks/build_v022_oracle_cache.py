"""Generate the complete frozen v0.2.2 RocketSim authority cache."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from rivalsim.arena import ArenaGeometry
from rivalsim.dfh_breadth import (
    build_breadth_catalog,
    cases_to_controls,
    cases_to_state,
    generate_breadth_cases,
)
from rivalsim.reference.rocketsim_oracle import RocketSimStaticWorldBatchOracle
from rivalsim.v022_oracle_cache import (
    CACHE_CAPTURE_TICKS,
    CACHE_CHUNK_SIZE,
    authority_cache_dir,
    authority_chunk_path,
    build_authority_identity,
    finalize_authority_cache,
    frame_arrays_from_sequence,
    validate_authority_chunk,
    validate_installed_rocketsim_extension,
    validate_or_write_identity,
    verify_complete_authority_cache,
    write_authority_chunk,
    write_frozen_corpus,
)

EXPECTED_GEOMETRY_SHA256 = "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
EXPECTED_CASE_COUNT = 39_236


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify a complete cache without launching RocketSim",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extension = validate_installed_rocketsim_extension()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError(
            f"unexpected collision geometry: {geometry.content_sha256}, "
            f"expected {EXPECTED_GEOMETRY_SHA256}"
        )
    cases = generate_breadth_cases(build_breadth_catalog(geometry))
    if len(cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"unexpected complete corpus size: {len(cases)}, expected {EXPECTED_CASE_COUNT}"
        )
    identity = build_authority_identity(geometry, cases)
    cache_dir = authority_cache_dir(args.cache_root, identity)
    validate_or_write_identity(cache_dir, identity)
    corpus_artifact = write_frozen_corpus(cache_dir, identity, cases)

    if args.verify_only:
        manifest = verify_complete_authority_cache(args.cache_root, identity, cases)
        summary = _summary(cache_dir, identity, manifest)
        summary["installed_rocketsim_extension"] = extension
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    started = time.perf_counter()
    collision_root = str(geometry.source_root)
    for chunk_number, start in enumerate(range(0, len(cases), CACHE_CHUNK_SIZE)):
        stop = min(start + CACHE_CHUNK_SIZE, len(cases))
        chunk_cases = cases[start:stop]
        case_ids = tuple(case.case_id for case in chunk_cases)
        path = authority_chunk_path(cache_dir, start, stop)
        if path.exists():
            try:
                validate_authority_chunk(path, identity, start, case_ids)
            except RuntimeError:
                pass
            else:
                print(
                    json.dumps(
                        {
                            "chunk": chunk_number,
                            "range": [start, stop],
                            "status": "resumed",
                        }
                    ),
                    flush=True,
                )
                continue

        chunk_started = time.perf_counter()
        intended = cases_to_state(chunk_cases)
        controls = cases_to_controls(chunk_cases)
        oracle = RocketSimStaticWorldBatchOracle(intended, collision_root)
        initial = oracle.authoritative_snapshot()
        oracle.set_controls(controls)
        native_frames = []
        for _tick in CACHE_CAPTURE_TICKS:
            oracle.step()
            native_frames.append(oracle.frame())
        frame_arrays = frame_arrays_from_sequence(native_frames)
        write_authority_chunk(path, identity, start, case_ids, initial, frame_arrays)
        validate_authority_chunk(path, identity, start, case_ids)
        del oracle, initial, native_frames, frame_arrays, controls, intended
        gc.collect()
        print(
            json.dumps(
                {
                    "chunk": chunk_number,
                    "range": [start, stop],
                    "status": "generated_native_only",
                    "seconds": round(time.perf_counter() - chunk_started, 3),
                }
            ),
            flush=True,
        )

    manifest = finalize_authority_cache(cache_dir, identity, cases, corpus_artifact)
    summary = _summary(cache_dir, identity, manifest)
    summary["installed_rocketsim_extension"] = extension
    summary["generation_seconds"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _summary(
    cache_dir: Path,
    identity: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "purpose": "oracle_data_generation_not_rivalsim_acceptance",
        "status": manifest["status"],
        "cache_dir": str(cache_dir.resolve()),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "case_count": manifest["case_count"],
        "frame_count": manifest["frame_count"],
        "chunk_count": manifest["chunk_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
