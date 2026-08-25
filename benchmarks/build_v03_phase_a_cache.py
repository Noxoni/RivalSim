"""Freeze and generate the complete v0.3 Phase A RocketSim authority cache."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from rivalsim.arena import ArenaGeometry
from rivalsim.reference.rocketsim_oracle import RocketSimBallWorldBatchOracle
from rivalsim.v03_corpus import generate_phase_a_cases, phase_a_cases_to_state
from rivalsim.v03_oracle_cache import (
    EXPECTED_SOCCAR_CMF_SHA256,
    V03_CACHE_CAPTURE_TICKS,
    V03_CACHE_CHUNK_SIZE,
    build_phase_a_identity,
    finalize_phase_a_cache,
    freeze_phase_a_corpus,
    phase_a_chunk_paths,
    phase_a_frame_arrays,
    phase_cache_dir,
    validate_installed_rocketsim_extension,
    validate_phase_a_chunk,
    verify_phase_a_cache,
    write_phase_a_chunk,
)

EXPECTED_PHASE_A_CASE_COUNT = 31_216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="freeze identity/corpus without launching RocketSim",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify a previously completed cache without launching RocketSim",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError(
            f"unexpected collision geometry: {geometry.content_sha256}, "
            f"expected {EXPECTED_SOCCAR_CMF_SHA256}"
        )
    _catalog, cases = generate_phase_a_cases(geometry)
    if len(cases) != EXPECTED_PHASE_A_CASE_COUNT:
        raise RuntimeError(
            f"unexpected Phase A corpus size: {len(cases)}, "
            f"expected {EXPECTED_PHASE_A_CASE_COUNT}"
        )
    identity = build_phase_a_identity(geometry, cases)
    cache_dir = phase_cache_dir(args.cache_root, identity)
    frozen = freeze_phase_a_corpus(cache_dir, identity, cases)

    if args.freeze_only:
        print(json.dumps(_summary(cache_dir, identity, frozen=frozen), indent=2))
        return 0

    extension = validate_installed_rocketsim_extension()
    if args.verify_only:
        manifest = verify_phase_a_cache(args.cache_root, identity, cases)
        print(
            json.dumps(
                _summary(cache_dir, identity, manifest=manifest, extension=extension),
                indent=2,
            )
        )
        return 0

    started = time.perf_counter()
    for chunk_number, start in enumerate(range(0, len(cases), V03_CACHE_CHUNK_SIZE)):
        stop = min(start + V03_CACHE_CHUNK_SIZE, len(cases))
        chunk_cases = cases[start:stop]
        case_ids = tuple(case.case_id for case in chunk_cases)
        npz_path, meta_path = phase_a_chunk_paths(cache_dir, start, stop)
        if npz_path.exists() and meta_path.exists():
            try:
                validate_phase_a_chunk(cache_dir, identity, start, case_ids)
            except RuntimeError:
                pass
            else:
                print(
                    json.dumps(
                        {"chunk": chunk_number, "range": [start, stop], "status": "resumed"}
                    ),
                    flush=True,
                )
                continue

        chunk_started = time.perf_counter()
        initial = phase_a_cases_to_state(chunk_cases)
        oracle = RocketSimBallWorldBatchOracle(initial, str(geometry.source_root))
        initial_readback = oracle.frame()
        frames = []
        for _tick in V03_CACHE_CAPTURE_TICKS:
            oracle.step()
            frames.append(oracle.frame())
        arrays = phase_a_frame_arrays(initial_readback, frames)
        write_phase_a_chunk(cache_dir, identity, start, case_ids, arrays)
        del oracle, initial_readback, frames, arrays, initial
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

    manifest = finalize_phase_a_cache(cache_dir, identity, cases, frozen)
    summary = _summary(cache_dir, identity, manifest=manifest, extension=extension)
    summary["generation_seconds"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _summary(
    cache_dir: Path,
    identity: dict[str, object],
    *,
    frozen: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    extension: dict[str, str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "purpose": "oracle_data_generation_not_rivalsim_acceptance",
        "phase": "A_ball_world",
        "cache_dir": str(cache_dir.resolve()),
        "authority_identity_sha256": identity["authority_identity_sha256"],
    }
    if frozen is not None:
        result.update(status="CORPUS_FROZEN_NATIVE_NOT_YET_COMPLETE", frozen_corpus=frozen)
    if manifest is not None:
        result.update(
            status=manifest["status"],
            case_count=manifest["case_count"],
            frame_count=manifest["frame_count"],
            chunk_count=manifest["chunk_count"],
        )
    if extension is not None:
        result["installed_rocketsim_extension"] = extension
    return result


if __name__ == "__main__":
    raise SystemExit(main())
