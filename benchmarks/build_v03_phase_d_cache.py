"""Freeze and generate the complete v0.3 Phase D native authority cache."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rivalsim.arena import ArenaGeometry
from rivalsim.reference.v03_phase_d_oracle import (
    IntegratedBatchOracleFrame,
    RocketSimIntegratedBatchOracle,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_SOCCAR_CMF_SHA256,
    phase_cache_dir,
    sha256_file,
    validate_installed_rocketsim_extension,
)
from rivalsim.v03_phase_d_cache import (
    EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256,
    PHASE_D_CACHE_CHUNK_SIZE,
    PHASE_D_CAPTURE_TICKS,
    PHASE_D_FIELDS,
    PHASE_D_NATIVE_BRANCHES,
    build_phase_d_identity,
    finalize_phase_d_cache,
    freeze_phase_d_corpus,
    phase_d_chunk_paths,
    phase_d_frame_arrays,
    validate_phase_d_chunk,
    verify_phase_d_cache,
    write_phase_d_chunk,
)
from rivalsim.v03_phase_d_corpus import (
    PHASE_D_CASE_COUNT,
    generate_phase_d_cases,
    phase_d_cases_to_state,
    phase_d_controls_at,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--order-diagnostic-extension", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prove-invariance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-existing", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError("unexpected Soccar collision geometry")
    cases = generate_phase_d_cases()
    if len(cases) != PHASE_D_CASE_COUNT:
        raise RuntimeError(f"unexpected Phase D corpus size: {len(cases)}")
    identity = build_phase_d_identity(geometry, cases)
    cache_dir = phase_cache_dir(args.cache_root, identity)
    frozen = freeze_phase_d_corpus(cache_dir, identity, cases)

    if args.worker_start is not None:
        diagnostic = _activate_order_diagnostic(args.order_diagnostic_extension)
        return _run_worker(args, cache_dir, identity, cases, geometry, diagnostic)
    if args.prove_invariance:
        _activate_order_diagnostic(args.order_diagnostic_extension)
        result = _prove_branch_determinism(cache_dir, cases, geometry)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.freeze_only:
        print(json.dumps(_summary(cache_dir, identity, frozen=frozen), indent=2))
        return 0

    extension = validate_installed_rocketsim_extension()
    diagnostic = _validate_order_diagnostic(args.order_diagnostic_extension)
    if args.verify_only:
        manifest = verify_phase_d_cache(args.cache_root, identity, cases)
        print(
            json.dumps(
                _summary(cache_dir, identity, manifest=manifest, extension=extension),
                indent=2,
            )
        )
        return 0

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--collision-dir",
            str(geometry.source_root),
            "--cache-root",
            str(args.cache_root.resolve()),
            "--order-diagnostic-extension",
            str(args.order_diagnostic_extension.resolve()),
            "--prove-invariance",
        ],
        check=True,
    )
    invariance = json.loads((cache_dir / "branch-determinism.json").read_text())
    started = time.perf_counter()
    for chunk_number, start in enumerate(range(0, len(cases), PHASE_D_CACHE_CHUNK_SIZE)):
        stop = min(start + PHASE_D_CACHE_CHUNK_SIZE, len(cases))
        case_ids = tuple(case.case_id for case in cases[start:stop])
        npz_path, meta_path = phase_d_chunk_paths(cache_dir, start, stop)
        if npz_path.exists() and meta_path.exists():
            try:
                validate_phase_d_chunk(cache_dir, identity, start, case_ids)
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
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--collision-dir",
            str(geometry.source_root),
            "--cache-root",
            str(args.cache_root.resolve()),
            "--order-diagnostic-extension",
            str(args.order_diagnostic_extension.resolve()),
            "--worker-start",
            str(start),
        ]
        for process_attempt in range(1, 17):
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode == 0:
                print(completed.stdout, end="", flush=True)
                break
            lines = completed.stderr.strip().splitlines()
            print(
                json.dumps(
                    {
                        "range": [start, stop],
                        "status": "discarded_worker_process_without_complete_branches",
                        "process_attempt": process_attempt,
                        "diagnostic": lines[-1] if lines else "worker failed",
                    }
                ),
                flush=True,
            )
        else:
            raise RuntimeError(
                f"Phase D source did not yield both branches for chunk {start}:{stop}"
            )

    manifest = finalize_phase_d_cache(cache_dir, identity, cases, frozen)
    summary = _summary(cache_dir, identity, manifest=manifest, extension=extension)
    summary["logical_order_diagnostic"] = diagnostic
    summary["source_branch_determinism"] = invariance
    summary["generation_seconds"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _run_worker(
    args: argparse.Namespace,
    cache_dir: Path,
    identity: dict[str, object],
    cases: tuple[object, ...],
    geometry: ArenaGeometry,
    diagnostic: dict[str, str],
) -> int:
    start = args.worker_start
    if start < 0 or start >= len(cases) or start % PHASE_D_CACHE_CHUNK_SIZE:
        raise ValueError(f"invalid Phase D worker start: {start}")
    stop = min(start + PHASE_D_CACHE_CHUNK_SIZE, len(cases))
    chunk_cases = cases[start:stop]
    case_ids = tuple(case.case_id for case in chunk_cases)
    initial_readbacks: list[IntegratedBatchOracleFrame] = []
    branch_frames: list[list[IntegratedBatchOracleFrame]] = []
    branch_construction: dict[str, object] = {}
    chunk_started = time.perf_counter()

    for branch_index, branch in enumerate(PHASE_D_NATIVE_BRANCHES):
        initial_case_frames: list[IntegratedBatchOracleFrame] = []
        tick_case_frames: list[list[IntegratedBatchOracleFrame]] = [
            [] for _tick in PHASE_D_CAPTURE_TICKS
        ]
        attempts = np.empty(len(chunk_cases), dtype=np.int32)
        for local_index, case in enumerate(chunk_cases):
            oracle = RocketSimIntegratedBatchOracle(
                phase_d_cases_to_state((case,)),
                str(geometry.source_root),
                pre_tick_visit_order=branch,
            )
            if int(oracle.pre_tick_first_car[0]) != branch_index:
                raise RuntimeError(f"native Phase D branch label mismatch: {branch}")
            attempts[local_index] = oracle.construction_attempts[0]
            initial_case_frames.append(oracle.frame())
            for tick_index, _tick in enumerate(PHASE_D_CAPTURE_TICKS):
                oracle.set_controls(phase_d_controls_at((case,), tick_index))
                oracle.step()
                tick_case_frames[tick_index].append(oracle.frame())
            del oracle
            if local_index % 8 == 7:
                gc.collect()
        initial_readbacks.append(_concatenate_frames(initial_case_frames))
        branch_frames.append(
            [_concatenate_frames(tick_frames) for tick_frames in tick_case_frames]
        )
        branch_construction[branch] = {
            "logical_first_car": branch_index,
            "observed_case_count": len(chunk_cases),
            "all_observed_as_requested": True,
            "fresh_arena_attempts": {
                "minimum": int(attempts.min()),
                "maximum": int(attempts.max()),
                "total": int(attempts.sum()),
            },
        }
    branch_construction["diagnostic_extension"] = {
        key: value for key, value in diagnostic.items() if key != "path"
    }
    arrays = phase_d_frame_arrays(initial_readbacks, branch_frames)
    if args.verify_existing:
        npz_path, _meta = phase_d_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as expected:
            for field, values in arrays.items():
                if not np.array_equal(values, expected[field]):
                    raise RuntimeError(
                        f"fresh-process Phase D replay mismatch at {start}:{stop}, field {field}"
                    )
        status = "fresh_process_replay_exact"
    else:
        write_phase_d_chunk(
            cache_dir, identity, start, case_ids, arrays, branch_construction
        )
        status = "generated_fresh_process_native_only"
    print(
        json.dumps(
            {
                "range": [start, stop],
                "status": status,
                "seconds": round(time.perf_counter() - chunk_started, 3),
            }
        ),
        flush=True,
    )
    return 0


def _prove_branch_determinism(
    cache_dir: Path, cases: tuple[object, ...], geometry: ArenaGeometry
) -> dict[str, object]:
    indices = (0, 63, 64, 127, 128, 191, 192, 255, 256, 319, 320, 383, 384, 447, 448, 511)
    compared_frames = 0
    attempts: dict[str, object] = {}
    for branch in PHASE_D_NATIVE_BRANCHES:
        branch_attempts = [0, 0]
        for index in indices:
            case = cases[index]
            runs: list[list[IntegratedBatchOracleFrame]] = []
            for replay in range(2):
                oracle = RocketSimIntegratedBatchOracle(
                    phase_d_cases_to_state((case,)),
                    str(geometry.source_root),
                    pre_tick_visit_order=branch,
                )
                branch_attempts[replay] += int(oracle.construction_attempts[0])
                frames = [oracle.frame()]
                for tick_index, _tick in enumerate(PHASE_D_CAPTURE_TICKS):
                    oracle.set_controls(phase_d_controls_at((case,), tick_index))
                    oracle.step()
                    frames.append(oracle.frame())
                runs.append(frames)
                del oracle
            for capture_index, (left, right) in enumerate(zip(*runs, strict=True)):
                for field in PHASE_D_FIELDS:
                    if not np.array_equal(getattr(left, field), getattr(right, field)):
                        raise RuntimeError(
                            "Phase D labeled native branch replay mismatch at "
                            f"case {index}, branch {branch}, capture {capture_index}, field {field}"
                        )
                compared_frames += 1
        attempts[branch] = {"first_total": branch_attempts[0], "second_total": branch_attempts[1]}
    result: dict[str, object] = {
        "status": "EXACT_SOURCE_BRANCH_REPLAY",
        "case_count": len(indices),
        "case_indices": list(indices),
        "native_branches": list(PHASE_D_NATIVE_BRANCHES),
        "branch_construction_attempts": attempts,
        "captures_per_case": len(PHASE_D_CAPTURE_TICKS) + 1,
        "compared_frames": compared_frames,
        "fields": list(PHASE_D_FIELDS),
    }
    (cache_dir / "branch-determinism.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _concatenate_frames(
    frames: list[IntegratedBatchOracleFrame],
) -> IntegratedBatchOracleFrame:
    if not frames:
        raise ValueError("cannot concatenate an empty Phase D frame sequence")
    return IntegratedBatchOracleFrame(
        **{
            field: np.ascontiguousarray(
                np.concatenate([getattr(frame, field) for frame in frames], axis=0)
            )
            for field in PHASE_D_FIELDS
        }
    )


def _validate_order_diagnostic(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing Phase D logical-order diagnostic: {resolved}")
    digest = sha256_file(resolved)
    if digest != EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256:
        raise RuntimeError(
            f"Phase D logical-order diagnostic hash mismatch: {digest}, "
            f"expected {EXPECTED_PHASE_D_ORDER_DIAGNOSTIC_SHA256}"
        )
    return {"path": str(resolved), "sha256": digest, "returns": "logical_car_ids_only"}


def _activate_order_diagnostic(path: Path) -> dict[str, str]:
    result = _validate_order_diagnostic(path)
    if "RocketSim" in sys.modules:
        raise RuntimeError("RocketSim was imported before activating the order diagnostic")
    sys.path.insert(0, str(path.resolve().parent))
    import RocketSim

    if Path(RocketSim.__file__).resolve() != path.resolve():
        raise RuntimeError("failed to activate the pinned Phase D order diagnostic")
    if not hasattr(RocketSim.Arena, "_get_pre_tick_visit_order"):
        raise RuntimeError("Phase D diagnostic lacks logical-order readback")
    return result


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
        "phase": "D_integrated",
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
