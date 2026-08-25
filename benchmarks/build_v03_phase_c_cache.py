"""Freeze and generate the complete v0.3 Phase C native authority cache."""

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
from rivalsim.reference.v03_phase_c_oracle import (
    CarCarBatchOracleFrame,
    RocketSimCarCarBatchOracle,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_SOCCAR_CMF_SHA256,
    phase_cache_dir,
    validate_installed_rocketsim_extension,
)
from rivalsim.v03_phase_c_cache import (
    EXPECTED_PHASE_C_ORDER_DIAGNOSTIC_SHA256,
    PHASE_C_CACHE_CHUNK_SIZE,
    PHASE_C_CAPTURE_TICKS,
    PHASE_C_FIELDS,
    PHASE_C_NATIVE_BRANCHES,
    build_phase_c_identity,
    finalize_phase_c_cache,
    freeze_phase_c_corpus,
    phase_c_chunk_paths,
    phase_c_frame_arrays,
    validate_phase_c_chunk,
    verify_phase_c_cache,
    write_phase_c_chunk,
)
from rivalsim.v03_phase_c_corpus import (
    PHASE_C_CASE_COUNT,
    generate_phase_c_cases,
    phase_c_cases_to_state,
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
        raise RuntimeError(
            f"unexpected collision geometry: {geometry.content_sha256}, "
            f"expected {EXPECTED_SOCCAR_CMF_SHA256}"
        )
    cases = generate_phase_c_cases()
    if len(cases) != PHASE_C_CASE_COUNT:
        raise RuntimeError(f"unexpected Phase C corpus size: {len(cases)}")
    identity = build_phase_c_identity(geometry, cases)
    cache_dir = phase_cache_dir(args.cache_root, identity)
    frozen = freeze_phase_c_corpus(cache_dir, identity, cases)
    if args.worker_start is not None:
        diagnostic = _activate_order_diagnostic(args.order_diagnostic_extension)
        return _run_worker(
            args,
            cache_dir,
            identity,
            cases,
            geometry,
            diagnostic,
        )
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
        manifest = verify_phase_c_cache(args.cache_root, identity, cases)
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
    for chunk_number, start in enumerate(
        range(0, len(cases), PHASE_C_CACHE_CHUNK_SIZE)
    ):
        stop = min(start + PHASE_C_CACHE_CHUNK_SIZE, len(cases))
        chunk_cases = cases[start:stop]
        case_ids = tuple(case.case_id for case in chunk_cases)
        npz_path, meta_path = phase_c_chunk_paths(cache_dir, start, stop)
        if npz_path.exists() and meta_path.exists():
            try:
                validate_phase_c_chunk(cache_dir, identity, start, case_ids)
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
        worker_command = [
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
            completed = subprocess.run(
                worker_command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                print(completed.stdout, end="", flush=True)
                break
            stderr_lines = completed.stderr.strip().splitlines()
            print(
                json.dumps(
                    {
                        "range": [start, stop],
                        "status": "discarded_worker_process_without_complete_branches",
                        "process_attempt": process_attempt,
                        "diagnostic": stderr_lines[-1] if stderr_lines else "worker failed",
                    }
                ),
                flush=True,
            )
        else:
            raise RuntimeError(
                f"Phase C source did not yield both branches for chunk {start}:{stop} "
                "across 16 fresh worker processes"
            )
    manifest = finalize_phase_c_cache(cache_dir, identity, cases, frozen)
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
    """Generate one chunk before this fresh native process has owned an arena."""

    start = args.worker_start
    if start < 0 or start >= len(cases) or start % PHASE_C_CACHE_CHUNK_SIZE:
        raise ValueError(f"invalid Phase C worker start: {start}")
    stop = min(start + PHASE_C_CACHE_CHUNK_SIZE, len(cases))
    chunk_cases = cases[start:stop]
    case_ids = tuple(case.case_id for case in chunk_cases)
    chunk_started = time.perf_counter()
    initial_readbacks: list[CarCarBatchOracleFrame] = []
    branch_frames: list[list[CarCarBatchOracleFrame]] = []
    branch_construction: dict[str, object] = {}
    for branch_index, branch in enumerate(PHASE_C_NATIVE_BRANCHES):
        initial_case_frames: list[CarCarBatchOracleFrame] = []
        tick_case_frames: list[list[CarCarBatchOracleFrame]] = [
            [] for _tick in PHASE_C_CAPTURE_TICKS
        ]
        attempts = np.empty(len(chunk_cases), dtype=np.int32)
        for local_index, case in enumerate(chunk_cases):
            oracle = RocketSimCarCarBatchOracle(
                phase_c_cases_to_state((case,)),
                str(geometry.source_root),
                pre_tick_visit_order=branch,
            )
            if int(oracle.pre_tick_first_car[0]) != branch_index:
                raise RuntimeError(f"native Phase C branch label mismatch: {branch}")
            attempts[local_index] = oracle.construction_attempts[0]
            initial_case_frames.append(oracle.frame())
            for tick_index, _tick in enumerate(PHASE_C_CAPTURE_TICKS):
                oracle.step()
                tick_case_frames[tick_index].append(oracle.frame())
            del oracle
            if local_index % 16 == 15:
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
    arrays = phase_c_frame_arrays(initial_readbacks, branch_frames)
    if args.verify_existing:
        npz_path, _meta_path = phase_c_chunk_paths(cache_dir, start, stop)
        with np.load(npz_path, allow_pickle=False) as expected:
            for field, values in arrays.items():
                if not np.array_equal(values, expected[field]):
                    raise RuntimeError(
                        f"fresh-process Phase C replay mismatch at {start}:{stop}, "
                        f"field {field}"
                    )
        status = "fresh_process_replay_exact"
    else:
        write_phase_c_chunk(
            cache_dir,
            identity,
            start,
            case_ids,
            arrays,
            branch_construction,
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
    cache_dir: Path,
    cases: tuple[object, ...],
    geometry: ArenaGeometry,
) -> dict[str, object]:
    """Prove each labeled native branch replays exactly across constructions."""

    requested_indices = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        464,
        472,
        1185,
        1665,
        2241,
        3122,
        3595,
        3763,
        3995,
        4563,
        6373,
        6565,
        7766,
    )
    sample_indices = tuple(index for index in requested_indices if index < len(cases))
    sample_count = len(sample_indices)
    sample = tuple(cases[index] for index in sample_indices)
    compared_frames = 0
    branch_attempts: dict[str, object] = {}
    for branch in PHASE_C_NATIVE_BRANCHES:
        first_total = 0
        second_total = 0
        for case in sample:
            first = RocketSimCarCarBatchOracle(
                phase_c_cases_to_state((case,)),
                str(geometry.source_root),
                pre_tick_visit_order=branch,
            )
            first_total += int(first.construction_attempts.sum())
            first_frames = [first.frame()]
            for _tick in PHASE_C_CAPTURE_TICKS:
                first.step()
                first_frames.append(first.frame())
            del first
            second = RocketSimCarCarBatchOracle(
                phase_c_cases_to_state((case,)),
                str(geometry.source_root),
                pre_tick_visit_order=branch,
            )
            second_total += int(second.construction_attempts.sum())
            second_frames = [second.frame()]
            for _tick in PHASE_C_CAPTURE_TICKS:
                second.step()
                second_frames.append(second.frame())
            del second
            for capture_index, (left_frame, right_frame) in enumerate(
                zip(first_frames, second_frames, strict=True)
            ):
                for field in PHASE_C_FIELDS:
                    if not np.array_equal(
                        getattr(left_frame, field), getattr(right_frame, field)
                    ):
                        raise RuntimeError(
                            "Phase C labeled native branch replay mismatch at "
                            f"branch {branch}, capture {capture_index}, field {field}"
                        )
                compared_frames += 1
        branch_attempts[branch] = {
            "first_total": first_total,
            "second_total": second_total,
        }
    result: dict[str, object] = {
        "status": "EXACT_SOURCE_BRANCH_REPLAY",
        "case_count": sample_count,
        "case_indices": list(sample_indices),
        "native_branches": list(PHASE_C_NATIVE_BRANCHES),
        "branch_construction_attempts": branch_attempts,
        "captures_per_case": len(PHASE_C_CAPTURE_TICKS) + 1,
        "compared_frames": compared_frames,
        "fields": list(PHASE_C_FIELDS),
    }
    path = cache_dir / "branch-determinism.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _concatenate_frames(
    frames: list[CarCarBatchOracleFrame],
) -> CarCarBatchOracleFrame:
    if not frames:
        raise ValueError("cannot concatenate an empty Phase C frame sequence")
    return CarCarBatchOracleFrame(
        **{
            field: np.ascontiguousarray(
                np.concatenate([getattr(frame, field) for frame in frames], axis=0)
            )
            for field in PHASE_C_FIELDS
        }
    )


def _validate_order_diagnostic(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing Phase C logical-order diagnostic: {resolved}")
    from rivalsim.v03_oracle_cache import sha256_file

    digest = sha256_file(resolved)
    if digest != EXPECTED_PHASE_C_ORDER_DIAGNOSTIC_SHA256:
        raise RuntimeError(
            f"Phase C logical-order diagnostic hash mismatch: {digest}, "
            f"expected {EXPECTED_PHASE_C_ORDER_DIAGNOSTIC_SHA256}"
        )
    return {"path": str(resolved), "sha256": digest, "returns": "logical_car_ids_only"}


def _activate_order_diagnostic(path: Path) -> dict[str, str]:
    result = _validate_order_diagnostic(path)
    if "RocketSim" in sys.modules:
        raise RuntimeError("RocketSim was imported before activating the order diagnostic")
    sys.path.insert(0, str(path.resolve().parent))
    import RocketSim

    if Path(RocketSim.__file__).resolve() != path.resolve():
        raise RuntimeError("failed to activate the pinned Phase C order diagnostic")
    if not hasattr(RocketSim.Arena, "_get_pre_tick_visit_order"):
        raise RuntimeError("Phase C diagnostic lacks logical-order readback")
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
        "phase": "C_car_car",
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
