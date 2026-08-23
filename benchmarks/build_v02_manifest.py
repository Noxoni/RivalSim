"""Build the compact, machine-verifiable RivalSim v0.2 evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
START_COMMIT = "7a6a6913fad6ceedd92d1170b373a0978edb05b6"
FROZEN_V01_COMMIT = "1f7a36cc6165273fb658ba07a8458e8d8e60628a"
IMPLEMENTATION_COMMIT = "f2363104a56a358276682e16110d16f37e8d0539"
PREFREEZE_PARITY_SHA256 = "7EB62CF97BE25EA5F7CF6540D9D6350829B0AE7887B09F0B63E3915E937B9BDF"
PREFREEZE_PARITY_SIZE = 170_872


def _run_git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _path_record(path: Path, *, canonical_lf: bool = False) -> dict[str, Any]:
    data = path.read_bytes()
    if canonical_lf:
        data = data.replace(b"\r\n", b"\n")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": len(data),
        "sha256": _sha256(data),
    }


def _git_blob_record(commit: str, path: str) -> dict[str, Any]:
    data = _run_git("show", f"{commit}:{path}")
    return {"path": path, "size_bytes": len(data), "sha256": _sha256(data)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dependency_versions(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        result[name] = version
    return result


def _implementation_blobs() -> list[dict[str, Any]]:
    names = _run_git(
        "diff", "--name-only", f"{START_COMMIT}..{IMPLEMENTATION_COMMIT}"
    ).decode("utf-8").splitlines()
    return [_git_blob_record(IMPLEMENTATION_COMMIT, name) for name in names]


def _v01_regression_record(path: Path, command: str) -> dict[str, Any]:
    record = _path_record(path)
    payload = _load_json(path)
    summary = payload["summary"]
    if "basic_parity_pass" in summary:
        compact_summary = {
            key: summary[key]
            for key in (
                "scenario_count",
                "same_equation_pass",
                "rocketsim_pass",
                "axis_sign_pass",
                "basic_parity_pass",
            )
        }
    else:
        compact_summary = {
            key: summary[key]
            for key in (
                "best_gpu_worlds",
                "best_gpu_aggregate_simulated_game_seconds_per_s",
                "best_cpu_worlds",
                "best_cpu_aggregate_simulated_game_seconds_per_s",
                "same_equation_gpu_speedup",
                "performance_conditions",
                "performance_gate_pass",
            )
        }
    record.update(
        {
            "tracked": False,
            "purpose": "post-v0.2 regression run; compact hash and summary only",
            "command": command,
            "summary": compact_summary,
        }
    )
    return record


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_path = ROOT / "results" / "v0.2" / "benchmark.json"
    parity_path = ROOT / "results" / "v0.2" / "parity.json"
    requirements_path = ROOT / "requirements-v0.2.txt"
    benchmark = _load_json(benchmark_path)
    parity = _load_json(parity_path)

    benchmark_record = _path_record(benchmark_path, canonical_lf=True)
    benchmark_record.update(
        {
            "created_utc": benchmark["created_utc"],
            "reproduce": (
                "python benchmarks/run_v02_benchmark.py --collision-dir "
                "$env:RIVALSIM_COLLISION_DIR --output <output.json> --repeats 5 "
                "--warmup-ticks 16 --graph-block-ticks 8 --ticks 64 "
                "--max-worlds 262144 --seed 20260823"
            ),
        }
    )
    parity_record = _path_record(parity_path, canonical_lf=True)
    parity_record.update(
        {
            "created_utc": parity["generated_at_utc"],
            "reproduce": (
                "python benchmarks/run_v02_parity.py --collision-dir "
                "$env:RIVALSIM_COLLISION_DIR --mode gate --output <output.json>"
            ),
        }
    )

    frozen_paths = [
        "results/v0.1/benchmark.json",
        "results/v0.1/manifest.json",
        "results/v0.1/parity.json",
    ]
    frozen_records = [_git_blob_record(FROZEN_V01_COMMIT, path) for path in frozen_paths]
    frozen_unchanged = all(
        _run_git("show", f"{FROZEN_V01_COMMIT}:{record['path']}")
        == _run_git("show", f"{IMPLEMENTATION_COMMIT}:{record['path']}")
        for record in frozen_records
    )

    tracked_assets = _run_git("ls-files", "--", "*.cmf", "*.pskx", "*.bin").decode(
        "utf-8"
    ).splitlines()
    source_root = Path(benchmark["arena"]["source_root"]).resolve()
    repository_root = ROOT.resolve()

    regressions = []
    parity_regression = ROOT / ".tools" / "v0.2-final" / "v0.1-parity.json"
    if parity_regression.exists():
        regressions.append(
            _v01_regression_record(
                parity_regression,
                (
                    "python benchmarks/run_parity.py --device cuda:0 "
                    "--output .tools/v0.2-final/v0.1-parity.json"
                ),
            )
        )
    benchmark_regression = ROOT / ".tools" / "v0.2-final" / "v0.1-benchmark.json"
    if benchmark_regression.exists():
        regressions.append(
            _v01_regression_record(
                benchmark_regression,
                (
                    "python benchmarks/run_benchmark.py --device cuda:0 --gpu-ticks 16 "
                    "--cpu-ticks 2 --gpu-repeats 3 --cpu-repeats 3 --warmup-ticks 8 "
                    "--graph-block-ticks 8 --max-worlds 16384 --seed 20260823 "
                    "--output .tools/v0.2-final/v0.1-benchmark.json"
                ),
            )
        )

    repository: dict[str, Any] = {
        "origin": _run_git("remote", "get-url", "origin").decode("utf-8").strip(),
        "branch": "main",
        "authority_start_commit": START_COMMIT,
        "frozen_v0_1_commit": FROZEN_V01_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "evidence_parent_commit": IMPLEMENTATION_COMMIT,
    }
    if args.evidence_package_commit:
        repository["evidence_package_commit"] = args.evidence_package_commit

    return {
        "schema_version": 1,
        "milestone": "v0.2",
        "manifest_created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PAUSE_RED",
        "repository": repository,
        "environment": {
            **benchmark["environment"],
            "cpu_model": "AMD Ryzen 7 9800X3D",
            "gpu_arch": "sm_120",
            "gpu_sm_count": 170,
            "cuda_driver_api": "13.3",
            "installed_cuda_compiler_toolkit": "13.3",
            "nvcc": "V13.3.73",
            "warp_bundled_cuda_toolkit": "12.9",
            "warp_mempool_enabled": True,
        },
        "resolved_dependencies": _dependency_versions(requirements_path),
        "dependency_lock": _path_record(requirements_path, canonical_lf=True),
        "source_custody": {
            "rocketsim_primary": {
                "repository": "https://github.com/ZealanL/RocketSim.git",
                "commit": "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02",
            },
            "rocketsim_python_binding": {
                "repository": "https://github.com/mtheall/RocketSim.git",
                "commit": "2da51b1dac7b8127127613a5ff30e490bdd70dd8",
                "package_version": "2.2.1",
                "installed_extension_sha256": (
                    "E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0"
                ),
            },
            "collision_asset_repository": {
                "repository": "https://github.com/Noxoni/Rival.git",
                "source_commit": "36cb14cf645c4f06b668c34d85ce1a500e4b53da",
                "asset_introducing_commit": "4f2b21c00e2fcb7108ab1006fd950b066fbd0484",
                "source_relative_directory": "bot/collision_meshes/soccar",
                "measured_local_source_root": str(source_root),
                "checkout_was_read_only": True,
            },
            "collision_dumper": {
                "repository": "https://github.com/ZealanL/RLArenaCollisionDumper.git",
                "invoked_for_this_run": False,
                "reason": "the exact RocketSim-oracle CMF set was already available",
            },
            "license_notices": "THIRD_PARTY_NOTICES.md",
        },
        "collision_assets": benchmark["arena"],
        "asset_audit": {
            "tracked_extracted_asset_paths": tracked_assets,
            "tracked_extracted_asset_count": len(tracked_assets),
            "measured_source_root_is_outside_repository": (
                source_root != repository_root and repository_root not in source_root.parents
            ),
            "policy": "external read-only input; hashes and metadata only",
        },
        "implementation_blobs": _implementation_blobs(),
        "tracked_evidence": [benchmark_record, parity_record],
        "ignored_local_measurement": {
            "path": ".tools/v0.2-parity-measurement.json",
            "size_bytes": PREFREEZE_PARITY_SIZE,
            "sha256": PREFREEZE_PARITY_SHA256,
            "purpose": "measurement-only pass created before the tolerance table was frozen",
        },
        "frozen_v0_1_evidence": {
            "unchanged_at_implementation_commit": frozen_unchanged,
            "files": frozen_records,
        },
        "post_v0_2_v0_1_regressions": regressions,
        "gates": {
            "geometry_query": benchmark["geometry_query_gate"],
            "performance": benchmark["summary"],
            "parity": {
                "scenario_count": parity["scenario_count"],
                "families": parity["families"],
                "horizons_ticks": parity["horizons_ticks"],
                "measurement_aggregates": parity["measurement_aggregates"],
                "frozen_tolerances": parity["frozen_tolerances"],
                "summary": parity["summary"],
            },
            "stress": benchmark["stress_gate"],
        },
        "validation": {
            "pytest": args.pytest_summary,
            "ruff": args.ruff,
            "compileall": args.compileall,
            "json_parse": args.json_parse,
            "git_diff_check": args.git_diff_check,
            "asset_and_path_audit": "passed" if not tracked_assets else "failed",
            "frozen_v0_1_diff": "empty" if frozen_unchanged else "changed",
            "pip_check": {
                "status": "known upstream wheel metadata warning",
                "message": "rocketsim 2.2.1 is not supported on this platform",
                "explanation": (
                    "The cp36-abi3 Windows extension imports on Python 3.14, but the wheel's "
                    "internal metadata advertises a conflicting tag. Live oracle tests pass; "
                    "installed metadata was not modified."
                ),
            },
        },
        "scope_boundary": {
            "v0_2_complete": True,
            "v0_3_begun": False,
            "excluded": [
                "ball-world collision",
                "car-ball collision",
                "car-car collision",
                "boost pads",
                "scoring and game rules",
                "RLGym/PPO/training integration",
                "Rival policy inference",
            ],
        },
        "self_hash_policy": "manifest.json intentionally does not hash itself",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/v0.2/manifest.json")
    parser.add_argument("--evidence-package-commit")
    parser.add_argument("--pytest-summary", default="pending final validation")
    parser.add_argument("--ruff", default="pending final validation")
    parser.add_argument("--compileall", default="pending final validation")
    parser.add_argument("--json-parse", default="pending final validation")
    parser.add_argument("--git-diff-check", default="pending final validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "verdict": manifest["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
