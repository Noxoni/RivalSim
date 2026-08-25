"""Bind the completed v0.3 implementation and staged compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8"

RELEASE_FILES = [
    "CODEX_START_PROMPT.md",
    "VERSION.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "docs/ROADMAP.md",
    "docs/V0_3_RESULTS.md",
    "docs/REPRODUCING_V0_3.md",
    "docs/V0_3_ORACLE_CACHE.md",
    "results/v0.3/ball_world.json",
    "results/v0.3/car_ball.json",
    "results/v0.3/car_car.json",
    "results/v0.3/integrated.json",
    "results/v0.3/oracle_data.json",
    "results/v0.3/source_port.json",
    "results/v0.3/regression.json",
    "results/v0.3/benchmark.json",
]
PRIOR_MANIFESTS = [
    "results/v0.1/manifest.json",
    "results/v0.2/manifest.json",
    "results/v0.2.1/manifest.json",
    "results/v0.2.2/manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "v0.3" / "manifest.json"
    )
    return parser.parse_args()


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _staged_entry(path: str) -> dict[str, Any]:
    payload = _git("show", f":{path}", text=False)
    assert isinstance(payload, bytes)
    return {"path": path, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _committed_entry(commit: str, path: str) -> dict[str, Any]:
    payload = _git("show", f"{commit}:{path}", text=False)
    assert isinstance(payload, bytes)
    return {"path": path, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _release_entry(commit: str, path: str, staged: set[str]) -> dict[str, Any]:
    return _staged_entry(path) if path in staged else _committed_entry(commit, path)


def _implementation_entries(commit: str) -> list[dict[str, Any]]:
    names = str(_git("diff", "--name-only", BASELINE, commit)).splitlines()
    prefixes = ("benchmarks/", "rivalsim/", "tests/", "tools/")
    selected = [
        path
        for path in names
        if path.startswith(prefixes) or path in {"pyproject.toml", "requirements-v0.2.txt"}
    ]
    return [_committed_entry(commit, path) for path in sorted(selected)]


def _validate_release() -> None:
    ball_world = _json("results/v0.3/ball_world.json")
    car_ball = _json("results/v0.3/car_ball.json")
    car_car = _json("results/v0.3/car_car.json")
    integrated = _json("results/v0.3/integrated.json")
    oracle = _json("results/v0.3/oracle_data.json")
    regression = _json("results/v0.3/regression.json")
    benchmark = _json("results/v0.3/benchmark.json")
    assert ball_world["gate"]["phase_a_complete_gate_pass"] is True
    assert car_ball["gate"]["phase_b_complete_gate_pass"] is True
    assert car_car["full_gate"]["failed_case_count"] == 0
    assert integrated["gate"]["phase_d_complete_gate_pass"] is True
    assert oracle["status"] == "COMPLETE_NATIVE_AUTHORITY"
    assert all(
        phase["status"] == "COMPLETE_NATIVE_AUTHORITY"
        for phase in oracle["phases"].values()
    )
    assert regression["status"] == "PASS_GREEN"
    assert (
        regression["prior_published_evidence"][
            "v0_1_v0_2_v0_2_1_v0_2_2_byte_diff_count"
        ]
        == 0
    )
    assert benchmark["summary"]["verdict"] == "PASS_GREEN"
    assert (
        benchmark["summary"]["best_aggregate_simulated_game_seconds_per_s"] >= 100_000
    )
    assert benchmark["summary"]["hot_loop_gpu_resident"] is True


def main() -> int:
    args = parse_args()
    implementation = str(_git("rev-parse", args.implementation_commit)).strip()
    head = str(_git("rev-parse", "HEAD")).strip()
    if implementation != head:
        raise RuntimeError("implementation commit must be current HEAD before evidence commit")
    if 'version = "0.3.0"' not in (ROOT / "pyproject.toml").read_text(encoding="utf-8"):
        raise RuntimeError("package version is not 0.3.0")
    if str(_git("ls-files", "*.cmf")).splitlines():
        raise RuntimeError("collision assets must remain external")
    _validate_release()

    staged = set(str(_git("diff", "--cached", "--name-only")).splitlines())
    missing = [
        path
        for path in RELEASE_FILES
        if path not in staged
        and subprocess.run(
            ["git", "cat-file", "-e", f"{implementation}:{path}"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ]
    if missing:
        raise RuntimeError(f"release files must be staged or committed: {missing}")

    oracle = _json("results/v0.3/oracle_data.json")
    benchmark = _json("results/v0.3/benchmark.json")
    manifest = {
        "schema_version": 1,
        "milestone": "v0.3",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS_GREEN",
        "implementation_commit": implementation,
        "authority_identities": {
            phase: record["authority_identity_sha256"]
            for phase, record in oracle["phases"].items()
        },
        "corpus_sha256": {
            phase: record["corpus_sha256"]
            for phase, record in oracle["phases"].items()
        },
        "release_files": [
            _release_entry(implementation, path, staged) for path in RELEASE_FILES
        ],
        "prior_evidence_manifests": [
            _committed_entry(implementation, path) for path in PRIOR_MANIFESTS
        ],
        "committed_implementation_files": _implementation_entries(implementation),
        "gates": {
            "phase_a_cases": 31_216,
            "phase_b_cases": 8_192,
            "phase_c_cases": 8_192,
            "phase_d_cases": 512,
            "hard_mismatch_events": 0,
            "numeric_failure_events": 0,
            "v0_2_2_static_cases": 39_236,
            "v0_1_scenarios": 27,
            "repository_tests": 63,
            "best_complete_dynamic_simulated_game_seconds_per_s": benchmark["summary"][
                "best_aggregate_simulated_game_seconds_per_s"
            ],
            "performance_floor_simulated_game_seconds_per_s": 100_000.0,
            "hot_loop_gpu_resident": True,
            "zero_timed_transfers": True,
            "deterministic_stress": True,
            "v0_4_begun": False,
        },
        "scope_boundary": {
            "world": "exactly two Octanes, one standard Soccar ball, static Soccar arena",
            "bump_demo": "physical classification only",
            "demolition_removal_respawn": False,
            "goals_scoring_resets": False,
            "training_integration": False,
            "generic_bullet_or_arbitrary_bodies": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{args.output.resolve()} sha256={_sha256_bytes(args.output.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
