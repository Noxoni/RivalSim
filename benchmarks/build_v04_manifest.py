"""Bind the completed v0.4 implementation and staged compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_BASELINE = "b5875c4b853a8ce844d0904e989b1d2a3854d0ac"
V03_RELEASE = "d6ca3912418a3dd7ca8979415142cd861e0c0ddb"
V03_IMPLEMENTATION = "a63d317b0de0522e6d3cbe243bf282c6b93a9d58"

RELEASE_FILES = [
    "CODEX_START_PROMPT.md",
    "VERSION.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "docs/ROADMAP.md",
    "docs/V0_4_RESULTS.md",
    "docs/REPRODUCING_V0_4.md",
    "docs/V0_4_AUTHORITY.md",
    "results/v0.4/boost_pads.json",
    "results/v0.4/goals_kickoff.json",
    "results/v0.4/demolition_respawn.json",
    "results/v0.4/match_lifecycle.json",
    "results/v0.4/oracle_data.json",
    "results/v0.4/rules_source.json",
    "results/v0.4/regression.json",
    "results/v0.4/benchmark.json",
]
PRIOR_MANIFESTS = [
    "results/v0.1/manifest.json",
    "results/v0.2/manifest.json",
    "results/v0.2.1/manifest.json",
    "results/v0.2.2/manifest.json",
    "results/v0.3/manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/v0.4/manifest.json")
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
    names = str(_git("diff", "--name-only", AUTHORIZED_BASELINE, commit)).splitlines()
    prefixes = ("benchmarks/", "rivalsim/", "tests/", "tools/")
    selected = [
        path
        for path in names
        if path.startswith(prefixes) or path in {"pyproject.toml", "requirements-v0.2.txt"}
    ]
    return [_committed_entry(commit, path) for path in sorted(selected)]


def _validate_release() -> None:
    boost = _json("results/v0.4/boost_pads.json")
    goals = _json("results/v0.4/goals_kickoff.json")
    demo = _json("results/v0.4/demolition_respawn.json")
    lifecycle = _json("results/v0.4/match_lifecycle.json")
    oracle = _json("results/v0.4/oracle_data.json")
    rules = _json("results/v0.4/rules_source.json")
    regression = _json("results/v0.4/regression.json")
    benchmark = _json("results/v0.4/benchmark.json")
    assert boost["gate"]["verdict"] == "PASS_GREEN"
    assert goals["gate"]["verdict"] == "PASS_GREEN"
    assert demo["gate"]["verdict"] == "PASS_GREEN"
    assert lifecycle["gate"]["verdict"] == "PASS_GREEN"
    assert oracle["status"] == "COMPLETE_NATIVE_AUTHORITY"
    assert oracle["cache"]["complete"] is True
    assert oracle["cache"]["live_fallback_after_freeze"] is False
    assert rules["policy_boundary"]["v0_5_begun"] is False
    assert regression["status"] == "PASS_GREEN"
    assert regression["prior_published_evidence"]["byte_diff_count"] == 0
    assert regression["repository_checks"]["pytest_passed"] >= 70
    assert benchmark["summary"]["verdict"] == "PASS_GREEN"
    assert benchmark["summary"]["performance_gate_pass"] is True
    assert benchmark["summary"]["best_aggregate_simulated_game_seconds_per_s"] >= 100_000
    assert benchmark["summary"]["hot_loop_gpu_resident"] is True


def main() -> int:
    args = parse_args()
    implementation = str(_git("rev-parse", args.implementation_commit)).strip()
    head = str(_git("rev-parse", "HEAD")).strip()
    if implementation != head:
        raise RuntimeError("implementation commit must be current HEAD before evidence commit")
    if 'version = "0.4.0"' not in (ROOT / "pyproject.toml").read_text(encoding="utf-8"):
        raise RuntimeError("package version is not 0.4.0")
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ]
    if missing:
        raise RuntimeError(f"release files must be staged or committed: {missing}")

    oracle = _json("results/v0.4/oracle_data.json")
    regression = _json("results/v0.4/regression.json")
    benchmark = _json("results/v0.4/benchmark.json")
    manifest = {
        "schema_version": 1,
        "milestone": "v0.4",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS_GREEN",
        "authorized_baseline_commit": AUTHORIZED_BASELINE,
        "v0_3_release_commit": V03_RELEASE,
        "v0_3_implementation_commit": V03_IMPLEMENTATION,
        "v0_4_implementation_commit": implementation,
        "authority_identity_sha256": oracle["authority_identity_sha256"],
        "authority_inputs": {
            "rocket_sim_primary_commit": oracle["identity_inputs"]["rocket_sim_primary_commit"],
            "rocket_sim_binding_commit": oracle["identity_inputs"]["rocket_sim_binding_commit"],
            "rocket_sim_extension": oracle["identity_inputs"]["rocket_sim_extension"],
            "collision_assets": oracle["identity_inputs"]["collision_assets"],
            "generator": oracle["identity_inputs"]["generator"],
            "corpus_sha256": oracle["derived_hashes"]["corpus_config_sha256"],
            "authority_settings_sha256": oracle["derived_hashes"]["authority_settings_sha256"],
            "bounded_contract_sha256": oracle["derived_hashes"]["bounded_contract_sha256"],
            "authority_artifact_sha256": oracle["cache"]["authority_sha256"],
            "live_fallback_after_freeze": False,
        },
        "release_files": [_release_entry(implementation, path, staged) for path in RELEASE_FILES],
        "prior_evidence_manifests": [
            _committed_entry(implementation, path) for path in PRIOR_MANIFESTS
        ],
        "prior_evidence_trees": regression["prior_published_evidence"]["trees"],
        "committed_implementation_files": _implementation_entries(implementation),
        "gates": {
            "boost_pad_pickup_cases": 68,
            "goal_boundary_cases": 6,
            "kickoff_layouts": 5,
            "demo_respawn_team_location_cases": 8,
            "demo_respawn_tick": 360,
            "v0_3_phase_a_cases": 31_216,
            "v0_3_phase_b_cases": 8_192,
            "v0_3_phase_c_cases": 8_192,
            "v0_3_phase_d_cases": 512,
            "v0_2_2_static_cases": 39_236,
            "v0_1_scenarios": 27,
            "arena_rays_per_backend": 4_608,
            "repository_tests": regression["repository_checks"]["pytest_passed"],
            "deterministic_mixed_lifecycle": True,
            "best_complete_simulated_game_seconds_per_s": benchmark["summary"][
                "best_aggregate_simulated_game_seconds_per_s"
            ],
            "performance_floor_simulated_game_seconds_per_s": 100_000.0,
            "reset_heavy_simulated_game_seconds_per_s": benchmark["summary"][
                "reset_heavy_sim_s_per_s"
            ],
            "reset_transitions_per_s": benchmark["summary"]["reset_transitions_per_s"],
            "hot_loop_gpu_resident": True,
            "zero_timed_transfers": True,
            "prior_published_evidence_byte_diff_count": 0,
            "v0_5_begun": False,
        },
        "scope_boundary": {
            "world": "exactly two Octanes, one standard Soccar ball, static Soccar arena",
            "complete_headless_game_transition": True,
            "raw_policy_neutral_terminal_outputs": True,
            "observation_or_reward_construction": False,
            "training_or_rollout_integration": False,
            "arbitrary_body_counts_or_game_modes": False,
            "generic_bullet_api": False,
            "v0_5_begun": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.output.resolve()} sha256={_sha256_bytes(args.output.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
