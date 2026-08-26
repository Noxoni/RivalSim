"""Assemble compact RivalSim v0.5 evidence from completed local gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.5"
HANDOFF_PARENT = "dbc4b2bebe802bed58c9e143c1f9bcdb61189ac4"
V04_RELEASE = "8a422a86c69f16f0d62073992e515575f88733b5"
V04_IMPLEMENTATION = "da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56"
PRIOR_RESULT_DIRS = tuple(
    f"results/{version}" for version in ("v0.1", "v0.2", "v0.2.1", "v0.2.2", "v0.3", "v0.4")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", type=Path, default=ROOT / ".tools/v0.5/acceptance_raw.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / ".tools/v0.5/benchmark_raw.json")
    parser.add_argument(
        "--v04-benchmark", type=Path, default=ROOT / ".tools/v0.5/v04-benchmark.json"
    )
    parser.add_argument(
        "--v04-lifecycle", type=Path, default=ROOT / ".tools/v0.5/lifecycle-gate.json"
    )
    parser.add_argument("--repository-tests", type=int, required=True)
    parser.add_argument("--ruff-pass", action="store_true", required=True)
    parser.add_argument("--compile-pass", action="store_true", required=True)
    parser.add_argument("--diff-check-pass", action="store_true", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, value: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def _prior_evidence() -> dict[str, Any]:
    changed = str(_git("diff", "--name-only", V04_RELEASE, "--", *PRIOR_RESULT_DIRS)).splitlines()
    if changed:
        raise RuntimeError(f"prior published evidence changed: {changed}")
    trees: dict[str, Any] = {}
    for directory in PRIOR_RESULT_DIRS:
        listing = _git("ls-tree", "-r", "--full-tree", V04_RELEASE, "--", directory, text=False)
        assert isinstance(listing, bytes)
        trees[directory] = {
            "file_count": len(listing.splitlines()),
            "git_tree_listing_sha256": _sha256_bytes(listing),
        }
    return {
        "v0_4_release_commit": V04_RELEASE,
        "v0_5_handoff_parent": HANDOFF_PARENT,
        "byte_diff_count": 0,
        "trees": trees,
    }


def _phase_summary(path: Path, letter: str) -> dict[str, Any]:
    value = _read(path)
    verdict = value.get("classification", value.get("status"))
    selected = value.get("selection", {}).get("selected_case_count", value.get("case_count"))
    if letter == "c":
        failures = value["blocking"]["failed_case_count"]
    elif letter == "d":
        failures = value["gate"]["relation_failure_count"]
    else:
        failures = value["counts"]["failed_cases"]
    if verdict != "PASS_GREEN" or failures != 0:
        raise RuntimeError(f"v0.3 Phase {letter.upper()} is not green")
    return {
        "verdict": verdict,
        "cases": selected,
        "failed_cases": failures,
        "authority_identity_sha256": value.get(
            "authority_identity_sha256", value.get("authority", {}).get("identity_sha256")
        ),
        "source_sha256": _sha256(path),
    }


def main() -> int:
    args = parse_args()
    acceptance = _read(args.acceptance)
    benchmark = _read(args.benchmark)
    v04_benchmark = _read(args.v04_benchmark)
    lifecycle = _read(args.v04_lifecycle)
    v022_path = ROOT / ".tools/v0.5/regression-v022-full/aggregate.json"
    v01_path = ROOT / ".tools/v0.5/regression-v01.json"
    v022 = _read(v022_path)
    v01 = _read(v01_path)

    if acceptance["verdict"] != "PASS_GREEN":
        raise RuntimeError("v0.5 correctness acceptance is not green")
    if benchmark["verdict"] != "PASS_GREEN":
        raise RuntimeError("v0.5 benchmark is not green")
    if v04_benchmark["summary"]["verdict"] != "PASS_GREEN":
        raise RuntimeError("inherited v0.4 benchmark/regression aggregate is not green")
    if lifecycle["verdict"] != "PASS_GREEN":
        raise RuntimeError("inherited v0.4 lifecycle gate is not green")
    if v022["gate"]["complete_v022_gate_pass"] is not True:
        raise RuntimeError("inherited v0.2.2 gate is not green")
    if v01["summary"]["basic_parity_pass"] is not True:
        raise RuntimeError("inherited v0.1 gate is not green")
    if not (args.ruff_pass and args.compile_pass and args.diff_check_pass):
        raise RuntimeError("repository quality gates were not declared green")

    generated = datetime.now(UTC).isoformat()
    common = {
        "schema_version": 1,
        "milestone": "v0.5",
        "created_utc": generated,
        "verdict": "PASS_GREEN",
        "contract_hashes": acceptance["contract_hashes"],
        "source_acceptance_sha256": _sha256(args.acceptance),
    }
    gates = acceptance["gates"]
    _write("tensor_bridge.json", {**common, "gate": gates["tensor_bridge"]})
    _write("observation.json", {**common, "gate": gates["observation"]})
    _write("action_distribution.json", {**common, "gate": gates["action_distribution"]})
    _write(
        "reward_episode.json",
        {
            **common,
            "reward_episode_gate": gates["reward_episode"],
            "mechanics4_cadence_gate": gates["mechanics4_cadence"],
        },
    )
    _write(
        "rollout_gae.json",
        {
            **common,
            "rollout_buffer_gate": gates["rollout_buffer"],
            "gae_gate": gates["gae"],
        },
    )
    _write(
        "ppo.json",
        {
            **common,
            "policy_config": acceptance["policy_config"],
            "policy_config_hash": acceptance["policy_config_hash"],
            "parameter_count": 626_190,
            "gate": gates["ppo"],
        },
    )
    _write("checkpoint_resume.json", {**common, "gate": gates["checkpoint_resume"]})
    _write("self_play.json", {**common, "gate": gates["self_play"]})
    _write("learning_smoke.json", {**common, "gate": gates["learning_smoke"]})
    _write(
        "benchmark.json",
        {
            **common,
            "source_benchmark_sha256": _sha256(args.benchmark),
            "workload": benchmark["workload"],
            "environment": benchmark["environment"],
            "ppo_config": benchmark["ppo_config"],
            "points": benchmark["points"],
            "boundaries": benchmark["boundaries"],
            "selected_worlds": benchmark["selected_worlds"],
            "selected_point": benchmark["selected_point"],
            "stability_cv_limit": benchmark["stability_cv_limit"],
        },
    )

    phases = {
        f"phase_{letter}": _phase_summary(
            ROOT / f".tools/v0.5/regression-v03-phase-{letter}.json", letter
        )
        for letter in "abcd"
    }
    regression = {
        **common,
        "prior_published_evidence": _prior_evidence(),
        "v0_4": {
            "implementation_commit": V04_IMPLEMENTATION,
            "release_commit": V04_RELEASE,
            "authority_identity_sha256": lifecycle["authority_identity"],
            "lifecycle_verdict": lifecycle["verdict"],
            "mixed_lifecycle_stress": lifecycle["phase_d"],
            "complete_path": v04_benchmark["summary"],
            "ray_backends": v04_benchmark["geometry_query_gate"],
            "source_lifecycle_sha256": _sha256(args.v04_lifecycle),
            "source_benchmark_sha256": _sha256(args.v04_benchmark),
        },
        "v0_3": phases,
        "v0_2_2": {
            "verdict": v022["gate"]["classification"],
            "cases": v022["counts"]["selected_starting_states"],
            "hard_mismatch_events": v022["counts"]["hard_mismatch_events"],
            "numeric_failure_events": v022["counts"]["numeric_failure_events"],
            "failed_cases": v022["counts"]["failed_cases"],
            "source_sha256": _sha256(v022_path),
        },
        "v0_1": {
            "verdict": "PASS_GREEN",
            "scenarios": v01["summary"]["scenario_count"],
            "basic_parity_pass": v01["summary"]["basic_parity_pass"],
            "source_sha256": _sha256(v01_path),
        },
        "repository": {
            "pytest_passed": args.repository_tests,
            "ruff_pass": args.ruff_pass,
            "compileall_pass": args.compile_pass,
            "git_diff_check_pass": args.diff_check_pass,
        },
        "v0_6_begun": False,
    }
    _write("regression.json", regression)
    print(RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
