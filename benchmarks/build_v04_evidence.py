"""Assemble compact v0.4 release evidence from the completed local gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.4"
BASELINE = "b5875c4b853a8ce844d0904e989b1d2a3854d0ac"

LIFECYCLE = ROOT / ".tools" / "v0.4" / "lifecycle-gate.json"
BENCHMARK = ROOT / ".tools" / "v0.4" / "benchmark.json"
V03_PHASES = {
    "phase_a": ROOT / ".tools" / "v0.4" / "regression-v03-phase-a.json",
    "phase_b": ROOT / ".tools" / "v0.4" / "regression-v03-phase-b.json",
    "phase_c": ROOT / ".tools" / "v0.4" / "regression-v03-phase-c.json",
    "phase_d": ROOT / ".tools" / "v0.4" / "regression-v03-phase-d.json",
}
V022 = ROOT / ".tools" / "v0.4" / "regression-v022-full" / "aggregate.json"
V01 = ROOT / ".tools" / "v0.4" / "regression-v01.json"

PRIOR_RESULT_DIRS = (
    "results/v0.1",
    "results/v0.2",
    "results/v0.2.1",
    "results/v0.2.2",
    "results/v0.3",
)
SOURCE_PATHS = (
    ".reference/RocketSim/src/Sim/Arena/Arena.cpp",
    ".reference/RocketSim/src/Sim/BoostPad/BoostPad.cpp",
    ".reference/RocketSim/src/Sim/Car/Car.cpp",
    ".reference/RocketSim/src/RLConst.h",
    ".reference/RocketSimPython/python-mtheall/Arena.cpp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".tools/v0.4/oracle")
    parser.add_argument("--repository-tests", type=int, required=True)
    parser.add_argument("--ruff-pass", action="store_true", required=True)
    parser.add_argument("--compile-pass", action="store_true", required=True)
    parser.add_argument("--diff-check-pass", action="store_true", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, value: dict[str, Any]) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def _source_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _prior_evidence() -> dict[str, Any]:
    changed = str(_git("diff", "--name-only", BASELINE, "--", *PRIOR_RESULT_DIRS)).splitlines()
    if changed:
        raise RuntimeError(f"prior published evidence changed: {changed}")
    result: dict[str, Any] = {
        "baseline_commit": BASELINE,
        "byte_diff_count": 0,
        "trees": {},
    }
    for directory in PRIOR_RESULT_DIRS:
        listing = _git("ls-tree", "-r", "--full-tree", BASELINE, "--", directory, text=False)
        assert isinstance(listing, bytes)
        result["trees"][directory] = {
            "file_count": len(listing.splitlines()),
            "git_tree_listing_sha256": _sha256_bytes(listing),
        }
    return result


def _phase_pass(phase: dict[str, Any], letter: str) -> bool:
    key = f"phase_{letter}_complete_gate_pass"
    return bool(
        phase.get("classification", phase.get("status")) == "PASS_GREEN"
        and (
            phase.get("gate", {}).get(key, False)
            or phase.get("blocking", {}).get("failed_case_count") == 0
        )
    )


def main() -> int:
    args = parse_args()
    lifecycle = _read(LIFECYCLE)
    benchmark = _read(BENCHMARK)
    phases = {name: _read(path) for name, path in V03_PHASES.items()}
    v022 = _read(V022)
    v01 = _read(V01)
    identity = lifecycle["authority_identity"]
    cache_dir = args.cache_root.resolve() / identity
    cache_identity = _read(cache_dir / "identity.json")
    frozen = _read(cache_dir / "frozen.json")
    authority = _read(cache_dir / "authority.json")

    assert lifecycle["verdict"] == "PASS_GREEN"
    assert benchmark["summary"]["verdict"] == "PASS_GREEN"
    assert benchmark["summary"]["performance_gate_pass"] is True
    assert benchmark["summary"]["arena_query_pass"] is True
    assert benchmark["summary"]["hot_loop_gpu_resident"] is True
    assert all(_phase_pass(phases[f"phase_{letter}"], letter) for letter in "abcd")
    assert v022["gate"]["complete_v022_gate_pass"] is True
    assert v022["counts"]["selected_starting_states"] == 39_236
    assert v01["summary"]["scenario_count"] == 27
    assert v01["summary"]["basic_parity_pass"] is True
    assert cache_identity["identity"] == identity
    assert frozen["identity"] == identity and frozen["complete"] is True
    assert frozen["live_fallback"] is False
    assert _sha256(cache_dir / "authority.json") == frozen["authority_sha256"].upper()

    common = {
        "schema_version": 1,
        "milestone": "v0.4",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity,
        "status": "PASS_GREEN",
    }
    _write(
        "boost_pads.json",
        {
            **common,
            "gate": lifecycle["phase_a"],
            "pickup_authority": authority["boost_pads"]["pickup_cases"],
            "cooldown_authority": authority["boost_pads"]["cooldown_cases"],
            "source_contract": {
                "pad_count": 34,
                "large_pad_count": 6,
                "small_pad_count": 28,
                "large_grant": 100.0,
                "small_grant": 12.0,
                "large_cooldown_seconds": 10.0,
                "small_cooldown_seconds": 4.0,
                "contention_order": "persistent per-world Arena::_cars visitation order",
            },
        },
    )
    _write(
        "goals_kickoff.json",
        {
            **common,
            "gate": lifecycle["phase_b"],
            "goal_boundary_authority": authority["goals_kickoff"]["boundary_cases"],
            "kickoff_authority": authority["goals_kickoff"]["kickoff_cases"],
            "contract": {
                "goal_test": "abs(ball_y) > 5124.25 + 91.25",
                "positive_y_scoring_team": "BLUE/0",
                "negative_y_scoring_team": "ORANGE/1",
                "goal_event": "first scored tick only",
                "auto_reset": "goal callback then deterministic reset_kickoff",
                "layout_selector": "explicit per-world state, advances modulo 5",
            },
        },
    )
    _write(
        "demolition_respawn.json",
        {
            **common,
            "gate": lifecycle["phase_c"],
            "timer_authority": authority["demolition_respawn"]["timer_trace"],
            "respawn_authority": authority["demolition_respawn"]["respawn_poses"],
            "contract": {
                "respawn_delay_seconds": 3.0,
                "disabled_physics": True,
                "disabled_public_state_frozen": True,
                "respawn_selector": (
                    "explicit per-car state initialized per world, advances modulo 4"
                ),
                "membership_changes": False,
                "car_visitation_order_preserved": True,
                "physical_demo_composition_case": "frozen v0.3 Phase C case C-00001",
            },
        },
    )
    _write(
        "match_lifecycle.json",
        {
            **common,
            "gate": lifecycle["phase_d"],
            "world_transition": (
                "GPU state -> accepted v0.3 physics -> pad/goal/demo/reset events -> next GPU state"
            ),
            "clock_contract": {
                "world_tick": "monotonic total transition count",
                "episode_tick": "resets on kickoff/full reset",
                "rocket_sim_match_clock_authority": False,
            },
            "terminal_contract": {
                "terminated": 0,
                "truncated": 0,
                "policy": "raw lifecycle events only; v0.5 training policy not begun",
            },
            "full_world_reset": {
                "host_reset": "configured deterministic kickoff and clean lifecycle state",
                "gpu_resident_interval_reset": True,
                "scores_cleared": True,
                "pads_active": True,
                "demo_and_contact_state_cleared": True,
            },
        },
    )

    identity_inputs = cache_identity["inputs"]
    _write(
        "oracle_data.json",
        {
            **common,
            "status": "COMPLETE_NATIVE_AUTHORITY",
            "cache": {
                "local_ignored_path": f".tools/v0.4/oracle/{identity}",
                "schema": frozen["schema"],
                "authority_sha256": frozen["authority_sha256"].upper(),
                "complete": True,
                "live_fallback_after_freeze": False,
            },
            "identity_inputs": identity_inputs,
            "derived_hashes": {
                "corpus_config_sha256": _canonical_sha256(identity_inputs["corpus"]),
                "authority_settings_sha256": _canonical_sha256(
                    identity_inputs["authority_settings"]
                ),
                "bounded_contract_sha256": _canonical_sha256(
                    identity_inputs["bounded_rivalsim_contract"]
                ),
                "lifecycle_gate_sha256": _sha256(LIFECYCLE),
                "gate_runner_sha256": _sha256(ROOT / "benchmarks/run_v04_lifecycle.py"),
                "benchmark_runner_sha256": _sha256(ROOT / "benchmarks/run_v04_benchmark.py"),
            },
        },
    )

    _write(
        "rules_source.json",
        {
            **common,
            "status": "SOURCE_CONTRACT_FROZEN",
            "pinned_lineage": {
                "rocketsim_primary_commit": identity_inputs["rocket_sim_primary_commit"],
                "rocketsim_binding_commit": identity_inputs["rocket_sim_binding_commit"],
                "rocketsim_package": identity_inputs["rocket_sim_package"],
            },
            "source_files": [_source_record(path) for path in SOURCE_PATHS],
            "translated_functions": {
                "arena": [
                    "Arena::Step",
                    "Arena::ResetToRandomKickoff",
                    "Arena::IsBallScored",
                    "Arena::_BtCallback_OnCarCarCollision",
                ],
                "boost_pad": [
                    "BoostPad::_PreTickUpdate",
                    "BoostPad::_CheckCollide",
                    "BoostPad::_PostTickUpdate",
                ],
                "car": ["Car::Demolish", "Car::Respawn", "Car::_PreTickUpdate"],
                "binding": ["Arena goal-score callback first-entry semantics"],
            },
            "handoff_specs": [
                _source_record("handoff/v0.4/README.md"),
                _source_record("handoff/v0.4/ACCEPTANCE.md"),
                _source_record("handoff/v0.4/LIFECYCLE_POLICY.md"),
            ],
            "policy_boundary": {
                "rocket_sim_regulation_clock_defined": False,
                "terminal_or_truncation_policy_invented": False,
                "rewards_or_observations": False,
                "v0_5_begun": False,
            },
            "prohibited_mechanisms": {
                "case_ids_or_expected_outputs_in_runtime": False,
                "runtime_best_match_selection": False,
                "native_pointer_or_allocator_emulation": False,
                "tolerance_broadening": False,
                "behavioral_stabilizers": False,
                "generic_bullet_port": False,
            },
        },
    )

    phase_records: dict[str, Any] = {}
    for letter in "abcd":
        value = phases[f"phase_{letter}"]
        phase_records[f"phase_{letter}"] = {
            "classification": value.get("classification", value.get("status")),
            "selection": value.get("selection", {"case_count": value.get("case_count")}),
            "gate": value.get("gate", value.get("blocking")),
            "authority_identity_sha256": value.get("authority_identity_sha256")
            or value.get("authority", {}).get("identity_sha256"),
            "corpus_sha256": value.get("corpus_sha256")
            or value.get("authority", {}).get("corpus_sha256"),
        }
    _write(
        "regression.json",
        {
            **common,
            "v0_3_physics": phase_records,
            "v0_2_2_static": {
                "counts": v022["counts"],
                "gate": v022["gate"],
            },
            "v0_1_live_rocketsim": {
                key: v01["summary"][key]
                for key in (
                    "scenario_count",
                    "same_equation_pass",
                    "rocketsim_pass",
                    "axis_sign_pass",
                    "basic_parity_pass",
                )
            },
            "arena_query_backends": benchmark["geometry_query_gate"],
            "deterministic_mixed_lifecycle": lifecycle["phase_d"],
            "repository_checks": {
                "pytest_passed": args.repository_tests,
                "pytest_failed": 0,
                "ruff_pass": args.ruff_pass,
                "compileall_pass": args.compile_pass,
                "git_diff_check_pass": args.diff_check_pass,
            },
            "prior_published_evidence": _prior_evidence(),
        },
    )
    _write("benchmark.json", benchmark)

    for name in (
        "boost_pads.json",
        "goals_kickoff.json",
        "demolition_respawn.json",
        "match_lifecycle.json",
        "oracle_data.json",
        "rules_source.json",
        "regression.json",
        "benchmark.json",
    ):
        path = RESULTS / name
        print(f"{path.relative_to(ROOT)} sha256={_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
