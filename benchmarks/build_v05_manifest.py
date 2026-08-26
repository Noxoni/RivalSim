"""Build the release manifest for a completed RivalSim v0.5 implementation commit."""

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
EVIDENCE_FILES = (
    "tensor_bridge.json",
    "observation.json",
    "action_distribution.json",
    "reward_episode.json",
    "rollout_gae.json",
    "ppo.json",
    "checkpoint_resume.json",
    "self_play.json",
    "learning_smoke.json",
    "benchmark.json",
    "regression.json",
)
RELEASE_FILES = (
    "CODEX_START_PROMPT.md",
    "VERSION.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "docs/ROADMAP.md",
    "docs/V0_5_RESULTS.md",
    "docs/REPRODUCING_V0_5.md",
    "docs/RIVAL2_TRAINING_CONTRACT.md",
    *(f"results/v0.5/{name}" for name in EVIDENCE_FILES),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    return parser.parse_args()


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _record(relative: str) -> dict[str, Any]:
    blob = _git("show", f":{relative}", text=False)
    assert isinstance(blob, bytes)
    return {
        "path": relative,
        "size_bytes": len(blob),
        "sha256": _sha256_bytes(blob),
    }


def _read(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    implementation = str(_git("rev-parse", args.implementation_commit)).strip()
    parent = str(_git("rev-parse", f"{implementation}^")).strip()
    if parent != HANDOFF_PARENT:
        raise RuntimeError(
            f"implementation commit parent {parent} is not handoff parent {HANDOFF_PARENT}"
        )
    changed_prior = str(
        _git("diff", "--name-only", V04_RELEASE, "--", *PRIOR_RESULT_DIRS)
    ).splitlines()
    if changed_prior:
        raise RuntimeError(f"prior evidence changed: {changed_prior}")

    evidence = {name: _read(f"results/v0.5/{name}") for name in EVIDENCE_FILES}
    if any(value["verdict"] != "PASS_GREEN" for value in evidence.values()):
        raise RuntimeError("one or more v0.5 evidence gates are not green")
    if evidence["regression.json"]["v0_6_begun"] is not False:
        raise RuntimeError("v0.6 boundary assertion is not false")

    implementation_names = str(
        _git("diff", "--name-only", HANDOFF_PARENT, implementation)
    ).splitlines()
    implementation_files = []
    for relative in implementation_names:
        blob = _git("show", f"{implementation}:{relative}", text=False)
        assert isinstance(blob, bytes)
        implementation_files.append(
            {"path": relative, "size_bytes": len(blob), "sha256": _sha256_bytes(blob)}
        )

    prior_trees: dict[str, Any] = {}
    prior_manifests = []
    for directory in PRIOR_RESULT_DIRS:
        listing = _git("ls-tree", "-r", "--full-tree", V04_RELEASE, "--", directory, text=False)
        assert isinstance(listing, bytes)
        prior_trees[directory] = {
            "file_count": len(listing.splitlines()),
            "git_tree_listing_sha256": _sha256_bytes(listing),
        }
        prior_manifests.append(_record(f"{directory}/manifest.json"))

    benchmark = evidence["benchmark.json"]
    learning = evidence["learning_smoke.json"]["gate"]
    tensor = evidence["tensor_bridge.json"]["gate"]
    regression = evidence["regression.json"]
    manifest = {
        "schema_version": 1,
        "milestone": "v0.5",
        "policy_generation": "Rival 2.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS_GREEN",
        "authorized_handoff_parent": HANDOFF_PARENT,
        "v0_4_release_commit": V04_RELEASE,
        "v0_4_implementation_commit": V04_IMPLEMENTATION,
        "v0_5_implementation_commit": implementation,
        "contract_hashes": evidence["observation.json"]["contract_hashes"],
        "policy": {
            "architecture": evidence["ppo.json"]["policy_config"],
            "config_hash": evidence["ppo.json"]["policy_config_hash"],
            "parameter_count": evidence["ppo.json"]["parameter_count"],
        },
        "selected_performance_point": {
            "worlds": benchmark["selected_worlds"],
            "complete_iteration_agent_samples_per_s_median": benchmark["selected_point"][
                "complete_iteration_agent_samples_per_s_median"
            ],
            "simulated_game_seconds_per_s_median": benchmark["selected_point"][
                "simulated_game_seconds_per_s_median"
            ],
            "wall_time_cv": benchmark["selected_point"]["wall_time_cv"],
            "vram_peak_observed_bytes": benchmark["selected_point"]["vram_peak_observed_bytes"],
        },
        "zero_copy_residency": {
            "verdict": tensor["verdict"],
            "alias_count": tensor["alias_count"],
            "all_aliases": tensor["all_aliases"],
            "hot_loop_h2d_bytes": benchmark["selected_point"]["hot_loop_h2d_bytes"],
            "hot_loop_d2h_bytes": benchmark["selected_point"]["hot_loop_d2h_bytes"],
        },
        "learning_smoke": {
            "verdict": learning["verdict"],
            "declared_metric": learning["declared_metric"],
            "objective_improvement": learning["objective_improvement"],
            "improvement_standard_errors": learning["improvement_standard_errors"],
        },
        "inherited_regressions": {
            "verdict": regression["verdict"],
            "v0_4": regression["v0_4"]["complete_path"]["verdict"],
            "v0_3": {name: value["verdict"] for name, value in regression["v0_3"].items()},
            "v0_2_2": regression["v0_2_2"]["verdict"],
            "v0_1": regression["v0_1"]["verdict"],
            "repository": regression["repository"],
        },
        "release_files": [_record(relative) for relative in RELEASE_FILES],
        "implementation_files": implementation_files,
        "prior_evidence_manifests": prior_manifests,
        "prior_evidence_trees": prior_trees,
        "prior_evidence_byte_diff_count": 0,
        "v0_6_begun": False,
    }
    path = RESULTS / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
