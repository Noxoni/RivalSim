"""Build the committed v0.2.2 release manifest after the implementation commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = (
    "CODEX_START_PROMPT.md",
    "VERSION.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "docs/ROADMAP.md",
    "docs/V0_2_2_ORACLE_CACHE.md",
    "docs/V0_2_2_RESULTS.md",
    "docs/REPRODUCING_V0_2_2.md",
    "results/v0.2.2/oracle_data.json",
    "results/v0.2.2/source_port.json",
    "results/v0.2.2/parity.json",
    "results/v0.2.2/regression.json",
    "results/v0.2.2/benchmark.json",
)
PRIOR_MANIFESTS = (
    "results/v0.1/manifest.json",
    "results/v0.2/manifest.json",
    "results/v0.2.1/manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v0.2.2" / "manifest.json",
    )
    return parser.parse_args()


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=text,
        encoding="utf-8" if text else None,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _validate_evidence() -> dict[str, dict[str, Any]]:
    evidence = {
        "oracle_data": _json("results/v0.2.2/oracle_data.json"),
        "source_port": _json("results/v0.2.2/source_port.json"),
        "parity": _json("results/v0.2.2/parity.json"),
        "regression": _json("results/v0.2.2/regression.json"),
        "benchmark": _json("results/v0.2.2/benchmark.json"),
    }
    parity = evidence["parity"]
    complete = parity["complete_acceptance"]
    if (
        parity.get("classification") != "PASS_GREEN"
        or complete["counts"].get("selected_starting_states") != 39236
        or complete["counts"].get("checkpoint_comparisons") != 156944
        or complete["counts"].get("hard_mismatch_events") != 0
        or complete["counts"].get("numeric_failure_events") != 0
        or complete["counts"].get("failed_cases") != 0
        or not complete["gate"].get("complete_v022_gate_pass")
    ):
        raise RuntimeError("v0.2.2 parity evidence is not complete PASS_GREEN")
    authority = parity["authority"]["identity_sha256"]
    if (
        evidence["oracle_data"]["authority"]["authority_identity_sha256"] != authority
        or evidence["source_port"]["authority_identity_sha256"] != authority
    ):
        raise RuntimeError("v0.2.2 evidence authority identities disagree")
    regression = evidence["regression"]["summary"]
    if regression.get("scenario_count") != 27 or not regression.get("basic_parity_pass"):
        raise RuntimeError("v0.1 regression evidence failed")
    benchmark = evidence["benchmark"]["summary"]
    if (
        benchmark.get("verdict") != "PASS_GREEN"
        or not benchmark.get("parity_gate_pass")
        or not benchmark.get("hot_loop_gpu_resident")
    ):
        raise RuntimeError("v0.2.2 benchmark evidence is not PASS_GREEN")
    if evidence["source_port"].get("prohibited_fitting_added") is not False:
        raise RuntimeError("source-port evidence does not preserve the no-fitting boundary")
    if evidence["oracle_data"]["scope_boundary"].get("v0_3_begun") is not False:
        raise RuntimeError("oracle evidence reports v0.3 work")
    return evidence


def _committed_files(commit: str) -> list[dict[str, Any]]:
    names = str(_git("ls-tree", "-r", "--name-only", commit)).splitlines()
    selected = sorted(
        path
        for path in names
        if path.startswith(("rivalsim/", "benchmarks/", "tests/"))
        or path in RELEASE_FILES
    )
    entries = []
    for path in selected:
        payload = _git("show", f"{commit}:{path}", text=False)
        assert isinstance(payload, bytes)
        entries.append(
            {
                "path": path,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return entries


def _committed_entry(commit: str, path: str) -> dict[str, Any]:
    payload = _git("show", f"{commit}:{path}", text=False)
    assert isinstance(payload, bytes)
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def main() -> int:
    args = parse_args()
    commit = str(_git("rev-parse", args.implementation_commit)).strip()
    head = str(_git("rev-parse", "HEAD")).strip()
    if commit != head:
        raise RuntimeError("implementation commit must be the current HEAD")
    if 'version = "0.2.2"' not in (ROOT / "pyproject.toml").read_text(encoding="utf-8"):
        raise RuntimeError("package version is not 0.2.2")
    tracked_cmfs = str(_git("ls-files", "*.cmf")).splitlines()
    if tracked_cmfs:
        raise RuntimeError(f"collision assets must remain external: {tracked_cmfs}")
    _validate_evidence()

    release_files = [_committed_entry(commit, path) for path in RELEASE_FILES]
    prior = [_committed_entry(commit, path) for path in PRIOR_MANIFESTS]
    manifest = {
        "schema_version": 1,
        "milestone": "v0.2.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS_GREEN",
        "implementation_commit": commit,
        "authority_identity_sha256": _json("results/v0.2.2/parity.json")["authority"][
            "identity_sha256"
        ],
        "release_files": release_files,
        "prior_evidence_manifests": prior,
        "committed_implementation_files": _committed_files(commit),
        "gates": {
            "representative_cases": 1043,
            "complete_cases": 39236,
            "checkpoint_comparisons": 156944,
            "hard_mismatch_events": 0,
            "numeric_failure_events": 0,
            "v0_1_scenarios": 27,
            "repository_tests": 46,
            "best_b3_aggregate_simulated_game_seconds_per_s": 511886.1503832971,
            "hot_loop_gpu_resident": True,
            "v0_3_begun": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output.resolve()} sha256={_sha256_bytes(args.output.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
