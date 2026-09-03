"""Build one immutable Rival checkpoint containing all trained capabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_CONTRACT_V2_120HZ_HASH,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
)
from rivalsim.rival2_official_bundle_v1 import (  # noqa: E402
    OFFICIAL_BUNDLE_V1_FORMAT,
    OfficialCapabilityRouterConfigV1,
    Rival2OfficialControllerV1,
)

SOURCES = {
    "base_blue": (
        ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt",
        "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C",
    ),
    "base_orange": (
        ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt",
        "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A",
    ),
    "aerial": (
        ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt",
        "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154",
    ),
    "capability_blue": (
        ROOT / "checkpoints/rival2/capability_curriculum_v2/rival2_blue.pt",
        "8F3942F30AFF00655D5A83CC2FA8EA8B9AC5314907CCB4D0AE2888C0F48C5442",
    ),
    "capability_orange": (
        ROOT / "checkpoints/rival2/capability_curriculum_v2/rival2_orange.pt",
        "E1E27300D58CFB57CEDE9A0E49E1F1C1392EA3DB59B44C22674BCB4B953AAC34",
    ),
}
DEFAULT_OUTPUT = ROOT / "checkpoints/rival2/official_v1/rival2_official_v1.pt"
DEFAULT_MANIFEST = ROOT / "results/rival2/official_v1/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def component(name: str, path: Path, expected: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{name} source identity changed: {observed} != {expected}")
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source.get("observation_version") != RIVAL2_OBS_V2_120HZ_VERSION:
        raise RuntimeError(f"{name} observation contract mismatch")
    if source.get("action_version") != RIVAL2_ACTION_V2_120HZ_VERSION:
        raise RuntimeError(f"{name} action contract mismatch")
    if int(source.get("policy_hz", 0)) != 120:
        raise RuntimeError(f"{name} is not a 120 Hz policy")
    return {
        "source_path": path.relative_to(ROOT).as_posix(),
        "source_sha256": observed,
        "policy_config": source["policy_config"],
        "policy_config_hash": source["policy_config_hash"],
        "model": {key: value.detach().cpu().clone() for key, value in source["model"].items()},
    }


def build(output: Path, manifest_path: Path) -> dict[str, Any]:
    router = OfficialCapabilityRouterConfigV1()
    router.validate()
    payload = {
        "format": OFFICIAL_BUNDLE_V1_FORMAT,
        "version": 1,
        "rivalsim_commit": git_head(),
        "observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
        "action_version": RIVAL2_ACTION_V2_120HZ_VERSION,
        "physics_hz": 120,
        "policy_hz": 120,
        "contract_hashes": {
            RIVAL2_OBS_V2_120HZ_VERSION: OBSERVATION_SCHEMA_V2_120HZ_HASH,
            RIVAL2_ACTION_V2_120HZ_VERSION: ACTION_CONTRACT_V2_120HZ_HASH,
        },
        "router_config": asdict(router),
        "components": {
            name: component(name, path, expected)
            for name, (path, expected) in SOURCES.items()
        },
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
        "deployment_semantics": (
            "V23 controls the fail-closed playable path; V3 aerial and capability "
            "V2 recovery/demo policies are embedded but automatic specialist "
            "takeovers remain disabled after failed whole-match physical gates"
        ),
    }
    # Construction-time load and finite inference prove that the single file is
    # self-contained before it is made visible at the destination path.
    Rival2OfficialControllerV1(payload, 2, device="cpu").action(
        torch.zeros((2, 182), dtype=torch.float32),
        torch.tensor((0, 1), dtype=torch.int64),
        kickoff_active=torch.ones(2, dtype=torch.bool),
        match_done=torch.zeros(2, dtype=torch.bool),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    manifest = {
        "format": f"{OFFICIAL_BUNDLE_V1_FORMAT}_MANIFEST",
        "checkpoint": {
            "path": output.relative_to(ROOT).as_posix(),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": expected,
            }
            for name, (path, expected) in SOURCES.items()
        },
        "router_config": asdict(router),
        "contracts": payload["contract_hashes"],
        "rivalsim_commit": payload["rivalsim_commit"],
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_changes": 0,
        "status": "candidate_pending_physical_validation",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.output.resolve(), args.manifest.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
