"""Run the deterministic five-layout-by-two-side Rival-vs-Nexto match matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarks.run_rival2_nexto_matches as nexto_matches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--seed", type=int, default=2_026_090_205)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256.upper():
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha} != "
            f"{args.checkpoint_sha256.upper()}"
        )
    collision_root = args.collision_root.resolve()
    if not collision_root.is_dir():
        raise FileNotFoundError(collision_root)

    # The established suite implementation owns full-match scoring and
    # telemetry. Only its checkpoint/collision globals are rebound here.
    nexto_matches.CHECKPOINT = checkpoint
    nexto_matches.COLLISION_ROOT = collision_root
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    canonical, _raw = nexto_matches._run_suite(
        name="codex_autonomous_deterministic_full_match",
        layout=layout,
        rival_side=rival_side,
        stochastic_rival=False,
        seed=int(args.seed),
    )

    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_FULL_MATCH_EVALUATION",
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_sha,
        },
        "opponent": "Nexto",
        "matrix": "five standard kickoff layouts x both Rival sides",
        "selection_use": "post-selection evaluation only",
        "policy_mutation": False,
        "canonical": canonical,
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / "full_match_evaluation.json", result)
    _write_json(
        output_dir / "full_match_ledger.json", canonical["canonical_match_ledger"]
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
