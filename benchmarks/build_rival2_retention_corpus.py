"""Build the fixed healthy-+239 Rival 2.0 retention observation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    OBS_FIELD_NAMES,
    POSITION_SCALE,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    Rival2MixedPPOSafetyConfig,
    build_retention_observation_corpus,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer  # noqa: E402

SOURCE_CHECKPOINT = Path("checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt")
SOURCE_CHECKPOINT_SHA256 = "77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21"
OUTPUT = Path("results/rival2/opponent_curriculum_v1/safe_transition/retention_corpus.pt")
SUMMARY = Path("results/rival2/opponent_curriculum_v1/safe_transition/retention_corpus.json")
COLLECTION_WORLDS = 8192
COLLECTION_DECISIONS = 240
COLLECTION_SEED = 2_026_082_721


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, object]:
    source_sha = _sha256(SOURCE_CHECKPOINT)
    if source_sha != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("healthy +239 source checkpoint SHA-256 mismatch")
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    source_ppo = Rival2PPOConfig(**source["ppo_config"])
    collection_ppo = Rival2PPOConfig(
        **{
            **asdict(source_ppo),
            "rollout_horizon": COLLECTION_DECISIONS,
        }
    )
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    kickoff_selector = (np.arange(COLLECTION_WORLDS, dtype=np.int32) + COLLECTION_SEED) % 5
    env = Rival2Env(
        COLLECTION_WORLDS,
        str(args.collision_dir),
        device=args.device,
        seed=COLLECTION_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=collection_ppo,
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        seed=COLLECTION_SEED,
    )
    trainer.model.load_state_dict(source["model"])
    trainer.policy_version = int(source["policy_version"])
    trainer.iteration = int(source["iteration"])
    rollout = trainer.collect_rollout()
    safety = Rival2MixedPPOSafetyConfig()
    observations, summary = build_retention_observation_corpus(
        rollout,
        corpus_size=safety.retention_corpus_size,
    )
    field = {name: OBS_FIELD_NAMES.index(name) for name in OBS_FIELD_NAMES}
    self_x = observations[:, field["self.position.x"]] * POSITION_SCALE[0]
    self_y = observations[:, field["self.position.y"]] * POSITION_SCALE[1]
    self_z = observations[:, field["self.position.z"]] * POSITION_SCALE[2]
    heading = torch.atan2(
        observations[:, field["self.forward.y"]],
        observations[:, field["self.forward.x"]],
    )
    x_region = torch.bucketize(self_x, torch.tensor([-1365.0, 1365.0], device=args.device))
    y_region = torch.bucketize(self_y, torch.tensor([-1706.0, 1706.0], device=args.device))
    heading_octant = torch.floor((heading + math.pi) / (math.pi / 4.0)).clamp(0, 7).to(torch.int64)
    state_coverage = {
        "self_position_x_min_max_uu": [float(self_x.min().item()), float(self_x.max().item())],
        "self_position_y_min_max_uu": [float(self_y.min().item()), float(self_y.max().item())],
        "self_position_z_min_max_uu": [float(self_z.min().item()), float(self_z.max().item())],
        "occupied_x_field_regions_of_3": int(torch.unique(x_region).numel()),
        "occupied_y_field_regions_of_3": int(torch.unique(y_region).numel()),
        "occupied_heading_octants_of_8": int(torch.unique(heading_octant).numel()),
        "heading_octant_counts": torch.bincount(heading_octant, minlength=8).cpu().tolist(),
    }
    summary.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_identity": {
                "checkpoint": SOURCE_CHECKPOINT.as_posix(),
                "checkpoint_sha256": source_sha,
                "iteration": int(source["iteration"]),
                "policy_version": int(source["policy_version"]),
                "reward_version": source["reward_version"],
                "episode_version": source["episode_version"],
            },
            "collection": {
                "worlds": COLLECTION_WORLDS,
                "decisions_per_world": COLLECTION_DECISIONS,
                "agent_observations_considered": int(rollout.train_mask.sum().item()),
                "seed": COLLECTION_SEED,
                "all_five_kickoff_layouts": True,
                "current_source_policy_on_both_sides": True,
                "stochastic_source_policy_actions": True,
                "training_performed": False,
            },
            "safety_config_hash": safety.content_hash,
            "state_coverage": state_coverage,
        }
    )
    required_categories = (
        "ordinary_ground_play",
        "possession_ball_approach",
        "near_ball_interaction",
        "recovery",
        "airborne",
    )
    coverage = {
        name: summary["category_counts"][name]["selected"] > 0 for name in required_categories
    }
    summary["checks"]["required_category_coverage"] = all(coverage.values())
    summary["checks"]["field_position_orientation_coverage"] = (
        state_coverage["occupied_x_field_regions_of_3"] == 3
        and state_coverage["occupied_y_field_regions_of_3"] == 3
        and state_coverage["occupied_heading_octants_of_8"] == 8
    )
    summary["required_category_coverage"] = coverage
    summary["verdict"] = "PASS_GREEN" if all(summary["checks"].values()) else "FAIL_RED"
    if summary["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"retention corpus category coverage failed: {coverage}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "RIVAL2_RETENTION_OBSERVATIONS_V1",
            "observations": observations.detach().cpu(),
            "summary": summary,
        },
        args.output,
    )
    summary["artifact"] = {
        "path": args.output.as_posix(),
        "sha256": _sha256(args.output),
        "size_bytes": args.output.stat().st_size,
    }
    _write_json(args.summary, summary)
    return summary


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = run(args)
    print(json.dumps({"verdict": result["verdict"], "artifact": result["artifact"]}, indent=2))
    return 0 if result["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
