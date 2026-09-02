"""Generate one fresh deployment-aligned PPO direction from the V19 parent."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign
from benchmarks.run_rival2_codex_autonomous_v17_branch import _window_evaluation


BRANCH_SEEDS = (2_026_090_261, 2_026_090_263, 2_026_090_267, 2_026_090_269)
SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v19/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v21/authority.json"


def _configure(branch_seed: int) -> Path:
    branch = str(branch_seed)
    run_dir = Path(f"G:/dev/RivalSim-runs/codex-autonomous-v21/branches/{branch}")
    campaign.SOURCE = SOURCE
    campaign.SOURCE_SHA256 = SOURCE_SHA256
    campaign.AUTHORITY = AUTHORITY
    campaign.RESULTS = ROOT / f"results/rival2/codex_autonomous_v21/branches/{branch}"
    campaign.CHECKPOINTS = ROOT / f"checkpoints/rival2/codex_autonomous_v21/branches/{branch}"
    campaign.DEFAULT_RUN_DIR = run_dir
    campaign.SEED = branch_seed
    campaign.POLICY_LR = 1.0e-7
    campaign.CRITIC_LR = 5.0e-7
    campaign.EXPLORATION_SIGMA = 0.005
    campaign.EXPLORATION_BUTTON_TEMPERATURE = 0.25
    campaign.HUMAN_REPLAY_STEPS = 0
    campaign.OPTIMIZER_STEP_LIMIT = 1
    campaign.POLICY_TRAINING_BOUNDARY = "analog_actor_only"
    campaign.CURRENT_OPPONENT_PROBABILITY = 0.5
    campaign.NEXTO_OPPONENT_PROBABILITY = 0.5
    campaign.AUTHORIZED_BRANCH_SEEDS = BRANCH_SEEDS
    campaign.MATERIAL_REGRESSION_PATIENCE = 100
    campaign.EVALUATION_INTERVAL = 1
    campaign.NEXTO_WIN_TARGET = 999.0
    campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V21"
    campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V21_AUTHORITY"
    campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V21_PREFLIGHT"
    campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V21_RESULT"
    campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V21_STATE"
    campaign.run_nexto_evaluation = _window_evaluation
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-seed", type=int, choices=BRANCH_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=campaign.DEFAULT_COLLISION_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    run_dir = _configure(args.branch_seed)
    return campaign.run(
        Namespace(
            collision_dir=args.collision_dir,
            device=args.device,
            run_dir=run_dir,
            worlds=campaign.WORLD_COUNT,
            target_updates=1,
            evaluation_interval=1,
            resume=False,
            preflight_only=args.preflight_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
