"""One prospectively authorized branch of the V6 analog microstep search."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign


BRANCH_SEEDS = (2_026_090_211, 2_026_090_213, 2_026_090_217, 2_026_090_219)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-seed", type=int, choices=BRANCH_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=campaign.DEFAULT_COLLISION_DIR)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    branch = str(args.branch_seed)
    campaign.SOURCE = (
        campaign.ROOT
        / "checkpoints/rival2/codex_autonomous_v4/rival2_codex_autonomous_best.pt"
    )
    campaign.SOURCE_SHA256 = (
        "172BA59786A2E08EB6DC95CFE29F20C21826F7CB9429FF3C89F4D7C4F4BD9E10"
    )
    campaign.AUTHORITY = campaign.ROOT / "results/rival2/codex_autonomous_v6/authority.json"
    campaign.RESULTS = campaign.ROOT / f"results/rival2/codex_autonomous_v6/branches/{branch}"
    campaign.CHECKPOINTS = (
        campaign.ROOT / f"checkpoints/rival2/codex_autonomous_v6/branches/{branch}"
    )
    campaign.SEED = args.branch_seed
    campaign.POLICY_LR = 2.0e-6
    campaign.CRITIC_LR = 1.0e-5
    campaign.EXPLORATION_SIGMA = 0.05
    campaign.EXPLORATION_BUTTON_TEMPERATURE = 1.15
    campaign.HUMAN_REPLAY_STEPS = 0
    campaign.OPTIMIZER_STEP_LIMIT = 1
    campaign.POLICY_TRAINING_BOUNDARY = "analog_actor_only"
    campaign.CURRENT_OPPONENT_PROBABILITY = 0.5
    campaign.NEXTO_OPPONENT_PROBABILITY = 0.5
    campaign.AUTHORIZED_BRANCH_SEEDS = BRANCH_SEEDS
    campaign.EVALUATION_INTERVAL = 1
    campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V6"
    campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V6_AUTHORITY"
    campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V6_PREFLIGHT"
    campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V6_RESULT"
    campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V6_STATE"
    run_args = Namespace(
        collision_dir=args.collision_dir,
        device=args.device,
        run_dir=Path(f"G:/dev/RivalSim-runs/codex-autonomous-v6/branches/{branch}"),
        worlds=campaign.WORLD_COUNT,
        target_updates=3,
        evaluation_interval=1,
        resume=False,
        preflight_only=False,
    )
    return campaign.run(run_args)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
