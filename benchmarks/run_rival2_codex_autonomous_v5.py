"""Nexto-emphasized analog microstep PPO from the promoted V4 checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign


campaign.SOURCE = (
    campaign.ROOT
    / "checkpoints/rival2/codex_autonomous_v4/rival2_codex_autonomous_best.pt"
)
campaign.SOURCE_SHA256 = (
    "172BA59786A2E08EB6DC95CFE29F20C21826F7CB9429FF3C89F4D7C4F4BD9E10"
)
campaign.AUTHORITY = campaign.ROOT / "results/rival2/codex_autonomous_v5/authority.json"
campaign.RESULTS = campaign.ROOT / "results/rival2/codex_autonomous_v5"
campaign.CHECKPOINTS = campaign.ROOT / "checkpoints/rival2/codex_autonomous_v5"
campaign.DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v5")
campaign.SEED = 2_026_090_205
campaign.POLICY_LR = 2.0e-6
campaign.CRITIC_LR = 1.0e-5
campaign.EXPLORATION_SIGMA = 0.05
campaign.EXPLORATION_BUTTON_TEMPERATURE = 1.15
campaign.HUMAN_REPLAY_STEPS = 0
campaign.OPTIMIZER_STEP_LIMIT = 1
campaign.POLICY_TRAINING_BOUNDARY = "analog_actor_only"
campaign.CURRENT_OPPONENT_PROBABILITY = 0.2
campaign.NEXTO_OPPONENT_PROBABILITY = 0.8
campaign.EVALUATION_INTERVAL = 1
campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V5"
campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V5_AUTHORITY"
campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V5_PREFLIGHT"
campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V5_RESULT"
campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V5_STATE"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
