"""Resume the promoted sustained-match PPO lineage for ten microsteps."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign


campaign.SOURCE = (
    campaign.ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
campaign.SOURCE_SHA256 = (
    "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
)
campaign.AUTHORITY = campaign.ROOT / "results/rival2/codex_autonomous_v12/authority.json"
campaign.RESULTS = campaign.ROOT / "results/rival2/codex_autonomous_v12"
campaign.CHECKPOINTS = campaign.ROOT / "checkpoints/rival2/codex_autonomous_v12"
campaign.DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v12")
campaign.SEED = 2_026_090_213
campaign.POLICY_LR = 2.0e-6
campaign.CRITIC_LR = 1.0e-5
campaign.EXPLORATION_SIGMA = 0.05
campaign.EXPLORATION_BUTTON_TEMPERATURE = 1.15
campaign.HUMAN_REPLAY_STEPS = 0
campaign.OPTIMIZER_STEP_LIMIT = 1
campaign.POLICY_TRAINING_BOUNDARY = "analog_actor_only"
campaign.CURRENT_OPPONENT_PROBABILITY = 0.5
campaign.NEXTO_OPPONENT_PROBABILITY = 0.5
campaign.AUTHORIZED_BRANCH_SEEDS = None
campaign.MATERIAL_REGRESSION_PATIENCE = 100
campaign.EVALUATION_INTERVAL = 1
campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V12"
campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V12_AUTHORITY"
campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V12_PREFLIGHT"
campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V12_RESULT"
campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V12_STATE"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
