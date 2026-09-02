"""Competitive-only continuation from the selected eight-step human anchor."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as campaign


campaign.SOURCE = (
    campaign.ROOT
    / "checkpoints/rival2/codex_autonomous_v2/rival2_codex_human_anchor_v1.pt"
)
campaign.SOURCE_SHA256 = (
    "9E8EC453944D8B0065611F54609B55C4FDE97DC3291803727CCB538246705CE2"
)
campaign.AUTHORITY = campaign.ROOT / "results/rival2/codex_autonomous_v2/authority.json"
campaign.RESULTS = campaign.ROOT / "results/rival2/codex_autonomous_v2"
campaign.CHECKPOINTS = campaign.ROOT / "checkpoints/rival2/codex_autonomous_v2"
campaign.DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v2")
campaign.POLICY_LR = 5.0e-7
campaign.CRITIC_LR = 1.0e-5
campaign.EXPLORATION_SIGMA = 0.05
campaign.EXPLORATION_BUTTON_TEMPERATURE = 1.15
campaign.HUMAN_REPLAY_STEPS = 0
campaign.EVALUATION_INTERVAL = 1
campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V2"
campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V2_AUTHORITY"
campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V2_PREFLIGHT"
campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V2_RESULT"
campaign.STATE_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V2_STATE"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
