"""Fine winning-direction line search around the promoted V9 region."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v9 as campaign


campaign.AUTHORITY = (
    campaign.ROOT / "results/rival2/codex_autonomous_v10/authority.json"
)
campaign.RESULTS = campaign.ROOT / "results/rival2/codex_autonomous_v10"
campaign.CHECKPOINT = (
    campaign.ROOT
    / "checkpoints/rival2/codex_autonomous_v10/rival2_codex_autonomous_best.pt"
)
campaign.DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v10")
campaign.LAMBDAS = (0.30, 0.35, 0.40, 0.45, 0.475, 0.525, 0.55, 0.60, 0.65, 0.70)
campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V10_FINE_DIRECTION_SEARCH"
campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V10_AUTHORITY"
campaign.PREFLIGHT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V10_PREFLIGHT"
campaign.RESULT_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V10_RESULT"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
