"""Long native-teacher continuation after the bounded V14 bridge diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v14 as campaign

campaign.SOURCE = Path(
    "G:/dev/RivalSim-runs/codex-autonomous-v14/candidate_s0300.pt"
)
campaign.SOURCE_SHA256 = (
    "6C21B078C73A4B4039326C4C49A6D663C8E7C6952BEE66060F1793D14955B924"
)
campaign.SOURCE_MODEL_SHA256 = (
    "6E7A7288B338C6825F04F5DB288B206961F876EAFFD88BBADF9B3FBBCFB44828"
)
campaign.BASELINE = Path(
    "G:/dev/RivalSim-runs/codex-autonomous-v14/candidate_s0300_window.json"
)
campaign.AUTHORITY = campaign.ROOT / "results/rival2/codex_autonomous_v15/authority.json"
campaign.RESULTS = campaign.ROOT / "results/rival2/codex_autonomous_v15"
campaign.CHECKPOINT = (
    campaign.ROOT
    / "checkpoints/rival2/codex_autonomous_v15/rival2_codex_autonomous_best.pt"
)
campaign.DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v15")
campaign.SEED = 2_026_090_221
campaign.NATIVE_BATCH = 3072
campaign.HUMAN_GAMEPLAY_BATCH = 512
campaign.HUMAN_MECHANIC_BATCH = 512
campaign.OPTIMIZER_STEPS = 1750
campaign.VALIDATION_INTERVAL = 250
campaign.CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V15_NATIVE_TEACHER_CONTINUATION"
campaign.AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V15_AUTHORITY"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
