"""Run the prospectively frozen high-intercept correction to aerial hybrid V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_intercept_hybrid_v1 as campaign  # noqa: E402

VERSION = "RIVAL2_AERIAL_INTERCEPT_HYBRID_V2"
campaign.HYBRID_VERSION = VERSION
campaign.AUTHORITY = ROOT / "results/rival2/aerial_intercept_hybrid_v2/authority.json"
campaign.AUTHORITY_SHA256 = "52ABF768B0FE8A30664BB826F69A7D0B5660795B756D2D0CEA4E1D091A15B94A"
campaign.RESULTS = ROOT / "results/rival2/aerial_intercept_hybrid_v2"
campaign.BUNDLE = ROOT / "checkpoints/rival2/aerial_intercept_hybrid_v2"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
