"""Run the frozen airborne-only trajectory-intercept hybrid campaign."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_intercept_hybrid_v1 as campaign  # noqa: E402

VERSION = "RIVAL2_AERIAL_INTERCEPT_HYBRID_V3"
campaign.HYBRID_VERSION = VERSION
campaign.AUTHORITY = ROOT / "results/rival2/aerial_intercept_hybrid_v3/authority.json"
campaign.AUTHORITY_SHA256 = "BA8E62E63ECB14F25CDFCDA8CA1EE51F1589D84C1EB52F85BFEBF5CDF094B418"
campaign.RESULTS = ROOT / "results/rival2/aerial_intercept_hybrid_v3"
campaign.BUNDLE = ROOT / "checkpoints/rival2/aerial_intercept_hybrid_v3"


if __name__ == "__main__":
    raise SystemExit(campaign.run(campaign.parse_args()))
