"""Run actor-head-only Rival human behavior cloning V2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarks.run_rival2_human_bc_continuation_v1 as runner  # noqa: E402

runner.CONTINUATION_VERSION = "RIVAL2_HUMAN_BEHAVIOR_CLONING_V2"
runner.CONTINUATION_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V2"
runner.FROZEN_CONFIG = Path("results/rival2/human_bc_v2/frozen_config.json")
runner.FROZEN_CONFIG_SHA256 = "88DB0BDB8C40ECB12C0B43E9A3FFA4C98C44D2D8B0334AB80DCC65259F4310B0"
runner.RESULT_ROOT = Path("results/rival2/human_bc_v2")
runner.WORK_ROOT = Path(".tools/rival2_human_bc_v2")


if __name__ == "__main__":
    raise SystemExit(runner.main())
