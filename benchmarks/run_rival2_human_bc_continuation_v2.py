"""Run corrected Rival human BC continuation V2 on the frozen V1 data path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarks.run_rival2_human_bc_continuation_v1 as runner  # noqa: E402

runner.CONTINUATION_VERSION = "RIVAL2_HUMAN_BC_CONTINUATION_V2"
runner.CONTINUATION_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BC_CONTINUATION_CHECKPOINT_V2"
runner.FROZEN_CONFIG = Path("results/rival2/human_bc_continuation_v2/frozen_config.json")
runner.FROZEN_CONFIG_SHA256 = "39C86ADDB7A7502F1826AF3CFE1AA44D2B86E66068AE05B06D5DB225D0A5AFAC"
runner.RESULT_ROOT = Path("results/rival2/human_bc_continuation_v2")
runner.WORK_ROOT = Path(".tools/rival2_human_bc_continuation_v2")


if __name__ == "__main__":
    raise SystemExit(runner.main())
