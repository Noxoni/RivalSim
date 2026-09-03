"""Train natural aerial continuation with the V8-selected jump timing."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_natural_v7 as balanced  # noqa: E402

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V9"
AUTHORITY = balanced.ROOT / "results/rival2/ground_to_air_natural_v9/authority.json"
AUTHORITY_SHA256 = "CF3250FA583951E78DEF6011B738CEA7CCDF19B4BA46ACAE79A7661FABFECCA2"
RESULTS = balanced.ROOT / "results/rival2/ground_to_air_natural_v9"
CHECKPOINTS = balanced.ROOT / "checkpoints/rival2/ground_to_air_natural_v9"
CHECKPOINT_NAME = "rival2_ground_to_air_natural_v9.pt"
STAGE_NAME = "ground_to_air_natural_v9"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v9")
POP_CONTROL_IDENTITY = (
    "RIVAL2_GROUND_TO_AIR_POP_CONTROL_V7+V8_HOLD24_RELEASE4"
)
OPTIMIZER_FORMAT = "RIVAL2_GROUND_TO_AIR_NATURAL_V9_FRESH_BALANCED_ADAMW"

_OVERRIDES: dict[str, Any] = {
    "VERSION": VERSION,
    "AUTHORITY": AUTHORITY,
    "AUTHORITY_SHA256": AUTHORITY_SHA256,
    "RESULTS": RESULTS,
    "CHECKPOINTS": CHECKPOINTS,
    "CHECKPOINT_NAME": CHECKPOINT_NAME,
    "STAGE_NAME": STAGE_NAME,
    "DEFAULT_RUN_DIR": DEFAULT_RUN_DIR,
    "POP_CONTROL_IDENTITY": POP_CONTROL_IDENTITY,
    "OPTIMIZER_FORMAT": OPTIMIZER_FORMAT,
    "VALIDATION_PHYSICAL_PROBE": True,
}


@contextmanager
def _configured_runner() -> Iterator[None]:
    """Temporarily bind the generic balanced runner to the frozen V9 authority."""

    original = {name: getattr(balanced, name) for name in _OVERRIDES}
    try:
        for name, value in _OVERRIDES.items():
            setattr(balanced, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(balanced, name, value)


def load_authority() -> dict[str, Any]:
    with _configured_runner():
        return balanced.load_authority()


def run(args: argparse.Namespace) -> int:
    with _configured_runner():
        return balanced.run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=balanced.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-stratum", type=int, default=256)
    parser.add_argument("--evaluation-worlds-per-row", type=int, default=256)
    parser.add_argument("--test-worlds-per-row", type=int, default=512)
    parser.add_argument("--maximum-blocks", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
