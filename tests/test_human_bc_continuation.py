from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run_rival2_human_bc_continuation_v1 import (
    FROZEN_CONFIG_SHA256,
    _distribution_guard,
    _transactional_retry_learning_rates,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "results/rival2/human_bc_continuation_v1/frozen_config.json"


def _healthy_actor_statistics() -> dict[str, object]:
    return {
        "analog_mean": {
            name: {
                "mean": 0.0,
                "std": 0.2,
                "min": -0.5,
                "max": 0.5,
                "absolute_ge_5_fraction": 0.0,
            }
            for name in ("throttle", "steer", "pitch", "yaw", "roll")
        },
        "button_probability": {
            name: {"mean": 0.5, "std": 0.2, "saturation_fraction": 0.0}
            for name in ("jump", "boost", "handbrake")
        },
        "log_std": {
            name: {"mean": -1.0, "at_min_fraction": 0.0, "at_max_fraction": 0.0}
            for name in ("throttle", "steer", "pitch", "yaw", "roll")
        },
        "finite": True,
        "sample_count": 8,
    }


def test_continuation_authority_hash_and_parent_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert file_sha256(CONFIG_PATH) == FROZEN_CONFIG_SHA256
    assert (
        config["authority"]["source_checkpoint_sha256"]
        == "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
    )
    assert config["inherit_exactly_from_base_bc_v1"]["human_action_objective"]
    assert config["prohibited"]["ppo"]
    assert config["prohibited"]["closed_loop_mechanic_framework"]


def test_transactional_retry_learning_rates_back_off_progressively() -> None:
    rates = _transactional_retry_learning_rates(3e-5, backoff_factor=0.5, retry_count=3)
    assert rates == (3e-5, 1.5e-5, 7.5e-6, 3.75e-6)
    assert len(set(rates)) == 4


def test_distribution_guard_accepts_health_and_rejects_collapse() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    human = {
        "finite": True,
        "families": {
            family: {"actor_output_statistics": _healthy_actor_statistics()}
            for family in ("gameplay", "mechanic")
        },
    }
    assert _distribution_guard(human, config)["accepted"]
    human["families"]["mechanic"]["actor_output_statistics"]["analog_mean"]["roll"]["std"] = 0.0
    result = _distribution_guard(human, config)
    assert not result["accepted"]
    assert not result["checks"]["mechanic.analog.roll.nonconstant"]
