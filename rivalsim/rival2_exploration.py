"""Frozen controller-wide exploration ramp for Fresh Human Seed PPO v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from rivalsim.rival2_policy import HybridDistributionOverride

RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_V1 = "RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_V1"
SIGMA_START = 0.01
SIGMA_END = 0.08
BUTTON_TEMPERATURE_START = 0.02
BUTTON_TEMPERATURE_END = 0.50
RAMP_START_UPDATE = 60
RAMP_END_UPDATE = 300

RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT = {
    "version": RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_V1,
    "normalized_progress": {
        "through_update_60": 0.0,
        "from_update_300": 1.0,
        "interior": "x*x*(3-2*x)",
        "x": "(accepted_update-60)/240",
    },
    "analog": {
        "channels": ["throttle", "steer", "pitch", "yaw", "roll"],
        "sigma_start": SIGMA_START,
        "sigma_end": SIGMA_END,
        "interpolation": "linear_in_log_sigma",
        "raw_actor_log_std_bypassed": True,
    },
    "buttons": {
        "channels": ["jump", "boost", "handbrake"],
        "temperature_start": BUTTON_TEMPERATURE_START,
        "temperature_end": BUTTON_TEMPERATURE_END,
        "effective_logits": "learned_logits/positive_temperature",
    },
    "coherence": [
        "rollout_sampling",
        "stored_old_log_probability",
        "ppo_recomputed_log_probability",
        "ppo_ratio",
        "entropy_diagnostic",
        "kl_guards",
    ],
}
RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH = (
    hashlib.sha256(
        json.dumps(
            RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    .hexdigest()
    .upper()
)


@dataclass(frozen=True, slots=True)
class FreshHumanSeedExploration:
    accepted_update: int
    normalized_progress: float
    analog_sigma: float
    analog_log_sigma: float
    button_temperature: float
    version: str = RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_V1
    contract_sha256: str = RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH

    @property
    def distribution_override(self) -> HybridDistributionOverride:
        return HybridDistributionOverride(
            analog_log_std=self.analog_log_sigma,
            button_temperature=self.button_temperature,
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def fresh_human_seed_exploration(accepted_update: int) -> FreshHumanSeedExploration:
    """Resolve the one immutable distribution used by a proposed accepted update."""

    update = int(accepted_update)
    if update < 0:
        raise ValueError("accepted update cannot be negative")
    if update <= RAMP_START_UPDATE:
        progress = 0.0
    elif update >= RAMP_END_UPDATE:
        progress = 1.0
    else:
        x = (update - RAMP_START_UPDATE) / (RAMP_END_UPDATE - RAMP_START_UPDATE)
        progress = x * x * (3.0 - 2.0 * x)
    log_start = math.log(SIGMA_START)
    log_end = math.log(SIGMA_END)
    log_sigma = log_start + (log_end - log_start) * progress
    sigma = (
        SIGMA_START if progress == 0.0 else SIGMA_END if progress == 1.0 else math.exp(log_sigma)
    )
    temperature = (
        BUTTON_TEMPERATURE_START + (BUTTON_TEMPERATURE_END - BUTTON_TEMPERATURE_START) * progress
    )
    return FreshHumanSeedExploration(
        accepted_update=update,
        normalized_progress=progress,
        analog_sigma=sigma,
        analog_log_sigma=log_sigma,
        button_temperature=temperature,
    )


__all__ = [
    "BUTTON_TEMPERATURE_END",
    "BUTTON_TEMPERATURE_START",
    "RAMP_END_UPDATE",
    "RAMP_START_UPDATE",
    "RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT",
    "RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH",
    "RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_V1",
    "SIGMA_END",
    "SIGMA_START",
    "FreshHumanSeedExploration",
    "fresh_human_seed_exploration",
]
