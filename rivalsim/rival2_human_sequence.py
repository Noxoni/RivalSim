"""Shared deterministic observation view for Human Sequence Seed v1.

The 182-slot Rival observation structure is preserved, but only physical fields that
are represented with the same semantics in the native recording and RivalSim remain
visible.  Every other slot is hard-zeroed.  This module contains no learned repair.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from rivalsim.human_demo.training_adapter import AdaptedSample
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES

HUMAN_SEQUENCE_OBS_VIEW_VERSION = "RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1"

_BALL_FIELDS = tuple(field for field in OBS_FIELD_NAMES if field.startswith("ball."))
_RELATIVE_FIELDS = tuple(field for field in OBS_FIELD_NAMES if field.startswith("relative."))
_CAR_RETAINED_SUFFIXES = (
    *(f"position.{axis}" for axis in "xyz"),
    *(f"linear_velocity.{axis}" for axis in "xyz"),
    *(f"forward.{axis}" for axis in "xyz"),
    *(f"up.{axis}" for axis in "xyz"),
    *(f"angular_velocity.{axis}" for axis in "xyz"),
    "boost",
    "on_ground",
    "has_jumped",
    "has_double_jumped",
    "jump_available",
    "wheel_contact.front_left",
    "wheel_contact.front_right",
    "wheel_contact.back_left",
    "wheel_contact.back_right",
    "is_supersonic",
)
_CAR_FIELDS = tuple(
    f"{prefix}.{suffix}" for prefix in ("self", "opponent") for suffix in _CAR_RETAINED_SUFFIXES
)

RETAINED_OBSERVATION_FIELDS = (*_BALL_FIELDS, *_CAR_FIELDS, *_RELATIVE_FIELDS)
RETAINED_OBSERVATION_INDICES = tuple(
    OBS_FIELD_NAMES.index(field) for field in RETAINED_OBSERVATION_FIELDS
)
ZEROED_OBSERVATION_FIELDS = tuple(
    field for field in OBS_FIELD_NAMES if field not in RETAINED_OBSERVATION_FIELDS
)
ZEROED_OBSERVATION_INDICES = tuple(
    OBS_FIELD_NAMES.index(field) for field in ZEROED_OBSERVATION_FIELDS
)

_MASK_ARRAY = np.zeros(OBS_DIM, dtype=np.float32)
_MASK_ARRAY[list(RETAINED_OBSERVATION_INDICES)] = 1.0
_MASK_ARRAY.flags.writeable = False


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


HUMAN_SEQUENCE_OBS_VIEW_CONTRACT = {
    "version": HUMAN_SEQUENCE_OBS_VIEW_VERSION,
    "structural_observation_dimensions": OBS_DIM,
    "retained_field_count": len(RETAINED_OBSERVATION_FIELDS),
    "zeroed_field_count": len(ZEROED_OBSERVATION_FIELDS),
    "retained_fields": list(RETAINED_OBSERVATION_FIELDS),
    "retained_indices": list(RETAINED_OBSERVATION_INDICES),
    "zeroed_fields": list(ZEROED_OBSERVATION_FIELDS),
    "zeroed_indices": list(ZEROED_OBSERVATION_INDICES),
    "human_boost_semantics": "native recorder fraction in [0,1], copied directly",
    "native_boost_semantics": "RivalSim boost amount divided by frozen BOOST_SCALE=100",
    "learned_repair": False,
    "observation_adapter_v2": False,
}
HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256 = _canonical_hash(HUMAN_SEQUENCE_OBS_VIEW_CONTRACT)


def observation_view_mask_numpy() -> np.ndarray:
    result = _MASK_ARRAY.copy()
    result.flags.writeable = False
    return result


def observation_view_mask_torch(*, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(_MASK_ARRAY.copy()).to(device=device)


def project_human_sequence_observation(
    observation: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Hard-zero all non-shared fields while preserving the input container type."""

    if observation.shape[-1] != OBS_DIM:
        raise ValueError(f"expected observation last dimension {OBS_DIM}")
    if isinstance(observation, torch.Tensor):
        mask = observation_view_mask_torch(device=observation.device).to(observation.dtype)
        return observation * mask
    value = np.asarray(observation)
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("observation projection requires floating-point input")
    return np.ascontiguousarray(value * _MASK_ARRAY.astype(value.dtype, copy=False))


def _human_and_opponent(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    humans = [car for car in frame.get("cars", ()) if car.get("flags", {}).get("is_local_human")]
    if len(humans) != 1:
        raise ValueError("human sequence view requires one unique local human")
    human = humans[0]
    opponents = [car for car in frame.get("cars", ()) if car is not human]
    if len(opponents) != 1:
        raise ValueError("human sequence view requires exactly one opponent")
    return human, opponents[0]


def direct_human_sequence_observation(
    frame: dict[str, Any], exact_sample: AdaptedSample
) -> np.ndarray:
    """Build the non-learned human-domain view from one exact adapter audit sample."""

    value = np.zeros(OBS_DIM, dtype=np.float32)
    boost_indices = {
        OBS_FIELD_NAMES.index("self.boost"),
        OBS_FIELD_NAMES.index("opponent.boost"),
    }
    for index in RETAINED_OBSERVATION_INDICES:
        if index in boost_indices:
            continue
        if not bool(exact_sample.exact_field_mask[index]):
            raise ValueError(
                f"retained field is not exact in frame {exact_sample.sequence}: "
                f"{OBS_FIELD_NAMES[index]}"
            )
        source = float(exact_sample.partial_observation[index])
        if not np.isfinite(source):
            raise ValueError(f"retained field is nonfinite: {OBS_FIELD_NAMES[index]}")
        value[index] = np.float32(source)

    human, opponent = _human_and_opponent(frame)
    for field, car in (("self.boost", human), ("opponent.boost", opponent)):
        boost = float(car.get("boost", float("nan")))
        if not np.isfinite(boost) or not 0.0 <= boost <= 1.0:
            raise ValueError(f"native recorder boost is not a fraction for {field}: {boost}")
        value[OBS_FIELD_NAMES.index(field)] = np.float32(boost)

    projected = project_human_sequence_observation(value)
    assert isinstance(projected, np.ndarray)
    if not np.isfinite(projected).all():
        raise ValueError("projected human sequence observation is nonfinite")
    return projected


__all__ = [
    "HUMAN_SEQUENCE_OBS_VIEW_CONTRACT",
    "HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256",
    "HUMAN_SEQUENCE_OBS_VIEW_VERSION",
    "RETAINED_OBSERVATION_FIELDS",
    "RETAINED_OBSERVATION_INDICES",
    "ZEROED_OBSERVATION_FIELDS",
    "ZEROED_OBSERVATION_INDICES",
    "direct_human_sequence_observation",
    "observation_view_mask_numpy",
    "observation_view_mask_torch",
    "project_human_sequence_observation",
]
