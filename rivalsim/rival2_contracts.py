"""Frozen Rival 2.0 v0.5 observation, action, reward, and episode identities."""

from __future__ import annotations

import hashlib
import json
from typing import Final

RIVAL2_OBS_VERSION: Final = "RIVAL2_OBS_V1"
RIVAL2_ACTION_VERSION: Final = "RIVAL2_ACTION_V1"
RIVAL2_REWARD_VERSION: Final = "RIVAL2_REWARD_V1"
RIVAL2_REWARD_V2_VERSION: Final = "RIVAL2_REWARD_V2"
RIVAL2_REWARD_GOAL_ONLY_VERSION: Final = "RIVAL2_REWARD_GOAL_ONLY_V1"
RIVAL2_EPISODE_VERSION: Final = "RIVAL2_EPISODE_V1"
RIVAL2_FULL_MATCH_EPISODE_VERSION: Final = "RIVAL2_EPISODE_FULL_MATCH_V1"

ANALOG_ACTION_NAMES: Final = ("throttle", "steer", "pitch", "yaw", "roll")
BUTTON_ACTION_NAMES: Final = ("jump", "boost", "handbrake")
ACTION_NAMES: Final = ANALOG_ACTION_NAMES + BUTTON_ACTION_NAMES

# For canonical pad index i, this is the physical world index read by an
# orange-perspective agent after the required 180-degree Z rotation.
ORANGE_PAD_REMAP: Final = (
    1,
    0,
    5,
    4,
    3,
    2,
    33,
    32,
    31,
    30,
    29,
    28,
    27,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
    19,
    18,
    17,
    16,
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
)

POSITION_SCALE: Final = (4096.0, 5120.0, 2044.0)
CAR_LINEAR_SPEED_SCALE: Final = 2300.0
BALL_LINEAR_SPEED_SCALE: Final = 6000.0
ANGULAR_SPEED_SCALE: Final = 6.0
BOOST_SCALE: Final = 100.0
DEMO_TIMER_SCALE: Final = 3.0
JUMP_TIME_SCALE: Final = 0.2
AIR_TIME_SCALE: Final = 1.25
FLIP_TIME_SCALE: Final = 0.95
BOOSTING_TIME_SCALE: Final = 0.1
TIME_SINCE_BOOSTED_SCALE: Final = 1.0
SUPERSONIC_TIME_SCALE: Final = 1.0
STICKY_TICK_SCALE: Final = 3.0
EPISODE_AGE_SCALE_TICKS: Final = 45 * 120
NO_TOUCH_AGE_SCALE_TICKS: Final = 15 * 120
PROGRESS_Y_SCALE: Final = 5120.0
APPROACH_DISTANCE_SCALE: Final = 4096.0

_CAR_FIELDS = (
    "position.x",
    "position.y",
    "position.z",
    "linear_velocity.x",
    "linear_velocity.y",
    "linear_velocity.z",
    "forward.x",
    "forward.y",
    "forward.z",
    "up.x",
    "up.y",
    "up.z",
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
    "boost",
    "on_ground",
    "has_jumped",
    "is_jumping",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "jump_available",
    "dodge_available",
    "is_demoed",
    "demo_timer_remaining",
    "wheel_contact.front_left",
    "wheel_contact.front_right",
    "wheel_contact.back_left",
    "wheel_contact.back_right",
    "jump_time",
    "air_time",
    "air_time_since_jump",
    "flip_time",
    "boosting_time",
    "time_since_boosted",
    "is_supersonic",
    "supersonic_time",
    "sticky_ticks",
)

OBS_FIELD_NAMES: Final = (
    "ball.position.x",
    "ball.position.y",
    "ball.position.z",
    "ball.linear_velocity.x",
    "ball.linear_velocity.y",
    "ball.linear_velocity.z",
    "ball.angular_velocity.x",
    "ball.angular_velocity.y",
    "ball.angular_velocity.z",
    *(f"self.{name}" for name in _CAR_FIELDS),
    *(f"opponent.{name}" for name in _CAR_FIELDS),
    "relative.ball_position.x",
    "relative.ball_position.y",
    "relative.ball_position.z",
    "relative.ball_velocity.x",
    "relative.ball_velocity.y",
    "relative.ball_velocity.z",
    "relative.opponent_position.x",
    "relative.opponent_position.y",
    "relative.opponent_position.z",
    "relative.opponent_velocity.x",
    "relative.opponent_velocity.y",
    "relative.opponent_velocity.z",
    *(
        name
        for pad in range(34)
        for name in (f"boost_pad.{pad}.active", f"boost_pad.{pad}.cooldown")
    ),
    *(f"previous_action.{name}" for name in ACTION_NAMES),
    "lifecycle.kickoff_reset",
    "lifecycle.self_touch_event",
    "lifecycle.opponent_touch_event",
    "lifecycle.self_demoed_event",
    "lifecycle.opponent_demoed_event",
    "lifecycle.episode_age",
    "lifecycle.no_touch_age",
)
OBS_DIM: Final = len(OBS_FIELD_NAMES)
assert OBS_DIM == 182


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest().upper()


OBSERVATION_SCHEMA: Final = {
    "version": RIVAL2_OBS_VERSION,
    "dtype": "float32",
    "shape": ["world", "agent", OBS_DIM],
    "team_canonicalization": "orange rotates world 180 degrees around +Z",
    "field_names": list(OBS_FIELD_NAMES),
    "normalization": {
        "position_xyz": list(POSITION_SCALE),
        "car_linear_speed": CAR_LINEAR_SPEED_SCALE,
        "ball_linear_speed": BALL_LINEAR_SPEED_SCALE,
        "angular_speed": ANGULAR_SPEED_SCALE,
        "boost": BOOST_SCALE,
        "demo_timer": DEMO_TIMER_SCALE,
        "jump_time": JUMP_TIME_SCALE,
        "air_time": AIR_TIME_SCALE,
        "flip_time": FLIP_TIME_SCALE,
        "boosting_time": BOOSTING_TIME_SCALE,
        "time_since_boosted": TIME_SINCE_BOOSTED_SCALE,
        "supersonic_time": SUPERSONIC_TIME_SCALE,
        "sticky_ticks": STICKY_TICK_SCALE,
        "episode_age_ticks": EPISODE_AGE_SCALE_TICKS,
        "no_touch_age_ticks": NO_TOUCH_AGE_SCALE_TICKS,
        "timer_outputs_clamped_0_1": True,
    },
    "orange_pad_remap": list(ORANGE_PAD_REMAP),
    "history_updates": "policy decision boundaries only",
}

ACTION_CONTRACT: Final = {
    "version": RIVAL2_ACTION_VERSION,
    "physics_hz": 120,
    "policy_hz": 30,
    "hold_ticks": 4,
    "external_order": list(ACTION_NAMES),
    "actor_output_order": [
        *(f"mu.{name}" for name in ANALOG_ACTION_NAMES),
        *(f"log_std.{name}" for name in ANALOG_ACTION_NAMES),
        *(f"logit.{name}" for name in BUTTON_ACTION_NAMES),
    ],
    "analog_distribution": "Normal pre-tanh with tanh change-of-variables correction",
    "button_distribution": "Bernoulli(sigmoid(logit))",
    "log_std_clamp": [-5.0, 1.0],
    "deterministic": "tanh(mu), sigmoid(logit)>=0.5",
    "fixed_lookup_table": False,
}

REWARD_CONTRACT: Final = {
    "version": RIVAL2_REWARD_VERSION,
    "cadence_hz": 30,
    "zero_sum": True,
    "goal": 10.0,
    "progress_coefficient": 0.5,
    "progress_y_scale": PROGRESS_Y_SCALE,
    "unique_touch": 0.05,
    "unique_demolition": 0.10,
    "other_shaping": [],
}

REWARD_V2_CONTRACT: Final = {
    "version": RIVAL2_REWARD_V2_VERSION,
    "base_reward_version": RIVAL2_REWARD_VERSION,
    "cadence_hz": 30,
    "zero_sum": False,
    "base_reward": REWARD_CONTRACT,
    "approach": {
        "coefficient": 1.0,
        "distance_scale": APPROACH_DISTANCE_SCALE,
        "distance": "true 3D Euclidean car-center to ball-center distance in unreal units",
        "before": "start of four-tick decision interval",
        "after": "final pre-reset transition state after four physics ticks",
        "composition": "(distance_before - distance_after) / distance_scale per agent",
        "reset_motion_excluded": True,
        "zero_sum": False,
    },
    "other_changes_from_v1": [],
}

REWARD_GOAL_ONLY_CONTRACT: Final = {
    "version": RIVAL2_REWARD_GOAL_ONLY_VERSION,
    "cadence_hz": 30,
    "zero_sum": True,
    "goal": 10.0,
    "progress_coefficient": 0.0,
    "unique_touch": 0.0,
    "unique_demolition": 0.0,
    "other_shaping": [],
}

EPISODE_CONTRACT: Final = {
    "version": RIVAL2_EPISODE_VERSION,
    "goal": "terminated",
    "no_touch_timeout_seconds": 15.0,
    "hard_limit_seconds": 45.0,
    "timeout": "truncated with final pre-reset bootstrap",
    "reset": "accepted deterministic standard kickoff lifecycle",
}

FULL_MATCH_EPISODE_CONTRACT: Final = {
    "version": RIVAL2_FULL_MATCH_EPISODE_VERSION,
    "mode": "standard 1v1 Soccar",
    "physics_hz": 120,
    "policy_hz": 30,
    "regulation_seconds": 300.0,
    "goal": "score persists; standard kickoff reset at the next policy boundary",
    "regulation_end": "terminate on a non-tied score",
    "tied_regulation_end": "standard kickoff into unbounded sudden-death overtime",
    "overtime_goal": "terminate with the scoring side as winner",
    "truncation": None,
    "no_touch_timeout_seconds": 15.0,
    "no_touch_timeout_effect": "diagnostic counter only; never resets or truncates the match",
    "reset": "new five-minute match with the next standard kickoff layout",
}

OBSERVATION_SCHEMA_HASH: Final = _canonical_hash(OBSERVATION_SCHEMA)
ACTION_CONTRACT_HASH: Final = _canonical_hash(ACTION_CONTRACT)
REWARD_CONTRACT_HASH: Final = _canonical_hash(REWARD_CONTRACT)
REWARD_V2_CONTRACT_HASH: Final = _canonical_hash(REWARD_V2_CONTRACT)
REWARD_GOAL_ONLY_CONTRACT_HASH: Final = _canonical_hash(REWARD_GOAL_ONLY_CONTRACT)
EPISODE_CONTRACT_HASH: Final = _canonical_hash(EPISODE_CONTRACT)
FULL_MATCH_EPISODE_CONTRACT_HASH: Final = _canonical_hash(FULL_MATCH_EPISODE_CONTRACT)

CONTRACT_HASHES: Final = {
    RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
    RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
    RIVAL2_REWARD_VERSION: REWARD_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION: EPISODE_CONTRACT_HASH,
}


def contract_hashes_for_reward(
    reward_version: str,
    episode_version: str = RIVAL2_EPISODE_VERSION,
) -> dict[str, str]:
    """Return the frozen contract identity for one explicitly selected reward."""

    if episode_version == RIVAL2_EPISODE_VERSION:
        episode_hash = EPISODE_CONTRACT_HASH
    elif episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION:
        episode_hash = FULL_MATCH_EPISODE_CONTRACT_HASH
    else:
        raise ValueError(f"unsupported Rival 2.0 episode version: {episode_version}")

    if reward_version == RIVAL2_REWARD_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_VERSION: REWARD_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_V2_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_V2_VERSION: REWARD_V2_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_GOAL_ONLY_VERSION: REWARD_GOAL_ONLY_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    raise ValueError(f"unsupported Rival 2.0 reward version: {reward_version}")


__all__ = [
    "ACTION_CONTRACT",
    "ACTION_CONTRACT_HASH",
    "ACTION_NAMES",
    "ANALOG_ACTION_NAMES",
    "APPROACH_DISTANCE_SCALE",
    "BUTTON_ACTION_NAMES",
    "CONTRACT_HASHES",
    "EPISODE_CONTRACT",
    "EPISODE_CONTRACT_HASH",
    "FULL_MATCH_EPISODE_CONTRACT",
    "FULL_MATCH_EPISODE_CONTRACT_HASH",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SCHEMA_HASH",
    "OBS_DIM",
    "OBS_FIELD_NAMES",
    "ORANGE_PAD_REMAP",
    "REWARD_CONTRACT",
    "REWARD_CONTRACT_HASH",
    "REWARD_GOAL_ONLY_CONTRACT",
    "REWARD_GOAL_ONLY_CONTRACT_HASH",
    "REWARD_V2_CONTRACT",
    "REWARD_V2_CONTRACT_HASH",
    "RIVAL2_FULL_MATCH_EPISODE_VERSION",
    "RIVAL2_REWARD_GOAL_ONLY_VERSION",
    "RIVAL2_REWARD_V2_VERSION",
    "contract_hashes_for_reward",
]
