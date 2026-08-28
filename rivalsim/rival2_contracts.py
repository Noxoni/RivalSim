"""Frozen Rival 2.0 v0.5 observation, action, reward, and episode identities."""

from __future__ import annotations

import hashlib
import json
from typing import Final

RIVAL2_OBS_VERSION: Final = "RIVAL2_OBS_V1"
RIVAL2_ACTION_VERSION: Final = "RIVAL2_ACTION_V1"
RIVAL2_OBS_V2_120HZ_VERSION: Final = "RIVAL2_OBS_V2_120HZ"
RIVAL2_ACTION_V2_120HZ_VERSION: Final = "RIVAL2_ACTION_V2_120HZ"
RIVAL2_REWARD_VERSION: Final = "RIVAL2_REWARD_V1"
RIVAL2_REWARD_V2_VERSION: Final = "RIVAL2_REWARD_V2"
RIVAL2_REWARD_ACQUISITION_V1_VERSION: Final = "RIVAL2_REWARD_ACQUISITION_V1"
RIVAL2_REWARD_GOAL_ONLY_VERSION: Final = "RIVAL2_REWARD_GOAL_ONLY_V1"
RIVAL2_REWARD_SCORING_V1_VERSION: Final = "RIVAL2_REWARD_SCORING_V1"
RIVAL2_REWARD_GAMEPLAY_V1_VERSION: Final = "RIVAL2_REWARD_GAMEPLAY_V1"
RIVAL2_REWARD_GAMEPLAY_V2_VERSION: Final = "RIVAL2_REWARD_GAMEPLAY_V2"
RIVAL2_REWARD_GAMEPLAY_V3_VERSION: Final = "RIVAL2_REWARD_GAMEPLAY_V3"
RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION: Final = "RIVAL2_REWARD_GAMEPLAY_120_V1"
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

SCORING_PROGRESS_COEFFICIENT: Final = 0.5
SCORING_APPROACH_COEFFICIENT: Final = 0.10
SCORING_TOUCH_REWARD: Final = 0.02
SCORING_DEMOLITION_REWARD: Final = 0.10
SCORING_JUMP_RISING_EDGE_COST: Final = -0.002
SCORING_FLIP_ONSET_COST: Final = -0.01

GAMEPLAY_SPEED_COEFFICIENT: Final = 0.00010
GAMEPLAY_SUPERSONIC_REWARD: Final = 0.00020
GAMEPLAY_BOOST_USE_REWARD: Final = 0.00005
GAMEPLAY_SMALL_PAD_PICKUP_REWARD: Final = 0.001
GAMEPLAY_BIG_PAD_PICKUP_REWARD: Final = 0.005
GAMEPLAY_SAVE_REWARD: Final = 0.75
GAMEPLAY_SAVE_THREAT_HORIZON_SECONDS: Final = 2.0
GAMEPLAY_STRICT_DOUBLE_DASH_REWARD: Final = 0.005
GAMEPLAY_V3_MECHANICS_EVENT_REWARD: Final = 0.005
GAMEPLAY_V3_MECHANICS_EPISODE_BUDGET: Final = 0.05
GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS: Final = 10
GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY: Final = -0.01
GAMEPLAY_120_SPEED_COEFFICIENT: Final = GAMEPLAY_SPEED_COEFFICIENT / 4.0
GAMEPLAY_120_SUPERSONIC_REWARD: Final = GAMEPLAY_SUPERSONIC_REWARD / 4.0
GAMEPLAY_120_BOOST_USE_REWARD: Final = GAMEPLAY_BOOST_USE_REWARD / 4.0

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

# V1 remains byte-for-byte immutable.  V2 deliberately preserves the 182-field
# shape/order/normalization while changing the temporal meaning of decision
# history and one-step event fields to the immediately preceding 120 Hz tick.
OBSERVATION_SCHEMA_V2_120HZ: Final = {
    **OBSERVATION_SCHEMA,
    "version": RIVAL2_OBS_V2_120HZ_VERSION,
    "history_updates": "every 120 Hz physics/policy tick",
    "temporal_semantics": {
        "previous_action.*": "action emitted on the immediately preceding 120 Hz tick",
        "lifecycle.kickoff_reset": "reset visible at the immediately preceding tick boundary",
        "lifecycle.self_touch_event": "unique touch onset during the immediately preceding tick",
        "lifecycle.opponent_touch_event": (
            "opponent unique touch onset during the immediately preceding tick"
        ),
        "lifecycle.self_demoed_event": "demolition during the immediately preceding tick",
        "lifecycle.opponent_demoed_event": "demolition during the immediately preceding tick",
    },
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

ACTION_CONTRACT_V2_120HZ: Final = {
    **ACTION_CONTRACT,
    "version": RIVAL2_ACTION_V2_120HZ_VERSION,
    "policy_hz": 120,
    "hold_ticks": 1,
    "temporal_alignment": (
        "one newly evaluated Rival action for each authoritative 120 Hz physics tick"
    ),
    "human_demonstration_alignment": (
        "native Rocket League physics frame N maps directly to Rival policy decision N"
    ),
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

REWARD_ACQUISITION_V1_CONTRACT: Final = {
    "version": RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    "cadence_hz": 30,
    "zero_sum": False,
    "goal": {"score": 10.0, "concede": -10.0, "zero_sum": True},
    "progress": {
        "coefficient": 0.5,
        "progress_y_scale": PROGRESS_Y_SCALE,
        "zero_sum": True,
    },
    "approach": {
        "coefficient": 1.0,
        "distance_scale": APPROACH_DISTANCE_SCALE,
        "distance": "true 3D Euclidean car-center to ball-center distance in unreal units",
        "before": "start of four-tick decision interval",
        "after": "final pre-reset transition state after four physics ticks",
        "composition": "(distance_before - distance_after) / distance_scale per agent",
        "positive_condition": "positive only when true distance decreases",
        "proximity_reward": False,
        "reset_motion_excluded": True,
        "zero_sum": False,
    },
    "first_legitimate_touch_per_player_per_episode": {
        "reward": 1.0,
        "stacks_with_unique_touch_reward": True,
        "zero_sum": False,
    },
    "unique_touch_per_player": {
        "reward": 0.20,
        "continuous_contact_latched": True,
        "zero_sum": False,
    },
    "unique_demolition": {
        "reward": 0.10,
        "zero_sum": True,
        "unchanged_from": RIVAL2_REWARD_VERSION,
    },
    "no_touch_failure": {
        "seconds": 15.0,
        "reward_per_player_without_any_episode_touch": -0.5,
        "player_with_prior_touch_penalized": False,
        "training_episode_effect": "RIVAL2_EPISODE_V1 truncation and kickoff reset",
    },
    "direct_mechanic_rewards": [],
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

REWARD_SCORING_V1_CONTRACT: Final = {
    "version": RIVAL2_REWARD_SCORING_V1_VERSION,
    "cadence_hz": 30,
    "zero_sum": False,
    "goal": {"score": 10.0, "concede": -10.0, "zero_sum": True},
    "progress": {
        "coefficient": SCORING_PROGRESS_COEFFICIENT,
        "progress_y_scale": PROGRESS_Y_SCALE,
        "composition": "signed canonical delta ball Y toward the opponent goal / progress_y_scale",
        "zero_sum": True,
    },
    "approach": {
        "coefficient": SCORING_APPROACH_COEFFICIENT,
        "distance_scale": APPROACH_DISTANCE_SCALE,
        "distance": "true 3D Euclidean car-center to ball-center distance in unreal units",
        "before": "start of four-tick decision interval",
        "after": "final pre-reset transition state after four physics ticks",
        "composition": "0.10 * (distance_before - distance_after) / distance_scale per agent",
        "reset_motion_excluded": True,
        "zero_sum": False,
    },
    "first_legitimate_touch_per_player_per_match": {
        "reward": 0.0,
    },
    "unique_touch_per_player": {
        "reward": SCORING_TOUCH_REWARD,
        "continuous_contact_latched": True,
        "zero_sum": False,
    },
    "unique_demolition": {
        "reward": SCORING_DEMOLITION_REWARD,
        "zero_sum": True,
        "unchanged_from": RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    },
    "mechanic_initiation_cost": {
        "jump_button_rising_edge": SCORING_JUMP_RISING_EDGE_COST,
        "actual_directional_flip_or_dodge_onset": SCORING_FLIP_ONSET_COST,
        "jump_hold_cost": 0.0,
        "airborne_occupancy_cost": 0.0,
        "named_mechanic_reward": 0.0,
    },
    "no_touch_failure": None,
    "direct_mechanic_rewards": [],
}

REWARD_GAMEPLAY_V1_CONTRACT: Final = {
    "version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    "base_reward_version": RIVAL2_REWARD_VERSION,
    "cadence_hz": 30,
    "zero_sum": True,
    "composition": (
        "calculate the complete Blue competitive reward, then set OrangeReward = -BlueReward"
    ),
    "base_reward": REWARD_CONTRACT,
    "speed": {
        "per_player": (
            f"{GAMEPLAY_SPEED_COEFFICIENT} * clamp(actual_linear_speed / "
            f"{CAR_LINEAR_SPEED_SCALE}, 0, 1)"
        ),
        "competitive_composition": "Blue minus Orange",
        "controller_input_rewarded": False,
        "wheel_speed_rewarded": False,
    },
    "supersonic": {
        "per_player_if_authoritative_state_active": GAMEPLAY_SUPERSONIC_REWARD,
        "competitive_composition": "Blue minus Orange",
        "controller_input_inference": False,
    },
    "boost_use": {
        "per_player_if_actual_boost_thrust_active_during_interval": GAMEPLAY_BOOST_USE_REWARD,
        "competitive_composition": "Blue minus Orange",
        "boost_button_alone_rewarded": False,
        "empty_boost_rewarded": False,
    },
    "boost_pickup": {
        "small_pad_event": GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
        "large_pad_event": GAMEPLAY_BIG_PAD_PICKUP_REWARD,
        "competitive_composition": "Blue minus Orange",
        "authoritative_resource_event": True,
        "unavailable_pad_proximity_rewarded": False,
    },
    "save": {
        "event_reward": GAMEPLAY_SAVE_REWARD,
        "competitive_composition": "Blue saves minus Orange saves",
        "requires_unique_touch_onset": True,
        "pre_touch_threat": {
            "model": "straight-line ball trajectory to the defending player's own scoring plane",
            "goal_geometry": "existing Soccar scoring plane and goal-mouth dimensions",
            "maximum_seconds": GAMEPLAY_SAVE_THREAT_HORIZON_SECONDS,
            "intersection_inside_goal_mouth": True,
        },
        "post_touch_same_threat_must_be_absent": True,
        "touch_tick_goal_must_be_absent": True,
        "continuous_contact_latched": True,
    },
    "approach_reward": None,
    "first_touch_bonus": None,
    "no_touch_reward_penalty": None,
    "proximity_reward": None,
    "direct_mechanic_rewards_or_costs": [],
}

REWARD_GAMEPLAY_V2_CONTRACT: Final = {
    "version": RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    "base_reward_version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    "cadence_hz": 30,
    "zero_sum": True,
    "composition": (
        "BlueReward = GameplayV1Blue + 0.005 * BlueSuccessfulStrictDoubleDashEvents "
        "- 0.005 * OrangeSuccessfulStrictDoubleDashEvents; OrangeReward = -BlueReward"
    ),
    "base_reward": REWARD_GAMEPLAY_V1_CONTRACT,
    "successful_strict_double_dash": {
        "event_reward": GAMEPLAY_STRICT_DOUBLE_DASH_REWARD,
        "competitive_composition": "Blue events minus Orange events",
        "paid_when": "second successful wavedash landing completes the strict sequence",
        "successful_wavedash": {
            "actual_flip_onset": True,
            "wheel_mask_before": 0,
            "on_ground_before": False,
            "maximum_air_time_ticks_at_120_hz": 42,
            "maximum_flip_to_first_landing_ticks_at_120_hz": 24,
        },
        "pair": {
            "maximum_flip_onset_separation_ticks_at_120_hz": 90,
            "intervening_wheel_contact_required": True,
        },
        "individual_flip_reward": 0.0,
        "individual_wavedash_reward": 0.0,
        "partial_attempt_reward": 0.0,
    },
    "other_changes_from_gameplay_v1": [],
}

REWARD_GAMEPLAY_V3_CONTRACT: Final = {
    "version": RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    "transition_from": RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    "cadence_hz": 30,
    "physics_hz": 120,
    "action_hold_ticks": 4,
    "episode_version": RIVAL2_EPISODE_VERSION,
    "zero_sum": True,
    "composition": (
        "BlueReward = Goal + Progress + Demo + Speed + Supersonic + BoostUse + "
        "BoostPickup + Save + (BluePaidMechanics - OrangePaidMechanics) + "
        "(-0.01 * BlueUnnecessaryFlipThroughContacts + 0.01 * "
        "OrangeUnnecessaryFlipThroughContacts); OrangeReward = -BlueReward"
    ),
    "retained_gameplay_v1_terms": {
        "goal": 10.0,
        "progress_coefficient": 0.5,
        "progress_y_scale": PROGRESS_Y_SCALE,
        "demo": 0.10,
        "speed": GAMEPLAY_SPEED_COEFFICIENT,
        "supersonic": GAMEPLAY_SUPERSONIC_REWARD,
        "boost_use": GAMEPLAY_BOOST_USE_REWARD,
        "small_pad_pickup": GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
        "large_pad_pickup": GAMEPLAY_BIG_PAD_PICKUP_REWARD,
        "save": GAMEPLAY_SAVE_REWARD,
    },
    "unconditional_unique_touch": 0.0,
    "gameplay_v2_standalone_double_dash_reward": 0.0,
    "mechanics": {
        "event_reward": GAMEPLAY_V3_MECHANICS_EVENT_REWARD,
        "episode_budget": GAMEPLAY_V3_MECHANICS_EPISODE_BUDGET,
        "max_paid_events_per_player_episode": GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS,
        "quality_scaled": False,
        "canonical_rewardable": [
            "speedflip",
            "half_flip",
            "musty",
            "breezi",
            "redirect",
            "pinch",
            "pogo",
            "successful_dash",
            "ball_reset_acquisition",
            "car_reset_acquisition",
        ],
        "subtype_no_extra_payout": [
            "landing_dash",
            "wall_dash",
            "curve_dash",
            "ceiling_dash",
            "zapdash",
            "rival_double_dash",
            "chain_reset",
            "preflip_reset",
        ],
        "dash_windows_ticks_at_120_hz": {
            "air": 42,
            "landing": 24,
            "zap_jump": 12,
            "zap_dodge": 30,
            "double_dash": 90,
        },
        "dash_tangent_speed_gain_strictly_greater_than_uu_s": 1.0,
        "surface_normal_classes_abs_nz": {"floor_ceiling_min": 0.85, "wall_max": 0.25},
        "reset": {
            "supporting_wheels_min": 3,
            "resource": "AIR_UNTIMED_AVAILABLE",
            "requires_real_resource_transition": True,
            "chain_requires_consumption_or_loss_before_reacquisition": True,
        },
    },
    "unnecessary_flip_through_contact": {
        "penalty_to_offender_before_zero_sum": GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY,
        "candidate": (
            "new legitimate car-ball contact onset during active directional dodge: "
            "is_flipping and has_flipped and non-zero directional flip_rel_torque"
        ),
        "pending_window_ticks_at_120_hz": 5,
        "physical_exemption_classifier_v2": {
            "corpus": "RIVAL2_GAMEPLAY_V3_PHYSICAL_CLASSIFIER_TRACE_V2",
            "contest": {
                "association_window_ticks_at_120_hz": 3,
                "association_ball_displacement_max": 26.472905158996582,
                "opponent_distance_max": 436.1062316894531,
                "self_closing_speed_min": 727.6748657226562,
                "opponent_closing_speed_min": 590.4700012207031,
                "time_to_ball_delta_max": 0.23664760693567713,
                "convergence_velocity_sample": "authoritative_pre_contact",
                "contact_order": "order-neutral; opponent-before and opponent-after are equivalent",
                "recent_opponent_contact_state": (
                    "per-car legitimate opponent-contact onset tick and ball position"
                ),
            },
            "dodge_powered_contact": {
                "total_closing_speed_min": 95.31548309326172,
                "rotational_closing_speed_min": 164.38546752929688,
                "rotational_share_min": 0.2549042037005253,
                "ball_delta_v_min": 303.5580596923828,
            },
            "controlled_flick": {
                "control_history_ticks_min": 5,
                "control_distance_max": 278.92047119140625,
                "control_relative_speed_max": 853.1399946212769,
                "release_window_ticks_at_120_hz": 5,
                "release_distance_min": 76.1766586303711,
                "release_outward_speed_min": 111.21610260009766,
                "release_ball_delta_v_min": 163.4154052734375,
                "control_snapshot": "frozen_at_directional_dodge_onset",
                "release_snapshot": (
                    "first post-contact outward-separation transition within release window"
                ),
                "positive_reward": 0.0,
            },
        },
        "primary_precedence": [
            "EXEMPT_RECOGNIZED_MECHANIC",
            "EXEMPT_CONTROLLED_FLICK",
            "EXEMPT_CONTESTED_50",
            "EXEMPT_POWER_CONTACT",
            "UNNECESSARY_FLIP_THROUGH_CONTACT",
        ],
        "recognized_same_contact_allowlist": ["musty", "breezi", "preflip_reset"],
        "generic_jump_penalty": 0.0,
        "generic_flip_penalty": 0.0,
        "mechanic_failure_penalty": 0.0,
    },
    "disabled_mechanics_reward": [
        "possession",
        "ground_carry_or_dribble",
        "generic_controlled_flick",
        "air_dribble_milestones",
        "pop_reset_beyond_reset_acquisition",
        "double_tap_or_rebound",
        "bare_stall",
        "recovery",
        "generic_jump_flip_or_aerial",
    ],
}

REWARD_GAMEPLAY_120_V1_CONTRACT: Final = {
    "version": RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    "cadence_hz": 120,
    "physics_hz": 120,
    "action_hold_ticks": 1,
    "observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
    "action_version": RIVAL2_ACTION_V2_120HZ_VERSION,
    "episode_version": RIVAL2_EPISODE_VERSION,
    "zero_sum": True,
    "composition": (
        "BlueReward = Goal + Progress + Demo + SpeedOccupancy + SupersonicOccupancy + "
        "BoostUseOccupancy + BoostPickup + Save + BadFlipGuard; OrangeReward = -BlueReward"
    ),
    "event_rewards": {
        "goal": 10.0,
        "demo": 0.10,
        "small_pad_pickup": GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
        "large_pad_pickup": GAMEPLAY_BIG_PAD_PICKUP_REWARD,
        "save": GAMEPLAY_SAVE_REWARD,
        "unnecessary_flip_through_contact": GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY,
        "cadence_scaling": "none; paid once per authoritative physical event",
    },
    "progress": {
        "coefficient": SCORING_PROGRESS_COEFFICIENT,
        "progress_y_scale": PROGRESS_Y_SCALE,
        "displacement": "signed ball displacement over exactly one 120 Hz physics tick",
        "four_tick_telescope": True,
    },
    "dense_time_occupancy": {
        "speed_coefficient": GAMEPLAY_120_SPEED_COEFFICIENT,
        "supersonic_reward": GAMEPLAY_120_SUPERSONIC_REWARD,
        "physical_boost_use_reward": GAMEPLAY_120_BOOST_USE_REWARD,
        "source_30hz_coefficients_divisor": 4,
    },
    "unconditional_unique_touch": 0.0,
    "named_mechanics_reward": 0.0,
    "named_mechanics_hot_path": False,
    "bad_flip_guard": {
        "penalty": GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY,
        "candidate": (
            "new legitimate car-ball contact onset during active directional dodge: "
            "is_flipping and has_flipped and non-zero directional flip_rel_torque"
        ),
        "pending_window_ticks_at_120_hz": 3,
        "active_exemptions_in_precedence_order": [
            "EXEMPT_CONTESTED_50",
            "EXEMPT_POWER_CONTACT",
        ],
        "recognized_mechanic_exemption": False,
        "controlled_flick_exemption": False,
        "generic_jump_penalty": 0.0,
        "generic_flip_penalty": 0.0,
    },
    "quarantined_experimental_labels": list(
        REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["canonical_rewardable"]
    ),
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
OBSERVATION_SCHEMA_V2_120HZ_HASH: Final = _canonical_hash(OBSERVATION_SCHEMA_V2_120HZ)
ACTION_CONTRACT_V2_120HZ_HASH: Final = _canonical_hash(ACTION_CONTRACT_V2_120HZ)
REWARD_CONTRACT_HASH: Final = _canonical_hash(REWARD_CONTRACT)
REWARD_V2_CONTRACT_HASH: Final = _canonical_hash(REWARD_V2_CONTRACT)
REWARD_ACQUISITION_V1_CONTRACT_HASH: Final = _canonical_hash(REWARD_ACQUISITION_V1_CONTRACT)
REWARD_GOAL_ONLY_CONTRACT_HASH: Final = _canonical_hash(REWARD_GOAL_ONLY_CONTRACT)
REWARD_SCORING_V1_CONTRACT_HASH: Final = _canonical_hash(REWARD_SCORING_V1_CONTRACT)
REWARD_GAMEPLAY_V1_CONTRACT_HASH: Final = _canonical_hash(REWARD_GAMEPLAY_V1_CONTRACT)
REWARD_GAMEPLAY_V2_CONTRACT_HASH: Final = _canonical_hash(REWARD_GAMEPLAY_V2_CONTRACT)
REWARD_GAMEPLAY_V3_CONTRACT_HASH: Final = _canonical_hash(REWARD_GAMEPLAY_V3_CONTRACT)
REWARD_GAMEPLAY_120_V1_CONTRACT_HASH: Final = _canonical_hash(
    REWARD_GAMEPLAY_120_V1_CONTRACT
)
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
    *,
    observation_version: str | None = None,
    action_version: str | None = None,
) -> dict[str, str]:
    """Return the frozen contract identity for one explicitly selected reward."""

    if episode_version == RIVAL2_EPISODE_VERSION:
        episode_hash = EPISODE_CONTRACT_HASH
    elif episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION:
        episode_hash = FULL_MATCH_EPISODE_CONTRACT_HASH
    else:
        raise ValueError(f"unsupported Rival 2.0 episode version: {episode_version}")

    if reward_version == RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION:
        selected_observation = observation_version or RIVAL2_OBS_V2_120HZ_VERSION
        selected_action = action_version or RIVAL2_ACTION_V2_120HZ_VERSION
        if selected_observation != RIVAL2_OBS_V2_120HZ_VERSION:
            raise ValueError("Gameplay 120 V1 requires RIVAL2_OBS_V2_120HZ")
        if selected_action != RIVAL2_ACTION_V2_120HZ_VERSION:
            raise ValueError("Gameplay 120 V1 requires RIVAL2_ACTION_V2_120HZ")
        return {
            RIVAL2_OBS_V2_120HZ_VERSION: OBSERVATION_SCHEMA_V2_120HZ_HASH,
            RIVAL2_ACTION_V2_120HZ_VERSION: ACTION_CONTRACT_V2_120HZ_HASH,
            RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION: (
                REWARD_GAMEPLAY_120_V1_CONTRACT_HASH
            ),
            episode_version: episode_hash,
        }

    selected_observation = observation_version or RIVAL2_OBS_VERSION
    selected_action = action_version or RIVAL2_ACTION_VERSION
    if selected_observation != RIVAL2_OBS_VERSION or selected_action != RIVAL2_ACTION_VERSION:
        raise ValueError("historical reward contracts require RIVAL2_OBS_V1/RIVAL2_ACTION_V1")

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
    if reward_version == RIVAL2_REWARD_ACQUISITION_V1_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_ACQUISITION_V1_VERSION: (REWARD_ACQUISITION_V1_CONTRACT_HASH),
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_GOAL_ONLY_VERSION: REWARD_GOAL_ONLY_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_SCORING_V1_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_SCORING_V1_VERSION: REWARD_SCORING_V1_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_GAMEPLAY_V1_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_GAMEPLAY_V1_VERSION: REWARD_GAMEPLAY_V1_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_GAMEPLAY_V2_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_GAMEPLAY_V2_VERSION: REWARD_GAMEPLAY_V2_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    if reward_version == RIVAL2_REWARD_GAMEPLAY_V3_VERSION:
        return {
            RIVAL2_OBS_VERSION: OBSERVATION_SCHEMA_HASH,
            RIVAL2_ACTION_VERSION: ACTION_CONTRACT_HASH,
            RIVAL2_REWARD_GAMEPLAY_V3_VERSION: REWARD_GAMEPLAY_V3_CONTRACT_HASH,
            episode_version: episode_hash,
        }
    raise ValueError(f"unsupported Rival 2.0 reward version: {reward_version}")


__all__ = [
    "ACTION_CONTRACT",
    "ACTION_CONTRACT_HASH",
    "ACTION_CONTRACT_V2_120HZ",
    "ACTION_CONTRACT_V2_120HZ_HASH",
    "ACTION_NAMES",
    "ANALOG_ACTION_NAMES",
    "APPROACH_DISTANCE_SCALE",
    "BUTTON_ACTION_NAMES",
    "CONTRACT_HASHES",
    "EPISODE_CONTRACT",
    "EPISODE_CONTRACT_HASH",
    "FULL_MATCH_EPISODE_CONTRACT",
    "FULL_MATCH_EPISODE_CONTRACT_HASH",
    "GAMEPLAY_120_BOOST_USE_REWARD",
    "GAMEPLAY_120_SPEED_COEFFICIENT",
    "GAMEPLAY_120_SUPERSONIC_REWARD",
    "GAMEPLAY_BIG_PAD_PICKUP_REWARD",
    "GAMEPLAY_BOOST_USE_REWARD",
    "GAMEPLAY_SAVE_REWARD",
    "GAMEPLAY_SAVE_THREAT_HORIZON_SECONDS",
    "GAMEPLAY_SMALL_PAD_PICKUP_REWARD",
    "GAMEPLAY_SPEED_COEFFICIENT",
    "GAMEPLAY_STRICT_DOUBLE_DASH_REWARD",
    "GAMEPLAY_SUPERSONIC_REWARD",
    "GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS",
    "GAMEPLAY_V3_MECHANICS_EPISODE_BUDGET",
    "GAMEPLAY_V3_MECHANICS_EVENT_REWARD",
    "GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SCHEMA_HASH",
    "OBSERVATION_SCHEMA_V2_120HZ",
    "OBSERVATION_SCHEMA_V2_120HZ_HASH",
    "OBS_DIM",
    "OBS_FIELD_NAMES",
    "ORANGE_PAD_REMAP",
    "REWARD_ACQUISITION_V1_CONTRACT",
    "REWARD_ACQUISITION_V1_CONTRACT_HASH",
    "REWARD_CONTRACT",
    "REWARD_CONTRACT_HASH",
    "REWARD_GAMEPLAY_120_V1_CONTRACT",
    "REWARD_GAMEPLAY_120_V1_CONTRACT_HASH",
    "REWARD_GAMEPLAY_V1_CONTRACT",
    "REWARD_GAMEPLAY_V1_CONTRACT_HASH",
    "REWARD_GAMEPLAY_V2_CONTRACT",
    "REWARD_GAMEPLAY_V2_CONTRACT_HASH",
    "REWARD_GAMEPLAY_V3_CONTRACT",
    "REWARD_GAMEPLAY_V3_CONTRACT_HASH",
    "REWARD_GOAL_ONLY_CONTRACT",
    "REWARD_GOAL_ONLY_CONTRACT_HASH",
    "REWARD_SCORING_V1_CONTRACT",
    "REWARD_SCORING_V1_CONTRACT_HASH",
    "REWARD_V2_CONTRACT",
    "REWARD_V2_CONTRACT_HASH",
    "RIVAL2_ACTION_V2_120HZ_VERSION",
    "RIVAL2_FULL_MATCH_EPISODE_VERSION",
    "RIVAL2_OBS_V2_120HZ_VERSION",
    "RIVAL2_REWARD_ACQUISITION_V1_VERSION",
    "RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION",
    "RIVAL2_REWARD_GAMEPLAY_V1_VERSION",
    "RIVAL2_REWARD_GAMEPLAY_V2_VERSION",
    "RIVAL2_REWARD_GAMEPLAY_V3_VERSION",
    "RIVAL2_REWARD_GOAL_ONLY_VERSION",
    "RIVAL2_REWARD_SCORING_V1_VERSION",
    "RIVAL2_REWARD_V2_VERSION",
    "SCORING_APPROACH_COEFFICIENT",
    "SCORING_DEMOLITION_REWARD",
    "SCORING_FLIP_ONSET_COST",
    "SCORING_JUMP_RISING_EDGE_COST",
    "SCORING_PROGRESS_COEFFICIENT",
    "SCORING_TOUCH_REWARD",
    "contract_hashes_for_reward",
]
