"""Offline observation-coverage and 120 Hz input-variation diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from rivalsim.human_demo.format import ACTION_NAMES
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_V2_120HZ_HASH,
    OBS_FIELD_NAMES,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
)


def rival_observation_mapping_report() -> dict[str, Any]:
    """Classify every current Rival observation field without inventing values."""

    rows = []
    for field in OBS_FIELD_NAMES:
        status, source, note = _classify_observation_field(field)
        rows.append({"field": field, "status": status, "source": source, "note": note})
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    return {
        "format": "RIVALRL_NATIVE_TO_RIVAL_OBSERVATION_MAPPING_V1",
        "rival_observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
        "rival_observation_schema_sha256": OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "field_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "fields": rows,
        "policy": "Unavailable fields are reported, never silently filled.",
    }


def _classify_observation_field(field: str) -> tuple[str, str, str]:
    if field.startswith("ball."):
        return (
            "exact_derivable",
            "frame.ball plus current Rival normalization",
            "Native RBActor value; team canonicalization and scaling are deterministic.",
        )
    if field.startswith("relative."):
        return (
            "exact_derivable",
            "frame ball/car transforms",
            "Subtraction and local-team canonicalization are deterministic.",
        )
    if field.startswith("previous_action."):
        return (
            "approximately_derivable",
            "120 Hz frame.rival_action history",
            "The immediately preceding frame is exact under V2; the first recorded frame "
            "has no pre-session predecessor and is never silently filled.",
        )
    if field.startswith("boost_pad."):
        return (
            "approximately_derivable",
            "event-discovered native boost pickups",
            "SDK exposes pickup/spawn events but no complete pad enumeration/cooldown array.",
        )
    if field.startswith("lifecycle."):
        event_fields = {
            "lifecycle.kickoff_reset",
            "lifecycle.self_touch_event",
            "lifecycle.opponent_touch_event",
            "lifecycle.self_demoed_event",
            "lifecycle.opponent_demoed_event",
        }
        if field in event_fields:
            return (
                "exact_derivable",
                "native event stream",
                "Exact for events observed after recording begins.",
            )
        return (
            "approximately_derivable",
            "frame/event history",
            "Session may begin after episode or last-touch origin.",
        )
    if field.startswith("self.") or field.startswith("opponent."):
        suffix = field.split(".", 1)[1]
        derived = {
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
            "jump_available",
        }
        direct = {
            "on_ground",
            "has_jumped",
            "has_double_jumped",
            "wheel_contact.front_left",
            "wheel_contact.front_right",
            "wheel_contact.back_left",
            "wheel_contact.back_right",
            "is_supersonic",
        }
        approximate = {
            "boost",
            "is_jumping",
            "has_flipped",
            "is_flipping",
            "dodge_available",
            "is_demoed",
            "demo_timer_remaining",
            "jump_time",
            "air_time",
            "air_time_since_jump",
            "flip_time",
            "boosting_time",
        }
        unavailable = {"time_since_boosted", "supersonic_time", "sticky_ticks"}
        if suffix in direct:
            return "exact_direct", "frame.cars", "Native wrapper/component value."
        if suffix in derived:
            return (
                "exact_derivable",
                "frame.cars plus current Rival normalization",
                "Team canonicalization, scaling, basis conversion, or boolean derivation "
                "is deterministic.",
            )
        if suffix in approximate:
            return (
                "approximately_derivable",
                "native components and event history",
                "Native and RivalSim timer/state semantics are not proven identical.",
            )
        if suffix in unavailable:
            return (
                "unavailable",
                "none in pinned SDK",
                "The recorder does not synthesize this state.",
            )
    return "unavailable", "none", "No exact native source is exposed by the pinned SDK."


def action_variation_report(
    frames: Iterable[dict[str, Any]],
    *,
    window_size: int = 4,
    analog_epsilon: float = 1e-4,
    session_class: str = "unlabeled",
) -> dict[str, Any]:
    """Preserve the historical four-tick variation diagnostic for comparison."""

    if window_size <= 1:
        raise ValueError("window_size must exceed one")
    frame_list = list(frames)
    channel_ranges: dict[str, list[float]] = {name: [] for name in ACTION_NAMES}
    changed = {name: 0 for name in ACTION_NAMES}
    valid_windows = 0
    skipped_gap_windows = 0
    all_constant = 0
    for start in range(0, len(frame_list) - window_size + 1, window_size):
        window = frame_list[start : start + window_size]
        sequences = [int(frame["sequence"]) for frame in window]
        physics = [int(frame["physics_frame"]) for frame in window]
        if any(b != a + 1 for a, b in pairwise(sequences)) or any(
            b != a + 1 for a, b in pairwise(physics)
        ):
            skipped_gap_windows += 1
            continue
        valid_windows += 1
        window_constant = True
        for name in ACTION_NAMES:
            values = [float(frame["rival_action"][name]) for frame in window]
            variation = max(values) - min(values)
            channel_ranges[name].append(variation)
            is_constant = variation <= (analog_epsilon if name in ACTION_NAMES[:5] else 0.0)
            if not is_constant:
                changed[name] += 1
                window_constant = False
        all_constant += int(window_constant)
    per_channel = {}
    for name in ACTION_NAMES:
        values = channel_ranges[name]
        per_channel[name] = {
            "changed_window_count": changed[name],
            "changed_window_fraction": changed[name] / valid_windows if valid_windows else 0.0,
            "mean_range": sum(values) / len(values) if values else 0.0,
            "max_range": max(values, default=0.0),
        }
    return {
        "format": "RIVALRL_120HZ_TO_CANDIDATE_30HZ_DIAGNOSTIC_V1",
        "window_size": window_size,
        "analog_epsilon": analog_epsilon,
        "session_class": session_class,
        "input_frame_count": len(frame_list),
        "valid_window_count": valid_windows,
        "skipped_gap_window_count": skipped_gap_windows,
        "all_channels_constant_window_count": all_constant,
        "all_channels_constant_window_fraction": (
            all_constant / valid_windows if valid_windows else 0.0
        ),
        "per_channel": per_channel,
        "decision": None,
        "decision_policy": (
            "Historical evidence only; RIVAL2_ACTION_V2_120HZ uses every native frame "
            "directly and performs no temporal reduction."
        ),
    }


def action_variation_collection_report(
    sessions: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    window_size: int = 4,
    analog_epsilon: float = 1e-4,
) -> dict[str, Any]:
    """Aggregate independent sessions and identify classes with the most variation."""

    session_reports = [
        action_variation_report(
            frames,
            window_size=window_size,
            analog_epsilon=analog_epsilon,
            session_class=session_class,
        )
        for session_class, frames in sessions
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in session_reports:
        grouped[str(report["session_class"])].append(report)
    classes = []
    for session_class, reports in grouped.items():
        valid_windows = sum(int(report["valid_window_count"]) for report in reports)
        all_constant = sum(
            int(report["all_channels_constant_window_count"]) for report in reports
        )
        per_channel = {}
        for channel in ACTION_NAMES:
            changed = sum(
                int(report["per_channel"][channel]["changed_window_count"])
                for report in reports
            )
            weighted_range = sum(
                float(report["per_channel"][channel]["mean_range"])
                * int(report["valid_window_count"])
                for report in reports
            )
            per_channel[channel] = {
                "changed_window_count": changed,
                "changed_window_fraction": changed / valid_windows if valid_windows else 0.0,
                "mean_range": weighted_range / valid_windows if valid_windows else 0.0,
                "max_range": max(
                    (
                        float(report["per_channel"][channel]["max_range"])
                        for report in reports
                    ),
                    default=0.0,
                ),
            }
        constant_fraction = all_constant / valid_windows if valid_windows else 0.0
        classes.append(
            {
                "session_class": session_class,
                "session_count": len(reports),
                "input_frame_count": sum(
                    int(report["input_frame_count"]) for report in reports
                ),
                "valid_window_count": valid_windows,
                "skipped_gap_window_count": sum(
                    int(report["skipped_gap_window_count"]) for report in reports
                ),
                "all_channels_constant_window_count": all_constant,
                "all_channels_constant_window_fraction": constant_fraction,
                "any_channel_changed_window_fraction": 1.0 - constant_fraction,
                "per_channel": per_channel,
            }
        )
    classes.sort(
        key=lambda row: (
            -float(row["any_channel_changed_window_fraction"]),
            str(row["session_class"]),
        )
    )
    return {
        "format": "RIVALRL_120HZ_TO_CANDIDATE_30HZ_COLLECTION_DIAGNOSTIC_V1",
        "window_size": window_size,
        "analog_epsilon": analog_epsilon,
        "session_count": len(session_reports),
        "classes_by_intra_window_variation": classes,
        "sessions": session_reports,
        "decision": None,
        "decision_policy": (
            "Historical evidence only; RIVAL2_ACTION_V2_120HZ uses every native frame "
            "directly and performs no temporal reduction."
        ),
    }


def human_action_alignment_report() -> dict[str, Any]:
    """Return the frozen direct native-frame to Rival V2 action relation."""

    return {
        "format": "RIVALRL_NATIVE_120_TO_RIVAL2_ACTION_V2_120HZ_V1",
        "source": {
            "schema": "RIVALRL_NATIVE_DEMO_V1",
            "cadence_hz": 120,
            "field": "frame.rival_action",
            "authoritative_source": (
                "Rocket League ControllerInput consumed at the physics application hook"
            ),
        },
        "target": {
            "action_version": RIVAL2_ACTION_V2_120HZ_VERSION,
            "action_contract_sha256": ACTION_CONTRACT_V2_120HZ_HASH,
            "policy_hz": 120,
            "channels": list(ACTION_NAMES),
        },
        "alignment": (
            "Rocket League native physics frame N -> one eight-channel target -> "
            "Rival 120 Hz policy decision N"
        ),
        "temporal_reduction": None,
        "averaging": False,
        "subsampling": False,
        "four_frame_combination": False,
        "observation_quality_policy": (
            "Action cadence is exact; observation fields retain the separate field-quality "
            "classifications and unavailable values are not synthesized."
        ),
    }
