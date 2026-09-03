"""Source-bound, no-learning dash timing schedules and trace analysis.

The helpers in this module deliberately do not define a production mechanic
detector or reward.  They turn accepted 120 Hz human timing evidence into a
bounded set of native-simulator control sweeps and report literal flip,
surface-contact, and tangent-velocity outcomes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

DASH_SCRIPTED_PROBE_VERSION = "RIVAL2_DASH_SCRIPTED_PHYSICS_PROBE_V1"
ACTION_NAMES = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


@dataclass(frozen=True, slots=True)
class DashProbeCase:
    """One deterministic grounded-origin control sequence."""

    case_id: str
    family: Literal["floor_wavedash", "wall_dash"]
    initial_speed_uu_per_second: float
    first_jump_hold_ticks: int
    second_jump_tick: int
    fire_second_jump: bool
    boost: bool
    wall_sign: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _observed_integer_values(
    attempts: list[dict[str, Any]],
    label: str,
    field: str,
) -> list[int]:
    values = {
        int(row[field])
        for row in attempts
        if row["declared_label"] == label and row[field] is not None
    }
    if not values:
        raise ValueError(f"human timing contains no {label} {field}")
    return sorted(values)


def build_dash_probe_cases(human_timing: dict[str, Any]) -> list[DashProbeCase]:
    """Build the frozen sweep directly from accepted human timing ranges."""

    attempts = human_timing.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("human timing attempts are missing")
    # The literal first-jump hold is release minus onset and is retained in
    # each source row, so derive it without any hand-selected timing.
    floor_jump_holds = sorted(
        {
            int(row["jump_release_physics_frame"])
            - int(row["jump_onset_physics_frame"])
            for row in attempts
            if row["declared_label"] == "wavedash"
        }
    )
    wall_jump_holds = sorted(
        {
            int(row["jump_release_physics_frame"])
            - int(row["jump_onset_physics_frame"])
            for row in attempts
            if row["declared_label"] == "walldash"
        }
    )
    floor_second_jump_ticks = _observed_integer_values(
        attempts,
        "wavedash",
        "jump_to_dodge_ticks",
    )
    wall_second_jump_ticks = _observed_integer_values(
        attempts,
        "walldash",
        "jump_to_dodge_ticks",
    )
    if floor_jump_holds != [5, 6, 7]:
        raise ValueError(f"unexpected human wavedash jump holds: {floor_jump_holds}")
    if wall_jump_holds != [4, 5]:
        raise ValueError(f"unexpected human wall-dash jump holds: {wall_jump_holds}")
    if (
        min(floor_second_jump_ticks) != 116
        or max(floor_second_jump_ticks) != 131
    ):
        raise ValueError("human wavedash dodge range changed")
    if (
        min(wall_second_jump_ticks) != 11
        or max(wall_second_jump_ticks) != 13
    ):
        raise ValueError("human wall-dash dodge range changed")

    # Include interior timing points without claiming that unrecorded points
    # were demonstrated.  They are a deterministic native-physics sweep bounded
    # by the observed extrema.
    floor_ticks = sorted(
        set(floor_second_jump_ticks)
        | set(range(min(floor_second_jump_ticks), max(floor_second_jump_ticks) + 1, 3))
    )
    wall_ticks = list(range(min(wall_second_jump_ticks), max(wall_second_jump_ticks) + 1))
    cases: list[DashProbeCase] = []
    floor_speeds = (0.0, 300.0, 700.0, 1100.0, 1500.0)
    for speed in floor_speeds:
        for hold in floor_jump_holds:
            for second_jump_tick in floor_ticks:
                cases.append(
                    DashProbeCase(
                        case_id=(
                            f"floor-s{int(speed):04d}-h{hold:02d}-"
                            f"d{second_jump_tick:03d}-positive"
                        ),
                        family="floor_wavedash",
                        initial_speed_uu_per_second=speed,
                        first_jump_hold_ticks=hold,
                        second_jump_tick=second_jump_tick,
                        fire_second_jump=True,
                        boost=False,
                        wall_sign=0,
                    )
                )
            cases.append(
                DashProbeCase(
                    case_id=f"floor-s{int(speed):04d}-h{hold:02d}-control",
                    family="floor_wavedash",
                    initial_speed_uu_per_second=speed,
                    first_jump_hold_ticks=hold,
                    second_jump_tick=round(np.median(floor_second_jump_ticks)),
                    fire_second_jump=False,
                    boost=False,
                    wall_sign=0,
                )
            )

    wall_speeds = (900.0, 1300.0, 1700.0, 2100.0)
    for wall_sign in (-1, 1):
        for speed in wall_speeds:
            for hold in wall_jump_holds:
                for boost in (False, True):
                    for second_jump_tick in wall_ticks:
                        cases.append(
                            DashProbeCase(
                                case_id=(
                                    f"wall{wall_sign:+d}-s{int(speed):04d}-"
                                    f"h{hold:02d}-d{second_jump_tick:02d}-"
                                    f"b{int(boost)}-positive"
                                ),
                                family="wall_dash",
                                initial_speed_uu_per_second=speed,
                                first_jump_hold_ticks=hold,
                                second_jump_tick=second_jump_tick,
                                fire_second_jump=True,
                                boost=boost,
                                wall_sign=wall_sign,
                            )
                        )
                    cases.append(
                        DashProbeCase(
                            case_id=(
                                f"wall{wall_sign:+d}-s{int(speed):04d}-"
                                f"h{hold:02d}-b{int(boost)}-control"
                            ),
                            family="wall_dash",
                            initial_speed_uu_per_second=speed,
                            first_jump_hold_ticks=hold,
                            second_jump_tick=round(np.median(wall_second_jump_ticks)),
                            fire_second_jump=False,
                            boost=boost,
                            wall_sign=wall_sign,
                        )
                    )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("dash probe case IDs are not unique")
    return cases


def action_for_case(case: DashProbeCase, tick: int) -> np.ndarray:
    """Return one exact Rival eight-channel action for a probe tick."""

    if tick < 0:
        raise ValueError("tick must be nonnegative")
    action = np.zeros(8, dtype=np.float32)
    if case.family == "wall_dash":
        action[0] = 1.0
        action[6] = float(case.boost)
        if tick < case.first_jump_hold_ticks:
            action[1] = float(case.wall_sign)
            action[2] = 1.0
            action[3] = float(case.wall_sign)
            action[5] = 1.0
        elif tick < case.second_jump_tick:
            span = max(1, case.second_jump_tick - case.first_jump_hold_ticks)
            fraction = (tick - case.first_jump_hold_ticks + 1) / span
            orientation = 1.0 - 2.0 * min(max(fraction, 0.0), 1.0)
            action[1] = float(case.wall_sign) * orientation
            action[2] = orientation
            action[3] = float(case.wall_sign) * orientation
        elif tick <= case.second_jump_tick + 4:
            action[1] = float(-case.wall_sign)
            action[2] = -1.0
            action[3] = float(-case.wall_sign)
            action[5] = float(case.fire_second_jump)
        return action

    if tick < case.first_jump_hold_ticks:
        action[5] = 1.0
        return action
    # The accepted forward human wavedash used a brief nose-up rotation near
    # mid-flight, neutral coast, then six ticks of forward-dodge pitch before
    # the second jump fired at surface arrival.
    nose_up_start = round(case.second_jump_tick * 0.42)
    nose_up_end = nose_up_start + 18
    if nose_up_start <= tick < nose_up_end:
        phase = (tick - nose_up_start + 1) / 18.0
        action[2] = float(math.sin(math.pi * min(phase, 1.0)))
    if case.second_jump_tick - 6 <= tick <= case.second_jump_tick + 1:
        action[2] = -1.0
    if case.second_jump_tick <= tick <= case.second_jump_tick + 1:
        action[5] = float(case.fire_second_jump)
    return action


def action_matrix(cases: list[DashProbeCase], tick: int) -> np.ndarray:
    """Vectorize :func:`action_for_case` for a native simulator batch."""

    if not cases:
        raise ValueError("dash probe case list is empty")
    return np.stack([action_for_case(case, tick) for case in cases])


def _first_true(mask: np.ndarray, start: int = 0) -> int | None:
    found = np.flatnonzero(mask[max(0, start) :])
    return None if not found.size else int(found[0] + max(0, start))


def _tangent_speed(velocity: np.ndarray, normal: np.ndarray) -> float | None:
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-6:
        return None
    unit = normal / length
    tangent = velocity - unit * float(np.dot(velocity, unit))
    return float(np.linalg.norm(tangent))


def analyze_dash_trace(
    cases: list[DashProbeCase],
    trace: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Measure literal flip, landing, and tangent-gain outcomes."""

    required = {
        "car_position",
        "car_velocity",
        "on_ground",
        "has_flipped",
        "is_flipping",
        "wheel_contact_count",
        "world_contact_normal",
        "action",
    }
    missing = required - set(trace)
    if missing:
        raise ValueError(f"dash trace is missing {sorted(missing)}")
    state_ticks = trace["car_position"].shape[0]
    worlds = len(cases)
    if state_ticks < 2 or trace["action"].shape != (state_ticks - 1, worlds, 8):
        raise ValueError("dash trace state/action lengths do not align")
    for name in required - {"action"}:
        if trace[name].shape[1] != worlds:
            raise ValueError(f"dash trace world count mismatch: {name}")

    results: list[dict[str, Any]] = []
    for world, case in enumerate(cases):
        flipped = (trace["has_flipped"][:, world] != 0) | (
            trace["is_flipping"][:, world] != 0
        )
        flip_onset_state = _first_true(flipped & ~np.r_[False, flipped[:-1]])
        wheel = trace["wheel_contact_count"][:, world] > 0
        contact_or_ground = wheel | (trace["on_ground"][:, world] != 0)
        airborne_state = _first_true(~contact_or_ground, start=1)
        landing_state: int | None = None
        search_start = (
            max(1, flip_onset_state - 1)
            if flip_onset_state is not None
            else (airborne_state or 1)
        )
        for state_tick in range(search_start, state_ticks):
            if not contact_or_ground[state_tick]:
                continue
            previously_airborne = bool(
                np.any(~contact_or_ground[max(0, state_tick - 8) : state_tick])
            )
            if previously_airborne:
                landing_state = state_tick
                break

        flip_action_tick = (
            None if flip_onset_state is None else max(0, flip_onset_state - 1)
        )
        landing_action_tick = (
            None if landing_state is None else max(0, landing_state - 1)
        )
        normal = (
            np.zeros(3, dtype=np.float32)
            if landing_state is None
            else trace["world_contact_normal"][landing_state, world].astype(
                np.float64
            )
        )
        if float(np.linalg.norm(normal)) <= 1.0e-6:
            normal = np.asarray(
                (0.0, 0.0, 1.0)
                if case.family == "floor_wavedash"
                else (-float(case.wall_sign), 0.0, 0.0),
                dtype=np.float64,
            )
        before_state = (
            max(0, case.second_jump_tick)
            if flip_onset_state is None
            else max(0, flip_onset_state - 1)
        )
        after_state = (
            None if landing_state is None else min(state_ticks - 1, landing_state + 6)
        )
        before_speed = _tangent_speed(
            trace["car_velocity"][before_state, world], normal
        )
        after_speed = (
            None
            if after_state is None
            else _tangent_speed(trace["car_velocity"][after_state, world], normal)
        )
        tangent_gain = (
            None
            if before_speed is None or after_speed is None
            else after_speed - before_speed
        )
        unit_normal = normal / max(float(np.linalg.norm(normal)), 1.0e-6)
        surface_ok = (
            abs(float(unit_normal[2])) >= 0.70
            if case.family == "floor_wavedash"
            else abs(float(unit_normal[2])) <= 0.30
        )
        requested_to_landing = (
            None
            if landing_action_tick is None
            else landing_action_tick - case.second_jump_tick
        )
        source_window_ok = (
            requested_to_landing is not None
            and (
                0 <= requested_to_landing <= 1
                if case.family == "floor_wavedash"
                else 0 <= requested_to_landing <= 5
            )
        )
        productive = bool(
            case.fire_second_jump
            and flip_action_tick is not None
            and landing_action_tick is not None
            and source_window_ok
            and surface_ok
            and tangent_gain is not None
            and tangent_gain >= 100.0
        )
        results.append(
            {
                **case.to_dict(),
                "origin_surface_supported": bool(contact_or_ground[0]),
                "airborne_state_tick": airborne_state,
                "flip_action_tick": flip_action_tick,
                "landing_action_tick": landing_action_tick,
                "requested_to_landing_ticks": requested_to_landing,
                "landing_contact_normal": unit_normal.tolist(),
                "surface_class_matches": bool(surface_ok),
                "surface_tangent_speed_before_uu_per_second": before_speed,
                "surface_tangent_speed_after_uu_per_second": after_speed,
                "surface_tangent_speed_gain_uu_per_second": tangent_gain,
                "source_timing_window_matches": bool(source_window_ok),
                "productive_source_timed_landing": productive,
            }
        )
    return results


def summarize_dash_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize positives and no-second-jump controls by surface family."""

    summary: dict[str, Any] = {}
    for family in ("floor_wavedash", "wall_dash"):
        grouped = [row for row in rows if row["family"] == family]
        positive = [row for row in grouped if row["fire_second_jump"]]
        controls = [row for row in grouped if not row["fire_second_jump"]]

        def success_fraction(selected: list[dict[str, Any]]) -> float:
            return sum(
                bool(row["productive_source_timed_landing"]) for row in selected
            ) / max(1, len(selected))

        gains = [
            float(row["surface_tangent_speed_gain_uu_per_second"])
            for row in positive
            if row["surface_tangent_speed_gain_uu_per_second"] is not None
        ]
        best = sorted(
            positive,
            key=lambda row: (
                bool(row["productive_source_timed_landing"]),
                float(row["surface_tangent_speed_gain_uu_per_second"] or -1e9),
            ),
            reverse=True,
        )[:12]
        summary[family] = {
            "positive_cases": len(positive),
            "control_cases": len(controls),
            "positive_success_fraction": success_fraction(positive),
            "control_success_fraction": success_fraction(controls),
            "positive_tangent_gain_uu_per_second": {
                "minimum": min(gains) if gains else None,
                "median": float(np.median(gains)) if gains else None,
                "maximum": max(gains) if gains else None,
            },
            "best_case_ids": [row["case_id"] for row in best],
        }
    return summary


__all__ = [
    "ACTION_NAMES",
    "DASH_SCRIPTED_PROBE_VERSION",
    "DashProbeCase",
    "action_for_case",
    "action_matrix",
    "analyze_dash_trace",
    "build_dash_probe_cases",
    "summarize_dash_results",
]
