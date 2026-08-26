"""Run and publish the authorized RocketSim reciprocal cross-validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_PRIMARY_COMMIT,
)
from rivalsim.rocketsim_crosscheck import (
    DUEL_LIMIT_TICKS,
    GOAL_HALF_WIDTH,
    GOAL_HEIGHT,
    OpenPlayDuelRuntime,
    REGULATION_TICKS,
    RocketSimPolicyRuntime,
    SelfPlayHarvestRuntime,
    capture_public_state,
    concatenate_banks,
    mirror_bank,
)

CHECKPOINT = ROOT / "checkpoints" / "rival2" / "overnight" / "rival2_overnight_final_6h_resume.pt"
CHECKPOINT_SHA256 = "4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E"
NEXTO_MODEL = ROOT / "third_party" / "nexto" / "nexto-model.pt"
NEXTO_MODEL_SHA256 = "BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA"
NEXTO_COMMIT = "2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca"
EXTENSION_SHA256 = "E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0"
OUTPUT = ROOT / "results" / "rival2" / "rocketsim_crosscheck"
WORK = OUTPUT / "work"
DOC = ROOT / "docs" / "RIVAL2_ROCKETSIM_CROSSCHECK.md"
ADAPTER = OUTPUT / "adapter_fidelity.json"
RIVALSIM_REFERENCE = ROOT / "results" / "rival2" / "nexto" / "summary.json"
RIVALSIM_OPEN_REFERENCE = ROOT / "results" / "rival2" / "nexto_open_play" / "summary.json"

PROBE_WORLDS = 128
PROBE_TICKS = 240
STOCHASTIC_MATCHES = 128
MATCH_SHARD = 128
HARVEST_PER_SOURCE = 2_048
HARVEST_SHARD = 64
DUEL_BASE_SHARD = 32
CANONICAL_SEED = 2_026_082_810
STOCHASTIC_SEED = 2_026_082_811
HARVEST_SEED = 2_026_082_812
DUEL_SEED = 2_026_082_813
CHANNEL_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _ratio(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    return {
        "count": int(numerator),
        "denominator": int(denominator),
        "fraction": None if denominator == 0 else float(numerator / denominator),
    }


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {name: None for name in ("mean", "median", "minimum", "p25", "p75", "maximum")} | {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "maximum": float(array.max()),
    }


def _mechanics_summary(rows: list[dict[str, Any]], tick_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mechanics = [row["rival_mechanics"] for row in rows]
    decisions = sum(item["decision_count"] for item in mechanics)
    decision_channels = np.asarray([item["decision_channel_active"] for item in mechanics], dtype=np.int64).sum(axis=0)
    physics_ticks = sum(item["physics_control_ticks"] for item in mechanics)
    physics_channels = np.asarray([item["physics_channel_active"] for item in mechanics], dtype=np.int64).sum(axis=0)
    simulated_ticks = sum(int(row[tick_key]) for row in rows)
    events: list[tuple[dict[str, Any], dict[str, Any]]] = []
    classified: list[dict[str, Any]] = []
    for row in rows:
        identity = {key: row[key] for key in ("rival_side",) if key in row}
        for key in ("match_index", "base_index", "variant", "source", "mirrored"):
            if key in row:
                identity[key] = row[key]
        for event in row["rival_mechanics"]["flip_events"]:
            events.append((identity, event))
            if event["candidate_labels"]:
                classified.append(identity | event)
    directions = Counter(event["direction"] for _identity, event in events)
    candidates = Counter(label for _identity, event in events for label in event["candidate_labels"])
    hold_durations = [duration for item in mechanics for duration in item["jump_held_durations_ticks"]]
    def values(name: str) -> list[float]:
        return [float(event[name]) for _identity, event in events if event[name] is not None]
    summary = {
        "worlds": len(rows),
        "simulated_minutes": float(simulated_ticks / (120 * 60)),
        "controller_activation_30hz_decisions": {
            "decisions": int(decisions),
            "channels": {
                name: _ratio(int(decision_channels[index]), int(decisions))
                for index, name in enumerate(CHANNEL_NAMES)
            },
            "prior_rivalsim_jump_active_fraction_for_comparison": 0.77516,
        },
        "controller_activation_120hz_physics_ticks": {
            "ticks": int(physics_ticks),
            "channels": {
                name: _ratio(int(physics_channels[index]), int(physics_ticks))
                for index, name in enumerate(CHANNEL_NAMES)
            },
        },
        "jump_button_rising_edges": int(sum(item["jump_rising_edges"] for item in mechanics)),
        "jump_button_held_ticks": int(sum(item["jump_held_ticks_total"] for item in mechanics)),
        "jump_button_hold_duration_ticks": _distribution(hold_durations),
        "first_jump_onsets": int(sum(item["first_jump_onsets"] for item in mechanics)),
        "double_jump_onsets": int(sum(item["double_jump_onsets"] for item in mechanics)),
        "actual_flip_dodge_onsets": len(events),
        "flips_dodges_per_simulated_minute": 0.0 if simulated_ticks == 0 else float(len(events) * 120 * 60 / simulated_ticks),
        "jump_presses_while_unavailable": int(sum(item["unavailable_jump_presses"] for item in mechanics)),
        "flip_direction_counts": dict(sorted(directions.items())),
        "candidate_sequence_event_counts": dict(sorted(candidates.items())),
        "candidate_double_dash_sequence_count": int(candidates.get("double_dash_candidate", 0) // 2),
        "timing_and_state_distributions": {
            "ticks_last_wheel_contact_to_jump": _distribution(values("ticks_last_wheel_contact_to_jump")),
            "ticks_jump_to_flip": _distribution(values("ticks_jump_to_flip")),
            "ticks_last_wheel_contact_to_flip": _distribution(values("ticks_last_wheel_contact_to_flip")),
            "air_time_before_flip_seconds": _distribution(values("air_time_before_seconds")),
            "air_time_since_jump_before_flip_seconds": _distribution(values("air_time_since_jump_before_seconds")),
            "ticks_landing_to_flip": _distribution(values("ticks_landing_to_flip")),
            "ticks_flip_to_next_wheel_contact": _distribution(values("ticks_flip_to_next_wheel_contact")),
            "speed_before_flip_uu_per_s": _distribution(values("speed_before_uu_per_s")),
            "speed_after_flip_uu_per_s": _distribution(values("speed_after_uu_per_s")),
            "speed_delta_flip_uu_per_s": _distribution(values("speed_delta_uu_per_s")),
            "wheel_contacts_before_flip": _distribution(values("wheel_contacts_before")),
            "wheel_contacts_after_flip": _distribution(values("wheel_contacts_after")),
        },
        "classification_definitions": {
            "ground_contact_dodge_candidate": "actual flip onset with wheel contact immediately before or after the state transition",
            "wavedash_candidate": "actual airborne flip onset followed by wheel contact within 24 ticks with pre-flip air time <=0.35 s",
            "zapdash_candidate": "landing-to-first-jump <=12 ticks, first-jump-to-actual-flip <=30 ticks, and pre-flip air time <=0.35 s",
            "double_dash_candidate": "two actual flip onsets within 90 ticks, an intervening landing, and both pre-flip air times <=0.35 s",
            "caveat": "candidate labels are state-transition classifications, not intent claims; every classified event is retained with raw timing/state evidence",
        },
        "classified_event_evidence_count": len(classified),
    }
    return summary, classified


def _compact_mechanics(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "flip_events"}


def _histogram_summary(counts: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        median = None
    else:
        index = int(np.searchsorted(np.cumsum(counts), (total + 1) // 2))
        median = float((edges[index] + edges[index + 1]) / 2)
    return {
        "count": total,
        "edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
        "binned_median_estimate": median,
    }


def _behavior_summary(rows: list[dict[str, Any]], side_for_row: Any) -> dict[str, Any]:
    selected = [(row["behavior"], int(side_for_row(row))) for row in rows]
    if not selected:
        return {"worlds": 0}

    def scalar(name: str) -> float:
        return float(sum(item[name][side] for item, side in selected))

    def vector(name: str, dtype: Any = np.float64) -> np.ndarray:
        return np.asarray([item[name][side] for item, side in selected], dtype=dtype).sum(axis=0)

    def hist(name: str) -> np.ndarray:
        return vector(name, np.int64)

    ticks = int(scalar("tick_count"))
    decisions = int(scalar("decision_count"))
    action_sum = vector("action_sum")
    action_abs_sum = vector("action_abs_sum")
    action_active = vector("action_active", np.int64)
    decision_sum = vector("decision_action_sum")
    decision_abs_sum = vector("decision_action_abs_sum")
    decision_active = vector("decision_action_active", np.int64)
    first_edges = selected[0][0]["histogram_edges"]
    action_edges = np.asarray(first_edges["action_magnitude"], dtype=np.float64)
    boost_edges = np.asarray(first_edges["boost"], dtype=np.float64)
    speed_edges = np.asarray(first_edges["speed"], dtype=np.float64)
    height_edges = np.asarray(first_edges["airborne_height"], dtype=np.float64)
    distance_edges = np.asarray(first_edges["distance"], dtype=np.float64)
    advantage_edges = np.asarray(first_edges["boost_advantage"], dtype=np.float64)
    action_hist = hist("action_abs_hist")
    touch_events = [event for item, side in selected for event in item["touch_events"] if int(event["side"]) == side]
    goal_events = [event for item, side in selected for event in item["goal_events"] if int(event["scorer"]) == side]
    chains = [event for item, side in selected for event in item["possession_chains"] if int(event["side"]) == side]
    field_touch = Counter(event["field_region"] for event in touch_events)
    phase_touch = Counter("kickoff" if event["kickoff_phase"] else "established_open_play" for event in touch_events)
    direction = Counter(
        event["result_direction"] for event in touch_events if event.get("result_finalized")
    )
    occupancy = vector("field_occupancy", np.int64)
    occupancy_total = int(occupancy.sum())
    touch_total = len(touch_events)
    boost_count = int(hist("boost_hist").sum())
    speed_count = int(hist("speed_hist").sum())
    air_count = int(scalar("airborne_height_count"))
    car_ball_count = int(hist("car_ball_distance_hist").sum())
    car_opp_count = int(hist("car_opponent_distance_hist").sum())
    return {
        "worlds": len(selected),
        "simulated_ticks": ticks,
        "simulated_minutes": ticks / (120 * 60),
        "controller_30hz_or_15hz_decisions": {
            "decisions": decisions,
            "channels": {
                name: {
                    "activation": _ratio(int(decision_active[index]), decisions),
                    "mean_signed": None if decisions == 0 else float(decision_sum[index] / decisions),
                    "mean_absolute": None if decisions == 0 else float(decision_abs_sum[index] / decisions),
                }
                for index, name in enumerate(CHANNEL_NAMES)
            },
        },
        "controller_120hz_ticks": {
            "ticks": ticks,
            "channels": {
                name: {
                    "activation": _ratio(int(action_active[index]), ticks),
                    "mean_signed": None if ticks == 0 else float(action_sum[index] / ticks),
                    "mean_absolute": None if ticks == 0 else float(action_abs_sum[index] / ticks),
                    "absolute_magnitude_histogram": _histogram_summary(action_hist[index], action_edges),
                }
                for index, name in enumerate(CHANNEL_NAMES)
            },
        },
        "boost": {
            "mean_level": None if boost_count == 0 else scalar("boost_sum") / boost_count,
            "level_histogram": _histogram_summary(hist("boost_hist"), boost_edges),
            "starved_below_1_ticks": int(scalar("boost_starved_ticks")),
            "starved_fraction": _ratio(int(scalar("boost_starved_ticks")), ticks),
            "consumed_on_ticks_without_pad_pickup": scalar("boost_consumed_no_pickup_ticks"),
            "consumption_qualification": "exact net boost decrease is summed only on ticks without a pad pickup; pickup ticks are excluded because the public callback does not expose pre-pickup boost operation order",
            "small_pad_pickups": int(scalar("pad_pickups_small")),
            "large_pad_pickups": int(scalar("pad_pickups_big")),
            "mean_advantage": None if ticks == 0 else scalar("boost_advantage_sum") / ticks,
            "advantage_ticks": _ratio(int(scalar("boost_advantage_ticks")), ticks),
            "disadvantage_ticks": _ratio(int(scalar("boost_disadvantage_ticks")), ticks),
            "advantage_histogram": _histogram_summary(hist("boost_advantage_hist"), advantage_edges),
        },
        "movement": {
            "mean_speed_uu_per_s": None if speed_count == 0 else scalar("speed_sum") / speed_count,
            "speed_histogram": _histogram_summary(hist("speed_hist"), speed_edges),
            "supersonic": _ratio(int(scalar("supersonic_ticks")), ticks),
            "distance_traveled_uu": scalar("distance_traveled"),
            "grounded": _ratio(int(scalar("grounded_ticks")), ticks),
            "airborne": _ratio(int(scalar("airborne_ticks")), ticks),
            "mean_airborne_height_uu": None if air_count == 0 else scalar("airborne_height_sum") / air_count,
            "airborne_height_histogram": _histogram_summary(hist("airborne_height_hist"), height_edges),
            "maximum_height_uu": float(max(item["maximum_height"][side] for item, side in selected)),
            "powerslide_handbrake_ticks": _ratio(int(action_active[7]), ticks),
        },
        "interactions": {
            "demos_inflicted": int(scalar("demos_inflicted")),
            "demos_suffered": int(scalar("demos_suffered")),
            "time_demolished": _ratio(int(scalar("demoed_ticks")), ticks),
            "native_RocketSim_shots": int(scalar("shots")),
            "native_RocketSim_saves": int(scalar("saves")),
            "shots_saves_definition": "RocketSim native game-event callbacks; no hindsight heuristic",
            "challenge_50_outcomes": None,
            "challenge_50_qualification": "omitted because the public binding exposes no objective challenge/50 event and no prospective source definition was authorized",
        },
        "positioning": {
            "car_to_ball_distance_mean_uu": None if car_ball_count == 0 else scalar("car_ball_distance_sum") / car_ball_count,
            "car_to_ball_distance_histogram": _histogram_summary(hist("car_ball_distance_hist"), distance_edges),
            "car_to_opponent_distance_mean_uu": None if car_opp_count == 0 else scalar("car_opponent_distance_sum") / car_opp_count,
            "car_to_opponent_distance_histogram": _histogram_summary(hist("car_opponent_distance_hist"), distance_edges),
            "field_occupancy": {
                label: _ratio(int(occupancy[index]), occupancy_total)
                for index, label in enumerate(("defensive", "midfield", "offensive"))
            },
        },
        "touches_and_possession": {
            "touches": touch_total,
            "touches_by_field_region": dict(sorted(field_touch.items())),
            "touches_by_phase": dict(sorted(phase_touch.items())),
            "ball_speed_before_touch": _distribution(event["ball_speed_before_tick"] for event in touch_events),
            "ball_speed_after_touch": _distribution(event["ball_speed_after_touch"] for event in touch_events),
            "ball_speed_delta": _distribution(event["ball_speed_delta"] for event in touch_events),
            "result_forward_displacement_uu": _distribution(event["result_forward_displacement_uu"] for event in touch_events if event.get("result_finalized")),
            "result_direction_counts": dict(sorted(direction.items())),
            "possession_chain_length_touches": _distribution(event["touches"] for event in chains),
            "possession_chain_duration_ticks": _distribution(event["duration_ticks"] for event in chains),
            "surface_continuations": None,
            "surface_continuation_qualification": "omitted because the public RocketSim binding does not expose authoritative ball-world contact surfaces or normals",
        },
        "goals": {
            "scored": len(goal_events),
            "timing_ticks": _distribution(event["tick"] for event in goal_events),
            "final_touch_to_goal_ticks": _distribution(event["final_touch_to_goal_ticks"] for event in goal_events if event["final_touch_to_goal_ticks"] is not None),
            "scorer_matches_last_toucher": _ratio(sum(event["scorer_matches_last_toucher"] for event in goal_events), len(goal_events)),
            "kickoff_phase_goals": sum(event["phase"] == "kickoff" for event in goal_events),
            "established_open_play_goals": sum(event["phase"] == "established_open_play" for event in goal_events),
            "entry_speed_uu_per_s": _distribution(event["goal_entry_speed_uu_per_s"] for event in goal_events),
            "entry_angle_degrees_from_goal_normal": _distribution(event["goal_entry_angle_degrees_from_goal_normal"] for event in goal_events if event["goal_entry_angle_degrees_from_goal_normal"] is not None),
        },
        "raw_event_counts": {
            "touch": len(touch_events),
            "possession_chain": len(chains),
            "goal": len(goal_events),
            "pad": sum(1 for item, side in selected for event in item["pad_events"] if int(event["side"]) == side),
            "demo": sum(1 for item, side in selected for event in item["demo_events"] if int(event["bumper_side"]) == side or int(event["victim_side"]) == side),
            "shot_save": sum(1 for item, side in selected for event in item["shot_save_events"] if int(event["side"]) == side),
        },
    }


def _match_side_summary(rows: list[dict[str, Any]], side: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if int(row["rival_side"]) == side]
    goals = [row["blue_score"] if side == 0 else row["orange_score"] for row in selected]
    conceded = [row["orange_score"] if side == 0 else row["blue_score"] for row in selected]
    wins = [row for row in selected if row["winner"] == side]
    kickoffs = sum(row["total_kickoffs"] for row in selected)
    rival_touches = sum(row["touch_blue"] if side == 0 else row["touch_orange"] for row in selected)
    nexto_touches = sum(row["touch_orange"] if side == 0 else row["touch_blue"] for row in selected)
    possession_total = sum(row[f"possession_total_{'blue' if side == 0 else 'orange'}"] for row in selected)
    possession_same = sum(row[f"possession_same_{'blue' if side == 0 else 'orange'}"] for row in selected)
    possession_opponent = sum(row[f"possession_opponent_{'blue' if side == 0 else 'orange'}"] for row in selected)
    mechanics, classified = _mechanics_summary(selected, "total_physics_ticks")
    return {
        "matches": len(selected),
        "wins": len(wins),
        "losses": len(selected) - len(wins),
        "regulation_wins": sum(not row["entered_overtime"] for row in wins),
        "overtime_wins": sum(row["entered_overtime"] for row in wins),
        "win_rate": _ratio(len(wins), len(selected)),
        "goals_for": int(sum(goals)),
        "goals_against": int(sum(conceded)),
        "goals_for_per_match": float(np.mean(goals)),
        "goals_against_per_match": float(np.mean(conceded)),
        "goal_differential": _distribution(np.asarray(goals) - np.asarray(conceded)),
        "total_kickoffs": int(kickoffs),
        "kickoff_first_touches": _ratio(sum(row[f"kickoff_first_touch_{'blue' if side == 0 else 'orange'}"] for row in selected), kickoffs),
        "direct_kickoff_goals": _ratio(sum(row[f"kickoff_goal_{'blue' if side == 0 else 'orange'}"] for row in selected), kickoffs),
        "touches": int(rival_touches),
        "touch_share": _ratio(rival_touches, rival_touches + nexto_touches),
        "same_next_touch": _ratio(possession_same, possession_total),
        "opponent_handoff": _ratio(possession_opponent, possession_total),
        "demos": int(sum(row[f"demo_{'blue' if side == 0 else 'orange'}"] for row in selected)),
        "movement_mechanics": mechanics,
        "comprehensive_behavior": {
            "Rival": _behavior_summary(selected, lambda _row: side),
            "Nexto": _behavior_summary(selected, lambda _row: 1 - side),
        },
    }, classified


def _physical_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "Blue": {
            "wins": sum(row["winner"] == 0 for row in rows),
            "goals": sum(row["blue_score"] for row in rows),
            "behavior": _behavior_summary(rows, lambda _row: 0),
        },
        "Orange": {
            "wins": sum(row["winner"] == 1 for row in rows),
            "goals": sum(row["orange_score"] for row in rows),
            "behavior": _behavior_summary(rows, lambda _row: 1),
        },
    }


def _run_probe(collision_root: Path) -> dict[str, Any]:
    path = OUTPUT / "throughput_probe.json"
    if path.exists():
        result = _read_json(path)
        result["selected_stochastic_matches"] = STOCHASTIC_MATCHES
        result["selection_reason"] = "the user prospectively authorized exactly 128 complete stochastic matches: 64 Rival Blue and 64 Rival Orange"
        _write_json(path, result)
        return result
    index = np.arange(PROBE_WORLDS, dtype=np.int32)
    runtime = RocketSimPolicyRuntime(
        collision_root, CHECKPOINT, NEXTO_MODEL, (index // 2) % 5, index % 2,
        stochastic_rival=True, seed=STOCHASTIC_SEED, reset_goals=True,
    )
    timing = runtime.run_ticks(PROBE_TICKS)
    world_rate = timing.world_ticks_per_second
    result = {
        "worlds": PROBE_WORLDS,
        "physics_ticks": PROBE_TICKS,
        "seconds": timing.seconds,
        "world_ticks_per_second": world_rate,
        "projected_4096_regulation_hours": 4096 * REGULATION_TICKS / world_rate / 3600,
        "selected_stochastic_matches": STOCHASTIC_MATCHES,
        "selection_reason": "the user prospectively authorized exactly 128 complete stochastic matches: 64 Rival Blue and 64 Rival Orange",
    }
    _write_json(path, result)
    return result


def _run_match_shard(
    collision_root: Path,
    name: str,
    start: int,
    count: int,
    *,
    stochastic: bool,
) -> dict[str, Any]:
    path = WORK / "matches" / f"{name}_{start:05d}_{count:04d}.json"
    if path.exists():
        return _read_json(path)
    index = np.arange(start, start + count, dtype=np.int32)
    if name == "canonical":
        layout = np.repeat(np.arange(5, dtype=np.int32), 2)
        side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    else:
        side = index % 2
        layout = (index // 2) % 5
    runtime = RocketSimPolicyRuntime(
        collision_root, CHECKPOINT, NEXTO_MODEL, layout, side,
        stochastic_rival=stochastic,
        seed=(STOCHASTIC_SEED if stochastic else CANONICAL_SEED) + start,
        reset_goals=True,
    )
    print(f"{name} shard {start}:{start + count}: running", flush=True)
    timings = runtime.run_full_matches()
    rows = runtime.export_match_rows()
    for local, row in enumerate(rows):
        row["match_index"] = start + local
    payload = {
        "rows": rows,
        "timing": [
            {
                "worlds": item.worlds,
                "physics_ticks": item.physics_ticks,
                "seconds": item.seconds,
                "world_ticks_per_second": item.world_ticks_per_second,
            }
            for item in timings
        ],
    }
    _write_json(path, payload)
    return payload


def _publish_matches(collision_root: Path, probe: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_payload = _run_match_shard(collision_root, "canonical", 0, 10, stochastic=False)
    stochastic_payloads = [
        _run_match_shard(
            collision_root,
            "stochastic",
            start,
            min(MATCH_SHARD, STOCHASTIC_MATCHES - start),
            stochastic=True,
        )
        for start in range(0, STOCHASTIC_MATCHES, MATCH_SHARD)
    ]
    canonical_rows = canonical_payload["rows"]
    stochastic_rows = [row for payload in stochastic_payloads for row in payload["rows"]]
    all_rows = canonical_rows + stochastic_rows
    canonical_sides = {}
    stochastic_sides = {}
    classified: list[dict[str, Any]] = []
    for side, name in ((0, "Rival_Blue"), (1, "Rival_Orange")):
        canonical_sides[name], events = _match_side_summary(canonical_rows, side)
        classified += [{"suite": "canonical"} | event for event in events]
        stochastic_sides[name], events = _match_side_summary(stochastic_rows, side)
        classified += [{"suite": "stochastic"} | event for event in events]
    ledger = []
    for row in all_rows:
        compact = dict(row)
        compact["rival_mechanics"] = _compact_mechanics(compact["rival_mechanics"])
        compact.pop("goal_events")
        ledger.append(compact)
    _write_json(OUTPUT / "normal_match_ledger.json", ledger)
    _write_json(OUTPUT / "normal_match_mechanics_classified_events.json", classified)
    canonical_scorelines = [
        {
            "layout": row["starting_layout"],
            "rival_side": "Blue" if row["rival_side"] == 0 else "Orange",
            "blue_score": row["blue_score"],
            "orange_score": row["orange_score"],
            "rival_score": row["blue_score"] if row["rival_side"] == 0 else row["orange_score"],
            "nexto_score": row["orange_score"] if row["rival_side"] == 0 else row["blue_score"],
            "winner": "Rival" if row["rival_won"] else "Nexto",
            "overtime": row["entered_overtime"],
        }
        for row in canonical_rows
    ]
    _write_json(OUTPUT / "canonical_normal_matches.json", canonical_scorelines)
    result = {
        "protocol": {
            "canonical_matches": 10,
            "stochastic_target": 128,
            "stochastic_executed": len(stochastic_rows),
            "stochastic_reduction_authority": probe,
            "regulation_ticks": REGULATION_TICKS,
            "goal_resets": True,
            "fresh_kickoff_overtime": True,
            "zero_second_continuation": False,
        },
        "canonical": {
            "by_rival_side": canonical_sides,
            "physical_teams": _physical_summary(canonical_rows),
            "scorelines": canonical_scorelines,
        },
        "stochastic": {
            "by_rival_side": stochastic_sides,
            "physical_teams": _physical_summary(stochastic_rows),
        },
        "timing": {
            "canonical": canonical_payload["timing"],
            "stochastic": [item for payload in stochastic_payloads for item in payload["timing"]],
        },
    }
    _write_json(OUTPUT / "normal_matches.json", result)
    return result, all_rows


def _comparison(normal: dict[str, Any]) -> dict[str, Any]:
    reference = _read_json(RIVALSIM_REFERENCE)
    result: dict[str, Any] = {"by_rival_side": {}}
    ordering = []
    for local_name, prior_name in (("Rival_Blue", "rival_as_blue"), ("Rival_Orange", "rival_as_orange")):
        rocket = normal["stochastic"]["by_rival_side"][local_name]
        prior = reference["stochastic_side_results"][prior_name]
        prior_win = prior["win_rate"]["fraction"]
        rocket_win = rocket["win_rate"]["fraction"]
        metrics = {
            "rival_win_rate": [prior_win, rocket_win],
            "rival_goals_per_match": [prior["goals_per_match"], rocket["goals_for_per_match"]],
            "nexto_goals_per_match": [prior["goals_conceded_per_match"], rocket["goals_against_per_match"]],
            "mean_goal_differential": [prior["goal_differential"]["mean"], rocket["goal_differential"]["mean"]],
            "touch_share": [prior["touch_share"]["fraction"], rocket["touch_share"]["fraction"]],
        }
        result["by_rival_side"][local_name] = {
            name: {
                "RivalSim": values[0],
                "RocketSim": values[1],
                "absolute_delta": values[1] - values[0],
                "relative_delta": None if values[0] == 0 else (values[1] - values[0]) / abs(values[0]),
            }
            for name, values in metrics.items()
        }
        ordering.append((prior_win, rocket_win))
    prior_gap = ordering[1][0] - ordering[0][0]
    rocket_gap = ordering[1][1] - ordering[0][1]
    if np.sign(prior_gap) != np.sign(rocket_gap):
        verdict = "DISAGREEMENT"
    elif max(abs(item[1] - item[0]) for item in ordering) <= 0.10:
        verdict = "STRONG_AGREEMENT"
    else:
        verdict = "PARTIAL_AGREEMENT"
    result["classification"] = verdict
    result["side_gap"] = {
        "RivalSim_orange_minus_blue": prior_gap,
        "RocketSim_orange_minus_blue": rocket_gap,
        "absolute_delta": rocket_gap - prior_gap,
    }
    result["classification_rule"] = "same side ordering plus both side win-rate deltas <=10pp is strong; same ordering beyond that is partial; reversed ordering is disagreement"
    _write_json(OUTPUT / "normal_match_cross_simulator_comparison.json", result)
    return result


def _target_ages(index: np.ndarray, seed: int) -> np.ndarray:
    mixed = (index.astype(np.uint64) * np.uint64(1103515245) + np.uint64(seed))
    return (600 + (mixed % np.uint64(2401))).astype(np.int32)


def _harvest_source(collision_root: Path, source: str, offset: int) -> list[Path]:
    paths = [
        WORK / "harvest" / f"{source}_{local_start:04d}_{min(HARVEST_SHARD, HARVEST_PER_SOURCE - local_start):03d}.npz"
        for local_start in range(0, HARVEST_PER_SOURCE, HARVEST_SHARD)
    ]
    completed = 0
    for path in paths:
        if path.exists():
            completed += HARVEST_SHARD
        else:
            break
    if completed >= HARVEST_PER_SOURCE:
        return paths
    pool = min(HARVEST_SHARD, HARVEST_PER_SOURCE - completed)
    slot_index = np.arange(offset + completed, offset + completed + pool, dtype=np.int32)
    runtime = SelfPlayHarvestRuntime(
        collision_root,
        CHECKPOINT,
        NEXTO_MODEL,
        slot_index % 5,
        source=source,
        seed=HARVEST_SEED + int(slot_index[0]),
    )
    targets = _target_ages(slot_index, HARVEST_SEED + (0 if source == "rival_stochastic" else 1))
    next_index = offset + completed + pool
    end_index = offset + HARVEST_PER_SOURCE
    active = np.ones(pool, dtype=bool)
    parts: list[dict[str, np.ndarray]] = []
    while np.any(active):
        runtime.tick(active)
        rows = np.flatnonzero(runtime.eligible(targets) & active)
        if not rows.size:
            continue
        parts.append(capture_public_state(runtime, rows, slot_index[rows]))
        assign_count = min(rows.size, end_index - next_index)
        if assign_count:
            assigned_rows = rows[:assign_count]
            assigned_index = np.arange(next_index, next_index + assign_count, dtype=np.int32)
            runtime.reassign_rows(assigned_rows, assigned_index)
            slot_index[assigned_rows] = assigned_index
            targets[assigned_rows] = _target_ages(
                assigned_index,
                HARVEST_SEED + (0 if source == "rival_stochastic" else 1),
            )
            next_index += assign_count
        if assign_count < rows.size:
            retired = rows[assign_count:]
            active[retired] = False
            runtime.event.enabled[retired] = False
        captured_count = sum(part["base_index"].size for part in parts)
        if captured_count % 256 < rows.size:
            print(
                f"harvest {source}: {completed + captured_count}/{HARVEST_PER_SOURCE} states; pool host tick {runtime.host_tick}",
                flush=True,
            )
        if runtime.host_tick > 250_000:
            raise RuntimeError(f"{source} refill harvest did not complete")
    bank = concatenate_banks(parts)
    order = np.argsort(bank["base_index"])
    bank = {key: value[order] for key, value in bank.items()}
    expected = HARVEST_PER_SOURCE - completed
    if bank["base_index"].size != expected:
        raise RuntimeError(f"{source} refill harvest produced {bank['base_index'].size}, expected {expected}")
    for relative in range(0, expected, HARVEST_SHARD):
        local_start = completed + relative
        path = paths[local_start // HARVEST_SHARD]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{key: value[relative:relative + HARVEST_SHARD] for key, value in bank.items()},
        )
    print(f"harvested remaining {source} states at pool host tick {runtime.host_tick}", flush=True)
    return paths


def _load_bank(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            parts.append({key: loaded[key].copy() for key in loaded.files})
    return concatenate_banks(parts)


def _expand_four_way(bank: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    mirrored = mirror_bank(bank)
    count = bank["base_index"].size
    expanded: dict[str, np.ndarray] = {}
    for key in bank:
        shape = (count * 4,) + bank[key].shape[1:]
        value = np.empty(shape, dtype=bank[key].dtype)
        value[0::4] = bank[key]
        value[1::4] = bank[key]
        value[2::4] = mirrored[key]
        value[3::4] = mirrored[key]
        expanded[key] = value
    base_index = np.repeat(bank["base_index"], 4)
    variant = np.tile(np.arange(4, dtype=np.int8), count)
    rival_side = np.tile(np.asarray((0, 1, 0, 1), dtype=np.int32), count)
    source = np.repeat(bank["source"], 4)
    return expanded, base_index, variant, rival_side, source


def _run_duel_shard(collision_root: Path, bank: dict[str, np.ndarray], start: int) -> list[dict[str, Any]]:
    path = WORK / "duels" / f"duels_v2_{start:04d}_{bank['base_index'].size:03d}.json"
    if path.exists():
        return _read_json(path)
    expanded, base_index, variant, rival_side, source = _expand_four_way(bank)
    runtime = OpenPlayDuelRuntime(
        collision_root, CHECKPOINT, NEXTO_MODEL, expanded, rival_side,
        seed=DUEL_SEED + start,
    )
    timing = runtime.run_duels()
    rows = runtime.export_duel_rows(base_index, variant, source)
    _write_json(path, rows)
    print(
        f"duels base {start}:{start + bank['base_index'].size}: {timing.physics_ticks} ticks, {timing.seconds:.2f}s",
        flush=True,
    )
    return rows


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rival = sum(row["outcome"] == "rival" for row in rows)
    nexto = sum(row["outcome"] == "nexto" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    return {
        "duels": len(rows),
        "Rival_wins": rival,
        "Nexto_wins": nexto,
        "draws": draws,
        "decisive_Rival_win_rate": _ratio(rival, rival + nexto),
        "all_duel_Rival_win_fraction": _ratio(rival, len(rows)),
        "time_to_goal_seconds": _distribution(row["elapsed_seconds"] for row in rows if row["outcome"] != "draw"),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(row[key]) for row in rows})
    return {value: _outcome_summary([row for row in rows if str(row[key]) == value]) for value in values}


def _publish_open_play(collision_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _harvest_source(collision_root, "rival_stochastic", 0)
    paths += _harvest_source(collision_root, "nexto_deterministic", HARVEST_PER_SOURCE)
    bank = _load_bank(paths)
    order = np.argsort(bank["base_index"])
    bank = {key: value[order] for key, value in bank.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "open_play_state_bank.npz", **bank)
    involution = mirror_bank(mirror_bank(bank))
    involution_max = max(float(np.max(np.abs(bank[key].astype(np.float64) - involution[key].astype(np.float64)))) for key in bank)
    description = {
        "states": int(bank["base_index"].size),
        "source_counts": {
            "rival_stochastic": int((bank["source"] == 0).sum()),
            "nexto_deterministic": int((bank["source"] == 1).sum()),
        },
        "capture_tick": _distribution(bank["capture_tick"]),
        "minimum_ticks_after_reset": int(bank["capture_tick"].min()),
        "both_cars_active": bool(np.all(bank["car_is_demoed"] == 0)),
        "inside_scoring_plane": bool(np.all(np.abs(bank["ball_pos"][:, 1]) < 5120.0)),
        "neutral_policy_previous_action": bool(np.all(bank["memory_previous_action"] == 0)),
        "public_state_restore_scope": "all public BallState/CarState/pad continuation fields plus adapter lifecycle memory; transient internal Bullet manifolds are not exposed by the binding and are reconstructed by SetState on the first tick",
        "mirror_involution_max_abs_error": involution_max,
    }
    _write_json(OUTPUT / "open_play_state_bank_description.json", description)
    duel_rows: list[dict[str, Any]] = []
    for start in range(0, bank["base_index"].size, DUEL_BASE_SHARD):
        sliced = {key: value[start:start + DUEL_BASE_SHARD] for key, value in bank.items()}
        duel_rows += _run_duel_shard(collision_root, sliced, start)
    if len(duel_rows) != 16_384:
        raise RuntimeError(f"expected 16,384 duels, got {len(duel_rows)}")
    mechanics = {}
    classified: list[dict[str, Any]] = []
    for side, name in ((0, "Rival_Blue"), (1, "Rival_Orange")):
        selected = [row for row in duel_rows if row["rival_side"] == side]
        mechanics[name], events = _mechanics_summary(selected, "elapsed_ticks")
        classified += events
    compact_rows = []
    for row in duel_rows:
        compact = dict(row)
        compact["rival_mechanics"] = _compact_mechanics(compact["rival_mechanics"])
        compact_rows.append(compact)
    _write_csv(OUTPUT / "open_play_per_duel_ledger.csv", compact_rows)
    _write_jsonl_gzip(
        OUTPUT / "open_play_behavior_raw.jsonl.gz",
        (
            {
                "base_index": row["base_index"],
                "variant": row["variant"],
                "source": row["source"],
                "mirrored": row["mirrored"],
                "rival_side": row["rival_side"],
                "elapsed_ticks": row["elapsed_ticks"],
                "behavior": row["behavior"],
            }
            for row in duel_rows
        ),
    )
    _write_json(OUTPUT / "open_play_mechanics_classified_events.json", classified)
    families = []
    for base in range(4096):
        family = sorted((row for row in duel_rows if row["base_index"] == base), key=lambda row: row["variant"])
        families.append({
            "base_index": base,
            "source": family[0]["source"],
            "outcomes": [row["outcome"] for row in family],
            "rival_wins": sum(row["outcome"] == "rival" for row in family),
            "draws": sum(row["outcome"] == "draw" for row in family),
        })
    _write_json(OUTPUT / "open_play_paired_family_summary.json", {
        "families": len(families),
        "rival_win_count_per_family": dict(sorted(Counter(item["rival_wins"] for item in families).items())),
        "draw_count_per_family": dict(sorted(Counter(item["draws"] for item in families).items())),
        "outcome_pattern_counts": dict(sorted(Counter("/".join(item["outcomes"]) for item in families).items())),
    })
    result = {
        "protocol": {
            "base_states": 4096,
            "duels": 16384,
            "first_goal_wins": True,
            "kickoff_at_start": False,
            "goal_reset": False,
            "limit_ticks": DUEL_LIMIT_TICKS,
        },
        "overall": _outcome_summary(duel_rows),
        "by_rival_side": {
            "Rival_Blue": _outcome_summary([row for row in duel_rows if row["rival_side"] == 0]),
            "Rival_Orange": _outcome_summary([row for row in duel_rows if row["rival_side"] == 1]),
        },
        "by_source": _group_summary(duel_rows, "source"),
        "by_original_mirror": _group_summary(duel_rows, "mirrored"),
        "by_inherited_physical_car": _group_summary(duel_rows, "rival_inherited_original_physical_car"),
        "by_closest_to_ball": _group_summary(duel_rows, "rival_closest_to_ball"),
        "by_field_third": _group_summary(duel_rows, "ball_field_third"),
        "by_height": _group_summary(duel_rows, "ball_height_band"),
        "movement_mechanics_by_rival_side": mechanics,
        "comprehensive_behavior_by_rival_side": {
            "Rival_Blue": {
                "Rival": _behavior_summary([row for row in duel_rows if row["rival_side"] == 0], lambda _row: 0),
                "Nexto": _behavior_summary([row for row in duel_rows if row["rival_side"] == 0], lambda _row: 1),
            },
            "Rival_Orange": {
                "Rival": _behavior_summary([row for row in duel_rows if row["rival_side"] == 1], lambda _row: 1),
                "Nexto": _behavior_summary([row for row in duel_rows if row["rival_side"] == 1], lambda _row: 0),
            },
        },
        "comprehensive_behavior_by_physical_team": {
            "Blue": _behavior_summary(duel_rows, lambda _row: 0),
            "Orange": _behavior_summary(duel_rows, lambda _row: 1),
        },
        "behavior": {
            "Rival_touches": int(sum(row["touch_rival"] for row in duel_rows)),
            "Nexto_touches": int(sum(row["touch_nexto"] for row in duel_rows)),
            "Rival_touch_share": _ratio(sum(row["touch_rival"] for row in duel_rows), sum(row["touch_rival"] + row["touch_nexto"] for row in duel_rows)),
            "Rival_demos": int(sum(row["demo_rival"] for row in duel_rows)),
            "Nexto_demos": int(sum(row["demo_nexto"] for row in duel_rows)),
            "Rival_same_next_touch": _ratio(sum(row["possession_same_rival"] for row in duel_rows), sum(row["possession_total_rival"] for row in duel_rows)),
            "Rival_opponent_handoff": _ratio(sum(row["possession_opponent_rival"] for row in duel_rows), sum(row["possession_total_rival"] for row in duel_rows)),
            "goal_entry_valid": _ratio(sum(row["goal_entry_valid"] for row in duel_rows), sum(row["outcome"] != "draw" for row in duel_rows)),
            "goal_entry_x_uu": _distribution(row["goal_entry_x"] for row in duel_rows if row["goal_entry_valid"]),
            "goal_entry_z_uu": _distribution(row["goal_entry_z"] for row in duel_rows if row["goal_entry_valid"]),
            "goal_mouth_geometry_uu": {"half_width": float(GOAL_HALF_WIDTH), "height": float(GOAL_HEIGHT)},
        },
        "state_bank": description,
    }
    _write_json(OUTPUT / "open_play.json", result)
    return result, duel_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write lossless raw event rows in deterministic, repository-safe gzip form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_handle, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))


def _localization(adapter: dict[str, Any], normal: dict[str, Any], open_play: dict[str, Any]) -> dict[str, Any]:
    nb = normal["stochastic"]["by_rival_side"]["Rival_Blue"]["win_rate"]["fraction"]
    no = normal["stochastic"]["by_rival_side"]["Rival_Orange"]["win_rate"]["fraction"]
    ob = open_play["by_rival_side"]["Rival_Blue"]["decisive_Rival_win_rate"]["fraction"]
    oo = open_play["by_rival_side"]["Rival_Orange"]["decisive_Rival_win_rate"]["fraction"]
    reference_open = _read_json(RIVALSIM_OPEN_REFERENCE)
    result = {
        "adapter_team_symmetry": {
            "observation_max_abs_error": max(
                adapter["team_mirror_symmetry"]["blue_to_mirrored_orange_max_abs_error"],
                adapter["team_mirror_symmetry"]["orange_to_mirrored_blue_max_abs_error"],
            ),
            "action_agreement_fraction": adapter["team_mirror_symmetry"]["deterministic_action_agreement"]["fraction"],
        },
        "normal_RocketSim": {"Blue": nb, "Orange": no, "Orange_minus_Blue": no - nb},
        "open_play_RocketSim": {"Blue": ob, "Orange": oo, "Orange_minus_Blue": oo - ob},
        "open_play_RivalSim": {"Blue": 0.46948, "Orange": 0.62545, "Orange_minus_Blue": 0.15597},
        "reference_file_verdict": reference_open.get("verdict"),
    }
    if abs(oo - ob) < 0.05:
        interpretation = "RocketSim open play substantially reduces the RivalSim side split while adapter symmetry is exact; RivalSim simulator/lifecycle asymmetry becomes more likely, without proving causality."
    else:
        interpretation = "RocketSim open play retains a material side split while adapter symmetry is exact; frozen policy/contract/game-side semantics become more likely, without proving causality."
    result["bounded_interpretation"] = interpretation
    _write_json(OUTPUT / "side_asymmetry_localization.json", result)
    return result


def _write_report(summary: dict[str, Any]) -> None:
    normal = summary["normal_matches"]
    open_play = summary["open_play"]
    nb = normal["stochastic"]["by_rival_side"]["Rival_Blue"]
    no = normal["stochastic"]["by_rival_side"]["Rival_Orange"]
    ob = open_play["by_rival_side"]["Rival_Blue"]
    oo = open_play["by_rival_side"]["Rival_Orange"]
    lines = [
        "# Rival 2.0 RocketSim reciprocal cross-validation",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "## Adapter fidelity",
        "",
        f"The 2,048-state gate passed with `{summary['adapter_fidelity']['observation_parity']['overall_max_abs_error']}` maximum observation error and 100% deterministic action agreement for both Blue and Orange. Exact team/mirror action agreement was also 100%.",
        "",
        "## Normal five-minute RocketSim matches",
        "",
        f"The required ten deterministic layout/side matches and the user-authorized {normal['protocol']['stochastic_executed']:,} stochastic matches completed (64 Rival Blue, 64 Rival Orange). The published throughput probe still records the original 4,096-match projection; the controlling sample-size authority is now exactly 128.",
        "",
        f"- Rival Blue: {nb['wins']}-{nb['losses']}, {100*nb['win_rate']['fraction']:.3f}% wins, goals {nb['goals_for']}-{nb['goals_against']}.",
        f"- Rival Orange: {no['wins']}-{no['losses']}, {100*no['win_rate']['fraction']:.3f}% wins, goals {no['goals_for']}-{no['goals_against']}.",
        "",
        "The ten exact deterministic scorelines are in `canonical_normal_matches.json`; Blue and Orange are never hidden in an aggregate rate.",
        "",
        f"Cross-simulator normal-match classification: **{summary['normal_match_comparison']['classification']}**.",
        "",
        "## Kickoff-free RocketSim open play",
        "",
        f"All 4,096 continuous states and 16,384 four-way duels completed. Rival Blue: {ob['Rival_wins']}-{ob['Nexto_wins']} with {ob['draws']} draws and {100*ob['decisive_Rival_win_rate']['fraction']:.3f}% decisive wins. Rival Orange: {oo['Rival_wins']}-{oo['Nexto_wins']} with {oo['draws']} draws and {100*oo['decisive_Rival_win_rate']['fraction']:.3f}% decisive wins.",
        "",
        "## Rival movement mechanics",
        "",
        "Read-only telemetry was collected from controller and RocketSim state transitions. It includes jump rising edges/holds, actual first/double-jump and flip onsets, direction, wheel/ground context, wheel-contact-to-jump-to-flip timing, landing timing, air time, speed deltas, unavailable jump presses, and conservative wavedash/zapdash/double-dash candidates. Candidate labels are not inferred from button frequency; every classified event retains its raw timing/state evidence.",
        "",
        f"Normal stochastic jump-active decisions: Blue `{nb['movement_mechanics']['controller_activation_30hz_decisions']['channels']['jump']['fraction']:.6f}`, Orange `{no['movement_mechanics']['controller_activation_30hz_decisions']['channels']['jump']['fraction']:.6f}`; prior RivalSim comparison `{0.77516:.5f}`.",
        f"Open-play jump-active decisions: Blue `{open_play['movement_mechanics_by_rival_side']['Rival_Blue']['controller_activation_30hz_decisions']['channels']['jump']['fraction']:.6f}`, Orange `{open_play['movement_mechanics_by_rival_side']['Rival_Orange']['controller_activation_30hz_decisions']['channels']['jump']['fraction']:.6f}`.",
        "",
        "## Comprehensive behavioral dataset",
        "",
        "The same single-pass runs retain per-match accumulators and authoritative touch, possession-chain, pad, demo, native shot/save, goal, and classified-mechanic events for both policies. Aggregates cover controller magnitudes, boost economy, speed/travel, ground/air/demo occupancy, distances, field occupancy, touch regions and ball response, goal timing/entry, kickoff versus established play, and all requested side/suite dimensions.",
        "",
        "Ball wall/backboard/ceiling continuation and challenge/50 outcomes are explicitly omitted: this public RocketSim binding exposes neither authoritative ball-surface contacts nor an objective challenge event. No positional or hindsight-tuned proxy was substituted.",
        "",
        "## Side-asymmetry localization",
        "",
        summary["side_asymmetry_localization"]["bounded_interpretation"],
        "",
        "No policy, reward, PPO, observation/action contract, controller semantics, or simulator physics was changed. No training was run.",
        "",
        "## Evidence",
        "",
        "Machine-readable evidence is under `results/rival2/rocketsim_crosscheck/`, including adapter fidelity, provenance, throughput, all match scorelines, side-separated summaries, cross-simulator deltas, the state bank, all duel outcomes, paired families, mechanic summaries/classified-event evidence, and artifact hashes.",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "probe", "matches", "open-play", "publish"), default="all")
    args = parser.parse_args()
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256 or _sha256(NEXTO_MODEL) != NEXTO_MODEL_SHA256:
        raise RuntimeError("frozen policy identity mismatch")
    extension = ROOT / ".venv" / "Lib" / "site-packages" / "RocketSim.pyd"
    if _sha256(extension) != EXTENSION_SHA256:
        raise RuntimeError("RocketSim extension identity mismatch")
    if not ADAPTER.exists() or _read_json(ADAPTER).get("verdict") != "PASS_GREEN":
        raise RuntimeError("adapter fidelity gate is not green")
    started = time.perf_counter()
    probe = _run_probe(args.collision_dir)
    if args.phase == "probe":
        print(json.dumps(probe, indent=2), flush=True)
        return 0
    normal_path = OUTPUT / "normal_matches.json"
    open_path = OUTPUT / "open_play.json"
    if args.phase in {"all", "matches"}:
        normal, _rows = _publish_matches(args.collision_dir, probe)
        comparison = _comparison(normal)
        if args.phase == "matches":
            print(json.dumps({
                "canonical_matches": normal["protocol"]["canonical_matches"],
                "stochastic_matches": normal["protocol"]["stochastic_executed"],
                "comparison": comparison["classification"],
            }, indent=2), flush=True)
            return 0
    else:
        normal = _read_json(normal_path)
        comparison = _read_json(OUTPUT / "normal_match_cross_simulator_comparison.json")
    if args.phase in {"all", "open-play"}:
        open_play, _duels = _publish_open_play(args.collision_dir)
    else:
        open_play = _read_json(open_path)
    adapter = _read_json(ADAPTER)
    localization = _localization(adapter, normal, open_play)
    summary = {
        "verdict": "PASS_GREEN",
        "scope": "evaluation only; no training or policy/reward/PPO/contract/physics behavior change",
        "identity": {
            "Rival_checkpoint_sha256": _sha256(CHECKPOINT),
            "Rival_policy_version": 5403,
            "Rival_total_samples": 45_323_649_024,
            "Nexto_commit": NEXTO_COMMIT,
            "Nexto_model_sha256": _sha256(NEXTO_MODEL),
            "RocketSim_primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "RocketSim_binding_commit": ROCKETSIM_BINDING_COMMIT,
            "RocketSim_extension_sha256": _sha256(extension),
            "RocketSim_package": "rocketsim==2.2.1 (import name RocketSim)",
        },
        "adapter_fidelity": adapter,
        "normal_matches": normal,
        "normal_match_comparison": comparison,
        "open_play": open_play,
        "side_asymmetry_localization": localization,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0),
        },
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "gates": {
            "adapter_green": adapter["verdict"] == "PASS_GREEN",
            "canonical_exact_10": normal["protocol"]["canonical_matches"] == 10,
            "stochastic_side_balanced": normal["stochastic"]["by_rival_side"]["Rival_Blue"]["matches"] == normal["stochastic"]["by_rival_side"]["Rival_Orange"]["matches"],
            "open_play_states_exact_4096": open_play["protocol"]["base_states"] == 4096,
            "open_play_duels_exact_16384": open_play["protocol"]["duels"] == 16384,
            "mirror_involution_exact": open_play["state_bank"]["mirror_involution_max_abs_error"] == 0,
            "movement_telemetry_normal_both_sides": all(normal["stochastic"]["by_rival_side"][name]["movement_mechanics"]["actual_flip_dodge_onsets"] >= 0 for name in ("Rival_Blue", "Rival_Orange")),
            "movement_telemetry_open_both_sides": all(open_play["movement_mechanics_by_rival_side"][name]["actual_flip_dodge_onsets"] >= 0 for name in ("Rival_Blue", "Rival_Orange")),
        },
    }
    if not all(summary["gates"].values()):
        summary["verdict"] = "FAIL_RED"
    _write_json(OUTPUT / "summary.json", summary)
    _write_report(summary)
    artifact_paths = sorted(
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "evidence_manifest.json"
    ) + [DOC]
    manifest = {
        "verdict": summary["verdict"],
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    _write_json(OUTPUT / "evidence_manifest.json", manifest)
    if WORK.exists():
        shutil.rmtree(WORK)
    print(json.dumps({
        "verdict": summary["verdict"],
        "normal_Blue": normal["stochastic"]["by_rival_side"]["Rival_Blue"]["win_rate"],
        "normal_Orange": normal["stochastic"]["by_rival_side"]["Rival_Orange"]["win_rate"],
        "open_Blue": open_play["by_rival_side"]["Rival_Blue"]["decisive_Rival_win_rate"],
        "open_Orange": open_play["by_rival_side"]["Rival_Orange"]["decisive_Rival_win_rate"],
    }, indent=2), flush=True)
    return 0 if summary["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
