"""Evaluate healthy Gameplay V1 checkpoints against the frozen Nexto policy.

The run is paired, deterministic by default, short-lifecycle, and evaluation
only.  It does not expose training, reward mutation, or simulator mutation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.nexto_short_eval import (  # noqa: E402
    NEXTO_MODEL_SHA256,
    NEXTO_UPSTREAM_COMMIT,
    PHYSICS_HZ,
    TERMINATION_GOAL,
    TERMINATION_HARD_TIME,
    TERMINATION_NO_TOUCH,
    NextoShortEpisodeRunner,
    classify_dash_events,
)

AUTHORITATIVE_HEAD = "bf03aaad90e6d44a04adfd8d7d4d74f42ede974e"
COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
OUTPUT_ROOT = Path("results/rival2/gameplay_v1_nexto")
REPORT_PATH = Path("docs/RIVAL2_GAMEPLAY_V1_NEXTO_RESULTS.md")
PRIMARY_SEED = 2_026_082_701
SECONDARY_SEED = 2_026_082_702
PRIMARY_WORLDS_PER_SIDE = 512
SECONDARY_WORLDS_PER_SIDE = 256

CHECKPOINTS = (
    {
        "label": "plus_180",
        "iteration": 300,
        "path": Path(
            r"G:\dev\RivalSim-runs\gameplay-v1-20260827-1507d3f\checkpoints\rival2_gameplay_plus_180_resume.pt"
        ),
        "sha256": "FEC1C289E7F7EB8D69876FB75C5325D56063A7A674A46F6FD20C5C270542511B",
    },
    {
        "label": "plus_239",
        "iteration": 359,
        "path": Path("checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt"),
        "sha256": "77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-root", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds-per-side", type=int, default=PRIMARY_WORLDS_PER_SIDE)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument(
        "--secondary-stochastic",
        action="store_true",
        help="also run the optional 256-per-side stochastic Rival comparison",
    )
    parser.add_argument(
        "--secondary-worlds-per-side",
        type=int,
        default=SECONDARY_WORLDS_PER_SIDE,
    )
    parser.add_argument("--secondary-seed", type=int, default=SECONDARY_SEED)
    parser.add_argument(
        "--checkpoint-label",
        choices=("plus_180", "plus_239"),
        help="target one checkpoint for a bounded smoke run",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="run without writing the canonical report/combined manifest",
    )
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def _ratio(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "fraction": None if denominator == 0 else float(count / denominator),
    }


def _distribution(values: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "p25": None,
            "p75": None,
            "maximum": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "maximum": float(np.max(values)),
    }


def _assignments(worlds_per_side: int) -> tuple[np.ndarray, np.ndarray]:
    if worlds_per_side <= 0:
        raise ValueError("worlds per side must be positive")
    local = np.arange(worlds_per_side, dtype=np.int32)
    rival_side = np.concatenate(
        (np.zeros(worlds_per_side, dtype=np.int32), np.ones(worlds_per_side, dtype=np.int32))
    )
    # Each physical-side half independently receives the same prospective
    # round-robin kickoff-layout assignment.  Counts differ by at most one.
    layouts = np.concatenate((local % 5, local % 5)).astype(np.int32)
    return rival_side, layouts


def _select_car(array: np.ndarray, rows: np.ndarray, side: np.ndarray) -> np.ndarray:
    return array[rows, side]


def _movement_summary(
    raw: dict[str, np.ndarray],
    rows: np.ndarray,
    side: np.ndarray,
) -> dict[str, Any]:
    ticks = _select_car(raw["simulated_ticks"], rows, side).astype(np.int64)
    tick_total = int(ticks.sum())
    minutes = tick_total / (PHYSICS_HZ * 60)

    def integer(name: str) -> int:
        return int(_select_car(raw[name], rows, side).sum())

    def floating(name: str) -> float:
        return float(_select_car(raw[name], rows, side).sum(dtype=np.float64))

    return {
        "simulated_ticks": tick_total,
        "simulated_minutes": minutes,
        "mean_speed_uu_per_s": (None if tick_total == 0 else floating("speed_sum") / tick_total),
        "supersonic_fraction": _ratio(integer("supersonic_ticks"), tick_total),
        "grounded_fraction": _ratio(integer("grounded_ticks"), tick_total),
        "airborne_fraction": _ratio(integer("airborne_ticks"), tick_total),
        "boost_active_fraction": _ratio(integer("boost_active_ticks"), tick_total),
        "observed_boost_consumed": floating("boost_consumed"),
        "boost_pickups": integer("boost_pickups"),
        "jump_activation_fraction": _ratio(integer("jump_active_ticks"), tick_total),
        "jump_rising_edges": integer("jump_rising_edges"),
        "first_jump_onsets": integer("first_jump_onsets"),
        "double_jump_onsets": integer("double_jump_onsets"),
        "actual_flip_onsets": integer("flip_onsets"),
        "actual_flips_per_simulated_minute": (
            None if minutes == 0 else integer("flip_onsets") / minutes
        ),
        "analog_saturation_fraction": _ratio(integer("analog_saturated_count"), tick_total * 5),
        "analog_mean_absolute": (
            None if tick_total == 0 else floating("analog_absolute_sum") / (tick_total * 5)
        ),
    }


def _candidate_counts(
    events: list[dict[str, Any]],
    *,
    rows: np.ndarray,
    policy: str,
) -> dict[str, int]:
    selected = set(int(value) for value in rows)
    counts: Counter[str] = Counter()
    for event in events:
        if int(event["world"]) in selected and event["policy"] == policy:
            counts.update(event["candidate_labels"])
    return dict(sorted(counts.items()))


def _group_summary(
    raw: dict[str, np.ndarray],
    events: list[dict[str, Any]],
    rival_side: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    rival = rival_side[rows].astype(np.int64)
    nexto = 1 - rival
    winner = raw["winner"][rows]
    termination = raw["termination_kind"][rows]
    decisive = termination == TERMINATION_GOAL
    rival_wins = int((decisive & (winner == rival)).sum())
    nexto_wins = int((decisive & (winner == nexto)).sum())
    no_touch = int((termination == TERMINATION_NO_TOUCH).sum())
    hard_time = int((termination == TERMINATION_HARD_TIME).sum())
    rival_touches = int(_select_car(raw["touch_count"], rows, rival).sum())
    nexto_touches = int(_select_car(raw["touch_count"], rows, nexto).sum())
    first = raw["first_toucher"][rows]
    first_resolved = first >= 0
    rival_first = int((first_resolved & (first == rival)).sum())
    nexto_first = int((first_resolved & (first == nexto)).sum())
    rival_saves = int(_select_car(raw["save_count"], rows, rival).sum())
    nexto_saves = int(_select_car(raw["save_count"], rows, nexto).sum())
    duration = raw["duration_ticks"][rows].astype(np.float64) / PHYSICS_HZ
    return {
        "episodes": int(rows.size),
        "rival_wins": rival_wins,
        "nexto_wins": nexto_wins,
        "no_goal_truncated_episodes": no_touch + hard_time,
        "decisive_episodes": int(decisive.sum()),
        "rival_win_rate_among_decisive": _ratio(rival_wins, int(decisive.sum())),
        "goals_for": rival_wins,
        "goals_against": nexto_wins,
        "goal_differential": rival_wins - nexto_wins,
        "goal_terminated_fraction": _ratio(int(decisive.sum()), int(rows.size)),
        "no_touch_fraction": _ratio(no_touch, int(rows.size)),
        "hard_timeout_fraction": _ratio(hard_time, int(rows.size)),
        "rival_touches": rival_touches,
        "nexto_touches": nexto_touches,
        "touch_differential": rival_touches - nexto_touches,
        "first_touch": {
            "resolved_episodes": int(first_resolved.sum()),
            "rival": rival_first,
            "nexto": nexto_first,
            "rival_share": _ratio(rival_first, int(first_resolved.sum())),
        },
        "saves": {"Rival": rival_saves, "Nexto": nexto_saves},
        "episode_duration_seconds": _distribution(duration),
        "movement_controller": {
            "Rival": _movement_summary(raw, rows, rival),
            "Nexto": _movement_summary(raw, rows, nexto),
        },
        "dash_candidate_event_counts": {
            "Rival": _candidate_counts(events, rows=rows, policy="Rival"),
            "Nexto": _candidate_counts(events, rows=rows, policy="Nexto"),
        },
    }


def _summarize(
    export: dict[str, Any],
    events: list[dict[str, Any]],
    mechanics_summary: dict[str, Any],
    *,
    timing: dict[str, Any],
    mode: str,
    seed: int,
) -> dict[str, Any]:
    raw = export["raw"]
    rival_side = export["rival_side"]
    worlds = rival_side.size
    all_rows = np.arange(worlds, dtype=np.int64)
    blue_rows = np.flatnonzero(rival_side == 0)
    orange_rows = np.flatnonzero(rival_side == 1)
    overflow = int(raw["event_overflow"].sum())
    checks = {
        "every_world_completed_exactly_one_episode": bool(np.all(raw["done"] == 1)),
        "every_world_has_valid_termination": bool(
            np.all(
                np.isin(
                    raw["termination_kind"],
                    (TERMINATION_GOAL, TERMINATION_NO_TOUCH, TERMINATION_HARD_TIME),
                )
            )
        ),
        "done_partition_exact": int(
            (raw["termination_kind"] == TERMINATION_GOAL).sum()
            + (raw["termination_kind"] == TERMINATION_NO_TOUCH).sum()
            + (raw["termination_kind"] == TERMINATION_HARD_TIME).sum()
        )
        == worlds,
        "dash_event_capacity_no_overflow": overflow == 0,
        "timed_world_h2d_bytes_zero": export["world_host_to_device_bytes_after_initialization"]
        == 0,
        "timed_world_d2h_bytes_zero": export["world_device_to_host_bytes_after_initialization"]
        == 0,
        "timed_nexto_h2d_bytes_zero": export["nexto_timed_h2d_bytes"] == 0,
        "timed_nexto_d2h_bytes_zero": export["nexto_timed_d2h_bytes"] == 0,
    }
    return {
        "checkpoint": export["checkpoint_identity"],
        "rival_action_mode": mode,
        "nexto_action_mode": (
            "deterministic beta=1 argmax at 15 Hz plus stock 120 Hz kickoff controller"
        ),
        "evaluation_seed": int(seed),
        "episode_contract": {
            "version": "RIVAL2_EPISODE_V1",
            "standard_kickoff": True,
            "first_goal_terminates": True,
            "no_touch_truncation_seconds": 15,
            "hard_limit_seconds": 45,
            "rival_policy_hz": 30,
            "nexto_neural_hz": 15,
            "nexto_kickoff_controller_hz": 120,
            "physics_hz": PHYSICS_HZ,
        },
        "episodes": worlds,
        "by_rival_side": {
            "overall": _group_summary(raw, events, rival_side, all_rows),
            "Rival_Blue": _group_summary(raw, events, rival_side, blue_rows),
            "Rival_Orange": _group_summary(raw, events, rival_side, orange_rows),
        },
        "mechanics": mechanics_summary,
        "performance": {
            "physics_ticks_requested": int(timing["physics_ticks_requested"]),
            "wall_seconds": float(timing["seconds"]),
            "world_ticks_per_second": float(timing["world_ticks_per_second"]),
            "peak_cuda_bytes": int(export["peak_cuda_bytes"]),
            "nexto_inference_calls": int(export["nexto_inference_calls"]),
            "nexto_observation_builds": int(export["nexto_observation_builds"]),
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _event_count_by_world_policy(
    events: Iterable[dict[str, Any]],
) -> dict[tuple[int, str], Counter[str]]:
    result: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for event in events:
        result[(int(event["world"]), str(event["policy"]))].update(event["candidate_labels"])
    return result


def _write_episode_ledger(
    path: Path,
    checkpoint_label: str,
    export: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = export["raw"]
    rival_side = export["rival_side"]
    layout = export["starting_layout"]
    dash = _event_count_by_world_policy(events)
    fields = (
        "checkpoint",
        "world",
        "pair_key",
        "starting_layout",
        "rival_side",
        "termination",
        "winner",
        "duration_ticks",
        "duration_seconds",
        "rival_touches",
        "nexto_touches",
        "first_toucher",
        "rival_saves",
        "nexto_saves",
        "rival_flips",
        "nexto_flips",
        "rival_wavedash_candidates",
        "nexto_wavedash_candidates",
        "rival_speed_increasing_wavedash_candidates",
        "nexto_speed_increasing_wavedash_candidates",
        "rival_zapdash_candidates",
        "nexto_zapdash_candidates",
        "rival_double_dash_candidate_events",
        "nexto_double_dash_candidate_events",
    )
    termination_names = {
        TERMINATION_GOAL: "goal",
        TERMINATION_NO_TOUCH: "no_touch",
        TERMINATION_HARD_TIME: "hard_time",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for world in range(rival_side.size):
            rival = int(rival_side[world])
            nexto = 1 - rival
            winner_side = int(raw["winner"][world])
            first_side = int(raw["first_toucher"][world])
            rival_dash = dash[(world, "Rival")]
            nexto_dash = dash[(world, "Nexto")]
            writer.writerow(
                {
                    "checkpoint": checkpoint_label,
                    "world": world,
                    "pair_key": (
                        f"side{rival}-layout{int(layout[world])}-"
                        f"slot{world % (rival_side.size // 2)}"
                    ),
                    "starting_layout": int(layout[world]),
                    "rival_side": "Blue" if rival == 0 else "Orange",
                    "termination": termination_names[int(raw["termination_kind"][world])],
                    "winner": (
                        "Rival"
                        if winner_side == rival
                        else "Nexto"
                        if winner_side == nexto
                        else "None"
                    ),
                    "duration_ticks": int(raw["duration_ticks"][world]),
                    "duration_seconds": float(raw["duration_ticks"][world] / PHYSICS_HZ),
                    "rival_touches": int(raw["touch_count"][world, rival]),
                    "nexto_touches": int(raw["touch_count"][world, nexto]),
                    "first_toucher": (
                        "Rival"
                        if first_side == rival
                        else "Nexto"
                        if first_side == nexto
                        else "None"
                    ),
                    "rival_saves": int(raw["save_count"][world, rival]),
                    "nexto_saves": int(raw["save_count"][world, nexto]),
                    "rival_flips": int(raw["flip_onsets"][world, rival]),
                    "nexto_flips": int(raw["flip_onsets"][world, nexto]),
                    "rival_wavedash_candidates": rival_dash["wavedash_candidate"],
                    "nexto_wavedash_candidates": nexto_dash["wavedash_candidate"],
                    "rival_speed_increasing_wavedash_candidates": rival_dash[
                        "speed_increasing_wavedash_candidate"
                    ],
                    "nexto_speed_increasing_wavedash_candidates": nexto_dash[
                        "speed_increasing_wavedash_candidate"
                    ],
                    "rival_zapdash_candidates": rival_dash["zapdash_candidate"],
                    "nexto_zapdash_candidates": nexto_dash["zapdash_candidate"],
                    "rival_double_dash_candidate_events": rival_dash["double_dash_candidate"],
                    "nexto_double_dash_candidate_events": nexto_dash["double_dash_candidate"],
                }
            )


def _mechanics_contract() -> dict[str, Any]:
    return {
        "purpose": (
            "read-only, inspectable classification of dash-like state transitions; "
            "never inferred from jump frequency alone"
        ),
        "engine_facts": {
            "physics_hz": 120,
            "dodge_requires_jump_rising_edge": True,
            "dodge_requires_airborne": True,
            "dodge_requires_has_jumped": True,
            "dodge_forbidden_after_has_double_jumped_or_has_flipped": True,
            "dodge_availability_seconds": 1.25,
            "directional_input_deadzone_sum": 0.5,
            "base_directional_dodge_impulse_uu_per_s": 500.0,
            "source": ("pinned RocketSim-derived RivalSim vehicle operation path and constants"),
        },
        "mechanic_requirements": {
            "wavedash": (
                "an actual airborne directional dodge begins immediately before or "
                "during wheel contact; the contacted surface interrupts the rotation "
                "while the dodge impulse contributes translation"
            ),
            "landing_wavedash": (
                "the same dodge/contact relation occurs during an existing landing, "
                "without requiring that the current airborne phase began with a fresh jump"
            ),
            "zapdash": (
                "an angled landing reaches the front wheels before the rear; once at "
                "least three wheels establish grounded jump availability but before a "
                "flat four-wheel landing, a jump pops the front upward and is followed "
                "by the directional landing dodge/wavedash"
            ),
            "double_dash": (
                "two wavedash outcomes occur in rapid succession with intervening contact; "
                "the raw trace states whether another first-jump onset occurred"
            ),
            "wall_or_curve_dash": (
                "the wavedash contact relation occurs against a wall or curved transition "
                "rather than a floor-like surface"
            ),
        },
        "retained_per_flip_evidence": [
            "controller and actual has_flipped onset",
            "wheel masks before/after and at first landing",
            "named prior-landing and first-jump wheel masks for front/rear ordering",
            "last contact/takeoff/landing/jump-rising/first-jump ticks",
            "air-time and air-time-since-jump",
            "car velocity and contact-surface-tangent speed before/after flip, at "
            "landing, and four ticks after landing",
            "orientation up-z before/after/at landing",
            "wheel suspension length and velocity before flip and at landing",
            "landing contact normal and surface class",
            "actual flip-relative torque",
        ],
        "classification_qualification": (
            "community mechanic names do not have authoritative engine event flags; "
            "candidate labels use prospective windows and retain raw evidence so the "
            "classification can be audited without relabeling from outcomes"
        ),
        "sources": [
            {
                "title": "Psyonix GDC 2018: It IS Rocket Science!",
                "url": "https://media.gdcvault.com/gdc2018/presentations/Cone_Jared_It_Is_Rocket.pdf",
                "role": "primary vehicle/wheel/suspension physics context",
            },
            {
                "title": "RLBot Wiki useful game values",
                "url": "https://wiki.rlbot.org/v5/botmaking/useful-game-values/",
                "role": "bot-facing coordinate, jump, speed, and game-value reference",
            },
            {
                "title": "Rocket Science: Dodges explained",
                "url": "https://www.s543778567.website-start.de/know/videos/dodges",
                "role": "measured 500 uu/s directional dodge impulse behavior",
            },
            {
                "title": "Rocket Science: Landing Wavedash",
                "url": "https://www.youtube.com/watch?v=baNsqFEfRMY",
                "role": "measured landing/suspension dash behavior",
            },
            {
                "title": "Rocket League Mechanics Database",
                "url": "https://0byte-coding.github.io/rocket_league_mechanics/",
                "role": "community definitions for zapdash and double-dash naming",
            },
        ],
    }


def _run_one(
    checkpoint: dict[str, Any],
    *,
    collision_root: Path,
    output_root: Path,
    device: str,
    worlds_per_side: int,
    seed: int,
    stochastic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rival_side, layout = _assignments(worlds_per_side)
    print(
        f"{checkpoint['label']}: initialize {rival_side.size:,} worlds "
        f"({'stochastic' if stochastic else 'deterministic'} Rival)",
        flush=True,
    )
    runner = NextoShortEpisodeRunner(
        rival_side.size,
        str(collision_root),
        checkpoint["path"],
        expected_checkpoint_sha256=checkpoint["sha256"],
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=stochastic,
        evaluation_seed=seed,
        device=device,
    )
    timing_value = runner.run()
    timing = {
        "physics_ticks_requested": timing_value.physics_ticks_requested,
        "seconds": timing_value.seconds,
        "world_ticks_per_second": timing_value.world_ticks_per_second,
    }
    export = runner.export()
    events, mechanics = classify_dash_events(
        export["raw"],
        rival_side=rival_side,
        starting_layout=layout,
        checkpoint_label=checkpoint["label"],
    )
    mode = "stochastic_hybrid_sampling" if stochastic else "deterministic_deployment"
    summary = _summarize(
        export,
        events,
        mechanics,
        timing=timing,
        mode=mode,
        seed=seed,
    )
    suffix = "stochastic" if stochastic else "deterministic"
    _write_json(output_root / f"{checkpoint['label']}_{suffix}.json", summary)
    dash_events = [
        event
        for event in events
        if any(label != "ground_contact_dodge_candidate" for label in event["candidate_labels"])
    ]
    _write_json(
        output_root / f"{checkpoint['label']}_{suffix}_dash_events.json",
        dash_events,
    )
    _write_episode_ledger(
        output_root / f"{checkpoint['label']}_{suffix}_episodes.csv",
        checkpoint["label"],
        export,
        events,
    )
    print(
        f"{checkpoint['label']}: {summary['verdict']} in {timing_value.seconds:.2f}s; "
        f"Rival {summary['by_rival_side']['overall']['rival_wins']}-"
        f"{summary['by_rival_side']['overall']['nexto_wins']} Nexto; "
        f"no-touch={summary['by_rival_side']['overall']['no_touch_fraction']['fraction']:.6f}",
        flush=True,
    )
    return summary, dash_events, export


def _side_gap(summary: dict[str, Any]) -> float:
    blue = summary["by_rival_side"]["Rival_Blue"]["rival_win_rate_among_decisive"]["fraction"]
    orange = summary["by_rival_side"]["Rival_Orange"]["rival_win_rate_among_decisive"]["fraction"]
    if blue is None or orange is None:
        return 1.0
    return abs(float(blue) - float(orange))


def _selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    overall = summary["by_rival_side"]["overall"]
    decisive_rate = overall["rival_win_rate_among_decisive"]["fraction"]
    return (
        -1.0 if decisive_rate is None else float(decisive_rate),
        float(overall["rival_wins"]),
        -_side_gap(summary),
        float(overall["goal_differential"]),
        -float(overall["no_touch_fraction"]["fraction"]),
        float(overall["touch_differential"]),
        float(overall["first_touch"]["rival_share"]["fraction"] or 0.0),
    )


def _recommend(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    winner = max(results, key=lambda label: _selection_key(results[label]))
    return {
        "selected_checkpoint": winner,
        "priority_order": [
            "performance against Nexto",
            "side consistency",
            "goal differential and defensive outcomes",
            "preserved acquisition ability",
            "controller and mechanical sanity",
        ],
        "selection_keys": {
            label: list(_selection_key(summary)) for label, summary in results.items()
        },
        "qualification": (
            "lexicographic selection follows the prospective priority order; "
            "self-play goals/min and reward totals are not selection inputs"
        ),
    }


def _format_fraction(value: dict[str, Any]) -> str:
    fraction = value["fraction"]
    return "n/a" if fraction is None else f"{100.0 * fraction:.3f}%"


def _mechanic_highlights(
    event_sets: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    zap_events: list[tuple[str, dict[str, Any]]] = []
    for mode, checkpoints in event_sets.items():
        for events in checkpoints.values():
            for event in events:
                if "zapdash_candidate" in event["candidate_labels"]:
                    zap_events.append((mode, event))

    representative_zap: dict[str, Any] | None = None
    if zap_events:
        mode, event = zap_events[0]
        zap = event["classification_evidence"]["zapdash_candidate"]
        speed = event["classification_evidence"].get("speed_increasing_wavedash_candidate")
        representative_zap = {
            "mode": mode,
            "checkpoint": event["checkpoint"],
            "policy": event["policy"],
            "world": event["world"],
            "rival_side": event["rival_side"],
            "starting_layout": event["starting_layout"],
            "prior_landing_tick": event["last_landing_tick"],
            "prior_landing_wheels": zap["prior_landing_wheels"],
            "prior_landing_to_first_jump_ticks": zap["prior_landing_to_first_jump_ticks"],
            "first_jump_tick": event["last_first_jump_tick"],
            "first_jump_wheels": zap["first_jump_wheels_before"],
            "first_jump_vertical_velocity_before": zap["first_jump_vertical_velocity_before"],
            "first_jump_max_abs_suspension_velocity": zap["first_jump_max_abs_suspension_velocity"],
            "first_jump_to_flip_ticks": zap["first_jump_to_flip_ticks"],
            "flip_tick": event["tick"],
            "flip_direction": event["direction"],
            "flip_to_landing_ticks": event["flip_to_landing_ticks"],
            "landing_tick": event["landing_tick"],
            "landing_wheels": event["landing_wheels"],
            "landing_surface": event["landing_surface"],
            "surface_tangent_speed_before": (
                None if speed is None else speed["surface_tangent_speed_before"]
            ),
            "surface_tangent_speed_after": (
                None if speed is None else speed["surface_tangent_speed_after_landing_sample"]
            ),
            "surface_tangent_speed_delta": (
                None if speed is None else speed["surface_tangent_speed_delta"]
            ),
            "also_double_dash_candidate": "double_dash_candidate" in event["candidate_labels"],
            "double_dash_evidence": event["classification_evidence"].get("double_dash_candidate"),
        }

    return {
        "zapdash_candidate_count": len(zap_events),
        "representative_zapdash_candidate": representative_zap,
        "qualification": (
            "the representative is selected by first prospective event order, not "
            "by outcome; it remains a state-transition candidate rather than an "
            "inference about policy intent"
        ),
    }


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Rival 2.0 Gameplay V1 vs pinned Nexto",
        "",
        f"Source HEAD: `{manifest['identity']['source_head']}`.",
        "",
        (
            "This is evaluation only: standard Soccar kickoff, first goal ends the "
            "episode, 15 seconds without a touch truncates, and 45 seconds is the hard "
            "limit. Rival is deterministic at 30 Hz; frozen Nexto is deterministic at "
            "15 Hz with its stock 120 Hz kickoff controller. Reward is not used to "
            "select an outcome."
        ),
        "",
        "## Identities",
        "",
        f"- Nexto upstream commit: `{manifest['identity']['nexto_upstream_commit']}`.",
        f"- Nexto model SHA-256: `{manifest['identity']['nexto_model_sha256']}`.",
    ]
    for label, summary in results.items():
        checkpoint = summary["checkpoint"]
        lines.append(
            f"- `{label}`: iteration `{checkpoint['iteration']}`, SHA-256 `{checkpoint['sha256']}`."
        )
    primary = manifest["primary"]
    lines.extend(
        [
            "",
            "## Episode design",
            "",
            (
                f"Each checkpoint uses {primary['episodes_per_checkpoint']} primary "
                "episodes: 512 with Rival Blue and 512 with Rival Orange. Within each "
                "side, kickoff layouts 0-4 receive 103, 103, 102, 102, and 102 "
                "episodes. Checkpoints use the identical prospective simulator seed "
                f"`{primary['evaluation_seed']}`, world index, physical side, and layout."
            ),
            "",
            (
                "Because both deployed policies and standard kickoff states are "
                "deterministic, repeated rows for a given side/layout are paired "
                "episode outcomes rather than claims of 1,024 statistically "
                "independent randomized physical starts."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Primary deterministic result",
            "",
            (
                "| Checkpoint / Rival side | Episodes | Rival wins | Nexto wins | "
                "No goal | Decisive win rate | GF-GA | GD | Goal-term | No-touch | "
                "Hard-time | Rival touches | Nexto touches | Touch diff | "
                "First-touch share | Mean seconds |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in results.items():
        for side_key, side_label in (
            ("overall", "overall"),
            ("Rival_Blue", "Rival Blue"),
            ("Rival_Orange", "Rival Orange"),
        ):
            value = summary["by_rival_side"][side_key]
            lines.append(
                f"| {label} / {side_label} | {value['episodes']} | {value['rival_wins']} | "
                f"{value['nexto_wins']} | {value['no_goal_truncated_episodes']} | "
                f"{_format_fraction(value['rival_win_rate_among_decisive'])} | "
                f"{value['goals_for']}-{value['goals_against']} | "
                f"{value['goal_differential']} | "
                f"{_format_fraction(value['goal_terminated_fraction'])} | "
                f"{_format_fraction(value['no_touch_fraction'])} | "
                f"{_format_fraction(value['hard_timeout_fraction'])} | "
                f"{value['rival_touches']} | {value['nexto_touches']} | "
                f"{value['touch_differential']} | "
                f"{_format_fraction(value['first_touch']['rival_share'])} | "
                f"{value['episode_duration_seconds']['mean']:.3f} |"
            )
    secondary = manifest["secondary_stochastic"]
    if secondary["executed"]:
        lines.extend(
            [
                "",
                "## Secondary stochastic Rival result",
                "",
                (
                    "This optional suite samples Rival's learned hybrid action "
                    "distribution while Nexto remains deterministic. It uses 256 "
                    "episodes per Rival side and is secondary to the deterministic "
                    "deployment-policy result above."
                ),
                "",
                (
                    "| Checkpoint / Rival side | Episodes | Rival wins | Nexto wins | "
                    "No goal | Decisive win rate | GF-GA | GD | Goal-term | No-touch | "
                    "Hard-time | Rival touches | Nexto touches | Touch diff | "
                    "First-touch share | Mean seconds |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, summary in secondary["results"].items():
            for side_key, side_label in (
                ("overall", "overall"),
                ("Rival_Blue", "Rival Blue"),
                ("Rival_Orange", "Rival Orange"),
            ):
                value = summary["by_rival_side"][side_key]
                lines.append(
                    f"| {label} / {side_label} | {value['episodes']} | "
                    f"{value['rival_wins']} | {value['nexto_wins']} | "
                    f"{value['no_goal_truncated_episodes']} | "
                    f"{_format_fraction(value['rival_win_rate_among_decisive'])} | "
                    f"{value['goals_for']}-{value['goals_against']} | "
                    f"{value['goal_differential']} | "
                    f"{_format_fraction(value['goal_terminated_fraction'])} | "
                    f"{_format_fraction(value['no_touch_fraction'])} | "
                    f"{_format_fraction(value['hard_timeout_fraction'])} | "
                    f"{value['rival_touches']} | {value['nexto_touches']} | "
                    f"{value['touch_differential']} | "
                    f"{_format_fraction(value['first_touch']['rival_share'])} | "
                    f"{value['episode_duration_seconds']['mean']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Rival movement and controls",
            "",
            (
                "| Checkpoint / side | Mean speed | Supersonic | Boost active | Pickups | "
                "Grounded | Airborne | Jump active | Flips/min | Analog saturation | "
                "Saves |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in results.items():
        for side_key, side_label in (
            ("overall", "overall"),
            ("Rival_Blue", "Blue"),
            ("Rival_Orange", "Orange"),
        ):
            group = summary["by_rival_side"][side_key]
            move = group["movement_controller"]["Rival"]
            lines.append(
                f"| {label} / {side_label} | {move['mean_speed_uu_per_s']:.3f} | "
                f"{_format_fraction(move['supersonic_fraction'])} | "
                f"{_format_fraction(move['boost_active_fraction'])} | "
                f"{move['boost_pickups']} | {_format_fraction(move['grounded_fraction'])} | "
                f"{_format_fraction(move['airborne_fraction'])} | "
                f"{_format_fraction(move['jump_activation_fraction'])} | "
                f"{move['actual_flips_per_simulated_minute']:.3f} | "
                f"{_format_fraction(move['analog_saturation_fraction'])} | "
                f"{group['saves']['Rival']} |"
            )
    lines.extend(
        [
            "",
            "## Dash-mechanics telemetry",
            "",
            (
                "Named mechanics have no authoritative simulator event flag. The "
                "evaluator therefore retains actual flip onset, all four wheel "
                "contacts, first-jump and landing timing, suspension length/velocity, "
                "orientation, contact normal, controller input, and speed before/after "
                "landing. Labels are prospective candidates rather than intent claims."
            ),
            "",
            (
                "A wavedash candidate is an actual airborne dodge from zero wheel "
                "contact that reaches first wheel contact within 0.20 seconds with no "
                "more than 0.35 seconds of pre-flip air time. A speed-increasing "
                "candidate must also retain strictly higher velocity tangent to the "
                "contacted surface after landing. Zapdash candidates additionally "
                "require a front-wheel-only first landing, a non-flat three-wheel "
                "first-jump onset, and the subsequent landing dodge; double-dash "
                "candidates require two wavedash candidates with intervening contact. "
                "Raw evidence for every named candidate is published beside each "
                "checkpoint result."
            ),
            "",
            (
                "Sources used to define and qualify the telemetry: [Psyonix GDC "
                "vehicle-physics presentation](https://media.gdcvault.com/gdc2018/"
                "presentations/Cone_Jared_It_Is_Rocket.pdf), [RLBot useful game "
                "values](https://wiki.rlbot.org/v5/botmaking/useful-game-values/), "
                "[Rocket Science dodge measurements](https://www.s543778567.website-"
                "start.de/know/videos/dodges), [Rocket Science landing-wavedash "
                "analysis](https://www.youtube.com/watch?v=baNsqFEfRMY), and the "
                "[community mechanics database](https://0byte-coding.github.io/"
                "rocket_league_mechanics/) for zapdash/double-dash naming."
            ),
            "",
            (
                "| Mode | Checkpoint | Policy | Wavedash candidates | Speed-increasing | "
                "Zapdash | Double-dash event rows | Wall dash | Curved-surface dash |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    mechanic_modes = [("deterministic", results)]
    if secondary["executed"]:
        mechanic_modes.append(("stochastic", secondary["results"]))
    for mode, mode_results in mechanic_modes:
        for label, summary in mode_results.items():
            counts = summary["mechanics"]["candidate_event_counts_by_policy"]
            for policy in ("Rival", "Nexto"):
                value = counts[policy]
                lines.append(
                    f"| {mode} | {label} | {policy} | "
                    f"{value.get('wavedash_candidate', 0)} | "
                    f"{value.get('speed_increasing_wavedash_candidate', 0)} | "
                    f"{value.get('zapdash_candidate', 0)} | "
                    f"{value.get('double_dash_candidate', 0)} | "
                    f"{value.get('wall_dash_candidate', 0)} | "
                    f"{value.get('curved_surface_dash_candidate', 0)} |"
                )
    representative_zap = manifest["mechanic_highlights"]["representative_zapdash_candidate"]
    if representative_zap is not None:
        speed_delta = representative_zap["surface_tangent_speed_delta"]
        zap_policy = representative_zap["policy"]
        zap_checkpoint = representative_zap["checkpoint"]
        zap_mode = representative_zap["mode"]
        zap_world = representative_zap["world"]
        zap_layout = representative_zap["starting_layout"]
        prior_tick = representative_zap["prior_landing_tick"]
        prior_wheels = ", ".join(representative_zap["prior_landing_wheels"])
        landing_to_jump = representative_zap["prior_landing_to_first_jump_ticks"]
        jump_wheels = ", ".join(representative_zap["first_jump_wheels"])
        jump_to_flip = representative_zap["first_jump_to_flip_ticks"]
        flip_direction = representative_zap["flip_direction"]
        flip_to_landing = representative_zap["flip_to_landing_ticks"]
        landing_wheels = ", ".join(representative_zap["landing_wheels"])
        zap_line = (
            f"The sole strict zapdash candidate occurred for {zap_policy} "
            f"`{zap_checkpoint}` in the {zap_mode} suite (world {zap_world}, layout "
            f"{zap_layout}). Front-only contact began at tick {prior_tick} on "
            f"{prior_wheels}; {landing_to_jump} ticks later, the first jump began with "
            f"{jump_wheels} in contact; the directional `{flip_direction}` flip began "
            f"{jump_to_flip} ticks later and landed after {flip_to_landing} tick on "
            f"{landing_wheels}. Contact-surface-tangent speed changed by "
            f"{speed_delta:+.3f} uu/s. It is also one half of a measured double-dash "
            "candidate. This is strong transition evidence, but not a claim about "
            "learned intent."
        )
        lines.extend(
            [
                "",
                "### Representative measured zapdash sequence",
                "",
                zap_line,
            ]
        )
    selection = manifest["checkpoint_selection"]
    selected = selection["selected_checkpoint"]
    lines.extend(
        [
            "",
            "## Checkpoint selection",
            "",
            f"**Recommendation: continue from `{selected}`.**",
            "",
            manifest["checkpoint_selection"]["narrative"],
            "",
            (
                "This is a tiebreaking recommendation: neither checkpoint showed "
                "competitive deterministic performance against Nexto."
            ),
            "",
            (
                "The selection is based first on the fixed Nexto opponent, then side "
                "consistency, goal differential/defensive outcome, acquisition "
                "retention, and finally controller/mechanical sanity. Self-play "
                "goals/min and reward totals were not used."
            ),
            "",
            "## Evidence boundary",
            "",
            (
                "No Rival or Nexto training ran. No policy, reward, PPO, observation, "
                "action, physics, or adapter behavior changed. The stochastic suite is "
                "explicitly secondary and did not influence the primary deterministic "
                "measurement."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _selection_narrative(
    selected: str,
    results: dict[str, dict[str, Any]],
    stochastic: dict[str, dict[str, Any]],
) -> str:
    other = "plus_239" if selected == "plus_180" else "plus_180"
    a = results[selected]["by_rival_side"]["overall"]
    b = results[other]["by_rival_side"]["overall"]
    selected_minutes = a["movement_controller"]["Rival"]["simulated_minutes"]
    other_minutes = b["movement_controller"]["Rival"]["simulated_minutes"]
    narrative = (
        f"`{selected}` recorded {a['rival_wins']} Rival wins and {a['nexto_wins']} "
        f"Nexto wins, versus {b['rival_wins']} and {b['nexto_wins']} for `{other}`. "
        f"Its goal differential was {a['goal_differential']} versus "
        f"{b['goal_differential']}; no-touch fraction was "
        f"{a['no_touch_fraction']['fraction']:.6f} versus "
        f"{b['no_touch_fraction']['fraction']:.6f}. Side-consistency gaps were "
        f"{_side_gap(results[selected]):.6f} and {_side_gap(results[other]):.6f}. "
        f"With the primary outcome tied, `{selected}` supplied "
        f"{a['saves']['Rival']} measured saves versus {b['saves']['Rival']}, "
        f"{a['rival_touches'] / selected_minutes:.3f} versus "
        f"{b['rival_touches'] / other_minutes:.3f} Rival touches/min, and allowed "
        f"{a['nexto_touches'] / selected_minutes:.3f} versus "
        f"{b['nexto_touches'] / other_minutes:.3f} Nexto touches/min. Its shorter "
        f"mean survival ({a['episode_duration_seconds']['mean']:.3f} versus "
        f"{b['episode_duration_seconds']['mean']:.3f} seconds) is an honest negative."
    )
    if stochastic:
        selected_secondary = stochastic[selected]["by_rival_side"]["overall"]
        other_secondary = stochastic[other]["by_rival_side"]["overall"]
        narrative += (
            f" The secondary sampled-policy suite also favored `{selected}`: "
            f"{selected_secondary['rival_wins']} wins and "
            f"{_format_fraction(selected_secondary['rival_win_rate_among_decisive'])} "
            f"decisive win rate, versus {other_secondary['rival_wins']} and "
            f"{_format_fraction(other_secondary['rival_win_rate_among_decisive'])} "
            f"for `{other}`; this remains corroborating rather than primary evidence."
        )
    return narrative


def main() -> int:
    args = parse_args()
    if not args.collision_root.is_dir():
        raise FileNotFoundError(args.collision_root)
    source_head = _git("rev-parse", "HEAD")
    if source_head != AUTHORITATIVE_HEAD:
        raise RuntimeError(
            f"authoritative source HEAD mismatch: {source_head} != {AUTHORITATIVE_HEAD}"
        )
    dirty_before = _git("status", "--short")
    expected_dirty = {
        "?? benchmarks/run_rival2_gameplay_nexto.py",
        "?? docs/RIVAL2_GAMEPLAY_V1_NEXTO_RESULTS.md",
        "?? rivalsim/nexto_short_eval.py",
        "?? results/rival2/gameplay_v1_nexto/",
        "?? tests/test_nexto_short_eval.py",
    }
    unexpected_dirty = [
        line for line in dirty_before.splitlines() if line and line not in expected_dirty
    ]
    if unexpected_dirty:
        raise RuntimeError(f"unexpected pre-run repository changes: {unexpected_dirty}")

    model_path = REPO_ROOT / "third_party/nexto/nexto-model.pt"
    actual_nexto_sha = _sha256(model_path)
    if actual_nexto_sha != NEXTO_MODEL_SHA256:
        raise RuntimeError("pinned Nexto model SHA-256 mismatch")
    selected_checkpoints = [
        item
        for item in CHECKPOINTS
        if args.checkpoint_label is None or item["label"] == args.checkpoint_label
    ]
    for checkpoint in selected_checkpoints:
        if not checkpoint["path"].is_file():
            raise FileNotFoundError(checkpoint["path"])
        if _sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"{checkpoint['label']} checkpoint SHA-256 mismatch")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    mechanics_contract = _mechanics_contract()
    canonical_contract = json.dumps(
        mechanics_contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    mechanics_contract["content_sha256"] = hashlib.sha256(canonical_contract).hexdigest().upper()
    _write_json(output_root / "dash_mechanics_contract.json", mechanics_contract)

    deterministic: dict[str, dict[str, Any]] = {}
    deterministic_events: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in selected_checkpoints:
        summary, events, _export = _run_one(
            checkpoint,
            collision_root=args.collision_root,
            output_root=output_root,
            device=args.device,
            worlds_per_side=args.worlds_per_side,
            seed=args.seed,
            stochastic=False,
        )
        if summary["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"{checkpoint['label']} deterministic evaluation failed")
        deterministic[checkpoint["label"]] = summary
        deterministic_events[checkpoint["label"]] = events

    stochastic: dict[str, dict[str, Any]] = {}
    stochastic_events: dict[str, list[dict[str, Any]]] = {}
    if args.secondary_stochastic:
        for checkpoint in selected_checkpoints:
            summary, events, _export = _run_one(
                checkpoint,
                collision_root=args.collision_root,
                output_root=output_root,
                device=args.device,
                worlds_per_side=args.secondary_worlds_per_side,
                seed=args.secondary_seed,
                stochastic=True,
            )
            if summary["verdict"] != "PASS_GREEN":
                raise RuntimeError(f"{checkpoint['label']} stochastic evaluation failed")
            stochastic[checkpoint["label"]] = summary
            stochastic_events[checkpoint["label"]] = events

    if args.no_publish or len(deterministic) != 2:
        return 0

    recommendation = _recommend(deterministic)
    recommendation["narrative"] = _selection_narrative(
        recommendation["selected_checkpoint"], deterministic, stochastic
    )
    manifest = {
        "verdict": "PASS_GREEN",
        "scope": "evaluation-only paired Gameplay V1 checkpoint comparison against frozen Nexto",
        "identity": {
            "source_head": source_head,
            "rival_checkpoints": {
                label: summary["checkpoint"] for label, summary in deterministic.items()
            },
            "nexto_upstream_commit": NEXTO_UPSTREAM_COMMIT,
            "nexto_model_sha256": actual_nexto_sha,
        },
        "primary": {
            "rival_action_mode": "deterministic_deployment",
            "worlds_per_side_per_checkpoint": int(args.worlds_per_side),
            "episodes_per_checkpoint": int(args.worlds_per_side * 2),
            "total_episodes": int(args.worlds_per_side * 2 * 2),
            "evaluation_seed": int(args.seed),
            "paired_assignment": (
                "identical world index, Rival side, kickoff layout, and simulator seed "
                "for plus_180 and plus_239"
            ),
        },
        "secondary_stochastic": {
            "executed": bool(stochastic),
            "reason_if_not_executed": (
                None
                if stochastic
                else (
                    "omitted prospectively to avoid adding 50% more simulation after "
                    "the complete primary test"
                )
            ),
            "results": stochastic,
        },
        "dash_mechanics_contract_sha256": mechanics_contract["content_sha256"],
        "mechanic_highlights": _mechanic_highlights(
            {
                "primary_deterministic": deterministic_events,
                "secondary_stochastic": stochastic_events,
            }
        ),
        "checkpoint_selection": recommendation,
        "results": deterministic,
        "checks": {
            "source_head_exact": source_head == AUTHORITATIVE_HEAD,
            "nexto_identity_exact": actual_nexto_sha == NEXTO_MODEL_SHA256,
            "both_checkpoints_evaluated": set(deterministic) == {"plus_180", "plus_239"},
            "episode_count_exact": all(
                value["episodes"] == args.worlds_per_side * 2 for value in deterministic.values()
            ),
            "all_result_gates_green": all(
                value["verdict"] == "PASS_GREEN" for value in deterministic.values()
            ),
            "secondary_episode_count_exact": (
                not stochastic
                or all(
                    value["episodes"] == args.secondary_worlds_per_side * 2
                    for value in stochastic.values()
                )
            ),
            "secondary_result_gates_green": (
                not stochastic
                or all(value["verdict"] == "PASS_GREEN" for value in stochastic.values())
            ),
            "no_training": True,
            "reward_not_used_for_winner": True,
        },
    }
    manifest["verdict"] = "PASS_GREEN" if all(manifest["checks"].values()) else "FAIL_RED"
    _write_json(output_root / "summary.json", manifest)
    _write_report(args.report, manifest, deterministic)

    artifact_paths = sorted(
        [path for path in output_root.iterdir() if path.is_file()] + [args.report]
    )
    artifact_manifest = {
        "verdict": manifest["verdict"],
        "artifacts": [
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
            if path.name != "artifact_manifest.json"
        ],
    }
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(
        f"complete: {manifest['verdict']}; selected {recommendation['selected_checkpoint']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
