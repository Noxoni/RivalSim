"""Run and publish the authorized Rival-vs-Nexto full-match suites."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.behavioral_telemetry import GOAL_HALF_WIDTH_UU, GOAL_HEIGHT_UU
from rivalsim.full_match import REGULATION_TICKS, FullMatchRunner, MatchRunTiming

CHECKPOINT = ROOT / "checkpoints" / "rival2" / "overnight" / "rival2_overnight_final_6h_resume.pt"
COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
OUTPUT_DIR = ROOT / "results" / "rival2" / "nexto"
DOC_PATH = ROOT / "docs" / "RIVAL2_NEXTO_RESULTS.md"
FIDELITY_PATH = OUTPUT_DIR / "fidelity.json"

CANONICAL_SEED = 2_026_082_601
STOCHASTIC_SEED = 2_026_082_602
PROFILE_TICKS = 8
OVERTIME_STATUS_TICKS = 3_600
LAYOUT_NAMES = {
    0: "Blue diagonal-left",
    1: "Blue diagonal-right",
    2: "Blue off-center-left",
    3: "Blue off-center-right",
    4: "center",
}
DIRECTION_NAMES = ("backward", "neutral", "forward")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


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


def _category_counts(counts: np.ndarray) -> dict[str, Any]:
    totals = np.asarray(counts, dtype=np.int64).sum(axis=0)
    denominator = int(totals.sum())
    return {
        "denominator": denominator,
        "classes": {
            name: _ratio(int(totals[index]), denominator)
            for index, name in enumerate(DIRECTION_NAMES)
        },
    }


def _goal_events(
    raw: dict[str, np.ndarray], rows: np.ndarray, policy_side: np.ndarray
) -> dict[str, np.ndarray]:
    capacity = raw["goal_scorer"].shape[1]
    selected_count = np.minimum(raw["match.goal_count"][rows], capacity)
    slot = np.arange(capacity, dtype=np.int32)[None, :]
    valid = slot < selected_count[:, None]
    scorer = raw["goal_scorer"][rows]
    policy_goal = valid & (scorer == policy_side[:, None])
    return {
        "x": raw["goal_entry_x"][rows][policy_goal],
        "z": raw["goal_entry_z"][rows][policy_goal],
        "crossing_valid": raw["goal_entry_valid"][rows][policy_goal],
        "kickoff": raw["goal_kickoff"][rows][policy_goal],
        "overtime": raw["goal_overtime"][rows][policy_goal],
        "tick": raw["goal_tick"][rows][policy_goal],
    }


def _goal_mouth_summary(events: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = events["crossing_valid"] != 0
    x = np.asarray(events["x"], dtype=np.float64)[valid]
    z = np.asarray(events["z"], dtype=np.float64)[valid]
    inside = (
        (np.abs(x) <= GOAL_HALF_WIDTH_UU)
        & (z >= 0.0)
        & (z <= GOAL_HEIGHT_UU)
    )
    x_edges = np.linspace(-GOAL_HALF_WIDTH_UU, GOAL_HALF_WIDTH_UU, 11)
    z_edges = np.linspace(0.0, GOAL_HEIGHT_UU, 7)
    histogram, _, _ = np.histogram2d(x[inside], z[inside], bins=(x_edges, z_edges))
    horizontal_edges = np.asarray(
        [-GOAL_HALF_WIDTH_UU, -GOAL_HALF_WIDTH_UU / 3, GOAL_HALF_WIDTH_UU / 3, GOAL_HALF_WIDTH_UU]
    )
    vertical_edges = np.asarray([0.0, GOAL_HEIGHT_UU / 3, 2 * GOAL_HEIGHT_UU / 3, GOAL_HEIGHT_UU])
    horizontal = np.histogram(x[inside], bins=horizontal_edges)[0]
    vertical = np.histogram(z[inside], bins=vertical_edges)[0]
    return {
        "goal_count": int(events["x"].size),
        "interpolated_crossing_valid": _ratio(int(valid.sum()), int(valid.size)),
        "inside_declared_goal_mouth": _ratio(int(inside.sum()), int(valid.sum())),
        "declared_geometry_uu": {
            "x": [-GOAL_HALF_WIDTH_UU, GOAL_HALF_WIDTH_UU],
            "z": [0.0, GOAL_HEIGHT_UU],
        },
        "canonical_x_uu": _distribution(x),
        "z_uu": _distribution(z),
        "horizontal_bins": {
            "edges_uu": horizontal_edges.tolist(),
            "labels": ["left", "center", "right"],
            "counts": horizontal.astype(int).tolist(),
        },
        "vertical_bins": {
            "edges_uu": vertical_edges.tolist(),
            "labels": ["low", "middle", "high"],
            "counts": vertical.astype(int).tolist(),
        },
        "histogram_x_by_z": {
            "x_edges_uu": x_edges.tolist(),
            "z_edges_uu": z_edges.tolist(),
            "counts": histogram.astype(int).tolist(),
        },
    }


def _policy_summary(
    raw: dict[str, np.ndarray], rows: np.ndarray, policy: str
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    rival_side = raw["match.rival_side"][rows].astype(np.int64)
    side = rival_side if policy == "rival" else 1 - rival_side
    opponent_side = 1 - side
    blue = raw["match.blue_score"][rows].astype(np.int64)
    orange = raw["match.orange_score"][rows].astype(np.int64)
    goals = np.where(side == 0, blue, orange)
    conceded = np.where(opponent_side == 0, blue, orange)
    winner = raw["match.winner"][rows].astype(np.int64)
    entered_overtime = raw["match.overtime"][rows] != 0
    wins = winner == side
    losses = (winner >= 0) & ~wins
    unresolved = winner < 0
    car_index = (rows, side)
    opponent_index = (rows, opponent_side)

    touches = raw["touch_count"][car_index].astype(np.int64)
    opponent_touches = raw["touch_count"][opponent_index].astype(np.int64)
    touch_total = int(touches.sum() + opponent_touches.sum())
    possession_total = raw["possession_total"][car_index].astype(np.int64)
    possession_same = raw["possession_same"][car_index].astype(np.int64)
    possession_opponent = raw["possession_opponent"][car_index].astype(np.int64)
    finalized = raw["displacement_count"][car_index].astype(np.int64).sum(axis=1)
    events = _goal_events(raw, rows, side)
    scorelines = Counter(f"{int(a)}-{int(b)}" for a, b in zip(goals, conceded, strict=True))
    return {
        "matches": int(rows.size),
        "goals": int(goals.sum()),
        "goals_per_match": None if rows.size == 0 else float(goals.mean()),
        "goals_conceded": int(conceded.sum()),
        "goals_conceded_per_match": None if rows.size == 0 else float(conceded.mean()),
        "regulation_wins": int((wins & ~entered_overtime).sum()),
        "overtime_wins": int((wins & entered_overtime).sum()),
        "total_wins": int(wins.sum()),
        "total_losses": int(losses.sum()),
        "unresolved": int(unresolved.sum()),
        "win_rate": _ratio(int(wins.sum()), int(rows.size)),
        "goal_differential": _distribution(goals - conceded),
        "scoreline_histogram_policy_first": dict(sorted(scorelines.items())),
        "touches": int(touches.sum()),
        "touch_share": _ratio(int(touches.sum()), touch_total),
        "kickoff_first_touches": int(raw["kickoff_first_touch_count"][car_index].sum()),
        "kickoff_goals": int(raw["kickoff_goal_count"][car_index].sum()),
        "next_touch_possession": {
            "total_resolved": int(possession_total.sum()),
            "same_player": _ratio(int(possession_same.sum()), int(possession_total.sum())),
            "opponent_handoff": _ratio(
                int(possession_opponent.sum()), int(possession_total.sum())
            ),
        },
        "immediate_touch_direction": _category_counts(raw["direction_count"][car_index]),
        "net_displacement_before_next_touch_or_goal": _category_counts(
            raw["displacement_count"][car_index]
        ),
        "wall_continuation": _ratio(
            int(raw["wall_continuation_count"][car_index].sum()), int(finalized.sum())
        ),
        "backboard_continuation": _ratio(
            int(raw["backboard_continuation_count"][car_index].sum()), int(finalized.sum())
        ),
        "demos": int(raw["demo_count"][car_index].sum()),
        "goal_entries": _goal_mouth_summary(events),
    }


def _side_stratified(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    all_rows = np.arange(raw["match.rival_side"].size)
    blue_rows = all_rows[raw["match.rival_side"] == 0]
    orange_rows = all_rows[raw["match.rival_side"] == 1]

    def pair(rows: np.ndarray) -> dict[str, Any]:
        return {
            "rival": _policy_summary(raw, rows, "rival"),
            "nexto": _policy_summary(raw, rows, "nexto"),
        }

    return {
        "rival_as_blue": pair(blue_rows),
        "rival_as_orange": pair(orange_rows),
        "overall_with_side_breakdowns": pair(all_rows),
    }


def _physical_team_summary(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    blue = raw["match.blue_score"].astype(np.int64)
    orange = raw["match.orange_score"].astype(np.int64)
    winner = raw["match.winner"].astype(np.int64)
    return {
        "blue": {
            "goals": int(blue.sum()),
            "wins": int((winner == 0).sum()),
            "losses": int((winner == 1).sum()),
            "goals_per_match": float(blue.mean()),
        },
        "orange": {
            "goals": int(orange.sum()),
            "wins": int((winner == 1).sum()),
            "losses": int((winner == 0).sum()),
            "goals_per_match": float(orange.mean()),
        },
    }


def _layout_side_matrix(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    layout = raw["match.starting_layout"]
    rival_side = raw["match.rival_side"]
    result: dict[str, Any] = {}
    for layout_id in range(5):
        result[str(layout_id)] = {
            "layout_name": LAYOUT_NAMES[layout_id],
            "rival_as_blue": _policy_summary(
                raw, np.flatnonzero((layout == layout_id) & (rival_side == 0)), "rival"
            ),
            "rival_as_orange": _policy_summary(
                raw, np.flatnonzero((layout == layout_id) & (rival_side == 1)), "rival"
            ),
        }
    return result


def _canonical_ledger(raw: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for world in range(raw["match.rival_side"].size):
        rival_side = int(raw["match.rival_side"][world])
        blue = int(raw["match.blue_score"][world])
        orange = int(raw["match.orange_score"][world])
        rival_goals, nexto_goals = (blue, orange) if rival_side == 0 else (orange, blue)
        winner = int(raw["match.winner"][world])
        ledger.append(
            {
                "match": world + 1,
                "starting_layout": int(raw["match.starting_layout"][world]),
                "starting_layout_name": LAYOUT_NAMES[int(raw["match.starting_layout"][world])],
                "rival_side": "Blue" if rival_side == 0 else "Orange",
                "blue_score": blue,
                "orange_score": orange,
                "rival_score": rival_goals,
                "nexto_score": nexto_goals,
                "scoreline_rival_nexto": f"{rival_goals}-{nexto_goals}",
                "winner": "Rival" if winner == rival_side else "Nexto",
                "entered_overtime": bool(raw["match.overtime"][world]),
                "overtime_ticks": int(raw["match.overtime_ticks"][world]),
                "total_physics_ticks": int(raw["match.total_ticks"][world]),
            }
        )
    return ledger


def _timing_dict(timing: MatchRunTiming) -> dict[str, Any]:
    return {
        "physics_ticks_requested": timing.physics_ticks_requested,
        "seconds": timing.seconds,
        "world_ticks_per_second": timing.world_ticks_per_second,
    }


def _run_suite(
    *,
    name: str,
    layout: np.ndarray,
    rival_side: np.ndarray,
    stochastic_rival: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = int(layout.size)
    print(f"{name}: initializing {count:,} worlds", flush=True)
    runner = FullMatchRunner(
        count,
        str(COLLISION_ROOT),
        CHECKPOINT,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=stochastic_rival,
        evaluation_seed=seed,
    )
    profile_timing, transfer_names = runner.profile_ticks(PROFILE_TICKS)
    print(f"{name}: running remaining regulation ticks", flush=True)
    regulation_timing = runner.run_ticks(REGULATION_TICKS - PROFILE_TICKS)
    status = runner.phase_status()
    boundary_exports = 1
    overtime_timings: list[MatchRunTiming] = []
    while np.any(status["done"] == 0):
        pending = int((status["done"] == 0).sum())
        if not np.all(status["overtime"][status["done"] == 0] != 0):
            raise RuntimeError(f"{name}: unfinished worlds did not enter overtime")
        print(f"{name}: {pending:,} tied worlds, advancing a 30-second OT block", flush=True)
        overtime_timings.append(runner.run_ticks(OVERTIME_STATUS_TICKS))
        status = runner.phase_status()
        boundary_exports += 1

    exported = runner.export()
    raw = exported.pop("raw")
    if np.any(raw["match.done"] == 0) or np.any(raw["match.winner"] < 0):
        raise RuntimeError(f"{name}: final export contains unresolved matches")
    if np.any(raw["goal_overflow"] != 0):
        raise RuntimeError(f"{name}: goal-event telemetry capacity overflowed")

    long_seconds = regulation_timing.seconds + sum(item.seconds for item in overtime_timings)
    long_world_ticks = count * (
        regulation_timing.physics_ticks_requested
        + sum(item.physics_ticks_requested for item in overtime_timings)
    )
    result = {
        "suite": name,
        "worlds": count,
        "rival_action_mode": "stochastic_hybrid_sampling" if stochastic_rival else "deterministic_deployment",
        "nexto_action_mode": "deterministic_beta_1_argmax_plus_stock_kickoff",
        "seed": seed,
        "regulation_physics_ticks": REGULATION_TICKS,
        "overtime_rule": "fresh kickoff, next goal wins",
        "zero_second_airborne_continuation": False,
        "side_assignment_counts": {
            "rival_as_blue": int((rival_side == 0).sum()),
            "rival_as_orange": int((rival_side == 1).sum()),
        },
        "starting_layout_counts": {
            str(index): int((layout == index).sum()) for index in range(5)
        },
        "results_by_rival_side": _side_stratified(raw),
        "physical_team_results": _physical_team_summary(raw),
        "results_by_starting_layout_and_rival_side": _layout_side_matrix(raw),
        "performance": {
            "profiled_first_ticks": _timing_dict(profile_timing),
            "profiled_h2d_d2h_event_count": len(transfer_names),
            "profiled_h2d_d2h_event_names": transfer_names,
            "long_timed_blocks_seconds": long_seconds,
            "long_timed_blocks_world_ticks": long_world_ticks,
            "long_timed_blocks_world_ticks_per_second": long_world_ticks / long_seconds,
            "peak_cuda_bytes": exported["peak_cuda_bytes"],
            "world_host_to_device_bytes_after_initialization": exported[
                "world_host_to_device_bytes_after_initialization"
            ],
            "world_device_to_host_bytes_before_final_export": exported[
                "world_device_to_host_bytes_before_export"
            ],
            "nexto_timed_h2d_bytes": exported["nexto_timed_h2d_bytes"],
            "nexto_timed_d2h_bytes": exported["nexto_timed_d2h_bytes"],
            "compact_phase_boundary_exports": boundary_exports,
        },
        "checkpoint": exported["checkpoint"],
        "goal_telemetry_capacity_per_world": int(raw["goal_scorer"].shape[1]),
        "goal_telemetry_overflow_worlds": int((raw["goal_overflow"] != 0).sum()),
    }
    if name == "canonical_deterministic":
        result["canonical_match_ledger"] = _canonical_ledger(raw)
    return result, raw


def _side_line(summary: dict[str, Any]) -> str:
    return (
        f"{summary['total_wins']}-{summary['total_losses']}, "
        f"goals {summary['goals']}-{summary['goals_conceded']}, "
        f"GD mean {summary['goal_differential']['mean']:.3f}, "
        f"median {summary['goal_differential']['median']:.3f}"
    )


def _write_document(
    canonical: dict[str, Any], stochastic: dict[str, Any], fidelity: dict[str, Any], manifest: dict[str, Any]
) -> None:
    cb = canonical["results_by_rival_side"]["rival_as_blue"]["rival"]
    co = canonical["results_by_rival_side"]["rival_as_orange"]["rival"]
    sb = stochastic["results_by_rival_side"]["rival_as_blue"]["rival"]
    so = stochastic["results_by_rival_side"]["rival_as_orange"]["rival"]
    lines = [
        "# Rival 2.0 vs pinned public Nexto",
        "",
        f"Verdict: **{manifest['verdict']}**.",
        "",
        "The result is intentionally reported by Rival's assigned team. Blue and Orange are never collapsed into a headline aggregate because team assignment materially changes the observed scoring distribution.",
        "",
        "## Canonical deterministic deployment suite (primary)",
        "",
        f"- Rival as **Blue**: {_side_line(cb)}",
        f"- Rival as **Orange**: {_side_line(co)}",
        "",
        "| Layout | Rival side | Blue | Orange | Rival-Nexto | Winner | OT |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in canonical["canonical_match_ledger"]:
        lines.append(
            f"| {item['starting_layout']} ({item['starting_layout_name']}) | {item['rival_side']} | "
            f"{item['blue_score']} | {item['orange_score']} | {item['scoreline_rival_nexto']} | "
            f"{item['winner']} | {'yes' if item['entered_overtime'] else 'no'} |"
        )
    lines += [
        "",
        "These ten trajectories are the complete deterministic 5-layout by 2-side matrix; duplicated copies would not be independent evidence.",
        "",
        f"Physical-team totals in the canonical matrix: Blue {canonical['physical_team_results']['blue']['goals']} goals / {canonical['physical_team_results']['blue']['wins']} wins; Orange {canonical['physical_team_results']['orange']['goals']} goals / {canonical['physical_team_results']['orange']['wins']} wins.",
        "",
        "## Stochastic Rival robustness suite (secondary)",
        "",
        f"- Rival as **Blue** ({sb['matches']:,} matches): {_side_line(sb)}",
        f"- Rival as **Orange** ({so['matches']:,} matches): {_side_line(so)}",
        f"- Physical-team totals: Blue {stochastic['physical_team_results']['blue']['goals']:,} goals / {stochastic['physical_team_results']['blue']['wins']:,} wins; Orange {stochastic['physical_team_results']['orange']['goals']:,} goals / {stochastic['physical_team_results']['orange']['wins']:,} wins.",
        "",
        "This suite samples Rival's ordinary hybrid policy distribution with a fixed seed. Nexto remains deterministic. It is a robustness measurement, not the headline deployment matchup.",
        "",
        "## Side-separated behavior",
        "",
        "| Suite / Rival side | Touches | Touch share | Kickoff first touches | Kickoff goals | Same next touch | Opponent handoff | Demos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in (
        ("Canonical / Blue", cb),
        ("Canonical / Orange", co),
        ("Stochastic / Blue", sb),
        ("Stochastic / Orange", so),
    ):
        lines.append(
            f"| {label} | {value['touches']:,} | {value['touch_share']['fraction']:.6f} | "
            f"{value['kickoff_first_touches']:,} | {value['kickoff_goals']:,} | "
            f"{value['next_touch_possession']['same_player']['fraction']:.6f} | "
            f"{value['next_touch_possession']['opponent_handoff']['fraction']:.6f} | {value['demos']:,} |"
        )
    lines += [
        "",
        "The machine-readable suite files additionally retain side-separated forward/neutral/backward touch direction, net displacement, wall/backboard continuation, complete goal-mouth X/Z histograms, and layout-by-side results. Backward, lateral, wall, and backboard touches are descriptive categories, not value judgments.",
        "",
        "## Fidelity and runtime gates",
        "",
        f"- Observation parity: q `{fidelity['observation_parity']['q_max_abs_error']}`, kv `{fidelity['observation_parity']['kv_max_abs_error']}`, mask `{fidelity['observation_parity']['mask_max_abs_error']}` max absolute error.",
        f"- Deterministic action agreement: `{fidelity['model_action_parity']['argmax_agreement_count']}/{fidelity['corpus']['states']}` (100%).",
        f"- Action table: `{fidelity['action_table']['count']}` actions, SHA-256 `{fidelity['action_table']['float32_sha256']}`.",
        f"- Stock kickoff: `{fidelity['kickoff_sequence']['physics_ticks']}` controls at 120 Hz, SHA-256 `{fidelity['kickoff_sequence']['float32_sha256']}`.",
        f"- Fidelity hot-path H2D/D2H events: `{fidelity['hot_path']['profiled_h2d_d2h_event_count']}`.",
        f"- Canonical match throughput: `{canonical['performance']['long_timed_blocks_world_ticks_per_second']:.2f}` world-ticks/s; peak CUDA `{canonical['performance']['peak_cuda_bytes'] / 2**30:.3f}` GiB.",
        f"- Stochastic match throughput: `{stochastic['performance']['long_timed_blocks_world_ticks_per_second']:.2f}` world-ticks/s; peak CUDA `{stochastic['performance']['peak_cuda_bytes'] / 2**30:.3f}` GiB.",
        f"- Timed match-loop transfer profiler events: canonical `{canonical['performance']['profiled_h2d_d2h_event_count']}`, stochastic `{stochastic['performance']['profiled_h2d_d2h_event_count']}`.",
        "",
        "Compact match-status exports occur only after regulation and at coarse overtime boundaries; they are outside the timed per-tick loop. The match clock intentionally omits Rocket League's zero-second airborne continuation rule, as authorized. Training episode timeouts and no-touch truncation do not control this runtime.",
        "",
        "## Identity",
        "",
        f"- Rival checkpoint SHA-256: `{manifest['identity']['rival_checkpoint_sha256']}`",
        f"- Rival policy version / samples: `{manifest['identity']['rival_policy_version']}` / `{manifest['identity']['rival_total_agent_samples']:,}`",
        f"- Nexto upstream commit: `{manifest['identity']['nexto_upstream_commit']}`",
        f"- Nexto model SHA-256: `{manifest['identity']['nexto_model_sha256']}`",
        "- Pinned upstream license: CC BY-NC-SA 4.0; exact source/model/license blobs are isolated under `third_party/nexto/`.",
        "",
        "## Evidence files",
        "",
        "- `results/rival2/nexto/fidelity.json`",
        "- `results/rival2/nexto/canonical_deterministic.json`",
        "- `results/rival2/nexto/canonical_match_ledger.json`",
        "- `results/rival2/nexto/stochastic_robustness.json`",
        "- `results/rival2/nexto/summary.json`",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary-worlds", type=int, default=4096)
    args = parser.parse_args()
    if args.secondary_worlds <= 0 or args.secondary_worlds & (args.secondary_worlds - 1):
        raise ValueError("secondary worlds must be a positive power of two")
    if not FIDELITY_PATH.exists():
        raise FileNotFoundError("run the targeted Nexto fidelity gate first")
    fidelity = json.loads(FIDELITY_PATH.read_text(encoding="utf-8"))
    if fidelity.get("verdict") != "PASS_GREEN":
        raise RuntimeError("Nexto fidelity gate is not green")
    if not COLLISION_ROOT.exists():
        raise FileNotFoundError(COLLISION_ROOT)

    started = time.perf_counter()
    canonical_layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    canonical_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    canonical, _canonical_raw = _run_suite(
        name="canonical_deterministic",
        layout=canonical_layout,
        rival_side=canonical_side,
        stochastic_rival=False,
        seed=CANONICAL_SEED,
    )
    _write_json(OUTPUT_DIR / "canonical_deterministic.json", canonical)
    _write_json(OUTPUT_DIR / "canonical_match_ledger.json", canonical["canonical_match_ledger"])
    del _canonical_raw
    gc.collect()
    torch.cuda.empty_cache()

    index = np.arange(args.secondary_worlds, dtype=np.int32)
    stochastic_side = index % 2
    stochastic_layout = (index // 2) % 5
    stochastic, _stochastic_raw = _run_suite(
        name="stochastic_robustness",
        layout=stochastic_layout,
        rival_side=stochastic_side,
        stochastic_rival=True,
        seed=STOCHASTIC_SEED,
    )
    _write_json(OUTPUT_DIR / "stochastic_robustness.json", stochastic)
    del _stochastic_raw

    manifest = {
        "verdict": "PASS_GREEN",
        "scope": "authorized Rival-vs-Nexto benchmark; no training or policy/physics change",
        "reporting_rule": "Rival-as-Blue and Rival-as-Orange are always published separately",
        "identity": {
            "rival_checkpoint_sha256": _sha256(CHECKPOINT),
            "rival_policy_version": canonical["checkpoint"]["policy_version"],
            "rival_total_agent_samples": canonical["checkpoint"]["total_agent_samples"],
            "nexto_upstream_commit": fidelity["provenance"]["upstream_commit"],
            "nexto_model_sha256": fidelity["provenance"]["model_sha256"],
            "nexto_action_table_sha256": fidelity["action_table"]["float32_sha256"],
            "nexto_kickoff_sequence_sha256": fidelity["kickoff_sequence"]["float32_sha256"],
        },
        "canonical_side_results": {
            "rival_as_blue": canonical["results_by_rival_side"]["rival_as_blue"]["rival"],
            "rival_as_orange": canonical["results_by_rival_side"]["rival_as_orange"]["rival"],
        },
        "stochastic_side_results": {
            "rival_as_blue": stochastic["results_by_rival_side"]["rival_as_blue"]["rival"],
            "rival_as_orange": stochastic["results_by_rival_side"]["rival_as_orange"]["rival"],
        },
        "fidelity_gate": fidelity["verdict"],
        "full_match_gates": {
            "canonical_matches_exactly_10": canonical["worlds"] == 10,
            "canonical_side_layout_matrix_complete": len(canonical["canonical_match_ledger"]) == 10,
            "secondary_side_assignments_equal": stochastic["side_assignment_counts"]["rival_as_blue"]
            == stochastic["side_assignment_counts"]["rival_as_orange"],
            "all_matches_resolved": all(
                group["rival"]["unresolved"] == 0
                for suite in (canonical, stochastic)
                for group_name, group in suite["results_by_rival_side"].items()
                if group_name != "overall_with_side_breakdowns"
            ),
            "goal_telemetry_no_overflow": canonical["goal_telemetry_overflow_worlds"] == 0
            and stochastic["goal_telemetry_overflow_worlds"] == 0,
            "timed_loop_h2d_d2h_zero": canonical["performance"]["profiled_h2d_d2h_event_count"] == 0
            and stochastic["performance"]["profiled_h2d_d2h_event_count"] == 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "secondary_worlds": args.secondary_worlds,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    if not all(manifest["full_match_gates"].values()):
        manifest["verdict"] = "FAIL_RED"
    _write_json(OUTPUT_DIR / "summary.json", manifest)
    _write_document(canonical, stochastic, fidelity, manifest)
    print(json.dumps({
        "verdict": manifest["verdict"],
        "canonical_side_results": {
            key: {field: value[field] for field in ("matches", "total_wins", "total_losses", "goals", "goals_conceded")}
            for key, value in manifest["canonical_side_results"].items()
        },
        "stochastic_side_results": {
            key: {field: value[field] for field in ("matches", "total_wins", "total_losses", "goals", "goals_conceded")}
            for key, value in manifest["stochastic_side_results"].items()
        },
        "wall_seconds": manifest["wall_seconds"],
    }, indent=2), flush=True)
    return 0 if manifest["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
