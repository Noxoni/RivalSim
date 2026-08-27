"""Run the bounded Rival 2.0 Gameplay V2 mixed-opponent curriculum."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
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
from rivalsim.rival2_contracts import (  # noqa: E402
    EPISODE_CONTRACT_HASH,
    GAMEPLAY_STRICT_DOUBLE_DASH_REWARD,
    REWARD_CONTRACT_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V2_CONTRACT,
    REWARD_GAMEPLAY_V2_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import Rival2MixedPPOSafetyConfig  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    NEXTO_ACTING_VERSION,
    OPPONENT_NAMES,
    WISP_ACTING_VERSION,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402
from rivalsim.wisp_short_eval import WispShortEpisodeRunner  # noqa: E402
from third_party.wisp75b.adapter import (  # noqa: E402
    WISP_BOTPACK_COMMIT,
    WISP_POLICY_SHA256,
    WISP_SHARED_HEAD_SHA256,
    WISP_UPSTREAM_COMMIT,
)

SCHEMA_VERSION = 1
AUTHORITY = Path("handoff/rival2-opponent-curriculum-v1/README.md")
AUTHORITATIVE_HEAD = "58c1578ccb719b1a7782a128842cf10761a6d227"
SAFE_TRANSITION_BASE = "77b1e3df3a9f7226458445a13128e784fd22c268"
SOURCE_CHECKPOINT = Path("checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt")
SOURCE_CHECKPOINT_SHA256 = "77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21"
WISP_FIDELITY = Path("results/rival2/opponent_curriculum_v1/wisp_fidelity.json")
NEXTO_FIDELITY = Path(".tools/opponent_curriculum_nexto_fidelity.json")
RESULTS_DIR = Path("results/rival2/opponent_curriculum_v1")
REPORT_PATH = Path("docs/RIVAL2_OPPONENT_CURRICULUM_V1_RESULTS.md")
KL_DIAGNOSIS_REPORT_PATH = Path("docs/RIVAL2_OPPONENT_CURRICULUM_V1_KL_DIAGNOSIS.md")
KL_TRANSITION_REPORT_PATH = Path("docs/RIVAL2_OPPONENT_CURRICULUM_V1_TRANSITION_STRATEGY.md")
FINAL_CHECKPOINT = Path(
    "checkpoints/rival2/opponent_curriculum_v1/rival2_opponent_curriculum_resume.pt"
)
RETENTION_CORPUS = Path("results/rival2/opponent_curriculum_v1/safe_transition/retention_corpus.pt")
RETENTION_CORPUS_SUMMARY = RETENTION_CORPUS.with_suffix(".json")
WORLDS = 131_072
CAMPAIGN_SEED = 2_026_082_703
DETERMINISTIC_EVALUATION_SEED = 2_026_082_711
STOCHASTIC_EVALUATION_SEED = 2_026_082_712
EVALUATION_EVENT_CAPACITY_PER_CAR = 256
ADDITIONAL_UPDATES = 120
CHECKPOINT_OFFSETS = (30, 60, 90, 120)
EXPECTED_ITERATIONS = (389, 419, 449, 479)
KL_GUARD = Rival2KLGuardConfig(
    minibatch_kl_limit=0.10,
    completed_update_mean_kl_limit=0.05,
)
MIXED_PPO_SAFETY = Rival2MixedPPOSafetyConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-smoke-checkpoint",
        type=Path,
        help="run only one 2-per-side Wisp/Nexto evaluator smoke and exit",
    )
    parser.add_argument("--evaluation-smoke-sha256")
    parser.add_argument(
        "--resume-source-evaluation",
        action="store_true",
        help="reuse a complete green source evaluation in the work directory",
    )
    parser.add_argument(
        "--finalize-existing-rejection",
        action="store_true",
        help="audit and finalize an already-published KL rejection without simulation",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


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
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def _assignments(worlds_per_side: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(worlds_per_side, dtype=np.int32)
    side = np.concatenate(
        (
            np.zeros(worlds_per_side, dtype=np.int32),
            np.ones(worlds_per_side, dtype=np.int32),
        )
    )
    layout = np.concatenate((local % 5, local % 5)).astype(np.int32)
    return side, layout


def _select_car(array: np.ndarray, rows: np.ndarray, side: np.ndarray) -> np.ndarray:
    return array[rows, side]


def _movement_summary(
    raw: dict[str, np.ndarray], rows: np.ndarray, side: np.ndarray
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
        "mean_speed_uu_per_s": None if tick_total == 0 else floating("speed_sum") / tick_total,
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


def _touch_heights(raw: dict[str, np.ndarray], rows: np.ndarray, side: np.ndarray) -> np.ndarray:
    capacity = raw["touch_ball_height"].shape[2]
    values: list[np.ndarray] = []
    for row, physical_side in zip(rows, side, strict=True):
        count = min(int(raw["touch_event_count"][row, physical_side]), capacity)
        if count:
            values.append(raw["touch_ball_height"][row, physical_side, :count])
    return np.concatenate(values).astype(np.float64) if values else np.empty(0)


def _candidate_counts(
    events: list[dict[str, Any]], rows: np.ndarray, policy: str
) -> dict[str, int]:
    selected = set(int(value) for value in rows)
    counts: Counter[str] = Counter()
    for event in events:
        if int(event["world"]) in selected and event["policy"] == policy:
            counts.update(event["candidate_labels"])
    return dict(sorted(counts.items()))


def _strict_pair_count(events: list[dict[str, Any]], rows: np.ndarray, policy: str) -> int:
    selected = set(int(value) for value in rows)
    pairs: set[tuple[int, int, int]] = set()
    for event in events:
        if int(event["world"]) not in selected or event["policy"] != policy:
            continue
        evidence = event["classification_evidence"].get("double_dash_candidate")
        if evidence is not None:
            pairs.add(
                (
                    int(event["world"]),
                    int(event["physical_side"]),
                    int(evidence["current_ordinal"]),
                )
            )
    return len(pairs)


def _group_summary(
    raw: dict[str, np.ndarray],
    events: list[dict[str, Any]],
    rival_side: np.ndarray,
    rows: np.ndarray,
    opponent_name: str,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    rival = rival_side[rows].astype(np.int64)
    opponent = 1 - rival
    termination = raw["termination_kind"][rows]
    winner = raw["winner"][rows]
    decisive = termination == TERMINATION_GOAL
    rival_wins = int((decisive & (winner == rival)).sum())
    opponent_wins = int((decisive & (winner == opponent)).sum())
    no_touch = int((termination == TERMINATION_NO_TOUCH).sum())
    hard_time = int((termination == TERMINATION_HARD_TIME).sum())
    first = raw["first_toucher"][rows]
    resolved = first >= 0
    rival_first = int((resolved & (first == rival)).sum())
    rival_touches = int(_select_car(raw["touch_count"], rows, rival).sum())
    opponent_touches = int(_select_car(raw["touch_count"], rows, opponent).sum())
    rival_touch_events = int(_select_car(raw["touch_event_count"], rows, rival).sum())
    opponent_touch_events = int(_select_car(raw["touch_event_count"], rows, opponent).sum())
    rival_airborne_touches = int(_select_car(raw["airborne_touch_count"], rows, rival).sum())
    opponent_airborne_touches = int(_select_car(raw["airborne_touch_count"], rows, opponent).sum())
    rival_pairs = _strict_pair_count(events, rows, "Rival")
    opponent_pairs = _strict_pair_count(events, rows, opponent_name)
    duration = raw["duration_ticks"][rows].astype(np.float64) / PHYSICS_HZ
    return {
        "episodes": int(rows.size),
        "rival_wins": rival_wins,
        "opponent_wins": opponent_wins,
        "no_goal_episodes": no_touch + hard_time,
        "decisive_episodes": int(decisive.sum()),
        "rival_win_rate_among_decisive": _ratio(rival_wins, int(decisive.sum())),
        "goals_for": rival_wins,
        "goals_against": opponent_wins,
        "goal_differential": rival_wins - opponent_wins,
        "goal_terminated_fraction": _ratio(int(decisive.sum()), int(rows.size)),
        "no_touch_fraction": _ratio(no_touch, int(rows.size)),
        "hard_timeout_fraction": _ratio(hard_time, int(rows.size)),
        "touches": {
            "Rival": rival_touches,
            opponent_name: opponent_touches,
            "differential": rival_touches - opponent_touches,
        },
        "first_touch": {
            "resolved_episodes": int(resolved.sum()),
            "Rival": rival_first,
            opponent_name: int((resolved & (first == opponent)).sum()),
            "rival_share": _ratio(rival_first, int(resolved.sum())),
        },
        "saves": {
            "Rival": int(_select_car(raw["save_count"], rows, rival).sum()),
            opponent_name: int(_select_car(raw["save_count"], rows, opponent).sum()),
        },
        "conceded_goals": {"Rival": opponent_wins, opponent_name: rival_wins},
        "episode_duration_seconds": _distribution(duration),
        "movement_controller": {
            "Rival": _movement_summary(raw, rows, rival),
            opponent_name: _movement_summary(raw, rows, opponent),
        },
        "airborne_touch_fraction": {
            "Rival": _ratio(rival_airborne_touches, rival_touch_events),
            opponent_name: _ratio(opponent_airborne_touches, opponent_touch_events),
        },
        "touch_ball_center_height_uu": {
            "Rival": _distribution(_touch_heights(raw, rows, rival)),
            opponent_name: _distribution(_touch_heights(raw, rows, opponent)),
        },
        "dash_candidate_event_counts": {
            "Rival": _candidate_counts(events, rows, "Rival"),
            opponent_name: _candidate_counts(events, rows, opponent_name),
        },
        "strict_double_dash": {
            "Rival": rival_pairs,
            opponent_name: opponent_pairs,
            "rival_reward_contribution": GAMEPLAY_STRICT_DOUBLE_DASH_REWARD
            * (rival_pairs - opponent_pairs),
        },
    }


def _summarize_evaluation(
    export: dict[str, Any],
    events: list[dict[str, Any]],
    mechanics: dict[str, Any],
    *,
    opponent_name: str,
    mode: str,
    seed: int,
    timing: dict[str, Any],
) -> dict[str, Any]:
    raw = export["raw"]
    rival_side = export["rival_side"]
    worlds = int(rival_side.size)
    all_rows = np.arange(worlds, dtype=np.int64)
    blue_rows = np.flatnonzero(rival_side == 0)
    orange_rows = np.flatnonzero(rival_side == 1)
    checks = {
        "every_world_completed_exactly_one_episode": bool(np.all(raw["done"] == 1)),
        "valid_termination_partition": bool(
            np.all(
                np.isin(
                    raw["termination_kind"],
                    (TERMINATION_GOAL, TERMINATION_NO_TOUCH, TERMINATION_HARD_TIME),
                )
            )
        ),
        "dash_event_capacity_no_overflow": int(raw["event_overflow"].sum()) == 0,
        "touch_event_capacity_no_overflow": int(raw["touch_event_overflow"].sum()) == 0,
        "world_hot_path_h2d_zero": export["world_host_to_device_bytes_after_initialization"] == 0,
        "world_hot_path_d2h_zero": export["world_device_to_host_bytes_after_initialization"] == 0,
    }
    if opponent_name == "Nexto":
        checks.update(
            {
                "nexto_timed_h2d_zero": export["nexto_timed_h2d_bytes"] == 0,
                "nexto_timed_d2h_zero": export["nexto_timed_d2h_bytes"] == 0,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "checkpoint": export["checkpoint_identity"],
        "opponent": opponent_name,
        "rival_action_mode": mode,
        "opponent_action_mode": "deterministic source-faithful frozen policy",
        "evaluation_seed": int(seed),
        "episode_contract": {
            "version": RIVAL2_EPISODE_VERSION,
            "standard_kickoff": True,
            "first_goal_terminates": True,
            "no_touch_seconds": 15,
            "hard_limit_seconds": 45,
            "rival_policy_hz": 30,
            "physics_hz": PHYSICS_HZ,
        },
        "episodes": worlds,
        "by_rival_side": {
            "overall": _group_summary(raw, events, rival_side, all_rows, opponent_name),
            "Rival_Blue": _group_summary(raw, events, rival_side, blue_rows, opponent_name),
            "Rival_Orange": _group_summary(raw, events, rival_side, orange_rows, opponent_name),
        },
        "mechanics": mechanics,
        "telemetry_capacity": {
            "per_car": EVALUATION_EVENT_CAPACITY_PER_CAR,
            "dash_event_overflow_total": int(raw["event_overflow"].sum()),
            "touch_event_overflow_total": int(raw["touch_event_overflow"].sum()),
        },
        "performance": {
            **timing,
            "peak_cuda_bytes": int(export["peak_cuda_bytes"]),
            "opponent_inference_calls": int(
                export.get("nexto_inference_calls", export.get("wisp_inference_calls", 0))
            ),
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _write_episode_ledger(
    path: Path,
    export: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    checkpoint_label: str,
    opponent_name: str,
) -> None:
    raw = export["raw"]
    rival_side = export["rival_side"]
    layout = export["starting_layout"]
    event_counts: dict[tuple[int, str], Counter[str]] = {}
    for event in events:
        key = (int(event["world"]), str(event["policy"]))
        event_counts.setdefault(key, Counter()).update(event["candidate_labels"])
    fields = (
        "checkpoint",
        "opponent",
        "world",
        "starting_layout",
        "rival_side",
        "termination",
        "winner",
        "duration_ticks",
        "rival_touches",
        "opponent_touches",
        "first_toucher",
        "rival_saves",
        "opponent_saves",
        "rival_airborne_touches",
        "opponent_airborne_touches",
        "rival_flips",
        "opponent_flips",
        "rival_wavedash_candidate_events",
        "opponent_wavedash_candidate_events",
        "rival_double_dash_candidate_events",
        "opponent_double_dash_candidate_events",
    )
    termination_name = {
        TERMINATION_GOAL: "goal",
        TERMINATION_NO_TOUCH: "no_touch",
        TERMINATION_HARD_TIME: "hard_time",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for world in range(rival_side.size):
            rival = int(rival_side[world])
            opponent = 1 - rival
            winner = int(raw["winner"][world])
            first = int(raw["first_toucher"][world])
            rival_events = event_counts.get((world, "Rival"), Counter())
            opponent_events = event_counts.get((world, opponent_name), Counter())
            writer.writerow(
                {
                    "checkpoint": checkpoint_label,
                    "opponent": opponent_name,
                    "world": world,
                    "starting_layout": int(layout[world]),
                    "rival_side": "Blue" if rival == 0 else "Orange",
                    "termination": termination_name[int(raw["termination_kind"][world])],
                    "winner": (
                        "Rival"
                        if winner == rival
                        else opponent_name
                        if winner == opponent
                        else "None"
                    ),
                    "duration_ticks": int(raw["duration_ticks"][world]),
                    "rival_touches": int(raw["touch_count"][world, rival]),
                    "opponent_touches": int(raw["touch_count"][world, opponent]),
                    "first_toucher": (
                        "Rival"
                        if first == rival
                        else opponent_name
                        if first == opponent
                        else "None"
                    ),
                    "rival_saves": int(raw["save_count"][world, rival]),
                    "opponent_saves": int(raw["save_count"][world, opponent]),
                    "rival_airborne_touches": int(raw["airborne_touch_count"][world, rival]),
                    "opponent_airborne_touches": int(raw["airborne_touch_count"][world, opponent]),
                    "rival_flips": int(raw["flip_onsets"][world, rival]),
                    "opponent_flips": int(raw["flip_onsets"][world, opponent]),
                    "rival_wavedash_candidate_events": rival_events["wavedash_candidate"],
                    "opponent_wavedash_candidate_events": opponent_events["wavedash_candidate"],
                    "rival_double_dash_candidate_events": rival_events["double_dash_candidate"],
                    "opponent_double_dash_candidate_events": opponent_events[
                        "double_dash_candidate"
                    ],
                }
            )


def run_evaluation(
    *,
    opponent_name: str,
    mode: str,
    checkpoint_label: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    collision_dir: Path,
    device: str,
    work_dir: Path,
    worlds_per_side: int,
    seed: int,
) -> dict[str, Any]:
    rival_side, layout = _assignments(worlds_per_side)
    runner_type = NextoShortEpisodeRunner if opponent_name == "Nexto" else WispShortEpisodeRunner
    runner = runner_type(
        rival_side.size,
        str(collision_dir),
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        starting_layout=layout,
        rival_side=rival_side,
        stochastic_rival=mode == "stochastic",
        evaluation_seed=seed,
        device=device,
        dash_event_capacity=EVALUATION_EVENT_CAPACITY_PER_CAR,
    )
    timing = asdict(runner.run())
    export = runner.export()
    events, mechanics = classify_dash_events(
        export["raw"],
        rival_side=rival_side,
        starting_layout=layout,
        checkpoint_label=checkpoint_label,
        opponent_name=opponent_name,
    )
    result = _summarize_evaluation(
        export,
        events,
        mechanics,
        opponent_name=opponent_name,
        mode=mode,
        seed=seed,
        timing=timing,
    )
    stem = f"evaluation_{checkpoint_label}_{opponent_name.lower()}_{mode}"
    _write_json(work_dir / f"{stem}.json", result)
    _write_json(work_dir / f"{stem}_dash_events.json", events)
    _write_episode_ledger(
        work_dir / f"{stem}_episodes.csv",
        export,
        events,
        checkpoint_label=checkpoint_label,
        opponent_name=opponent_name,
    )
    # The runner temporarily makes its Warp-backed Torch stream current.  Put
    # both runtimes back on the process default before releasing the runner's
    # captured graph and stream wrappers; otherwise the next environment may
    # inherit a destroyed CUDA handle.
    torch.cuda.synchronize(device)
    default_torch_stream = torch.cuda.default_stream(torch.device(device))
    torch.cuda.set_stream(default_torch_stream)
    wp.set_stream(
        wp.stream_from_torch(default_torch_stream),
        device=device,
        sync=True,
    )
    del runner, export, events
    gc.collect()
    torch.cuda.empty_cache()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"{stem} failed: {result['checks']}")
    return result


def evaluate_checkpoint(
    *,
    label: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    collision_dir: Path,
    device: str,
    work_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint_label": label,
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "opponents": {},
    }
    for opponent_name in ("Nexto", "Wisp"):
        result["opponents"][opponent_name] = {
            "deterministic": run_evaluation(
                opponent_name=opponent_name,
                mode="deterministic",
                checkpoint_label=label,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                collision_dir=collision_dir,
                device=device,
                work_dir=work_dir,
                worlds_per_side=5,
                seed=DETERMINISTIC_EVALUATION_SEED,
            ),
            "stochastic": run_evaluation(
                opponent_name=opponent_name,
                mode="stochastic",
                checkpoint_label=label,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                collision_dir=collision_dir,
                device=device,
                work_dir=work_dir,
                worlds_per_side=128,
                seed=STOCHASTIC_EVALUATION_SEED,
            ),
        }
    return result


def frozen_configuration(source: dict[str, Any]) -> dict[str, Any]:
    policy = Rival2PolicyConfig(**source["policy_config"])
    ppo = Rival2PPOConfig(**source["ppo_config"])
    self_play = Rival2SelfPlayConfig(**source["self_play_config"])
    curriculum = Rival2OpponentCurriculumConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY.as_posix(),
        "authoritative_head": AUTHORITATIVE_HEAD,
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_iteration": int(source["iteration"]),
        "source_policy_version": int(source["policy_version"]),
        "source_agent_decision_samples": int(source["total_agent_samples"]),
        "source_reward_version": source["reward_version"],
        "destination_reward_version": RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        "destination_reward_contract": REWARD_GAMEPLAY_V2_CONTRACT,
        "destination_reward_contract_hash": REWARD_GAMEPLAY_V2_CONTRACT_HASH,
        "episode_version": RIVAL2_EPISODE_VERSION,
        "episode_contract_hash": EPISODE_CONTRACT_HASH,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "mixed_ppo_safety": asdict(MIXED_PPO_SAFETY),
        "mixed_ppo_safety_hash": MIXED_PPO_SAFETY.content_hash,
        "retention_corpus": RETENTION_CORPUS.as_posix(),
        "self_play_config": asdict(self_play),
        "opponent_curriculum": asdict(curriculum),
        "worlds": WORLDS,
        "campaign_seed": CAMPAIGN_SEED,
        "training": {
            "additional_updates": ADDITIONAL_UPDATES,
            "checkpoint_offsets": list(CHECKPOINT_OFFSETS),
            "expected_iterations": list(EXPECTED_ITERATIONS),
            "trainable_samples_are_measured_not_assumed": True,
        },
        "kl_guard": asdict(KL_GUARD),
        "evaluation": {
            "labels": ["source", "plus_030", "plus_060", "plus_090", "plus_120"],
            "opponents": ["Nexto", "Wisp"],
            "deterministic": {
                "episodes_per_opponent": 10,
                "five_layouts_each_rival_side": True,
                "seed": DETERMINISTIC_EVALUATION_SEED,
            },
            "stochastic": {
                "episodes_per_opponent": 256,
                "episodes_per_rival_side": 128,
                "seed": STOCHASTIC_EVALUATION_SEED,
            },
        },
        "frozen_opponents": {
            "Nexto": {
                "upstream_commit": NEXTO_UPSTREAM_COMMIT,
                "model_sha256": NEXTO_MODEL_SHA256,
            },
            "Wisp": {
                "upstream_commit": WISP_UPSTREAM_COMMIT,
                "botpack_commit": WISP_BOTPACK_COMMIT,
                "policy_sha256": WISP_POLICY_SHA256,
                "shared_head_sha256": WISP_SHARED_HEAD_SHA256,
            },
        },
        "five_minute_training_matches": False,
        "imitation_learning": False,
        "opponent_training": False,
    }


def verify_launch(configuration: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    source_sha = _sha256(SOURCE_CHECKPOINT)
    wisp = json.loads(WISP_FIDELITY.read_text(encoding="utf-8"))
    nexto = json.loads(NEXTO_FIDELITY.read_text(encoding="utf-8"))
    retention = json.loads(RETENTION_CORPUS_SUMMARY.read_text(encoding="utf-8"))
    ppo = configuration["ppo_config"]
    mix = configuration["opponent_curriculum"]
    checks = {
        "safe_transition_base_in_head_history": _git_is_ancestor(SAFE_TRANSITION_BASE, "HEAD"),
        "safe_transition_base_in_origin_main_history": _git_is_ancestor(
            SAFE_TRANSITION_BASE, "origin/main"
        ),
        "authority_present": AUTHORITY.is_file(),
        "source_checkpoint_sha256_exact": source_sha == SOURCE_CHECKPOINT_SHA256,
        "source_iteration_359": int(source["iteration"]) == 359,
        "source_policy_version_359": int(source["policy_version"]) == 359,
        "source_reward_gameplay_v1": source["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        "source_episode_short_v1": source["episode_version"] == RIVAL2_EPISODE_VERSION,
        "source_contracts_exact": source["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V1_VERSION, RIVAL2_EPISODE_VERSION),
        "historical_reward_v1_immutable": REWARD_CONTRACT_HASH
        == "E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F",
        "gameplay_v1_immutable": REWARD_GAMEPLAY_V1_CONTRACT_HASH
        == "48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072",
        "gameplay_v2_zero_sum": REWARD_GAMEPLAY_V2_CONTRACT["zero_sum"] is True,
        "gameplay_v2_only_new_term": (
            REWARD_GAMEPLAY_V2_CONTRACT["base_reward_version"] == RIVAL2_REWARD_GAMEPLAY_V1_VERSION
            and REWARD_GAMEPLAY_V2_CONTRACT["successful_strict_double_dash"]["event_reward"]
            == GAMEPLAY_STRICT_DOUBLE_DASH_REWARD
            and REWARD_GAMEPLAY_V2_CONTRACT["other_changes_from_gameplay_v1"] == []
        ),
        "wisp_fidelity_pass_green": wisp.get("verdict") == "PASS_GREEN",
        "wisp_policy_sha_exact": wisp["identity"]["policy_sha256"] == WISP_POLICY_SHA256,
        "wisp_shared_head_sha_exact": wisp["identity"]["shared_head_sha256"]
        == WISP_SHARED_HEAD_SHA256,
        "nexto_fidelity_pass_green": nexto.get("verdict") == "PASS_GREEN",
        "nexto_model_sha_exact": nexto["provenance"]["model_sha256"] == NEXTO_MODEL_SHA256,
        "retention_corpus_pass_green": retention.get("verdict") == "PASS_GREEN",
        "retention_source_checkpoint_exact": retention["source_identity"]["checkpoint_sha256"]
        == SOURCE_CHECKPOINT_SHA256,
        "retention_safety_config_exact": retention["safety_config_hash"]
        == MIXED_PPO_SAFETY.content_hash,
        "retention_artifact_sha_exact": retention["artifact"]["sha256"]
        == _sha256(RETENTION_CORPUS),
        "world_count_exact": configuration["worlds"] == WORLDS,
        "mix_exact": mix
        == {
            "nexto_probability": 0.35,
            "wisp_probability": 0.35,
            "current_probability": 0.20,
            "historical_probability": 0.10,
            "seed": CAMPAIGN_SEED,
        },
        "entropy_zero": ppo["entropy_coefficient"] == 0.0,
        "base_ppo_learning_rate_identity_unchanged": ppo["learning_rate"] == 0.0003,
        "mixed_policy_learning_rate_exact": configuration["mixed_ppo_safety"][
            "initial_policy_learning_rate"
        ]
        == 0.0001,
        "mixed_critic_learning_rate_exact": configuration["mixed_ppo_safety"][
            "critic_learning_rate"
        ]
        == 0.0003,
        "mixed_soft_minibatch_kl_exact": configuration["mixed_ppo_safety"][
            "soft_minibatch_kl_target"
        ]
        == 0.02,
        "mixed_retention_kl_exact": configuration["mixed_ppo_safety"][
            "retention_soft_mean_kl_target"
        ]
        == 0.02,
        "mixed_policy_minimum_learning_rate_exact": configuration["mixed_ppo_safety"][
            "minimum_policy_learning_rate"
        ]
        == 0.000025,
        "clip_range_unchanged": ppo["clip_range"] == 0.2,
        "value_loss_coefficient_unchanged": ppo["value_loss_coefficient"] == 0.5,
        "max_gradient_norm_unchanged": ppo["max_gradient_norm"] == 0.5,
        "horizon_32": ppo["rollout_horizon"] == 32,
        "epochs_unchanged": ppo["epochs"] == 2,
        "minibatch_unchanged": ppo["minibatch_size"] == 65_536,
        "update_boundary_exact": configuration["training"]["additional_updates"] == 120,
        "no_five_minute_training": not configuration["five_minute_training_matches"],
        "no_opponent_training": not configuration["opponent_training"],
        "no_imitation": not configuration["imitation_learning"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "source_checkpoint_sha256": source_sha,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"opponent curriculum launch gate failed: {checks}")
    return result


def transition_preservation_gate(
    source: dict[str, Any],
    trainer: Rival2OpponentCurriculumTrainer,
    transition: dict[str, Any],
) -> dict[str, Any]:
    before_assignment = trainer.checkpoint_payload()
    preserved = (
        "model",
        "optimizer",
        "policy_config",
        "ppo_config",
        "self_play_config",
        "policy_config_hash",
        "ppo_config_hash",
        "policy_version",
        "iteration",
        "total_agent_samples",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "policy_generator_state",
        "opponent_generator_state",
        "opponent_assignment",
        "historical_opponents",
    )
    checks = {
        f"{name}_exact_before_new_assignment": _nested_exact(source[name], before_assignment[name])
        for name in preserved
    }
    trainer.initialize_curriculum_assignments()
    checks.update(
        {
            "reward_and_fresh_world_are_only_transition_changes": transition["changed_semantics"]
            == ["reward_contract", "fresh_world_state"],
            "episode_contract_unchanged": transition["source_episode_version"]
            == transition["destination_episode_version"]
            == RIVAL2_EPISODE_VERSION,
            "fresh_short_episode_state": bool(
                torch.all(trainer.env.bridge.views["rival2.episode_ticks"] == 0).item()
                and torch.all(trainer.env.bridge.views["rival2.no_touch_ticks"] == 0).item()
            ),
            "new_family_assignment_complete": bool(
                torch.all((trainer.opponent_family >= 0) & (trainer.opponent_family < 4)).item()
            ),
            "new_assignment_count_exact": int(trainer.realized_family_assignments.sum().item())
            == trainer.env.num_envs,
            "dedicated_curriculum_rng_recorded": (
                trainer.curriculum_transition is not None
                and "opponent_curriculum_initialization" in trainer.curriculum_transition
            ),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "preserved_fields": list(preserved),
        "initial_family_counts": {
            name: int(trainer.realized_family_assignments[index].item())
            for index, name in enumerate(OPPONENT_NAMES)
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _training_integrity(
    trainer: Rival2OpponentCurriculumTrainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    *,
    policy_before: int,
    samples_before: int,
) -> dict[str, Any]:
    expected_samples = int(rollout.train_mask.sum().item())
    curriculum = trainer.last_rollout_curriculum_metrics
    adaptive = trainer.last_adaptive_ppo_diagnostics
    checks = {
        "finite_metrics": all(torch.isfinite(value).item() for value in metrics.values()),
        "finite_rewards": bool(torch.isfinite(rollout.rewards).all().item()),
        "finite_returns": bool(torch.isfinite(rollout.returns).all().item()),
        "finite_advantages": bool(torch.isfinite(rollout.advantages).all().item()),
        "reward_zero_sum_exact": bool((rollout.rewards.sum(dim=-1) == 0.0).all().item()),
        "policy_increment_exact": trainer.policy_version == policy_before + 1,
        "iteration_matches_policy": trainer.iteration == trainer.policy_version,
        "sample_increment_exact": trainer.total_agent_samples - samples_before == expected_samples,
        "trainable_policy_is_current": bool(
            torch.all(rollout.policy_version[rollout.train_mask] == policy_before).item()
        ),
        "nexto_never_trainable": bool(
            torch.all(~rollout.train_mask[rollout.policy_version == NEXTO_ACTING_VERSION]).item()
        ),
        "wisp_never_trainable": bool(
            torch.all(~rollout.train_mask[rollout.policy_version == WISP_ACTING_VERSION]).item()
        ),
        "family_sample_ledger_exact": curriculum is not None
        and sum(curriculum["trainable_agent_samples"].values()) == expected_samples,
        "family_world_decision_ledger_exact": curriculum is not None
        and sum(curriculum["world_decisions"].values())
        == trainer.env.num_envs * trainer.ppo_config.rollout_horizon,
        "completed_update_kl_within_guard": float(metrics["approx_kl"].item())
        <= KL_GUARD.completed_update_mean_kl_limit,
        "minibatch_kl_within_guard": float(metrics["optimizer_post_step_approx_kl_max"].item())
        <= KL_GUARD.minibatch_kl_limit,
        "adaptive_ppo_diagnostics_present": adaptive is not None,
        "adaptive_ppo_pass_green": adaptive is not None and adaptive.get("verdict") == "PASS_GREEN",
        "value_loss_to_trunk_gradient_zero": adaptive is not None
        and adaptive["checks"]["value_loss_to_shared_trunk_gradient_exact_zero"],
        "value_loss_to_actor_gradient_zero": adaptive is not None
        and adaptive["checks"]["value_loss_to_actor_gradient_exact_zero"],
        "retention_kl_within_soft_target": adaptive is not None
        and adaptive["retention_corpus_mean_kl"] <= MIXED_PPO_SAFETY.retention_soft_mean_kl_target,
        "world_hot_path_zero_transfer": trainer.env.hot_path_transfer_bytes()
        == {"h2d": 0, "d2h": 0},
    }
    return {
        "expected_trainable_agent_samples": expected_samples,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _checkpoint(
    label: str,
    trainer: Rival2OpponentCurriculumTrainer,
    work_dir: Path,
) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_opponent_curriculum_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    finite_model = all(torch.isfinite(value).all().item() for value in payload["model"].values())
    curriculum = payload.get("opponent_curriculum")
    checks = {
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "iteration_exact": int(payload["iteration"]) == trainer.iteration,
        "policy_version_exact": int(payload["policy_version"]) == trainer.policy_version,
        "sample_count_exact": int(payload["total_agent_samples"]) == trainer.total_agent_samples,
        "reward_exact": payload["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        "episode_exact": payload["episode_version"] == RIVAL2_EPISODE_VERSION,
        "contracts_exact": payload["contract_hashes"] == trainer.env.contract_hashes,
        "model_finite": finite_model,
        "curriculum_transition_present": "curriculum_transition" in payload,
        "opponent_curriculum_present": curriculum is not None,
        "curriculum_rng_present": curriculum is not None and "generator_state" in curriculum,
        "wisp_temporal_state_present": curriculum is not None and "wisp" in curriculum,
        "nexto_temporal_state_present": curriculum is not None and "nexto" in curriculum,
        "adaptive_ppo_present": curriculum is not None
        and curriculum.get("adaptive_ppo") is not None,
        "retention_corpus_present": curriculum is not None
        and (curriculum.get("adaptive_ppo") or {}).get("retention_observations") is not None,
        "split_optimizer_groups_present": len(payload["optimizer"]["param_groups"]) == 2,
    }
    return {
        "label": label,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "historical_pool_versions": list(trainer.opponent_pool.versions),
        "realized_family_assignments": {
            name: int(trainer.realized_family_assignments[index].item())
            for index, name in enumerate(OPPONENT_NAMES)
        },
        "audit": {
            "checks": checks,
            "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        },
    }


def _write_report(
    summary: dict[str, Any], evaluations: list[dict[str, Any]], checkpoints: list[dict[str, Any]]
) -> None:
    lines = [
        "# Rival 2.0 Opponent Curriculum V1 results",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"Source checkpoint: `{SOURCE_CHECKPOINT_SHA256}` at iteration `359`.",
        "",
        f"Gameplay V2 reward hash: `{REWARD_GAMEPLAY_V2_CONTRACT_HASH}`.",
        "",
        "Opponent mix per newly reset episode: 35% Nexto, 35% Wisp, "
        "20% current Rival, and 10% historical Rival.",
        "",
        "## Held-out opponent curve",
        "",
        "| checkpoint | opponent | mode | episodes | Rival W-L-NG | goal diff | "
        "no-touch | first-touch | Rival/opp touches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for evaluation in evaluations:
        for opponent, modes in evaluation["opponents"].items():
            for mode, result in modes.items():
                overall = result["by_rival_side"]["overall"]
                lines.append(
                    f"| {evaluation['checkpoint_label']} | {opponent} | {mode} | "
                    f"{result['episodes']} | {overall['rival_wins']}-"
                    f"{overall['opponent_wins']}-{overall['no_goal_episodes']} | "
                    f"{overall['goal_differential']} | "
                    f"{overall['no_touch_fraction']['fraction']:.6f} | "
                    f"{overall['first_touch']['rival_share']['fraction']:.6f} | "
                    f"{overall['touches']['Rival']}/{overall['touches'][opponent]} |"
                )
    lines.extend(
        [
            "",
            "## Held-out side splits",
            "",
            "| checkpoint | opponent | mode | Rival side | episodes | Rival W-L-NG | "
            "goal diff | no-touch | first-touch | Rival/opp touches |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for evaluation in evaluations:
        for opponent, modes in evaluation["opponents"].items():
            for mode, result in modes.items():
                for side_key, side_label in (
                    ("Rival_Blue", "Blue"),
                    ("Rival_Orange", "Orange"),
                ):
                    side = result["by_rival_side"][side_key]
                    lines.append(
                        f"| {evaluation['checkpoint_label']} | {opponent} | {mode} | "
                        f"{side_label} | {side['episodes']} | {side['rival_wins']}-"
                        f"{side['opponent_wins']}-{side['no_goal_episodes']} | "
                        f"{side['goal_differential']} | "
                        f"{side['no_touch_fraction']['fraction']:.6f} | "
                        f"{side['first_touch']['rival_share']['fraction']:.6f} | "
                        f"{side['touches']['Rival']}/{side['touches'][opponent]} |"
                    )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| label | iteration | cumulative samples | SHA-256 | audit |",
            "|---|---:|---:|---|---|",
        ]
    )
    for checkpoint in checkpoints:
        lines.append(
            f"| {checkpoint['label']} | {checkpoint['iteration']} | "
            f"{checkpoint['agent_decision_samples']} | `{checkpoint['sha256']}` | "
            f"`{checkpoint['audit']['verdict']}` |"
        )
    if summary["status"] == "STOPPED_KL_GUARD_REJECTION":
        diagnostic = summary["diagnostic"]
        lines.extend(
            [
                "",
                "## Mandatory KL-guard stop",
                "",
                f"Rejected update: `{diagnostic['rejected_iteration']}`.",
                "",
                f"Reason: `{diagnostic['reason']}`; post-step minibatch KL "
                f"`{diagnostic['post_step_approx_kl']:.9f}` exceeded the hard "
                f"`{diagnostic['minibatch_kl_limit']}` limit.",
                "",
                "No PPO update completed. Model, optimizer, gradients, and relevant "
                "RNG state were restored, and no later training or evaluation ran.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Boundary",
                "",
                "Training stopped at the authorized +120 boundary.",
            ]
        )
    lines.extend(
        [
            "",
            "No v0.6 work, five-minute training, opponent training, imitation, or "
            "continuation was run.",
            "",
            "Machine-readable PPO safety evidence, per-family sample ledgers, "
            "side-separated opponent evaluations, touch-height distributions, and "
            "dash-event evidence are stored under `results/rival2/opponent_curriculum_v1/`.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def publish(
    work_dir: Path,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    allowed_existing = {"wisp_fidelity.json"}
    unexpected = {path.name for path in RESULTS_DIR.iterdir() if path.name not in allowed_existing}
    if unexpected:
        raise RuntimeError(f"results directory has unexpected existing artifacts: {unexpected}")
    for path in sorted(work_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}:
            shutil.copy2(path, RESULTS_DIR / path.name)
    _write_json(
        RESULTS_DIR / "nexto_fidelity_regression.json",
        json.loads(NEXTO_FIDELITY.read_text(encoding="utf-8")),
    )
    final_source = Path(checkpoints[-1]["path"])
    FINAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_source, FINAL_CHECKPOINT)
    final_identity = {
        "path": FINAL_CHECKPOINT.as_posix(),
        "sha256": _sha256(FINAL_CHECKPOINT),
        "size_bytes": FINAL_CHECKPOINT.stat().st_size,
        "iteration": checkpoints[-1]["iteration"],
        "policy_version": checkpoints[-1]["policy_version"],
        "agent_decision_samples": checkpoints[-1]["agent_decision_samples"],
    }
    _write_json(RESULTS_DIR / "final_checkpoint.json", final_identity)
    summary["final_repository_checkpoint"] = final_identity
    _write_json(RESULTS_DIR / "run_summary.json", summary)
    _write_report(summary, evaluations, checkpoints)
    _write_artifact_manifest()


def _write_artifact_manifest() -> None:
    def committed_identity(path: Path) -> tuple[str, int]:
        content = path.read_bytes()
        if path.suffix.lower() in {".csv", ".json", ".jsonl", ".md"}:
            content = content.replace(b"\r\n", b"\n")
        return hashlib.sha256(content).hexdigest().upper(), len(content)

    artifacts = []
    for path in sorted(RESULTS_DIR.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            digest, size_bytes = committed_identity(path)
            artifacts.append(
                {
                    "path": path.as_posix(),
                    "sha256": digest,
                    "size_bytes": size_bytes,
                }
            )
    for path in (
        FINAL_CHECKPOINT,
        REPORT_PATH,
        KL_DIAGNOSIS_REPORT_PATH,
        KL_TRANSITION_REPORT_PATH,
    ):
        if not path.is_file():
            continue
        digest, size_bytes = committed_identity(path)
        artifacts.append(
            {
                "path": path.as_posix(),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
    _write_json(
        RESULTS_DIR / "artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "artifacts": artifacts,
        },
    )


def finalize_existing_rejection(work_dir: Path) -> dict[str, Any]:
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    restored = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    work_summary = json.loads((work_dir / "run_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text(encoding="utf-8"))
    if (
        work_summary.get("status") != "STOPPED_KL_GUARD_REJECTION"
        or summary.get("status") != "STOPPED_KL_GUARD_REJECTION"
    ):
        raise RuntimeError("existing result is not the governed KL rejection")
    family_values = restored["opponent_curriculum"]["realized_family_assignments"].tolist()
    family_counts = {name: int(family_values[index]) for index, name in enumerate(OPPONENT_NAMES)}
    horizon = int(source["ppo_config"]["rollout_horizon"])
    family_samples = {
        "current": family_counts["current"] * horizon * 2,
        "historical": family_counts["historical"] * horizon,
        "nexto": family_counts["nexto"] * horizon,
        "wisp": family_counts["wisp"] * horizon,
    }
    sample_delta = int(restored["total_agent_samples"] - source["total_agent_samples"])
    work_checkpoint = Path(summary["diagnostic"]["restored_checkpoint"]["path"])
    checks = {
        "model_exactly_restored_to_source": _nested_exact(source["model"], restored["model"]),
        "optimizer_exactly_restored_to_source": _nested_exact(
            source["optimizer"], restored["optimizer"]
        ),
        "policy_configuration_exact": _nested_exact(
            source["policy_config"], restored["policy_config"]
        ),
        "ppo_configuration_exact": _nested_exact(source["ppo_config"], restored["ppo_config"]),
        "self_play_configuration_exact": _nested_exact(
            source["self_play_config"], restored["self_play_config"]
        ),
        "historical_pool_exact": _nested_exact(
            source["historical_opponents"], restored["historical_opponents"]
        ),
        "cpu_rng_exact": _nested_exact(
            source["torch_cpu_rng_state"], restored["torch_cpu_rng_state"]
        ),
        "cuda_rng_exact": _nested_exact(
            source["torch_cuda_rng_state"], restored["torch_cuda_rng_state"]
        ),
        "opponent_generator_rng_exact": _nested_exact(
            source["opponent_generator_state"], restored["opponent_generator_state"]
        ),
        "iteration_not_advanced": int(restored["iteration"]) == 359,
        "policy_version_not_advanced": int(restored["policy_version"]) == 359,
        "rollout_sample_delta_exact": sample_delta == sum(family_samples.values()),
        "restored_checkpoint_hash_exact": _sha256(FINAL_CHECKPOINT)
        == summary["final_repository_checkpoint"]["sha256"],
        "work_and_repository_checkpoint_bytes_exact": _sha256(work_checkpoint)
        == _sha256(FINAL_CHECKPOINT),
        "transactional_rollback_reported": summary["diagnostic"].get(
            "transactional_rollback_completed"
        )
        is True,
        "no_later_training_reported": summary["diagnostic"].get("no_later_training_performed")
        is True,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "source_checkpoint": {
            "path": SOURCE_CHECKPOINT.as_posix(),
            "sha256": _sha256(SOURCE_CHECKPOINT),
            "iteration": int(source["iteration"]),
            "policy_version": int(source["policy_version"]),
            "agent_decision_samples": int(source["total_agent_samples"]),
        },
        "restored_checkpoint": summary["final_repository_checkpoint"],
        "rollout_trainable_agent_samples": sample_delta,
        "rollout_trainable_agent_samples_by_family": family_samples,
        "initial_episode_assignments_by_family": family_counts,
        "policy_generator_rng_note": (
            "The rollout legitimately advanced the policy generator. The KL transaction "
            "restored it to the exact state immediately before the rejected PPO update, "
            "not to the source checkpoint state before rollout collection."
        ),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if audit["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"rollback audit failed: {checks}")
    _write_json(work_dir / "rollback_audit.json", audit)
    _write_json(RESULTS_DIR / "rollback_audit.json", audit)
    summary["rollout_trainable_agent_samples"] = sample_delta
    summary["rollout_trainable_agent_samples_by_family"] = family_samples
    summary["initial_episode_assignments_by_family"] = family_counts
    summary["rollback_audit_verdict"] = audit["verdict"]
    summary["no_completed_ppo_update"] = True
    _write_json(work_dir / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "run_summary.json", summary)
    evaluations = json.loads((RESULTS_DIR / "evaluation_curve.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((RESULTS_DIR / "checkpoints.json").read_text(encoding="utf-8"))
    _write_report(summary, evaluations, checkpoints)
    _write_artifact_manifest()
    return audit


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    configuration = frozen_configuration(source)
    launch = verify_launch(configuration, source)
    _write_json(args.work_dir / "config.json", configuration)
    _write_json(args.work_dir / "launch_gate.json", launch)

    evaluations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    if args.resume_source_evaluation:
        evaluations = json.loads(
            (args.work_dir / "evaluation_curve.json").read_text(encoding="utf-8")
        )
        if (
            len(evaluations) != 1
            or evaluations[0].get("checkpoint_label") != "source"
            or evaluations[0].get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or any(
                result.get("verdict") != "PASS_GREEN"
                for modes in evaluations[0]["opponents"].values()
                for result in modes.values()
            )
        ):
            raise RuntimeError("reusable source evaluation is absent or not fully green")
    else:
        source_evaluation = evaluate_checkpoint(
            label="source",
            checkpoint_path=SOURCE_CHECKPOINT,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            collision_dir=args.collision_dir,
            device=args.device,
            work_dir=args.work_dir,
        )
        evaluations.append(source_evaluation)
        _write_json(args.work_dir / "evaluation_curve.json", evaluations)

    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    kickoff_selector = (np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        WORLDS,
        str(args.collision_dir),
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(),
        seed=CAMPAIGN_SEED,
    )
    transition = trainer.load_checkpoint_curriculum_transition(
        SOURCE_CHECKPOINT,
        source_reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY.as_posix(),
            "authorized_change": (
                "Gameplay V1 +239 to fresh short-lifecycle Gameplay V2 mixed opponents"
            ),
            "source_commit": AUTHORITATIVE_HEAD,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "collapsed_scoring_v1_lineage_used": False,
            "five_minute_world_state_carried": False,
        },
    )
    transition_gate = transition_preservation_gate(source, trainer, transition)
    if transition_gate["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"transition gate failed: {transition_gate['checks']}")
    optimizer_migration = trainer.enable_safe_mixed_ppo(MIXED_PPO_SAFETY)
    retention_payload = torch.load(RETENTION_CORPUS, map_location=args.device, weights_only=False)
    if retention_payload.get("format") != "RIVAL2_RETENTION_OBSERVATIONS_V1":
        raise ValueError("unsupported Rival 2.0 retention corpus format")
    trainer.install_retention_corpus(
        retention_payload["observations"], retention_payload["summary"]
    )
    _write_json(args.work_dir / "transition.json", trainer.curriculum_transition)
    _write_json(args.work_dir / "transition_gate.json", transition_gate)
    _write_json(args.work_dir / "optimizer_migration.json", optimizer_migration)
    _write_json(args.work_dir / "retention_corpus.json", trainer.retention_corpus_summary)

    ledger = args.work_dir / "training_curve.jsonl"
    started = time.perf_counter()
    snapshot_records: list[dict[str, Any]] = []
    for offset in range(1, ADDITIONAL_UPDATES + 1):
        policy_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        trainer.env.reset_transfer_counters()
        update_started = time.perf_counter()
        rollout = trainer.collect_rollout()
        try:
            metrics = trainer.update(rollout, kl_guard=KL_GUARD)
        except Rival2PolicyDisplacementRejected as error:
            torch.cuda.synchronize(args.device)
            checkpoint = _checkpoint(
                f"pre_rejected_update_{trainer.iteration + 1:05d}", trainer, args.work_dir
            )
            diagnostic = {
                **error.diagnostics,
                "created_utc": _utc_now(),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "restored_checkpoint": checkpoint,
                "no_later_training_performed": True,
                "rollout_curriculum_metrics": trainer.last_rollout_curriculum_metrics,
            }
            summary = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_KL_GUARD_REJECTION",
                "completed_additional_updates": trainer.iteration - int(source["iteration"]),
                "restored_iteration": trainer.iteration,
                "restored_policy_version": trainer.policy_version,
                "agent_decision_samples_after_rejected_rollout": trainer.total_agent_samples,
                "diagnostic": diagnostic,
            }
            _write_json(args.work_dir / "kl_rejection.json", diagnostic)
            _write_json(args.work_dir / "run_summary.json", summary)
            _write_json(args.work_dir / "checkpoints.json", [*checkpoints, checkpoint])
            _write_json(args.work_dir / "evaluation_curve.json", evaluations)
            publish(args.work_dir, summary, evaluations, [*checkpoints, checkpoint])
            return summary

        torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - update_started
        integrity = _training_integrity(
            trainer,
            rollout,
            metrics,
            policy_before=policy_before,
            samples_before=samples_before,
        )
        values = {name: float(value.item()) for name, value in metrics.items()}
        values.update(
            {
                "rollout_reward_mean_blue": float(rollout.rewards[..., 0].mean().item()),
                "rollout_reward_mean_orange": float(rollout.rewards[..., 1].mean().item()),
                "rollout_reward_mean_absolute": float(rollout.rewards.abs().mean().item()),
                "rollout_reward_max_absolute": float(rollout.rewards.abs().amax().item()),
            }
        )
        point = {
            "phase": "GAMEPLAY_V2_MIXED_OPPONENT_SHORT_EPISODE",
            "offset": offset,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
            "reward_version": trainer.env.reward_version,
            "episode_version": trainer.env.episode_version,
            "wall_seconds": seconds,
            "trainable_agent_decisions_per_second": (
                (trainer.total_agent_samples - samples_before) / seconds
            ),
            "family": trainer.last_rollout_curriculum_metrics,
            "adaptive_ppo": trainer.last_adaptive_ppo_diagnostics,
            "metrics": values,
            "integrity": integrity,
            "verdict": integrity["verdict"],
        }
        _append_jsonl(ledger, point)
        print(
            f"opponent-curriculum update={trainer.iteration} offset={offset}/120 "
            f"samples={trainer.total_agent_samples} delta="
            f"{trainer.total_agent_samples - samples_before} seconds={seconds:.3f} "
            f"kl={values['approx_kl']:.6f} "
            f"mb_kl_max={values['optimizer_post_step_approx_kl_max']:.6f} "
            f"verdict={point['verdict']}",
            flush=True,
        )
        if integrity["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"training integrity failure at iteration {trainer.iteration}")
        del rollout, metrics
        gc.collect()
        if offset not in CHECKPOINT_OFFSETS:
            continue

        # Persist the normal boundary snapshot in the resumable checkpoint.
        pool_before = list(trainer.opponent_pool.versions)
        trainer.add_historical_snapshot()
        pool_after = list(trainer.opponent_pool.versions)
        evicted = [version for version in pool_before if version not in pool_after]
        snapshot_record = {
            "offset": offset,
            "iteration": trainer.iteration,
            "added_version": trainer.policy_version,
            "pool_before": pool_before,
            "pool_after": pool_after,
            "evicted_versions": evicted,
        }
        snapshot_records.append(snapshot_record)
        label = f"plus_{offset:03d}"
        checkpoint = _checkpoint(label, trainer, args.work_dir)
        if checkpoint["audit"]["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"checkpoint audit failed at {label}")
        evaluation = evaluate_checkpoint(
            label=label,
            checkpoint_path=Path(checkpoint["path"]),
            checkpoint_sha256=checkpoint["sha256"],
            collision_dir=args.collision_dir,
            device=args.device,
            work_dir=args.work_dir,
        )
        checkpoints.append(checkpoint)
        evaluations.append(evaluation)
        _write_json(args.work_dir / "snapshot_records.json", snapshot_records)
        _write_json(args.work_dir / "checkpoints.json", checkpoints)
        _write_json(args.work_dir / "evaluation_curve.json", evaluations)

    final = checkpoints[-1]
    realized_total = {
        name: int(trainer.realized_family_assignments[index].item())
        for index, name in enumerate(OPPONENT_NAMES)
    }
    assignment_denominator = sum(realized_total.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "COMPLETE_120_UPDATE_BOUNDARY",
        "source_commit": AUTHORITATIVE_HEAD,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "reward_version": RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        "reward_contract_hash": REWARD_GAMEPLAY_V2_CONTRACT_HASH,
        "episode_version": RIVAL2_EPISODE_VERSION,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - int(source["iteration"]),
        "additional_agent_decision_samples": trainer.total_agent_samples
        - int(source["total_agent_samples"]),
        "realized_family_assignments": realized_total,
        "realized_family_assignment_fraction": {
            name: value / assignment_denominator for name, value in realized_total.items()
        },
        "historical_snapshot_records": snapshot_records,
        "final_work_checkpoint": final,
        "evaluation_labels": [item["checkpoint_label"] for item in evaluations],
        "kl_guard_rejections": 0,
        "wall_seconds_including_training_and_boundary_evaluations": time.perf_counter() - started,
        "five_minute_training_matches_run": False,
        "opponent_training_run": False,
        "imitation_learning_run": False,
        "v0_6_started": False,
        "recommendation": (
            "Compare side-separated Nexto/Wisp goal, acquisition, high-ball, and "
            "strict-dash curves before authorizing any curriculum beyond +120."
        ),
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    publish(args.work_dir, summary, evaluations, checkpoints)
    return summary


def run_evaluation_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.evaluation_smoke_checkpoint
    if checkpoint is None or args.evaluation_smoke_sha256 is None:
        raise ValueError("evaluation smoke requires checkpoint and SHA-256")
    result: dict[str, Any] = {}
    for opponent in ("Nexto", "Wisp"):
        result[opponent] = run_evaluation(
            opponent_name=opponent,
            mode="deterministic",
            checkpoint_label="smoke",
            checkpoint_path=checkpoint,
            checkpoint_sha256=args.evaluation_smoke_sha256,
            collision_dir=args.collision_dir,
            device=args.device,
            work_dir=args.work_dir,
            worlds_per_side=2,
            seed=DETERMINISTIC_EVALUATION_SEED,
        )
    _write_json(args.work_dir / "evaluation_smoke.json", result)
    return result


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    if (
        args.work_dir.exists()
        and any(args.work_dir.iterdir())
        and not args.resume_source_evaluation
        and not args.finalize_existing_rejection
    ):
        raise RuntimeError("work directory must be absent or empty")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.evaluation_smoke_checkpoint is not None:
        result = run_evaluation_smoke(args)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.finalize_existing_rejection:
        audit = finalize_existing_rejection(args.work_dir)
        print(json.dumps(audit, indent=2), flush=True)
        return 0
    summary = run_campaign(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETE_120_UPDATE_BOUNDARY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
