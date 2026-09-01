"""Fresh recurrent Human Sequence Seed v1 preparation and Stage-1 training.

Preparation freezes the direct shared observation view, countdown-aligned whole-
episode split, and recurrent training authority before any optimizer step.  Training
then rematerializes the immutable recording and ranks checkpoints only by validation
complete-action RMSE.  This module contains no PPO or reward optimization path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.human_demo.training_adapter import action_target, adapt_frame  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM  # noqa: E402
from rivalsim.rival2_human_sequence import (  # noqa: E402
    HUMAN_SEQUENCE_OBS_VIEW_CONTRACT,
    HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256,
    RETAINED_OBSERVATION_FIELDS,
    RETAINED_OBSERVATION_INDICES,
    ZEROED_OBSERVATION_INDICES,
    direct_human_sequence_observation,
    project_human_sequence_observation,
)
from rivalsim.rival2_recurrent_policy import (  # noqa: E402
    FROZEN_STAGE1_LOG_STD,
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
)

FORMAT = "RIVAL2_HUMAN_SEQUENCE_SEED_V1"
CHECKPOINT_FORMAT = f"{FORMAT}_STAGE1_CHECKPOINT"
ORIGINAL_PACKAGE_COMMIT = "AD01B15AA949111F2EFABA803FC3175BC26E9D0E"
ADDENDUM_COMMIT = "AFA4AFB36FEA60FD566D22487E42EC138C51B3E3"
SESSION_UUID = "CD6E7DB1-2761-4B8B-BD37-F21C7F135722"
SOURCE_FRAME_COUNT = 58_306
SOURCE_FILE_SET_SHA256 = "D7195C22964FDB096EDCB6D6ECBCE10C152F2D5F645B2095E64C2418F68927B9"
DATASET_MANIFEST = ROOT / "results/rival2/human_demo_dataset_v1/dataset_manifest.json"
REVIEW_RECORD = ROOT / "results/rival2/human_demo_review_v2/sessions" / f"{SESSION_UUID}.json"
RESULTS = ROOT / "results/rival2/human_sequence_seed_v1"
AUTHORITY = RESULTS / "authority.json"
SPLIT_MANIFEST = RESULTS / "source_split_manifest.json"
PARITY = RESULTS / "projected_matched_state_parity.json"
CURVE = RESULTS / "stage1_curve.jsonl"
SELECTED = RESULTS / "stage1_selected.json"
TEST_METRICS = RESULTS / "stage1_untouched_test.json"
CHECKPOINT = ROOT / "checkpoints/rival2/human_sequence_seed_v1" / "rival2_human_sequence_seed_v1.pt"

INITIALIZATION_SEED = 2026090201
TRAINING_SEED = 2026090202
PARITY_SEED = 2026090203
WINDOW_TICKS = 256
BURN_IN_TICKS = 64
BATCH_WINDOWS = 12
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
MAX_STEPS = 10_000
VALIDATION_INTERVAL = 100
MIN_PLATEAU_STEP = 1_500
PLATEAU_CHECKS = 20
SIGNIFICANT_IMPROVEMENT = 1.0e-4
GRADIENT_CLIP_NORM = 1.0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class PlayableSegment:
    identity: str
    raw_start: int
    playable_start: int
    playable_end: int
    raw_end: int
    start_sequence: int
    end_sequence: int
    start_physics_frame: int
    end_physics_frame: int
    start_round_number: int
    frame_count: int
    frozen_prefix_frames: int
    post_goal_suffix_frames: int
    playable_rule: str

    def indices(self) -> range:
        return range(self.playable_start, self.playable_end)


def _human(frame: dict[str, Any]) -> dict[str, Any]:
    rows = [car for car in frame.get("cars", ()) if car.get("flags", {}).get("is_local_human")]
    if len(rows) != 1:
        raise RuntimeError(f"frame {frame.get('sequence')} lacks one unique human")
    return rows[0]


def identify_playable_segments(frames: list[dict[str, Any]]) -> list[PlayableSegment]:
    """Apply the addendum's native playable-boundary rule."""

    raw_starts = [0]
    for index in range(1, len(frames)):
        if int(frames[index]["physics_frame"]) != int(frames[index - 1]["physics_frame"]) + 1:
            raw_starts.append(index)
    raw_ends = [*raw_starts[1:], len(frames)]
    segments: list[PlayableSegment] = []
    for ordinal, (raw_start, raw_end) in enumerate(zip(raw_starts, raw_ends, strict=True)):
        if ordinal == 0:
            baseline_time = float(frames[raw_start]["match"].get("total_game_time_played", 0.0))
            playable_start = next(
                (
                    index
                    for index in range(raw_start, raw_end)
                    if bool(frames[index]["match"]["flags"].get("round_active"))
                    and float(frames[index]["match"].get("total_game_time_played", 0.0))
                    > baseline_time + 1.0e-6
                ),
                None,
            )
            rule = "initial_total_game_time_begins_advancing_while_round_active"
        else:
            playable_start = next(
                (
                    index
                    for index in range(raw_start, raw_end)
                    if bool(frames[index]["match"]["flags"].get("round_active"))
                ),
                None,
            )
            rule = "post_goal_round_active_false_to_true"
        if playable_start is None:
            raise RuntimeError(f"raw segment {ordinal} has no playable boundary")
        playable_end = next(
            (
                index
                for index in range(playable_start + 1, raw_end)
                if not bool(frames[index]["match"]["flags"].get("round_active"))
            ),
            raw_end,
        )
        if playable_end <= playable_start:
            raise RuntimeError(f"raw segment {ordinal} has no playable frames")
        start_frame = frames[playable_start]
        end_frame = frames[playable_end - 1]
        human = _human(start_frame)
        planar_speed = float(np.linalg.norm(np.asarray(human["linear_velocity"][:2])))
        if not bool(human["flags"].get("on_ground")) or planar_speed > 30.0:
            raise RuntimeError(
                f"segment {ordinal} playable boundary is not a settled kickoff state"
            )
        segments.append(
            PlayableSegment(
                identity=f"episode-{ordinal:02d}",
                raw_start=raw_start,
                playable_start=playable_start,
                playable_end=playable_end,
                raw_end=raw_end,
                start_sequence=int(start_frame["sequence"]),
                end_sequence=int(end_frame["sequence"]),
                start_physics_frame=int(start_frame["physics_frame"]),
                end_physics_frame=int(end_frame["physics_frame"]),
                start_round_number=int(start_frame["match"].get("round_number", ordinal + 1)),
                frame_count=playable_end - playable_start,
                frozen_prefix_frames=playable_start - raw_start,
                post_goal_suffix_frames=raw_end - playable_end,
                playable_rule=rule,
            )
        )
    return segments


def choose_whole_segment_splits(
    segments: list[PlayableSegment],
) -> dict[str, list[PlayableSegment]]:
    if len(segments) < 3:
        raise RuntimeError("whole-episode split needs at least three playable episodes")
    cumulative = np.cumsum([segment.frame_count for segment in segments])
    total = int(cumulative[-1])
    candidates: list[tuple[float, int, int]] = []
    for train_end in range(1, len(segments) - 1):
        for validation_end in range(train_end + 1, len(segments)):
            score = abs(int(cumulative[train_end - 1]) - 0.8 * total) + abs(
                int(cumulative[validation_end - 1]) - 0.9 * total
            )
            candidates.append((float(score), train_end, validation_end))
    _score, train_end, validation_end = min(candidates)
    return {
        "train": segments[:train_end],
        "validation": segments[train_end:validation_end],
        "test": segments[validation_end:],
    }


def _split_indices(segments: Iterable[PlayableSegment]) -> np.ndarray:
    return np.asarray(
        [index for segment in segments for index in segment.indices()], dtype=np.int64
    )


def materialize(
    source_root: Path, *, retain_frames: bool
) -> tuple[
    np.ndarray, np.ndarray, list[PlayableSegment], list[dict[str, Any]] | None, dict[str, Any]
]:
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    gameplay = dataset["general_gameplay"]
    review = json.loads(REVIEW_RECORD.read_text(encoding="utf-8"))
    if (
        gameplay["session_uuid"] != SESSION_UUID
        or int(gameplay["source_frame_count"]) != SOURCE_FRAME_COUNT
    ):
        raise RuntimeError("reviewed gameplay source identity changed")
    if review["source_file_set_sha256"] != SOURCE_FILE_SET_SHA256:
        raise RuntimeError("reviewed gameplay source file-set hash changed")
    if review["classification"] != "gameplay" or review["declared_label"] != "nexto_1v1":
        raise RuntimeError("reviewed source is not the authoritative nexto_1v1 gameplay")
    session_dir = source_root / SESSION_UUID
    reader = SessionReader(session_dir)
    validation = reader.validate()
    if not validation.container_valid or validation.frame_count != SOURCE_FRAME_COUNT:
        raise RuntimeError(f"source container validation failed: {validation.as_dict()}")
    frames = list(SessionReader(session_dir).iter_frames())
    if [int(frame["sequence"]) for frame in frames] != list(range(SOURCE_FRAME_COUNT)):
        raise RuntimeError("source recorder sequence is not contiguous 0..58305")
    events_by_physics: dict[int, list[dict[str, Any]]] = {}
    for event in SessionReader(session_dir).iter_events():
        events_by_physics.setdefault(int(event.get("physics_frame", -1)), []).append(event)

    segments = identify_playable_segments(frames)
    playable = np.zeros(SOURCE_FRAME_COUNT, dtype=bool)
    for segment in segments:
        playable[segment.playable_start : segment.playable_end] = True
    # Keep excluded countdown/post-goal rows structurally neutral only so original
    # source indices remain stable.  They never enter loss, burn-in, or evaluation.
    observations = np.zeros((SOURCE_FRAME_COUNT, OBS_DIM), dtype=np.float32)
    actions = np.empty((SOURCE_FRAME_COUNT, 8), dtype=np.float32)
    previous: dict[str, Any] | None = None
    for index, frame in enumerate(frames):
        contiguous = bool(
            previous is not None
            and int(frame["physics_frame"]) == int(previous["physics_frame"]) + 1
        )
        if playable[index]:
            exact = adapt_frame(
                frame,
                session_uuid=SESSION_UUID,
                previous_frame=previous,
                events_at_physics_frame=events_by_physics.get(int(frame["physics_frame"]), ()),
                lifecycle_boundary_before=not contiguous,
            )
            observations[index] = direct_human_sequence_observation(frame, exact)
        actions[index] = action_target(frame)
        previous = frame
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise RuntimeError("materialized sequence contains nonfinite values")
    if np.count_nonzero(observations[:, list(ZEROED_OBSERVATION_INDICES)]):
        raise RuntimeError("shared observation projection exposed a required-zero field")
    source = {
        "session_uuid": SESSION_UUID,
        "declared_label": "nexto_1v1",
        "source_frame_count": SOURCE_FRAME_COUNT,
        "source_file_set_sha256": SOURCE_FILE_SET_SHA256,
        "dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": file_sha256(DATASET_MANIFEST),
        "review_record": REVIEW_RECORD.relative_to(ROOT).as_posix(),
        "review_record_sha256": file_sha256(REVIEW_RECORD),
        "container_valid": True,
        "mechanic_practice_frames_loaded": 0,
        "observation_adapter_v2_used": False,
        "source_frame_observation_sha256": array_sha256(observations),
        "source_frame_action_sha256": array_sha256(actions),
    }
    return observations, actions, segments, frames if retain_frames else None, source


def build_split_manifest(
    observations: np.ndarray,
    actions: np.ndarray,
    segments: list[PlayableSegment],
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[PlayableSegment]]]:
    split_segments = choose_whole_segment_splits(segments)
    splits: dict[str, Any] = {}
    memberships: set[int] = set()
    for name, rows in split_segments.items():
        indices = _split_indices(rows)
        overlap = memberships.intersection(indices.tolist())
        if overlap:
            raise RuntimeError(f"whole-episode split overlap in {name}")
        memberships.update(indices.tolist())
        splits[name] = {
            "frame_count": int(indices.size),
            "percentage_of_playable_frames": float(
                indices.size / sum(s.frame_count for s in segments)
            ),
            "segment_count": len(rows),
            "segment_identities": [row.identity for row in rows],
            "first_source_sequence": int(indices[0]),
            "last_source_sequence": int(indices[-1]),
            "observation_tensor_sha256": array_sha256(observations[indices]),
            "action_tensor_sha256": array_sha256(actions[indices]),
        }
    manifest = {
        "format": f"{FORMAT}_SOURCE_SPLIT_AUTHORITY",
        "created_utc": utc_now(),
        "source": source,
        "sequence_alignment_addendum_commit": ADDENDUM_COMMIT,
        "playable_boundary": {
            "rule": (
                "first post-countdown controllable tick; initial total game time begins "
                "advancing, subsequent rounds use round_active false-to-true"
            ),
            "frozen_countdown_used_for_loss": False,
            "frozen_countdown_used_for_burn_in": False,
            "hidden_state_zeroed_at_each_playable_start": True,
            "source_frames_excluded": SOURCE_FRAME_COUNT - sum(s.frame_count for s in segments),
        },
        "segments": [asdict(segment) for segment in segments],
        "split_policy": "chronological_whole_playable_segments_nearest_80_10_10",
        "splits": splits,
        "whole_segment_disjoint": True,
        "test_evaluations_before_selection": 0,
        "action_order": list(ACTION_NAMES),
    }
    return manifest, split_segments


def projected_parity(
    observations: np.ndarray,
    frames: list[dict[str, Any]],
    segments: list[PlayableSegment],
    *,
    collision_root: Path,
    device: str,
) -> dict[str, Any]:
    from benchmarks.diagnose_rival2_no_previous_action_observation_domains import (
        LAYOUTS,
        _layout_for_frame,
        construct_matched_native_observations,
    )

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for segment in segments:
        index = segment.playable_start
        layout = _layout_for_frame(frames[index])
        if layout is None or layout in seen:
            continue
        seen.add(layout)
        selected.append(
            {
                "layout": layout,
                "layout_name": LAYOUTS[layout][0],
                "phase": "aligned_playable_start",
                "frame_index": index,
                "sequence": int(frames[index]["sequence"]),
                "physics_frame": int(frames[index]["physics_frame"]),
            }
        )
    if len(selected) != 5:
        raise RuntimeError(f"playable segments did not cover all five kickoff layouts: {seen}")
    native, construction = construct_matched_native_observations(
        frames, selected, observations, collision_root, device
    )
    native = np.asarray(project_human_sequence_observation(native), dtype=np.float32)
    human = observations[[int(row["frame_index"]) for row in selected]]
    difference = np.abs(human - native)
    retained = difference[:, list(RETAINED_OBSERVATION_INDICES)]
    field_rows = []
    for field, index in zip(RETAINED_OBSERVATION_FIELDS, RETAINED_OBSERVATION_INDICES, strict=True):
        field_rows.append(
            {
                "field": field,
                "maximum_absolute": float(difference[:, index].max()),
                "rmse": float(np.sqrt(np.mean(np.square(difference[:, index])))),
            }
        )
    result = {
        "format": f"{FORMAT}_PROJECTED_MATCHED_STATE_PARITY",
        "created_utc": utc_now(),
        "seed": PARITY_SEED,
        "samples": selected,
        "native_construction": construction,
        "retained_field_count": len(RETAINED_OBSERVATION_INDICES),
        "zeroed_field_count": len(ZEROED_OBSERVATION_INDICES),
        "aggregate_retained_rmse": float(np.sqrt(np.mean(np.square(retained)))),
        "maximum_retained_absolute": float(retained.max()),
        "zeroed_human_nonzero_count": int(
            np.count_nonzero(human[:, list(ZEROED_OBSERVATION_INDICES)])
        ),
        "zeroed_native_nonzero_count": int(
            np.count_nonzero(native[:, list(ZEROED_OBSERVATION_INDICES)])
        ),
        "normal_float_tolerance": 2.0e-5,
        "passed": bool(float(retained.max()) <= 2.0e-5),
        "fields": field_rows,
    }
    if not result["passed"]:
        worst = sorted(field_rows, key=lambda row: row["maximum_absolute"], reverse=True)[:8]
        raise RuntimeError(f"projected matched-state parity failed: {worst}")
    return result


def fresh_model(device: str) -> Rival2RecurrentActorCritic:
    torch.manual_seed(INITIALIZATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    return Rival2RecurrentActorCritic(Rival2RecurrentPolicyConfig()).to(device)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    for commit in (ORIGINAL_PACKAGE_COMMIT, ADDENDUM_COMMIT):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT
        ).returncode:
            raise RuntimeError(f"HEAD does not contain required authority commit {commit}")
    if not args.collision_root.is_dir():
        raise FileNotFoundError(args.collision_root)
    observations, actions, segments, frames, source = materialize(
        args.human_source_root, retain_frames=True
    )
    assert frames is not None
    split_manifest, split_segments = build_split_manifest(observations, actions, segments, source)
    parity = projected_parity(
        observations,
        frames,
        segments,
        collision_root=args.collision_root,
        device=args.device,
    )
    write_json(SPLIT_MANIFEST, split_manifest)
    write_json(PARITY, parity)
    model = fresh_model("cpu")
    authority = {
        "format": f"{FORMAT}_PRE_OPTIMIZER_AUTHORITY",
        "created_utc": utc_now(),
        "implementation_commit": git("rev-parse", "HEAD").upper(),
        "original_package_commit": ORIGINAL_PACKAGE_COMMIT,
        "authoritative_addendum_commit": ADDENDUM_COMMIT,
        "observation_view": HUMAN_SEQUENCE_OBS_VIEW_CONTRACT,
        "observation_view_sha256": HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256,
        "policy_config": asdict(model.config),
        "policy_config_sha256": model.config.content_hash,
        "fresh_initialization": {
            "seed": INITIALIZATION_SEED,
            "model_tensor_sha256": tensor_tree_sha256(model.state_dict()),
            "prior_checkpoint_loaded": False,
        },
        "training": {
            "seed": TRAINING_SEED,
            "window_ticks": WINDOW_TICKS,
            "maximum_burn_in_ticks": BURN_IN_TICKS,
            "batch_windows": BATCH_WINDOWS,
            "optimizer": "fresh AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_steps": MAX_STEPS,
            "validation_interval": VALIDATION_INTERVAL,
            "minimum_plateau_step": MIN_PLATEAU_STEP,
            "plateau_checks": PLATEAU_CHECKS,
            "significant_improvement": SIGNIFICANT_IMPROVEMENT,
            "analog_loss": "MSE(tanh(actor_mean), exact target)",
            "button_loss": "BCEWithLogits(raw logits, exact 0/1 target)",
            "ranking_metric_only": "lowest_validation_complete_action_rmse",
            "validation_action_representation": "tanh(mean)+sigmoid(button_logits)",
            "critic_optimizer_steps": 0,
            "log_std_value": FROZEN_STAGE1_LOG_STD,
        },
        "split_identity": {
            "path": SPLIT_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(SPLIT_MANIFEST),
            "segment_counts": {key: len(value) for key, value in split_segments.items()},
        },
        "matched_state_parity": {
            "path": PARITY.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(PARITY),
            "aggregate_rmse": parity["aggregate_retained_rmse"],
            "maximum_absolute": parity["maximum_retained_absolute"],
            "passed": parity["passed"],
        },
        "scope": {
            "ppo": False,
            "reward_optimization": False,
            "mechanic_practice_data": False,
            "observation_adapter_v2": False,
            "previous_action_visible": False,
            "test_evaluation_before_selection": False,
        },
    }
    write_json(AUTHORITY, authority)
    return authority


@dataclass(frozen=True, slots=True)
class Window:
    segment_identity: str
    context_start: int
    loss_start: int
    end: int


def build_windows(segments: list[PlayableSegment]) -> list[Window]:
    windows: list[Window] = []
    for segment in segments:
        for loss_start in range(segment.playable_start, segment.playable_end, WINDOW_TICKS):
            windows.append(
                Window(
                    segment_identity=segment.identity,
                    context_start=max(segment.playable_start, loss_start - BURN_IN_TICKS),
                    loss_start=loss_start,
                    end=min(segment.playable_end, loss_start + WINDOW_TICKS),
                )
            )
    return windows


def _prediction(actor: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.tanh(actor[..., :5]), torch.sigmoid(actor[..., 10:13])), dim=-1)


@torch.inference_mode()
def evaluate_segments(
    model: Rival2RecurrentActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    segments: list[PlayableSegment],
) -> dict[str, Any]:
    squared = torch.zeros(8, dtype=torch.float64, device=observations.device)
    count = 0
    for segment in segments:
        actor, _value, _hidden = model(
            observations[segment.playable_start : segment.playable_end].unsqueeze(0)
        )
        prediction = _prediction(actor[0])
        target = actions[segment.playable_start : segment.playable_end]
        squared += (prediction - target).to(torch.float64).square().sum(dim=0)
        count += segment.frame_count
    per_channel = torch.sqrt(squared / count)
    return {
        "frame_count": count,
        "complete_action_rmse": float(torch.sqrt(squared.sum() / (count * 8)).item()),
        "per_channel_rmse": {
            name: float(per_channel[index].item()) for index, name in enumerate(ACTION_NAMES)
        },
    }


def _batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
    windows: list[Window],
    choices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [windows[int(index)] for index in choices]
    maximum = max(window.end - window.context_start for window in selected)
    batch_observation = torch.zeros(
        (len(selected), maximum, OBS_DIM), dtype=torch.float32, device=observations.device
    )
    batch_action = torch.zeros(
        (len(selected), maximum, 8), dtype=torch.float32, device=observations.device
    )
    loss_mask = torch.zeros((len(selected), maximum), dtype=torch.bool, device=observations.device)
    for row, window in enumerate(selected):
        length = window.end - window.context_start
        batch_observation[row, :length] = observations[window.context_start : window.end]
        batch_action[row, :length] = actions[window.context_start : window.end]
        loss_offset = window.loss_start - window.context_start
        loss_mask[row, loss_offset:length] = True
    return batch_observation, batch_action, loss_mask


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not AUTHORITY.is_file() or not SPLIT_MANIFEST.is_file() or not PARITY.is_file():
        raise RuntimeError("run --prepare and commit the authority before training")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if authority["split_identity"]["sha256"] != file_sha256(SPLIT_MANIFEST):
        raise RuntimeError("frozen split authority hash changed")
    if authority["matched_state_parity"]["sha256"] != file_sha256(PARITY):
        raise RuntimeError("frozen matched-state parity hash changed")
    observations_np, actions_np, segments, _frames, source = materialize(
        args.human_source_root, retain_frames=False
    )
    if (
        source["source_frame_observation_sha256"]
        != split_manifest["source"]["source_frame_observation_sha256"]
    ):
        raise RuntimeError("deterministic observation rematerialization changed")
    if (
        source["source_frame_action_sha256"]
        != split_manifest["source"]["source_frame_action_sha256"]
    ):
        raise RuntimeError("exact action rematerialization changed")
    split_segments = choose_whole_segment_splits(segments)
    for name, rows in split_segments.items():
        expected = split_manifest["splits"][name]["segment_identities"]
        if [row.identity for row in rows] != expected:
            raise RuntimeError(f"{name} whole-segment assignment changed")

    device = torch.device(args.device)
    observations = torch.from_numpy(observations_np).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    model = fresh_model(args.device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hash = tensor_tree_sha256(initial_state)
    if initial_hash != authority["fresh_initialization"]["model_tensor_sha256"]:
        raise RuntimeError("fresh recurrent initialization changed after authority freeze")
    critic_initial = {
        key: value.detach().cpu().clone() for key, value in model.critic.state_dict().items()
    }
    trainable = [
        *model.encoder.parameters(),
        *model.gru.parameters(),
        *model.post.parameters(),
        *model.actor.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(TRAINING_SEED)
    windows = build_windows(split_segments["train"])
    if not windows:
        raise RuntimeError("no aligned recurrent training windows")
    baseline_validation = evaluate_segments(
        model, observations, actions, split_segments["validation"]
    )
    best_rmse = math.inf
    best_step = 0
    checks_without_significant = 0
    last_significant = math.inf
    stop_reason = "maximum_10000_steps"
    executed_steps = 0
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CURVE.exists():
        CURVE.unlink()
    started = time.monotonic()

    for step in range(1, MAX_STEPS + 1):
        choices = generator.integers(0, len(windows), size=BATCH_WINDOWS)
        observation_batch, action_batch, loss_mask = _batch(observations, actions, windows, choices)
        actor, _value, _hidden = model(observation_batch)
        analog_prediction = torch.tanh(actor[..., :5])
        analog_loss = F.mse_loss(analog_prediction[loss_mask], action_batch[..., :5][loss_mask])
        button_loss = F.binary_cross_entropy_with_logits(
            actor[..., 10:13][loss_mask], action_batch[..., 5:8][loss_mask]
        )
        loss = analog_loss + button_loss
        if not torch.isfinite(loss):
            stop_reason = "nonfinite_supervised_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model.critic.weight.grad is not None or model.critic.bias.grad is not None:
            raise RuntimeError("critic received a Stage-1 gradient")
        if model.actor.weight.grad is None or model.actor.bias.grad is None:
            raise RuntimeError("actor gradient is missing")
        model.actor.weight.grad[5:10].zero_()
        model.actor.bias.grad[5:10].zero_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
        if not torch.isfinite(gradient_norm):
            stop_reason = "nonfinite_supervised_gradient"
            break
        optimizer.step()
        model.freeze_log_std_value(FROZEN_STAGE1_LOG_STD)
        executed_steps = step
        if step % VALIDATION_INTERVAL:
            continue

        validation = evaluate_segments(model, observations, actions, split_segments["validation"])
        rmse = float(validation["complete_action_rmse"])
        if not math.isfinite(rmse):
            stop_reason = "nonfinite_validation_rmse"
            break
        strict_improvement = rmse < best_rmse
        significant = rmse <= last_significant - SIGNIFICANT_IMPROVEMENT
        if significant:
            last_significant = rmse
            checks_without_significant = 0
        else:
            checks_without_significant += 1
        if strict_improvement:
            best_rmse = rmse
            best_step = step
            payload = {
                "format": CHECKPOINT_FORMAT,
                "created_utc": utc_now(),
                "lineage": {
                    "name": "Human Sequence Seed v1",
                    "fresh_random_initialization": True,
                    "initialization_seed": INITIALIZATION_SEED,
                    "initial_model_tensor_sha256": initial_hash,
                    "prior_checkpoint_loaded": False,
                },
                "model": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "policy_config": asdict(model.config),
                "policy_config_sha256": model.config.content_hash,
                "selected_step": step,
                "selection_metric_only": "lowest_validation_complete_action_rmse",
                "validation": validation,
                "authority": {
                    "path": AUTHORITY.relative_to(ROOT).as_posix(),
                    "sha256": file_sha256(AUTHORITY),
                    "split_sha256": file_sha256(SPLIT_MANIFEST),
                    "parity_sha256": file_sha256(PARITY),
                },
                "optimizer": {
                    "type": "fresh AdamW",
                    "state": optimizer.state_dict(),
                },
                "critic_optimizer_steps": 0,
                "log_std_rows_fixed": True,
                "ppo_resumable": False,
            }
            temporary = CHECKPOINT.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, CHECKPOINT)
        row = {
            "step": step,
            "created_utc": utc_now(),
            "train_analog_mse": float(analog_loss.detach().item()),
            "train_button_bce": float(button_loss.detach().item()),
            "train_total_loss": float(loss.detach().item()),
            "validation": validation,
            "best_validation_rmse": best_rmse,
            "best_step": best_step,
            "strict_improvement": strict_improvement,
            "significant_improvement": significant,
            "checks_without_significant_improvement": checks_without_significant,
            "gradient_norm": float(gradient_norm.detach().item()),
            "elapsed_seconds": time.monotonic() - started,
        }
        append_jsonl(CURVE, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if step >= MIN_PLATEAU_STEP and checks_without_significant >= PLATEAU_CHECKS:
            stop_reason = "validation_complete_action_rmse_plateau"
            break

    if best_step == 0 or not CHECKPOINT.is_file():
        raise RuntimeError(f"no finite validation checkpoint selected: {stop_reason}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    selected_validation = evaluate_segments(
        model, observations, actions, split_segments["validation"]
    )
    if not math.isclose(
        selected_validation["complete_action_rmse"], best_rmse, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("selected checkpoint does not reproduce best validation RMSE")
    for key, initial in critic_initial.items():
        if not torch.equal(model.critic.state_dict()[key].detach().cpu(), initial):
            raise RuntimeError("critic head changed despite zero Stage-1 optimizer steps")
    # The test is first opened only after the lowest-validation checkpoint is fixed.
    test = evaluate_segments(model, observations, actions, split_segments["test"])
    selected_record = {
        "format": f"{FORMAT}_STAGE1_SELECTED",
        "created_utc": utc_now(),
        "checkpoint": {
            "path": CHECKPOINT.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(CHECKPOINT),
            "bytes": CHECKPOINT.stat().st_size,
            "model_tensor_sha256": tensor_tree_sha256(checkpoint["model"]),
        },
        "steps_executed": executed_steps,
        "selected_step": best_step,
        "baseline_validation": baseline_validation,
        "selected_validation": selected_validation,
        "untouched_test": test,
        "stop_reason": stop_reason,
        "ranking_metric_only": "lowest_validation_complete_action_rmse",
        "test_evaluation_count": 1,
        "test_opened_after_selection": True,
        "critic_optimizer_steps": 0,
        "mechanic_practice_frames_loaded": 0,
        "ppo_steps": 0,
    }
    write_json(SELECTED, selected_record)
    write_json(
        TEST_METRICS,
        {
            "format": f"{FORMAT}_UNTOUCHED_HUMAN_TEST",
            "created_utc": utc_now(),
            "selected_before_test": True,
            "evaluation_count": 1,
            "selection_reopened": False,
            "metrics": test,
        },
    )
    return selected_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=Path(os.environ["APPDATA"]) / "bakkesmod/bakkesmod/data/rival2/human_demos",
    )
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare == args.train:
        raise SystemExit("choose exactly one of --prepare or --train")
    result = prepare(args) if args.prepare else train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
