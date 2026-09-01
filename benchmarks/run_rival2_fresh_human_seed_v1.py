"""Fresh-random Rival gameplay imitation for the Fresh Human Seed v1 lineage.

This runner deliberately has no previous-policy checkpoint argument.  It streams the
single reviewed gameplay session, applies the frozen Observation Adapter V2, freezes a
temporal 80/10/10 authority, and optimizes only eight-channel human action MSE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.human_demo.bc_observation_bridge import (  # noqa: E402
    BCBridgeTrajectoryAdapter,
    FieldQuality,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.human_demo.observation_adapter_v2 import (  # noqa: E402
    OBSERVATION_ADAPTER_CHECKPOINT_FORMAT,
    OBSERVATION_ADAPTER_VERSION,
    AdapterProfile,
    HumanDemoObservationAdapterV2,
    ObservationAdapterConfig,
    apply_native_pad_overlay,
    native_pad_overlay,
)
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_NAMES,
    OBS_DIM,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_policy import (  # noqa: E402
    PREVIOUS_ACTION_OBSERVATION_INDICES,
    Rival2ActorCritic,
    Rival2PolicyConfig,
)
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
)

PACKAGE_COMMIT = "84A05B1E050D746AFDD25BDBA7530B4583FE709D"
FORMAT = "RIVAL2_FRESH_HUMAN_SEED_V1"
CHECKPOINT_FORMAT = "RIVAL2_FRESH_HUMAN_SEED_V1_STAGE1_CHECKPOINT"
RESULTS = ROOT / "results/rival2/fresh_human_seed_v1"
AUTHORITY = RESULTS / "authority.json"
SPLIT_MANIFEST = RESULTS / "source_split_manifest.json"
CURVE = RESULTS / "stage1_curve.jsonl"
CHECKPOINT = ROOT / "checkpoints/rival2/fresh_human_seed_v1/rival2_fresh_human_seed_v1.pt"
DATASET_MANIFEST = ROOT / "results/rival2/human_demo_dataset_v1/dataset_manifest.json"
ADAPTER_CHECKPOINT = (
    ROOT / "checkpoints/rival2/observation_adapter_v2/rival2_human_demo_observation_adapter_v2.pt"
)
SESSION_UUID = "CD6E7DB1-2761-4B8B-BD37-F21C7F135722"
FRAME_COUNT = 58_306
TRAIN_END = int(FRAME_COUNT * 0.8)
VALIDATION_END = int(FRAME_COUNT * 0.9)
INITIALIZATION_SEED = 2026090102
TRAINING_SEED = 2026090103
BATCH_SIZE = 4096
VALIDATION_INTERVAL = 100
MAX_STEPS = 30_000
MIN_PLATEAU_STEP = 5_000
PLATEAU_CHECKS = 30
TARGET_PLATEAU_CHECKS = 10
SIGNIFICANT_IMPROVEMENT = 0.0001
TARGET_RMSE = 0.30
LEARNING_RATE = 3.0e-4
MINIMUM_LEARNING_RATE = 1.0e-5
ADAPTER_EXPECTED_SHA256 = "EDEDC9CCDE3269B393FB4C944F641CF4D34A78AB5944662F9019009BBA914C99"
AUTHORITY_PREPARATION_REQUIRES_EXACT_PACKAGE_COMMIT = True
NEUTRALIZE_PREVIOUS_ACTION = False
ZERO_PREVIOUS_ACTION_POLICY_INPUTS = False


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


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
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate a short-lived Windows scanner handle on a freshly written checkpoint."""

    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def _load_adapter(device: str) -> tuple[HumanDemoObservationAdapterV2, dict[str, Any]]:
    actual_sha = file_sha256(ADAPTER_CHECKPOINT)
    if actual_sha != ADAPTER_EXPECTED_SHA256:
        raise RuntimeError(f"Observation Adapter V2 SHA mismatch: {actual_sha}")
    payload = torch.load(ADAPTER_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("format") != OBSERVATION_ADAPTER_CHECKPOINT_FORMAT:
        raise RuntimeError("Observation Adapter V2 format mismatch")
    if payload.get("adapter_version") != OBSERVATION_ADAPTER_VERSION:
        raise RuntimeError("Observation Adapter V2 version mismatch")
    adapter = HumanDemoObservationAdapterV2(
        ObservationAdapterConfig(**payload["adapter_config"])
    ).to(device)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval().requires_grad_(False)
    return adapter, {
        "path": ADAPTER_CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": actual_sha,
        "version": OBSERVATION_ADAPTER_VERSION,
        "tensor_sha256": tensor_tree_sha256(adapter.state_dict()),
        "frozen": True,
    }


def neutralize_previous_action_before_adapter(
    degraded: torch.Tensor, quality: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neutralize the optional shortcut before Adapter V2 can inspect it."""

    indices = torch.as_tensor(
        PREVIOUS_ACTION_OBSERVATION_INDICES,
        dtype=torch.long,
        device=degraded.device,
    )
    degraded.index_fill_(1, indices, 0.0)
    quality.index_fill_(1, indices, int(FieldQuality.UNAVAILABLE))
    return degraded, quality


def hard_zero_previous_action_after_adapter(observation: torch.Tensor) -> torch.Tensor:
    """Enforce the final human-domain observation contract after all reconstruction."""

    indices = torch.as_tensor(
        PREVIOUS_ACTION_OBSERVATION_INDICES,
        dtype=torch.long,
        device=observation.device,
    )
    observation.index_fill_(1, indices, 0.0)
    return observation


@torch.no_grad()
def load_gameplay(
    source_root: Path, *, device: str, neutralize_previous_action: bool = False
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    gameplay = dataset["general_gameplay"]
    if gameplay["session_uuid"] != SESSION_UUID or gameplay["source_frame_count"] != FRAME_COUNT:
        raise RuntimeError("reviewed gameplay source identity changed")
    if any(row.get("declared_label") != "nexto_1v1" for row in gameplay["regions"]):
        raise RuntimeError("non-gameplay source entered the gameplay trajectory")
    if sum(int(row["source_frame_count"]) for row in gameplay["regions"]) != FRAME_COUNT:
        raise RuntimeError("reviewed gameplay regions no longer cover 58,306 frames")

    session_dir = source_root / SESSION_UUID
    reader = SessionReader(session_dir)
    report = reader.validate()
    if not report.container_valid or report.frame_count != FRAME_COUNT:
        raise RuntimeError(f"reviewed source container failed validation: {report.as_dict()}")
    adapter, adapter_identity = _load_adapter(device)
    trajectory = BCBridgeTrajectoryAdapter(session_dir)
    native_iterator = iter(SessionReader(session_dir).iter_frames())
    native_frame = next(native_iterator, None)
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    sequences: list[int] = []
    physics_frames: list[int] = []
    quality_counts = np.zeros(4, dtype=np.int64)
    previous_action_indices = torch.tensor(
        PREVIOUS_ACTION_OBSERVATION_INDICES, dtype=torch.long, device=device
    )
    neutralized_input_sample_count = 0
    neutralized_output_sample_count = 0
    buffer: list[tuple[Any, np.ndarray, np.ndarray]] = []

    def flush() -> None:
        nonlocal neutralized_input_sample_count, neutralized_output_sample_count
        if not buffer:
            return
        degraded = torch.from_numpy(
            np.stack([np.asarray(row[0].observation) for row in buffer])
        ).to(device)
        quality = torch.from_numpy(np.stack([np.asarray(row[0].quality) for row in buffer])).to(
            device
        )
        if neutralize_previous_action:
            degraded, quality = neutralize_previous_action_before_adapter(
                degraded, quality
            )
            neutralized_input_sample_count += int(degraded.shape[0])
        repaired = adapter(degraded, quality, profile=AdapterProfile.GAMEPLAY)
        pads = torch.from_numpy(np.stack([row[1] for row in buffer])).to(device)
        support = torch.from_numpy(np.stack([row[2] for row in buffer])).to(device)
        repaired = apply_native_pad_overlay(repaired, pads, support)
        if neutralize_previous_action:
            repaired = hard_zero_previous_action_after_adapter(repaired)
            if not torch.count_nonzero(
                repaired.index_select(1, previous_action_indices)
            ).eq(0):
                raise RuntimeError("previous-action output mask failed")
            neutralized_output_sample_count += int(repaired.shape[0])
        observations.append(repaired.cpu())
        actions.append(
            torch.from_numpy(np.stack([np.asarray(row[0].action) for row in buffer])).float()
        )
        buffer.clear()

    for identity, sample in trajectory.iter_spans((("full-gameplay", 0, FRAME_COUNT - 1),)):
        del identity
        if not sample.bc_usable or not sample.action_unchanged_from_exact_adapter:
            raise RuntimeError(f"gameplay sample {sample.sequence} is not BC-usable/exact-action")
        while native_frame is not None and int(native_frame["sequence"]) < sample.sequence:
            native_frame = next(native_iterator, None)
        if native_frame is None or int(native_frame["sequence"]) != sample.sequence:
            raise RuntimeError("native gameplay overlay sequence alignment failed")
        overlay = native_pad_overlay(native_frame)
        quality_counts += np.bincount(np.asarray(sample.quality), minlength=4)
        sequences.append(sample.sequence)
        physics_frames.append(sample.physics_frame)
        buffer.append(
            (
                sample,
                np.asarray(overlay.values).copy(),
                np.asarray(overlay.supported).copy(),
            )
        )
        if len(buffer) >= 8192:
            flush()
    flush()
    observation = torch.cat(observations)
    action = torch.cat(actions)
    if observation.shape != (FRAME_COUNT, OBS_DIM) or action.shape != (FRAME_COUNT, 8):
        raise RuntimeError("materialized gameplay tensor shape mismatch")
    if sequences != list(range(FRAME_COUNT)):
        raise RuntimeError("gameplay source is not the exact monotonically ordered trajectory")
    if not torch.isfinite(observation).all() or not torch.isfinite(action).all():
        raise RuntimeError("materialized gameplay contains nonfinite values")
    if not torch.all((action[:, :5] >= -1) & (action[:, :5] <= 1)):
        raise RuntimeError("analog action target outside native contract")
    if not torch.isin(action[:, 5:], torch.tensor([0.0, 1.0])).all():
        raise RuntimeError("button action target outside native Boolean contract")
    source_identity = {
        "dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": file_sha256(DATASET_MANIFEST),
        "session_uuid": SESSION_UUID,
        "session_file_set_sha256": gameplay["regions"][0]["source_file_set_sha256"],
        "frame_count": FRAME_COUNT,
        "first_sequence": sequences[0],
        "last_sequence": sequences[-1],
        "first_physics_frame": physics_frames[0],
        "last_physics_frame": physics_frames[-1],
        "container_valid": report.container_valid,
        "mechanic_practice_sessions_loaded": 0,
        "adapter": adapter_identity,
        "previous_action_input_contract": {
            "enabled": bool(neutralize_previous_action),
            "indices": list(PREVIOUS_ACTION_OBSERVATION_INDICES),
            "before_adapter": "zero_value_and_unavailable_quality",
            "after_adapter_and_pad_overlay": "hard_zero",
            "input_samples_verified": neutralized_input_sample_count,
            "output_samples_verified": neutralized_output_sample_count,
        },
        "quality_field_sample_counts": {
            "unavailable": int(quality_counts[int(FieldQuality.UNAVAILABLE)]),
            "approximate": int(quality_counts[int(FieldQuality.APPROXIMATE)]),
            "exactly_derived": int(quality_counts[int(FieldQuality.EXACT_DERIVED)]),
            "exact_direct": int(quality_counts[int(FieldQuality.EXACT_DIRECT)]),
        },
        "sequences": sequences,
        "physics_frames": physics_frames,
    }
    return observation, action, source_identity


def _slice_hash(tensor: torch.Tensor) -> str:
    return tensor_tree_sha256({"value": tensor.contiguous()})


def build_split_manifest(
    observation: torch.Tensor, action: torch.Tensor, source: dict[str, Any]
) -> dict[str, Any]:
    sequences = source.pop("sequences")
    physics_frames = source.pop("physics_frames")
    ranges = {
        "train": (0, TRAIN_END),
        "validation": (TRAIN_END, VALIDATION_END),
        "test": (VALIDATION_END, FRAME_COUNT),
    }
    splits: dict[str, Any] = {}
    for name, (start, end) in ranges.items():
        splits[name] = {
            "start_index_inclusive": start,
            "end_index_exclusive": end,
            "frame_count": end - start,
            "start_sequence": sequences[start],
            "end_sequence": sequences[end - 1],
            "start_physics_frame": physics_frames[start],
            "end_physics_frame": physics_frames[end - 1],
            "observation_tensor_sha256": _slice_hash(observation[start:end]),
            "action_tensor_sha256": _slice_hash(action[start:end]),
        }
    return {
        "format": f"{FORMAT}_SOURCE_SPLIT_MANIFEST",
        "created_utc": utc_now(),
        "policy": "chronological_first_80_next_10_final_10",
        "rounding": "train_end=floor(N*0.8); validation_end=floor(N*0.9)",
        "shuffle": {"train": True, "validation": False, "test": False},
        "source": source,
        "splits": splits,
        "disjoint": ranges["train"][1] == ranges["validation"][0]
        and ranges["validation"][1] == ranges["test"][0],
        "complete_coverage": sum(row[1] - row[0] for row in ranges.values()) == FRAME_COUNT,
        "action_order": list(ACTION_NAMES),
    }


def fresh_model(*, zero_previous_action_inputs: bool = False) -> Rival2ActorCritic:
    torch.manual_seed(INITIALIZATION_SEED)
    model = Rival2ActorCritic(
        Rival2PolicyConfig(
            zero_previous_action_inputs=zero_previous_action_inputs
        )
    )
    with torch.no_grad():
        model.actor.weight[5:10].zero_()
        model.actor.bias[5:10].fill_(-1.0)
    model.critic.requires_grad_(False)
    return model


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if AUTHORITY_PREPARATION_REQUIRES_EXACT_PACKAGE_COMMIT:
        if git("rev-parse", "HEAD").upper() != PACKAGE_COMMIT:
            raise RuntimeError("authority preparation must occur on exact package commit")
    elif git("merge-base", "--is-ancestor", PACKAGE_COMMIT, "HEAD") != "":
        raise RuntimeError("authorization parent is not an ancestor of current HEAD")
    observation, action, source = load_gameplay(
        args.human_source_root,
        device=args.device,
        neutralize_previous_action=NEUTRALIZE_PREVIOUS_ACTION,
    )
    split = build_split_manifest(observation, action, source)
    model = fresh_model(
        zero_previous_action_inputs=ZERO_PREVIOUS_ACTION_POLICY_INPUTS
    )
    authority = {
        "format": f"{FORMAT}_AUTHORITY",
        "created_utc": utc_now(),
        "package_commit": PACKAGE_COMMIT,
        "lineage": {
            "fresh_random_initialization": True,
            "prior_rival_checkpoint_loaded": False,
            "prior_rival_checkpoint_allowed": False,
            "initialization_seed": INITIALIZATION_SEED,
            "initial_model_tensor_sha256": tensor_tree_sha256(model.state_dict()),
        },
        "source_split_manifest": SPLIT_MANIFEST.relative_to(ROOT).as_posix(),
        "source_split_manifest_sha256_pending_commit": True,
        "stage1": {
            "trainable": ["shared_trunk", "actor_mean_rows_0_4", "actor_button_rows_10_12"],
            "frozen": ["critic_head", "actor_log_std_rows_5_9"],
            "log_std_rows": {"weight": 0.0, "bias": -1.0},
            "objective": "mean_squared_error_over_exact_8_channel_action",
            "optimizer": {
                "type": "AdamW",
                "learning_rate": LEARNING_RATE,
                "minimum_learning_rate": MINIMUM_LEARNING_RATE,
                "weight_decay": 1.0e-5,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "gradient_clip": 1.0,
                "batch_size": BATCH_SIZE,
            },
            "selection_metric": "lowest_complete_action_validation_rmse_only",
            "validation_interval_steps": VALIDATION_INTERVAL,
            "maximum_steps": MAX_STEPS,
            "minimum_plateau_step": MIN_PLATEAU_STEP,
            "plateau_validation_checks": PLATEAU_CHECKS,
            "target_rmse": TARGET_RMSE,
            "target_plateau_validation_checks": TARGET_PLATEAU_CHECKS,
            "significant_improvement": SIGNIFICANT_IMPROVEMENT,
            "training_seed": TRAINING_SEED,
        },
        "stage2": {
            "bootstrap_fields": ["model", "policy_config"],
            "critic_reinitialized": True,
            "fresh_optimizer_rng_counters": True,
            "ppo_version": RIVAL2_PPO_120HZ_V1,
            "ppo_contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
            "reward": contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
            "opponents": {"current": 1.0, "historical": 0.0, "nexto": 0.0, "wisp": 0.0},
            "accepted_updates": 600,
            "snapshots": [*range(30, 481, 30), 500, 510, 540, 570, 600],
        },
        "forbidden": {
            "mechanic_practice_data": True,
            "mechanic_labels_in_objective_sampling_selection_stopping": True,
            "old_policy_teacher_or_retention": True,
            "previous_rival_checkpoint_load": True,
            "nexto_training": True,
        },
    }
    write_json(SPLIT_MANIFEST, split)
    authority["source_split_manifest_sha256"] = file_sha256(SPLIT_MANIFEST)
    authority.pop("source_split_manifest_sha256_pending_commit")
    write_json(AUTHORITY, authority)
    return authority


@torch.no_grad()
def evaluate_rmse(
    model: Rival2ActorCritic, observation: torch.Tensor, action: torch.Tensor, device: str
) -> float:
    squared_sum = 0.0
    count = 0
    model.eval()
    for start in range(0, observation.shape[0], 8192):
        obs = observation[start : start + 8192].to(device)
        target = action[start : start + 8192].to(device)
        actor, _ = model(obs)
        prediction = torch.cat((torch.tanh(actor[:, :5]), torch.sigmoid(actor[:, 10:13])), dim=-1)
        squared_sum += float((prediction - target).double().square().sum().item())
        count += target.numel()
    return math.sqrt(squared_sum / count)


def _verify_materialization(
    observation: torch.Tensor, action: torch.Tensor, manifest: dict[str, Any]
) -> None:
    for name, (start, end) in {
        "train": (0, TRAIN_END),
        "validation": (TRAIN_END, VALIDATION_END),
        "test": (VALIDATION_END, FRAME_COUNT),
    }.items():
        row = manifest["splits"][name]
        checks = (
            row["frame_count"] == end - start,
            row["observation_tensor_sha256"] == _slice_hash(observation[start:end]),
            row["action_tensor_sha256"] == _slice_hash(action[start:end]),
        )
        if not all(checks):
            raise RuntimeError(f"{name} materialization does not match frozen split authority")


def train(args: argparse.Namespace) -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if authority.get("format") != f"{FORMAT}_AUTHORITY":
        raise RuntimeError("fresh-human authority format mismatch")
    if authority["source_split_manifest_sha256"] != file_sha256(SPLIT_MANIFEST):
        raise RuntimeError("frozen source/split manifest hash mismatch")
    if not git("merge-base", "--is-ancestor", PACKAGE_COMMIT, "HEAD") == "":
        raise RuntimeError("package commit is not an ancestor of current HEAD")
    observation, action, source = load_gameplay(
        args.human_source_root,
        device=args.device,
        neutralize_previous_action=NEUTRALIZE_PREVIOUS_ACTION,
    )
    del source
    _verify_materialization(observation, action, manifest)

    model = fresh_model(
        zero_previous_action_inputs=ZERO_PREVIOUS_ACTION_POLICY_INPUTS
    )
    initial_hash = tensor_tree_sha256(model.state_dict())
    if initial_hash != authority["lineage"]["initial_model_tensor_sha256"]:
        raise RuntimeError("fresh deterministic initialization identity mismatch")
    model = model.to(args.device)
    model.train()

    actor_row_mask = torch.ones_like(model.actor.weight)
    actor_row_mask[5:10].zero_()
    actor_bias_mask = torch.ones_like(model.actor.bias)
    actor_bias_mask[5:10].zero_()
    model.actor.weight.register_hook(lambda gradient: gradient * actor_row_mask)
    model.actor.bias.register_hook(lambda gradient: gradient * actor_bias_mask)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    if optimizer.state:
        raise RuntimeError("fresh Stage-1 optimizer unexpectedly has state")

    train_observation = observation[:TRAIN_END]
    train_action = action[:TRAIN_END]
    validation_observation = observation[TRAIN_END:VALIDATION_END]
    validation_action = action[TRAIN_END:VALIDATION_END]
    test_observation = observation[VALIDATION_END:]
    test_action = action[VALIDATION_END:]
    generator = torch.Generator(device="cpu").manual_seed(TRAINING_SEED)
    permutation = torch.randperm(TRAIN_END, generator=generator)
    position = 0
    baseline_validation_rmse = evaluate_rmse(
        model, validation_observation, validation_action, args.device
    )
    best_rmse = math.inf
    best_step = 0
    last_significant_rmse = baseline_validation_rmse
    checks_without_significant_improvement = 0
    target_reached = baseline_validation_rmse <= TARGET_RMSE
    executed_steps = 0
    stop_reason = "maximum_30000_steps_reached"
    started = time.monotonic()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CURVE.exists():
        raise RuntimeError("Stage-1 curve already exists; refusing ambiguous rerun")

    for step in range(1, MAX_STEPS + 1):
        if position + BATCH_SIZE > TRAIN_END:
            permutation = torch.randperm(TRAIN_END, generator=generator)
            position = 0
        indices = permutation[position : position + BATCH_SIZE]
        position += BATCH_SIZE
        obs = train_observation.index_select(0, indices).to(args.device)
        target = train_action.index_select(0, indices).to(args.device)
        actor, _ = model(obs)
        prediction = torch.cat((torch.tanh(actor[:, :5]), torch.sigmoid(actor[:, 10:13])), dim=-1)
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            stop_reason = "nonfinite_supervised_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        if not torch.isfinite(gradient_norm):
            stop_reason = "nonfinite_supervised_gradient"
            break
        optimizer.step()
        with torch.no_grad():
            model.actor.weight[5:10].zero_()
            model.actor.bias[5:10].fill_(-1.0)
        executed_steps = step

        if step % VALIDATION_INTERVAL:
            continue
        validation_rmse = evaluate_rmse(
            model, validation_observation, validation_action, args.device
        )
        if not math.isfinite(validation_rmse):
            stop_reason = "nonfinite_validation_rmse"
            break
        strict_improvement = validation_rmse < best_rmse
        significant_improvement = validation_rmse <= (
            last_significant_rmse - SIGNIFICANT_IMPROVEMENT
        )
        if significant_improvement:
            last_significant_rmse = validation_rmse
            checks_without_significant_improvement = 0
        else:
            checks_without_significant_improvement += 1
        if strict_improvement:
            best_rmse = validation_rmse
            best_step = step
            payload = {
                "format": CHECKPOINT_FORMAT,
                "created_utc": utc_now(),
                "lineage": {
                    "fresh_random_initialization": True,
                    "initialization_seed": INITIALIZATION_SEED,
                    "initial_model_tensor_sha256": initial_hash,
                    "prior_rival_checkpoint_loaded": False,
                    "previous_rival_lineage": None,
                },
                "model": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "policy_config": asdict(model.config),
                "selected_step": step,
                "selection_metric": "complete_action_validation_rmse",
                "validation_rmse": validation_rmse,
                "authority": {
                    "path": AUTHORITY.relative_to(ROOT).as_posix(),
                    "sha256": file_sha256(AUTHORITY),
                    "source_split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
                },
                "stage1_optimizer": {
                    "type": "AdamW",
                    "fresh": True,
                    "state": optimizer.state_dict(),
                },
                "critic_trained": False,
                "log_std_rows_frozen_weight_zero_bias_minus_one": True,
                "ppo_resumable": False,
            }
            temporary = CHECKPOINT.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            replace_with_retry(temporary, CHECKPOINT)
        if validation_rmse <= TARGET_RMSE:
            target_reached = True
        row = {
            "step": step,
            "created_utc": utc_now(),
            "train_batch_mse": float(loss.detach().item()),
            "validation_complete_action_rmse": validation_rmse,
            "best_validation_rmse": best_rmse,
            "best_step": best_step,
            "strict_improvement": strict_improvement,
            "significant_improvement": significant_improvement,
            "checks_without_significant_improvement": checks_without_significant_improvement,
            "target_reached": target_reached,
            "gradient_norm": float(gradient_norm.detach().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        append_jsonl(CURVE, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if step >= MIN_PLATEAU_STEP:
            patience = TARGET_PLATEAU_CHECKS if target_reached else PLATEAU_CHECKS
            if checks_without_significant_improvement >= patience:
                stop_reason = (
                    "validation_plateau_after_target"
                    if target_reached
                    else "validation_plateau_after_minimum_5000_steps"
                )
                break

    if best_step == 0 or not CHECKPOINT.is_file():
        raise RuntimeError(f"no finite validation checkpoint was selected: {stop_reason}")
    selected = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    selected_validation_rmse = evaluate_rmse(
        model, validation_observation, validation_action, args.device
    )
    if selected_validation_rmse != best_rmse:
        raise RuntimeError("selected checkpoint no longer reproduces lowest validation RMSE")
    test_rmse = evaluate_rmse(model, test_observation, test_action, args.device)
    selected_record = {
        "format": f"{FORMAT}_STAGE1_SELECTED",
        "created_utc": utc_now(),
        "checkpoint": {
            "path": CHECKPOINT.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(CHECKPOINT),
            "bytes": CHECKPOINT.stat().st_size,
            "model_tensor_sha256": tensor_tree_sha256(selected["model"]),
        },
        "steps_executed": executed_steps,
        "selected_step": best_step,
        "baseline_validation_rmse": baseline_validation_rmse,
        "best_validation_rmse": best_rmse,
        "untouched_test_rmse": test_rmse,
        "target_0_30_reached": best_rmse <= TARGET_RMSE,
        "stop_reason": stop_reason,
        "selection_used_test": False,
        "ranking_metric_only": "complete_action_validation_rmse",
        "test_evaluation_count": 1,
        "critic_head_gradient_steps": 0,
        "mechanic_frames_loaded": 0,
    }
    write_json(RESULTS / "stage1_selected.json", selected_record)
    write_json(
        RESULTS / "stage1_test_metrics.json",
        {
            "format": f"{FORMAT}_UNTOUCHED_HUMAN_TEST",
            "created_utc": utc_now(),
            "evaluation_count": 1,
            "selected_before_test": True,
            "test_frame_count": FRAME_COUNT - VALIDATION_END,
            "complete_action_rmse": test_rmse,
            "selection_reopened": False,
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
