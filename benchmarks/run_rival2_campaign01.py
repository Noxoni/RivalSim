"""Execute the frozen, bounded Rival 2.0 Campaign 01 training authority."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pynvml
import torch
import warp as wp

from benchmarks.run_v02_benchmark import TelemetrySampler
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ANALOG_ACTION_NAMES,
    BUTTON_ACTION_NAMES,
    CONTRACT_HASHES,
    OBS_DIM,
    RIVAL2_REWARD_VERSION,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

EXPECTED_HEAD = "4235963a0648d7148b93f073311bb3343dd68ac4"
EXPECTED_RELEASE = "cc3aa34e0bac4531c2750e0d05e2b4980621c642"
EXPECTED_IMPLEMENTATION = "676ef6bd3ca48376d706a2dbccbdec26fce3e4fb"
EXPECTED_CONTRACT_HASHES = {
    "RIVAL2_OBS_V1": "10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF",
    "RIVAL2_ACTION_V1": "145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B",
    "RIVAL2_REWARD_V1": "E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F",
    "RIVAL2_EPISODE_V1": "E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E",
}
EXPECTED_POLICY_CONFIG_HASH = (
    "58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136"
)
CAMPAIGN_SEED = 20260826
EVALUATION_SEED = 920260826
CAPACITY_ORDER = (131072, 65536, 32768)
VRAM_MARGIN_REQUIRED_BYTES = 4 * 1024**3
TARGET_SAMPLES = 100_000_000
THRESHOLDS = (
    ("000m", 0),
    ("010m", 10_000_000),
    ("025m", 25_000_000),
    ("050m", 50_000_000),
    ("100m", 100_000_000),
)
EVALUATION_WORLDS = 4096
EVALUATION_MAX_DECISIONS = 45 * 30
EVALUATION_MODES = (
    "stochastic_self_play",
    "deterministic_vs_initialization",
    "stochastic_vs_initialization",
)
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--phase", choices=("all", "preflight-candidate", "train"), default="all"
    )
    parser.add_argument("--worlds", type=int)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def _prior_results_manifest() -> dict[str, Any]:
    paths: list[str] = []
    for version in ("v0.1", "v0.2", "v0.2.1", "v0.2.2", "v0.3", "v0.4", "v0.5"):
        output = _git("ls-files", f"results/{version}")
        paths.extend(line for line in output.splitlines() if line)
    entries = [
        {
            "path": path.replace("\\", "/"),
            "size_bytes": Path(path).stat().st_size,
            "sha256": _sha256_file(Path(path)),
        }
        for path in sorted(set(paths))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "tracked results/v0.1 through results/v0.5",
        "files": entries,
        "manifest_sha256": _sha256_json(entries),
    }


def freeze_prior_results_manifest(work_dir: Path) -> None:
    path = work_dir / "prior_results_baseline.json"
    current = _prior_results_manifest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_comparable = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        if existing_comparable != current:
            raise RuntimeError("published v0.1-v0.5 results changed after Campaign 01 freeze")
    else:
        current["created_utc"] = _utc_now()
        _write_json(path, current)


def threshold_label_for_samples(samples: int) -> str | None:
    """Return the highest notional Campaign 01 threshold crossed by samples."""

    crossed = [label for label, threshold in THRESHOLDS if samples >= threshold]
    return crossed[-1] if crossed else None


def first_update_at_or_above(worlds: int, threshold: int) -> tuple[int, int]:
    samples_per_update = worlds * 2 * Rival2PPOConfig().rollout_horizon
    updates = (threshold + samples_per_update - 1) // samples_per_update
    return updates, updates * samples_per_update


def frozen_configuration() -> dict[str, Any]:
    policy = Rival2PolicyConfig()
    ppo = Rival2PPOConfig()
    self_play = Rival2SelfPlayConfig()
    evaluation = {
        "seed": EVALUATION_SEED,
        "worlds": EVALUATION_WORLDS,
        "maximum_decisions": EVALUATION_MAX_DECISIONS,
        "physics_ticks_per_decision": 4,
        "policy_hz": 30,
        "episode_scope": "first completed episode in every held-out world",
        "kickoff_layout_assignment": "(world_index + evaluation_seed) modulo 5",
        "current_side_assignment": "blue on even world index, orange on odd world index",
        "modes": list(EVALUATION_MODES),
        "sampling_seed_rule": "evaluation_seed + fixed mode ordinal; reset per checkpoint",
        "opponent": "frozen fresh initialization from the same campaign",
        "metrics": [
            "goal_terminated_fraction",
            "no_touch_truncated_fraction",
            "hard_truncated_fraction",
            "touches_per_simulated_minute",
            "goals_per_simulated_minute",
            "demolitions_per_simulated_minute",
            "mean_episode_duration_seconds",
            "mean_absolute_analog_action_by_channel",
            "button_activation_rate_by_channel",
            "mean_analog_policy_std_by_channel",
            "mean_button_probability_by_channel",
            "mean_button_entropy_by_channel",
            "score_goal_touch_differential_and_episode_outcomes_for_vs_initialization",
        ],
    }
    evaluation["protocol_sha256"] = _sha256_json(evaluation)
    threshold_schedule = {
        str(worlds): {
            label: {
                "first_update": first_update_at_or_above(worlds, threshold)[0],
                "actual_samples": first_update_at_or_above(worlds, threshold)[1],
            }
            for label, threshold in THRESHOLDS[1:]
        }
        for worlds in CAPACITY_ORDER
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "Rival 2.0 Campaign 01",
        "authorized_head": EXPECTED_HEAD,
        "parent_release": EXPECTED_RELEASE,
        "implementation_basis": EXPECTED_IMPLEMENTATION,
        "campaign_seed": CAMPAIGN_SEED,
        "target_agent_decision_samples": TARGET_SAMPLES,
        "stop_rule": "first completed PPO update at or above target, then fixed closeout only",
        "capacity_order": list(CAPACITY_ORDER),
        "minimum_vram_margin_bytes": VRAM_MARGIN_REQUIRED_BYTES,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "self_play_config": asdict(self_play),
        "contract_hashes": dict(CONTRACT_HASHES),
        "thresholds": {label: threshold for label, threshold in THRESHOLDS},
        "threshold_schedule_by_capacity": threshold_schedule,
        "evaluation": evaluation,
        "hard_boundaries": [
            "no contract, reward, curriculum, timeout, action, observation, or "
            "architecture changes",
            "no hyperparameter search or scripted teacher",
            "no legacy Rival/Wisp",
            "no RocketSim/RLBot deployment",
            "no v0.6",
        ],
    }


def verify_authority(configuration: dict[str, Any], *, require_head: bool = True) -> None:
    if CONTRACT_HASHES != EXPECTED_CONTRACT_HASHES:
        raise RuntimeError(f"contract hash mismatch: {CONTRACT_HASHES}")
    if Rival2PolicyConfig().content_hash != EXPECTED_POLICY_CONFIG_HASH:
        raise RuntimeError("policy configuration hash mismatch")
    if asdict(Rival2PPOConfig()) != configuration["ppo_config"]:
        raise RuntimeError("PPO defaults differ from the prospectively frozen configuration")
    if require_head and _git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("Campaign 01 must start from the authorized HEAD")
    for required in (EXPECTED_RELEASE, EXPECTED_IMPLEMENTATION):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", required, "HEAD"],
            check=True,
            capture_output=True,
        )


def _initialize_runtime(device: str) -> None:
    wp.init()
    pynvml.nvmlInit()
    torch.cuda.set_device(torch.device(device))


def _nvml_memory(device: str) -> tuple[int, int]:
    index = torch.device(device).index or 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int(info.total), int(info.used)


def _runtime_identity(device: str) -> dict[str, Any]:
    total, used = _nvml_memory(device)
    return {
        "created_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "warp": wp.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_total_memory_bytes": total,
        "cuda_used_memory_bytes_at_identity": used,
        "nvidia_driver": pynvml.nvmlSystemGetDriverVersion(),
    }


def _all_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not _all_finite(value):
                return False
    return True


def _parameter_integrity(model: Rival2ActorCritic) -> dict[str, Any]:
    parameters = list(model.parameters())
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return {
        "parameters_finite": all(_all_finite(parameter) for parameter in parameters),
        "gradients_present": len(gradients),
        "gradients_finite": bool(gradients)
        and all(_all_finite(gradient) for gradient in gradients),
    }


def _rollout_integrity(
    trainer: Rival2Trainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    *,
    policy_version_before: int,
    samples_before: int,
) -> dict[str, Any]:
    finite_fields = {
        name: _all_finite(getattr(rollout, name))
        for name in (
            "observations",
            "actions",
            "pre_tanh",
            "old_log_probability",
            "values",
            "rewards",
            "next_values",
            "advantages",
            "returns",
        )
    }
    analog = rollout.actions[..., :5]
    buttons = rollout.actions[..., 5:]
    terminated = rollout.terminated
    truncated = rollout.truncated
    active_versions = set(trainer.opponent_pool.versions)
    observed_versions = set(
        int(value) for value in rollout.opponent_version.unique().cpu().tolist()
    )
    allowed_versions = active_versions | {policy_version_before, -1}
    historical = []
    for version, policy in zip(
        trainer.opponent_pool.versions, trainer.opponent_pool.policies, strict=True
    ):
        historical.append(
            {
                "version": version,
                "requires_grad": any(parameter.requires_grad for parameter in policy.parameters()),
                "has_gradient": any(
                    parameter.grad is not None for parameter in policy.parameters()
                ),
                "parameters_finite": all(
                    _all_finite(parameter) for parameter in policy.parameters()
                ),
                "state_sha256": _state_dict_sha256(policy.state_dict()),
            }
        )
    parameter = _parameter_integrity(trainer.model)
    checks = {
        "finite_rollout_fields": all(finite_fields.values()),
        "finite_metrics": all(_all_finite(value) for value in metrics.values()),
        "analog_bounds": bool(((analog >= -1.0) & (analog <= 1.0)).all().item()),
        "buttons_exact_binary": bool(((buttons == 0.0) | (buttons == 1.0)).all().item()),
        "termination_team_consistent": bool(
            torch.equal(terminated[..., 0], terminated[..., 1])
        ),
        "truncation_team_consistent": bool(torch.equal(truncated[..., 0], truncated[..., 1])),
        "termination_and_truncation_exclusive": bool((~(terminated & truncated)).all().item()),
        "post_reset_observation_finite": _all_finite(trainer.env.observation),
        "policy_version_increment_exact": trainer.policy_version == policy_version_before + 1,
        "iteration_matches_policy_version": trainer.iteration == trainer.policy_version,
        "sample_increment_exact": trainer.total_agent_samples - samples_before
        == rollout.horizon * rollout.num_envs * 2,
        "behavior_policy_version_exact": bool(
            (rollout.policy_version[rollout.train_mask] == policy_version_before).all().item()
        ),
        "opponent_versions_eligible": observed_versions <= allowed_versions,
        "blue_always_trainable": bool(rollout.train_mask[..., 0].all().item()),
        "historical_frozen_gradient_free": all(
            not item["requires_grad"] and not item["has_gradient"] for item in historical
        ),
        "historical_parameters_finite": all(item["parameters_finite"] for item in historical),
        "parameters_finite": parameter["parameters_finite"],
        "gradients_finite": parameter["gradients_finite"],
        "optimizer_finite": _optimizer_finite(trainer.optimizer),
    }
    return {
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "checks": checks,
        "finite_fields": finite_fields,
        "metrics": {name: float(value.item()) for name, value in metrics.items()},
        "observed_opponent_versions": sorted(observed_versions),
        "historical_policies": historical,
        "trainable_samples": int(rollout.train_mask.sum().item()),
        "total_samples": rollout.horizon * rollout.num_envs * 2,
        "sample_age_policy_versions": 0,
    }


def run_preflight_candidate(args: argparse.Namespace, configuration: dict[str, Any]) -> int:
    if args.worlds not in CAPACITY_ORDER:
        raise ValueError("preflight candidate must be in the authorized capacity order")
    output = args.work_dir / "preflight" / f"candidate_{args.worlds}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "worlds": args.worlds,
        "horizon": Rival2PPOConfig().rollout_horizon,
        "started_utc": _utc_now(),
        "status": "FAIL",
    }
    try:
        _initialize_runtime(args.device)
        verify_authority(configuration)
        geometry = ArenaGeometry.load_soccar(args.collision_dir)
        meshes = WarpArenaMeshes(geometry, args.device)
        torch.manual_seed(CAMPAIGN_SEED ^ args.worlds)
        torch.cuda.manual_seed(CAMPAIGN_SEED ^ args.worlds)
        env = Rival2Env(
            args.worlds,
            args.collision_dir,
            device=args.device,
            seed=CAMPAIGN_SEED ^ args.worlds,
            geometry=geometry,
            meshes=meshes,
            kickoff_selector=np.arange(args.worlds, dtype=np.int32) % 5,
        )
        trainer = Rival2Trainer(env, seed=CAMPAIGN_SEED ^ args.worlds)
        torch.cuda.reset_peak_memory_stats(args.device)
        env.reset_transfer_counters()
        telemetry = TelemetrySampler(interval_s=0.005)
        telemetry.start()
        update_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started
        telemetry_result = telemetry.stop()
        integrity = _rollout_integrity(
            trainer, rollout, metrics, policy_version_before=0, samples_before=0
        )
        transfer = env.hot_path_transfer_bytes()
        checkpoint = args.work_dir / "preflight" / f"candidate_{args.worlds}.pt"
        trainer.save_checkpoint(checkpoint)
        with torch.no_grad():
            action, value = trainer.deterministic_action_value(
                env.observation[:16].reshape(-1, OBS_DIM)
            )
        checkpoint_safe = checkpoint.is_file() and _all_finite(action) and _all_finite(value)
        checkpoint_identity = {
            "sha256": _sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "load_and_evaluation_allocation_safe": checkpoint_safe,
        }
        checkpoint.unlink()
        total_memory, current_used = _nvml_memory(args.device)
        peak_observed = max(int(telemetry_result.vram_max_bytes or 0), current_used)
        margin = total_memory - peak_observed
        checks = {
            "update_completed": True,
            "integrity_pass": integrity["verdict"] == "PASS_GREEN",
            "zero_hot_h2d": transfer["h2d"] == 0,
            "zero_hot_d2h": transfer["d2h"] == 0,
            "checkpoint_and_evaluation_safe": checkpoint_safe,
            "vram_margin_adequate": margin >= VRAM_MARGIN_REQUIRED_BYTES,
        }
        result.update(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "integrity": integrity,
                "hot_path_transfer_bytes": transfer,
                "update_seconds": update_seconds,
                "agent_decision_samples": trainer.total_agent_samples,
                "agent_decisions_per_second": trainer.total_agent_samples / update_seconds,
                "rollout_logical_bytes": rollout.logical_bytes,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                "vram_total_bytes": total_memory,
                "vram_peak_observed_bytes": peak_observed,
                "vram_margin_bytes": margin,
                "vram_margin_required_bytes": VRAM_MARGIN_REQUIRED_BYTES,
                "checkpoint_probe": checkpoint_identity,
                "runtime": _runtime_identity(args.device),
            }
        )
    except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        result.update(
            {
                "status": "FAIL",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
            }
        )
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["finished_utc"] = _utc_now()
        _write_json(output, result)
    return 0 if result["status"] == "PASS" else 2


def run_preflight_subprocesses(args: argparse.Namespace) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    for worlds in CAPACITY_ORDER:
        command = [
            sys.executable,
            "-m",
            "benchmarks.run_rival2_campaign01",
            "--collision-dir",
            str(Path(args.collision_dir).resolve()),
            "--work-dir",
            str(args.work_dir.resolve()),
            "--device",
            args.device,
            "--phase",
            "preflight-candidate",
            "--worlds",
            str(worlds),
        ]
        completed = subprocess.run(command, check=False)
        path = args.work_dir / "preflight" / f"candidate_{worlds}.json"
        if not path.is_file():
            attempt = {
                "worlds": worlds,
                "status": "FAIL",
                "failure": "candidate process did not publish evidence",
                "process_exit_code": completed.returncode,
            }
        else:
            attempt = json.loads(path.read_text(encoding="utf-8"))
            attempt["process_exit_code"] = completed.returncode
        attempts.append(attempt)
        if attempt.get("status") == "PASS" and completed.returncode == 0:
            selected = worlds
            break
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "ordered_capacity_candidates": list(CAPACITY_ORDER),
        "attempts": attempts,
        "selected_worlds": selected,
        "selection_reason": (
            "first authorized horizon-32 capacity with a finite real PPO update, zero hot-path "
            "state transfers, successful checkpoint/evaluation probe, and at least 4 GiB "
            "VRAM margin"
            if selected is not None
            else "no authorized capacity passed every resource and integrity condition"
        ),
        "verdict": "PASS_GREEN" if selected is not None else "STOP_RESOURCE",
    }
    _write_json(args.work_dir / "preflight.json", result)
    return result


@torch.no_grad()
def _evaluate_mode(
    *,
    mode: str,
    checkpoint_model: Rival2ActorCritic,
    initialization_model: Rival2ActorCritic,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    reward_version: str = RIVAL2_REWARD_VERSION,
) -> dict[str, Any]:
    mode_ordinal = EVALUATION_MODES.index(mode)
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED
    ) % 5
    env = Rival2Env(
        EVALUATION_WORLDS,
        collision_dir,
        device=device,
        seed=EVALUATION_SEED + mode_ordinal,
        reward_version=reward_version,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    checkpoint_model.eval()
    initialization_model.eval()
    generator = torch.Generator(device=device).manual_seed(EVALUATION_SEED + mode_ordinal)
    current_is_blue = torch.arange(EVALUATION_WORLDS, device=device) % 2 == 0
    current_agent_mask = torch.stack((current_is_blue, ~current_is_blue), dim=1)
    active = torch.ones(EVALUATION_WORLDS, dtype=torch.bool, device=device)
    completed = torch.zeros(EVALUATION_WORLDS, dtype=torch.bool, device=device)
    episode_steps = torch.zeros(EVALUATION_WORLDS, dtype=torch.int32, device=device)
    terminated_count = torch.zeros((), dtype=torch.float64, device=device)
    no_touch_count = torch.zeros((), dtype=torch.float64, device=device)
    hard_count = torch.zeros((), dtype=torch.float64, device=device)
    touch_count = torch.zeros((), dtype=torch.float64, device=device)
    demolition_count = torch.zeros((), dtype=torch.float64, device=device)
    world_decisions = torch.zeros((), dtype=torch.float64, device=device)
    duration_sum = torch.zeros((), dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    analog_abs_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    policy_count = torch.zeros((), dtype=torch.float64, device=device)
    std_sum = torch.zeros(5, dtype=torch.float64, device=device)
    probability_sum = torch.zeros(3, dtype=torch.float64, device=device)
    button_entropy_sum = torch.zeros(3, dtype=torch.float64, device=device)
    current_goals = torch.zeros((), dtype=torch.float64, device=device)
    initialization_goals = torch.zeros((), dtype=torch.float64, device=device)
    current_touches = torch.zeros((), dtype=torch.float64, device=device)
    initialization_touches = torch.zeros((), dtype=torch.float64, device=device)
    current_wins = torch.zeros((), dtype=torch.float64, device=device)
    initialization_wins = torch.zeros((), dtype=torch.float64, device=device)
    draws = torch.zeros((), dtype=torch.float64, device=device)
    integrity = {
        "observations_finite": True,
        "actor_outputs_finite": True,
        "values_finite": True,
        "actions_finite": True,
        "analog_bounds": True,
        "buttons_exact_binary": True,
        "all_worlds_completed": False,
        "exclusive_done_kind": True,
        "zero_hot_h2d": False,
        "zero_hot_d2h": False,
    }
    env.reset_transfer_counters()
    decisions_executed = 0
    for _decision in range(EVALUATION_MAX_DECISIONS):
        decisions_executed += 1
        observation = env.observation
        integrity["observations_finite"] &= _all_finite(observation)
        flat = observation.reshape(-1, OBS_DIM)
        current_actor, current_value = checkpoint_model(flat)
        current_actor = current_actor.reshape(EVALUATION_WORLDS, 2, 13)
        current_value = current_value.reshape(EVALUATION_WORLDS, 2)
        if mode == "stochastic_self_play":
            actor = current_actor
            value = current_value
        else:
            initial_actor, initial_value = initialization_model(flat)
            initial_actor = initial_actor.reshape(EVALUATION_WORLDS, 2, 13)
            initial_value = initial_value.reshape(EVALUATION_WORLDS, 2)
            actor = torch.where(current_agent_mask[..., None], current_actor, initial_actor)
            value = torch.where(current_agent_mask, current_value, initial_value)
        integrity["actor_outputs_finite"] &= _all_finite(actor)
        integrity["values_finite"] &= _all_finite(value)
        if mode == "deterministic_vs_initialization":
            action = deterministic_hybrid_action(actor)
        else:
            action = sample_hybrid_action(actor, generator=generator).action
        action = torch.where(active[:, None, None], action, torch.zeros_like(action))
        integrity["actions_finite"] &= _all_finite(action)
        integrity["analog_bounds"] &= bool(
            ((action[..., :5] >= -1.0) & (action[..., :5] <= 1.0)).all().item()
        )
        integrity["buttons_exact_binary"] &= bool(
            ((action[..., 5:] == 0.0) | (action[..., 5:] == 1.0)).all().item()
        )
        mask = active[:, None]
        analog_abs_sum += (action[..., :5].abs() * mask[..., None]).sum((0, 1)).double()
        button_sum += (action[..., 5:] * mask[..., None]).sum((0, 1)).double()
        action_count += active.sum().double() * 2.0
        log_std = actor[..., 5:10].clamp(-5.0, 1.0)
        probability = torch.sigmoid(actor[..., 10:13])
        entropy = -probability * torch.log(probability.clamp_min(1e-12)) - (
            1.0 - probability
        ) * torch.log((1.0 - probability).clamp_min(1e-12))
        std_sum += (torch.exp(log_std) * mask[..., None]).sum((0, 1)).double()
        probability_sum += (probability * mask[..., None]).sum((0, 1)).double()
        button_entropy_sum += (entropy * mask[..., None]).sum((0, 1)).double()
        policy_count += active.sum().double() * 2.0
        transition = env.step(action)
        transition_observation = transition.transition_observation
        self_touch = transition_observation[..., 176] > 0.5
        self_demoed = transition_observation[..., 178] > 0.5
        touch_count += (self_touch & mask).sum().double()
        demolition_count += (self_demoed & mask).sum().double()
        if mode != "stochastic_self_play":
            current_touches += (self_touch & current_agent_mask & mask).sum().double()
            initialization_touches += (
                self_touch & ~current_agent_mask & mask
            ).sum().double()
        episode_steps += active.to(torch.int32)
        world_decisions += active.sum().double()
        done = active & (transition.terminated | transition.truncated)
        simultaneous = transition.terminated & transition.truncated & active
        integrity["exclusive_done_kind"] &= not bool(simultaneous.any().item())
        terminated_now = done & transition.terminated
        truncated_now = done & transition.truncated
        no_touch_now = truncated_now & (transition_observation[:, 0, 181] >= 1.0)
        hard_now = truncated_now & ~no_touch_now
        terminated_count += terminated_now.sum().double()
        no_touch_count += no_touch_now.sum().double()
        hard_count += hard_now.sum().double()
        duration_sum += episode_steps[done].sum().double() / 30.0
        if mode != "stochastic_self_play":
            blue_scored = transition_observation[:, 0, 1] > 0.0
            current_scored = terminated_now & torch.where(
                current_is_blue, blue_scored, ~blue_scored
            )
            initialization_scored = terminated_now & ~current_scored
            current_goals += current_scored.sum().double()
            initialization_goals += initialization_scored.sum().double()
            current_wins += current_scored.sum().double()
            initialization_wins += initialization_scored.sum().double()
            draws += truncated_now.sum().double()
        completed |= done
        active &= ~done
        if not bool(active.any().item()):
            break
    transfer = env.hot_path_transfer_bytes()
    integrity["all_worlds_completed"] = bool(completed.all().item())
    integrity["zero_hot_h2d"] = transfer["h2d"] == 0
    integrity["zero_hot_d2h"] = transfer["d2h"] == 0
    episodes = float(completed.sum().item())
    simulated_minutes = float(world_decisions.item()) / 30.0 / 60.0

    def channels(
        values: torch.Tensor, names: tuple[str, ...], denominator: float
    ) -> dict[str, float]:
        return {
            name: float(value.item()) / denominator
            for name, value in zip(names, values, strict=True)
        }

    result: dict[str, Any] = {
        "mode": mode,
        "worlds": EVALUATION_WORLDS,
        "completed_episodes": int(episodes),
        "decisions_executed": decisions_executed,
        "world_decisions": int(world_decisions.item()),
        "simulated_minutes": simulated_minutes,
        "goal_terminated_episodes": int(terminated_count.item()),
        "no_touch_truncated_episodes": int(no_touch_count.item()),
        "hard_truncated_episodes": int(hard_count.item()),
        "goal_terminated_fraction": float(terminated_count.item()) / episodes,
        "no_touch_truncated_fraction": float(no_touch_count.item()) / episodes,
        "hard_truncated_fraction": float(hard_count.item()) / episodes,
        "touch_entries": int(touch_count.item()),
        "demolition_events": int(demolition_count.item()),
        "touches_per_simulated_minute": float(touch_count.item()) / simulated_minutes,
        "goals_per_simulated_minute": float(terminated_count.item()) / simulated_minutes,
        "demolitions_per_simulated_minute": float(demolition_count.item())
        / simulated_minutes,
        "mean_episode_duration_seconds": float(duration_sum.item()) / episodes,
        "mean_absolute_analog_action": channels(
            analog_abs_sum, ANALOG_ACTION_NAMES, float(action_count.item())
        ),
        "button_activation_rate": channels(
            button_sum, BUTTON_ACTION_NAMES, float(action_count.item())
        ),
        "mean_analog_policy_std": channels(
            std_sum, ANALOG_ACTION_NAMES, float(policy_count.item())
        ),
        "mean_button_probability": channels(
            probability_sum, BUTTON_ACTION_NAMES, float(policy_count.item())
        ),
        "mean_button_entropy": channels(
            button_entropy_sum, BUTTON_ACTION_NAMES, float(policy_count.item())
        ),
        "integrity": integrity,
        "hot_path_transfer_bytes": transfer,
    }
    if mode != "stochastic_self_play":
        result["versus_initialization"] = {
            "current_goals": int(current_goals.item()),
            "initialization_goals": int(initialization_goals.item()),
            "score_differential": int(current_goals.item() - initialization_goals.item()),
            "goal_differential": int(current_goals.item() - initialization_goals.item()),
            "current_touches": int(current_touches.item()),
            "initialization_touches": int(initialization_touches.item()),
            "touch_differential": int(current_touches.item() - initialization_touches.item()),
            "current_wins": int(current_wins.item()),
            "initialization_wins": int(initialization_wins.item()),
            "draws": int(draws.item()),
        }
    result["verdict"] = "PASS_GREEN" if all(integrity.values()) else "FAIL_RED"
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_checkpoint(
    *,
    label: str,
    samples: int,
    checkpoint_model: Rival2ActorCritic,
    initialization_state: dict[str, torch.Tensor],
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    initialization_model = Rival2ActorCritic().to(device)
    initialization_model.load_state_dict(initialization_state)
    initialization_model.eval().requires_grad_(False)
    started = time.perf_counter()
    modes = {
        mode: _evaluate_mode(
            mode=mode,
            checkpoint_model=checkpoint_model,
            initialization_model=initialization_model,
            collision_dir=collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=device,
        )
        for mode in EVALUATION_MODES
    }
    del initialization_model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "checkpoint_label": label,
        "agent_decision_samples": samples,
        "evaluation_protocol_sha256": protocol_sha256,
        "modes": modes,
        "wall_seconds": time.perf_counter() - started,
        "verdict": (
            "PASS_GREEN"
            if all(result["verdict"] == "PASS_GREEN" for result in modes.values())
            else "FAIL_RED"
        ),
    }


def _checkpoint_record(path: Path, label: str, trainer: Rival2Trainer) -> dict[str, Any]:
    return {
        "label": label,
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "agent_decision_samples": trainer.total_agent_samples,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "historical_policy_versions": list(trainer.opponent_pool.versions),
    }


def _save_threshold(
    *,
    label: str,
    trainer: Rival2Trainer,
    initialization_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    configuration: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer.add_historical_snapshot()
    path = args.work_dir / "checkpoints" / f"rival2_campaign01_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    checkpoint = _checkpoint_record(path, label, trainer)
    evaluation = evaluate_checkpoint(
        label=label,
        samples=trainer.total_agent_samples,
        checkpoint_model=trainer.model,
        initialization_state=initialization_state,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        protocol_sha256=configuration["evaluation"]["protocol_sha256"],
    )
    _write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
    return checkpoint, evaluation


def _exact_reload_gate(
    trainer: Rival2Trainer,
    checkpoint_path: Path,
) -> dict[str, Any]:
    fixed_observation = trainer.env.observation[:64].detach().clone()
    generator_state = trainer.policy_generator.get_state()
    with torch.no_grad():
        expected_actor, expected_value = trainer.model(fixed_observation.reshape(-1, OBS_DIM))
        expected_sample = sample_hybrid_action(
            expected_actor, generator=trainer.policy_generator, config=trainer.policy_config
        )
        expected_deterministic = deterministic_hybrid_action(expected_actor)
    trainer.policy_generator.set_state(generator_state)
    restored = Rival2Trainer(trainer.env, seed=0)
    restored.load_checkpoint(checkpoint_path)
    with torch.no_grad():
        actual_actor, actual_value = restored.model(fixed_observation.reshape(-1, OBS_DIM))
        actual_sample = sample_hybrid_action(
            actual_actor, generator=restored.policy_generator, config=restored.policy_config
        )
        actual_deterministic = deterministic_hybrid_action(actual_actor)
    checks = {
        "contract_and_config_load_succeeded": True,
        "iteration_exact": restored.iteration == trainer.iteration,
        "policy_version_exact": restored.policy_version == trainer.policy_version,
        "sample_count_exact": restored.total_agent_samples == trainer.total_agent_samples,
        "historical_versions_exact": restored.opponent_pool.versions
        == trainer.opponent_pool.versions,
        "actor_output_exact": torch.equal(expected_actor, actual_actor),
        "value_output_exact": torch.equal(expected_value, actual_value),
        "next_stochastic_action_exact": torch.equal(
            expected_sample.action, actual_sample.action
        ),
        "next_stochastic_pre_tanh_exact": torch.equal(
            expected_sample.pre_tanh, actual_sample.pre_tanh
        ),
        "next_stochastic_log_probability_exact": torch.equal(
            expected_sample.log_probability, actual_sample.log_probability
        ),
        "deterministic_inference_exact": torch.equal(
            expected_deterministic, actual_deterministic
        ),
    }
    result = {
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "checks": checks,
        "fixed_observation_sha256": _sha256_tensor(fixed_observation),
        "actor_output_sha256": _sha256_tensor(actual_actor),
        "value_output_sha256": _sha256_tensor(actual_value),
        "next_stochastic_action_sha256": _sha256_tensor(actual_sample.action),
        "deterministic_action_sha256": _sha256_tensor(actual_deterministic),
    }
    del restored
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_training(
    args: argparse.Namespace, configuration: dict[str, Any], selected_worlds: int
) -> int:
    if selected_worlds not in CAPACITY_ORDER:
        raise ValueError("training capacity was not selected by the authorized preflight")
    _initialize_runtime(args.device)
    verify_authority(configuration)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    torch.manual_seed(CAMPAIGN_SEED)
    torch.cuda.manual_seed(CAMPAIGN_SEED)
    kickoff_selector = (
        np.arange(selected_worlds, dtype=np.int32) + CAMPAIGN_SEED
    ) % 5
    env = Rival2Env(
        selected_worlds,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    trainer = Rival2Trainer(env, seed=CAMPAIGN_SEED)
    initialization_state = {
        name: tensor.detach().cpu().clone() for name, tensor in trainer.model.state_dict().items()
    }
    initialization_sha256 = _state_dict_sha256(initialization_state)
    checkpoints: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    training_curve: list[dict[str, Any]] = []
    checkpoint, evaluation = _save_threshold(
        label="000m",
        trainer=trainer,
        initialization_state=initialization_state,
        args=args,
        configuration=configuration,
        geometry=geometry,
        meshes=meshes,
    )
    checkpoints.append(checkpoint)
    evaluations.append(evaluation)
    next_threshold_index = 1
    campaign_started = time.perf_counter()
    execution_status = "COMPLETE"
    stop_detail = "first completed PPO update crossing 100M agent decision samples"
    while trainer.total_agent_samples < TARGET_SAMPLES:
        policy_version_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        env.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(args.device)
        iteration_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        transfer = env.hot_path_transfer_bytes()
        integrity = _rollout_integrity(
            trainer,
            rollout,
            metrics,
            policy_version_before=policy_version_before,
            samples_before=samples_before,
        )
        integrity["checks"]["zero_hot_h2d"] = transfer["h2d"] == 0
        integrity["checks"]["zero_hot_d2h"] = transfer["d2h"] == 0
        integrity["hot_path_transfer_bytes"] = transfer
        integrity["verdict"] = (
            "PASS_GREEN" if all(integrity["checks"].values()) else "FAIL_RED"
        )
        point = {
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
            "wall_seconds": iteration_seconds,
            "agent_decisions_per_second": (
                trainer.total_agent_samples - samples_before
            )
            / iteration_seconds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "integrity": integrity,
        }
        training_curve.append(point)
        _write_json(args.work_dir / "training_curve.json", training_curve)
        print(
            f"campaign update={trainer.iteration} samples={trainer.total_agent_samples} "
            f"seconds={iteration_seconds:.3f} integrity={integrity['verdict']}",
            flush=True,
        )
        if integrity["verdict"] != "PASS_GREEN":
            execution_status = "STOP_NUMERICAL"
            stop_detail = f"integrity failure at update {trainer.iteration}"
            failure_path = (
                args.work_dir / "checkpoints" / f"integrity_failure_{trainer.iteration}.pt"
            )
            trainer.save_checkpoint(failure_path)
            break
        del rollout
        gc.collect()
        while (
            next_threshold_index < len(THRESHOLDS)
            and trainer.total_agent_samples >= THRESHOLDS[next_threshold_index][1]
        ):
            label = THRESHOLDS[next_threshold_index][0]
            checkpoint, evaluation = _save_threshold(
                label=label,
                trainer=trainer,
                initialization_state=initialization_state,
                args=args,
                configuration=configuration,
                geometry=geometry,
                meshes=meshes,
            )
            checkpoints.append(checkpoint)
            evaluations.append(evaluation)
            if evaluation["verdict"] != "PASS_GREEN":
                execution_status = "STOP_NUMERICAL"
                stop_detail = f"evaluation integrity failure at {label}"
                break
            next_threshold_index += 1
        if execution_status != "COMPLETE":
            break
    final_checkpoint = Path(checkpoints[-1]["path"])
    reload_gate = (
        _exact_reload_gate(trainer, final_checkpoint)
        if execution_status == "COMPLETE"
        else {"verdict": "NOT_RUN", "reason": stop_detail}
    )
    if execution_status == "COMPLETE" and reload_gate["verdict"] != "PASS_GREEN":
        execution_status = "STOP_ARCHITECTURAL"
        stop_detail = "final v0.5 checkpoint reload/continuation gate failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "execution_status": execution_status,
        "stop_detail": stop_detail,
        "selected_worlds": selected_worlds,
        "campaign_seed": CAMPAIGN_SEED,
        "initialization_model_sha256": initialization_sha256,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "final_iteration": trainer.iteration,
        "first_update_crossing_target": trainer.iteration
        if trainer.total_agent_samples >= TARGET_SAMPLES
        else None,
        "campaign_training_wall_seconds": time.perf_counter() - campaign_started,
        "checkpoints": checkpoints,
        "evaluations": [
            {
                "label": item["checkpoint_label"],
                "agent_decision_samples": item["agent_decision_samples"],
                "verdict": item["verdict"],
            }
            for item in evaluations
        ],
        "final_checkpoint_reload": reload_gate,
        "runtime": _runtime_identity(args.device),
        "frozen_contract_hashes": dict(CONTRACT_HASHES),
        "policy_config_hash": trainer.policy_config.content_hash,
        "ppo_config_hash": trainer.ppo_config.content_hash,
        "no_v06_work": True,
    }
    _write_json(args.work_dir / "checkpoints.json", {"checkpoints": checkpoints})
    _write_json(args.work_dir / "run_summary.json", result)
    return 0 if execution_status == "COMPLETE" else 3


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    configuration_path = args.work_dir / "config_frozen_before_training.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("existing prospectively frozen campaign configuration differs")
    else:
        _write_json(configuration_path, configuration)
    if args.phase == "preflight-candidate":
        if args.worlds is None:
            raise ValueError("--worlds is required for a preflight candidate")
        return run_preflight_candidate(args, configuration)
    if args.phase == "train":
        if args.worlds is None:
            raise ValueError("--worlds is required for training")
        return run_training(args, configuration, args.worlds)
    verify_authority(configuration)
    freeze_prior_results_manifest(args.work_dir)
    preflight = run_preflight_subprocesses(args)
    if preflight["selected_worlds"] is None:
        return 4
    return run_training(args, configuration, int(preflight["selected_worlds"]))


if __name__ == "__main__":
    raise SystemExit(main())
