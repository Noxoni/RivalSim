"""Execute the controlled entropy-off Rival 2.0 Campaign 02 authority."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ANALOG_ACTION_NAMES,
    BUTTON_ACTION_NAMES,
    CONTRACT_HASHES,
    OBS_DIM,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import deterministic_hybrid_action, sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2Trainer

AUTHORIZED_HEAD = "52713ef13309d8c5c219456ca6e66bdc10a5586a"
CAMPAIGN01_CLOSEOUT = "1ce5932cadd66b14032e61750836763499567bc9"
EXPECTED_INITIALIZATION_SHA256 = (
    "890F224879DB6E458472985B226A664D8AE49B8303C21CFB0FD83A485CF42848"
)
CAMPAIGN02_WORLDS = 131072
CAMPAIGN02_ENTROPY_COEFFICIENT = 0.0
KL_DIAGNOSTIC_THRESHOLD = 0.1
CLIP_FRACTION_DIAGNOSTIC_THRESHOLD = 0.3
DIAGNOSTIC_WORLDS = 4096
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def campaign02_ppo_config() -> Rival2PPOConfig:
    return Rival2PPOConfig(
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.20,
        value_loss_coefficient=0.50,
        entropy_coefficient=CAMPAIGN02_ENTROPY_COEFFICIENT,
        max_gradient_norm=0.50,
        learning_rate=3e-4,
        epochs=2,
        rollout_horizon=32,
        minibatch_size=65536,
    )


def ppo_configuration_differences(
    campaign01_config: dict[str, Any], campaign02_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = set(campaign01_config) | set(campaign02_config)
    return {
        key: {"campaign01": campaign01_config.get(key), "campaign02": campaign02_config.get(key)}
        for key in sorted(keys)
        if campaign01_config.get(key) != campaign02_config.get(key)
    }


def frozen_configuration() -> dict[str, Any]:
    base = copy.deepcopy(campaign01.frozen_configuration())
    ppo = campaign02_ppo_config()
    campaign01_ppo = dict(base["ppo_config"])
    base.update(
        {
            "authority": "Rival 2.0 Campaign 02 entropy-off controlled rerun",
            "authorized_head": AUTHORIZED_HEAD,
            "required_parent_closeout": CAMPAIGN01_CLOSEOUT,
            "worlds": CAMPAIGN02_WORLDS,
            "ppo_config": asdict(ppo),
            "ppo_config_hash": ppo.content_hash,
            "expected_initialization_model_sha256": EXPECTED_INITIALIZATION_SHA256,
            "controlled_variable": {
                "name": "entropy_coefficient",
                "campaign01": 0.01,
                "campaign02": 0.0,
                "diagnostic_entropy_still_logged": True,
                "optimization_entropy_contribution": 0.0,
            },
            "campaign01_ppo_config": campaign01_ppo,
            "ppo_configuration_differences": ppo_configuration_differences(
                campaign01_ppo, asdict(ppo)
            ),
            "per_update_diagnostics": {
                "frozen_observation_worlds": DIAGNOSTIC_WORLDS,
                "approximate_kl_flag_threshold": KL_DIAGNOSTIC_THRESHOLD,
                "clip_fraction_flag_threshold": CLIP_FRACTION_DIAGNOSTIC_THRESHOLD,
                "automatic_kl_stopping": False,
            },
        }
    )
    base["hard_boundaries"] = [
        "entropy coefficient is the only Campaign 01 to Campaign 02 learning change",
        "no reward, observation, action, episode, model, optimizer, self-play, or seed change",
        "no action mask, curriculum, imitation data, tuning, KL stop, or hyperparameter search",
        "no v0.6",
    ]
    return base


def _campaign01_config() -> dict[str, Any]:
    return json.loads(
        Path("results/rival2/campaign01/config.json").read_text(encoding="utf-8")
    )


def verify_authority(configuration: dict[str, Any]) -> dict[str, Any]:
    if campaign01._git("rev-parse", "HEAD") != AUTHORIZED_HEAD:
        raise RuntimeError("Campaign 02 must start from the authorized HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CAMPAIGN01_CLOSEOUT, "HEAD"],
        check=True,
        capture_output=True,
    )
    if CONTRACT_HASHES != campaign01.EXPECTED_CONTRACT_HASHES:
        raise RuntimeError("frozen Rival 2.0 contract hashes changed")
    c01 = _campaign01_config()
    invariants = (
        "campaign_seed",
        "target_agent_decision_samples",
        "policy_config",
        "policy_config_hash",
        "self_play_config",
        "contract_hashes",
        "thresholds",
        "evaluation",
    )
    invariant_checks = {name: configuration[name] == c01[name] for name in invariants}
    differences = ppo_configuration_differences(c01["ppo_config"], configuration["ppo_config"])
    checks = {
        **{f"campaign01_invariant_{name}": value for name, value in invariant_checks.items()},
        "only_ppo_difference_is_entropy_coefficient": set(differences)
        == {"entropy_coefficient"},
        "campaign01_entropy_is_0_01": c01["ppo_config"]["entropy_coefficient"] == 0.01,
        "campaign02_entropy_is_zero": configuration["ppo_config"]["entropy_coefficient"]
        == 0.0,
        "world_count_unchanged": c01["capacity_selection"]["selected_worlds"]
        == configuration["worlds"]
        == CAMPAIGN02_WORLDS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Campaign 02 controlled-variable proof failed: {checks}")
    return {
        "verdict": "PASS_GREEN",
        "checks": checks,
        "ppo_configuration_differences": differences,
    }


def _tracked_paths(*specifications: str) -> list[str]:
    paths: list[str] = []
    for specification in specifications:
        output = campaign01._git("ls-files", specification)
        paths.extend(line for line in output.splitlines() if line)
    return sorted(set(paths))


def immutable_parent_manifest() -> dict[str, Any]:
    prior_results = _tracked_paths(
        "results/v0.1",
        "results/v0.2",
        "results/v0.2.1",
        "results/v0.2.2",
        "results/v0.3",
        "results/v0.4",
        "results/v0.5",
    )
    campaign01_artifacts = _tracked_paths(
        "results/rival2/campaign01",
        "checkpoints/rival2/campaign01",
        "docs/RIVAL2_CAMPAIGN01_RESULTS.md",
        "benchmarks/run_rival2_campaign01.py",
        "benchmarks/build_rival2_campaign01_evidence.py",
        "tests/test_rival2_campaign01.py",
        "tests/test_rival2_campaign01_evidence.py",
        "handoff/rival2-c01",
    )
    frozen_implementation = _tracked_paths(
        "rivalsim/rival2_contracts.py",
        "rivalsim/rival2_env.py",
        "rivalsim/rival2_policy.py",
        "rivalsim/rival2_ppo.py",
        "rivalsim/rival2_training.py",
        "rivalsim/kernels/rival2.py",
    )

    def entries(paths: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "path": path.replace("\\", "/"),
                "size_bytes": Path(path).stat().st_size,
                "sha256": campaign01._sha256_file(Path(path)),
            }
            for path in paths
        ]

    sections = {
        "results_v01_through_v05": entries(prior_results),
        "campaign01_artifacts": entries(campaign01_artifacts),
        "frozen_v05_training_implementation": entries(frozen_implementation),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sections": sections,
        "section_sha256": {
            name: campaign01._sha256_json(value) for name, value in sections.items()
        },
        "manifest_sha256": campaign01._sha256_json(sections),
    }


def freeze_immutable_parent_manifest(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "immutable_parent_manifest.json"
    current = immutable_parent_manifest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_comparable = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        if existing_comparable != current:
            raise RuntimeError("immutable Campaign 01/v0.5 parent artifacts changed after freeze")
        return existing
    current["created_utc"] = campaign01._utc_now()
    campaign01._write_json(path, current)
    return current


@torch.no_grad()
def policy_distribution_diagnostic(
    trainer: Rival2Trainer, frozen_observation: torch.Tensor
) -> dict[str, Any]:
    trainer.model.eval()
    actor, value = trainer.model(frozen_observation.reshape(-1, OBS_DIM))
    log_std = actor[..., 5:10].clamp(-5.0, 1.0)
    standard_deviation = torch.exp(log_std)
    button_probability = torch.sigmoid(actor[..., 10:13])

    def channels(values: torch.Tensor, names: tuple[str, ...]) -> dict[str, float]:
        means = values.mean(dim=0).cpu().tolist()
        return {name: float(value) for name, value in zip(names, means, strict=True)}

    return {
        "observation_worlds": frozen_observation.shape[0],
        "observation_sha256": campaign01._sha256_tensor(frozen_observation),
        "mean_log_std": channels(log_std, ANALOG_ACTION_NAMES),
        "mean_analog_policy_std": channels(standard_deviation, ANALOG_ACTION_NAMES),
        "mean_button_probability": channels(button_probability, BUTTON_ACTION_NAMES),
        "actor_output_sha256": campaign01._sha256_tensor(actor),
        "value_output_sha256": campaign01._sha256_tensor(value),
        "finite": campaign01._all_finite(actor) and campaign01._all_finite(value),
    }


def _deep_differences(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [{"path": path, "expected": expected, "actual": actual}]
    if isinstance(expected, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                differences.append(
                    {"path": child, "expected": expected.get(key), "actual": actual.get(key)}
                )
            else:
                differences.extend(_deep_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "expected_length": len(expected), "actual_length": len(actual)}]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_deep_differences(left, right, f"{path}[{index}]"))
        return differences
    return [] if expected == actual else [{"path": path, "expected": expected, "actual": actual}]


def initialization_evaluation_control(evaluation: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads(
        Path("results/rival2/campaign01/evaluation_000m.json").read_text(encoding="utf-8")
    )
    compared_keys = (
        "schema_version",
        "checkpoint_label",
        "agent_decision_samples",
        "evaluation_protocol_sha256",
        "modes",
        "verdict",
    )
    expected_semantics = {key: expected[key] for key in compared_keys}
    actual_semantics = {key: evaluation[key] for key in compared_keys}
    differences = _deep_differences(expected_semantics, actual_semantics)
    return {
        "verdict": "PASS_GREEN" if not differences else "FAIL_RED",
        "semantic_metrics_exact": not differences,
        "semantic_difference_count": len(differences),
        "semantic_differences": differences[:100],
        "non_semantic_metadata": {
            "campaign01_created_utc": expected["created_utc"],
            "campaign02_created_utc": evaluation["created_utc"],
            "campaign01_wall_seconds": expected["wall_seconds"],
            "campaign02_wall_seconds": evaluation["wall_seconds"],
        },
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
    path = args.work_dir / "checkpoints" / f"rival2_campaign02_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    checkpoint = campaign01._checkpoint_record(path, label, trainer)
    evaluation = campaign01.evaluate_checkpoint(
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
    campaign01._write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
    return checkpoint, evaluation


def _exact_reload_gate(
    trainer: Rival2Trainer, checkpoint_path: Path, ppo_config: Rival2PPOConfig
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
    restored = Rival2Trainer(trainer.env, ppo_config=ppo_config, seed=0)
    restored.load_checkpoint(checkpoint_path)
    with torch.no_grad():
        actual_actor, actual_value = restored.model(fixed_observation.reshape(-1, OBS_DIM))
        actual_sample = sample_hybrid_action(
            actual_actor, generator=restored.policy_generator, config=restored.policy_config
        )
        actual_deterministic = deterministic_hybrid_action(actual_actor)
    checks = {
        "contract_and_config_load_succeeded": True,
        "entropy_coefficient_zero_after_load": restored.ppo_config.entropy_coefficient == 0.0,
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
        "fixed_observation_sha256": campaign01._sha256_tensor(fixed_observation),
        "actor_output_sha256": campaign01._sha256_tensor(actual_actor),
        "value_output_sha256": campaign01._sha256_tensor(actual_value),
        "next_stochastic_action_sha256": campaign01._sha256_tensor(actual_sample.action),
        "deterministic_action_sha256": campaign01._sha256_tensor(actual_deterministic),
    }
    del restored
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_campaign(args: argparse.Namespace, configuration: dict[str, Any]) -> int:
    campaign01._initialize_runtime(args.device)
    controlled_variable = verify_authority(configuration)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    torch.manual_seed(campaign01.CAMPAIGN_SEED)
    torch.cuda.manual_seed(campaign01.CAMPAIGN_SEED)
    kickoff_selector = (
        np.arange(CAMPAIGN02_WORLDS, dtype=np.int32) + campaign01.CAMPAIGN_SEED
    ) % 5
    env = Rival2Env(
        CAMPAIGN02_WORLDS,
        args.collision_dir,
        device=args.device,
        seed=campaign01.CAMPAIGN_SEED,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    ppo_config = campaign02_ppo_config()
    trainer = Rival2Trainer(env, ppo_config=ppo_config, seed=campaign01.CAMPAIGN_SEED)
    initialization_state = {
        name: tensor.detach().cpu().clone() for name, tensor in trainer.model.state_dict().items()
    }
    initialization_sha256 = campaign01._state_dict_sha256(initialization_state)
    frozen_diagnostic_observation = env.observation[:DIAGNOSTIC_WORLDS].detach().clone()
    initialization_diagnostic = policy_distribution_diagnostic(
        trainer, frozen_diagnostic_observation
    )
    initialization_hash_exact = initialization_sha256 == EXPECTED_INITIALIZATION_SHA256
    checkpoints: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    training_curve: list[dict[str, Any]] = []
    checkpoint, initialization_evaluation = _save_threshold(
        label="000m",
        trainer=trainer,
        initialization_state=initialization_state,
        args=args,
        configuration=configuration,
        geometry=geometry,
        meshes=meshes,
    )
    checkpoints.append(checkpoint)
    evaluations.append(initialization_evaluation)
    evaluation_control = initialization_evaluation_control(initialization_evaluation)
    initialization_control = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "verdict": (
            "PASS_GREEN"
            if initialization_hash_exact and evaluation_control["verdict"] == "PASS_GREEN"
            else "FAIL_RED"
        ),
        "campaign_seed": campaign01.CAMPAIGN_SEED,
        "evaluation_seed": campaign01.EVALUATION_SEED,
        "expected_model_sha256": EXPECTED_INITIALIZATION_SHA256,
        "actual_model_sha256": initialization_sha256,
        "model_sha256_exact": initialization_hash_exact,
        "evaluation_control": evaluation_control,
        "policy_distribution": initialization_diagnostic,
        "controlled_variable_proof": controlled_variable,
    }
    campaign01._write_json(args.work_dir / "initialization_control.json", initialization_control)
    if initialization_control["verdict"] != "PASS_GREEN":
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": campaign01._utc_now(),
            "execution_status": "STOP_ARCHITECTURAL",
            "stop_detail": "Campaign 02 initialization control did not reproduce Campaign 01",
            "initialization_control": initialization_control,
            "final_agent_decision_samples": 0,
            "final_iteration": 0,
            "no_v06_work": True,
        }
        campaign01._write_json(args.work_dir / "run_summary.json", summary)
        return 3
    next_threshold_index = 1
    execution_status = "COMPLETE"
    stop_detail = "first completed PPO update crossing 100M agent decision samples"
    campaign_started = time.perf_counter()
    while trainer.total_agent_samples < campaign01.TARGET_SAMPLES:
        policy_version_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        env.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(args.device)
        iteration_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        transfer = env.hot_path_transfer_bytes()
        integrity = campaign01._rollout_integrity(
            trainer,
            rollout,
            metrics,
            policy_version_before=policy_version_before,
            samples_before=samples_before,
        )
        metric_values = integrity["metrics"]
        loss_without_entropy = metric_values["policy_loss"] + (
            ppo_config.value_loss_coefficient * metric_values["value_loss"]
        )
        loss_identity_error = abs(metric_values["total_loss"] - loss_without_entropy)
        integrity["checks"].update(
            {
                "entropy_coefficient_exact_zero": ppo_config.entropy_coefficient == 0.0,
                "entropy_optimization_contribution_zero": True,
                "total_loss_excludes_entropy": loss_identity_error <= 2e-7,
                "zero_hot_h2d": transfer["h2d"] == 0,
                "zero_hot_d2h": transfer["d2h"] == 0,
            }
        )
        integrity["hot_path_transfer_bytes"] = transfer
        integrity["verdict"] = (
            "PASS_GREEN" if all(integrity["checks"].values()) else "FAIL_RED"
        )
        distribution = policy_distribution_diagnostic(
            trainer, frozen_diagnostic_observation
        )
        diagnostic_flags = {
            "approximate_kl_ge_0_1": metric_values["approx_kl"]
            >= KL_DIAGNOSTIC_THRESHOLD,
            "clip_fraction_ge_0_3": metric_values["clip_fraction"]
            >= CLIP_FRACTION_DIAGNOSTIC_THRESHOLD,
        }
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
            "policy_distribution_on_frozen_observations": distribution,
            "optimizer_diagnosis": {
                "entropy_coefficient": ppo_config.entropy_coefficient,
                "diagnostic_entropy": metric_values["entropy"],
                "entropy_optimization_contribution": 0.0,
                "loss_without_entropy": loss_without_entropy,
                "reported_total_loss": metric_values["total_loss"],
                "loss_identity_absolute_error": loss_identity_error,
                "flags": diagnostic_flags,
            },
        }
        training_curve.append(point)
        campaign01._write_json(args.work_dir / "training_curve.json", training_curve)
        print(
            f"campaign02 update={trainer.iteration} samples={trainer.total_agent_samples} "
            f"seconds={iteration_seconds:.3f} kl={metric_values['approx_kl']:.6f} "
            f"clip={metric_values['clip_fraction']:.6f} integrity={integrity['verdict']}",
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
            next_threshold_index < len(campaign01.THRESHOLDS)
            and trainer.total_agent_samples
            >= campaign01.THRESHOLDS[next_threshold_index][1]
        ):
            label = campaign01.THRESHOLDS[next_threshold_index][0]
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
        _exact_reload_gate(trainer, final_checkpoint, ppo_config)
        if execution_status == "COMPLETE"
        else {"verdict": "NOT_RUN", "reason": stop_detail}
    )
    if execution_status == "COMPLETE" and reload_gate["verdict"] != "PASS_GREEN":
        execution_status = "STOP_ARCHITECTURAL"
        stop_detail = "final frozen-v0.5 checkpoint continuation gate failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": execution_status,
        "stop_detail": stop_detail,
        "selected_worlds": CAMPAIGN02_WORLDS,
        "campaign_seed": campaign01.CAMPAIGN_SEED,
        "initialization_model_sha256": initialization_sha256,
        "initialization_control_verdict": initialization_control["verdict"],
        "final_agent_decision_samples": trainer.total_agent_samples,
        "final_iteration": trainer.iteration,
        "first_update_crossing_target": trainer.iteration
        if trainer.total_agent_samples >= campaign01.TARGET_SAMPLES
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
        "runtime": campaign01._runtime_identity(args.device),
        "frozen_contract_hashes": dict(CONTRACT_HASHES),
        "policy_config_hash": trainer.policy_config.content_hash,
        "ppo_config": asdict(ppo_config),
        "ppo_config_hash": ppo_config.content_hash,
        "entropy_optimization_contribution": 0.0,
        "no_v06_work": True,
    }
    campaign01._write_json(args.work_dir / "checkpoints.json", {"checkpoints": checkpoints})
    campaign01._write_json(args.work_dir / "run_summary.json", result)
    return 0 if execution_status == "COMPLETE" else 4


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    configuration_path = args.work_dir / "config_frozen_before_training.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("existing prospectively frozen Campaign 02 config differs")
    else:
        campaign01._write_json(configuration_path, configuration)
    freeze_immutable_parent_manifest(args.work_dir)
    return run_campaign(args, configuration)


if __name__ == "__main__":
    raise SystemExit(main())
