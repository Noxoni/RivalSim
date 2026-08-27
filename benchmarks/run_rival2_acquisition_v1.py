"""Train Rival 2.0 ball acquisition in the original short episode lifecycle."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign03 as campaign03
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ACTION_NAMES,
    CAR_LINEAR_SPEED_SCALE,
    EPISODE_CONTRACT_HASH,
    OBS_DIM,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
    REWARD_ACQUISITION_V1_CONTRACT,
    REWARD_ACQUISITION_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import Rival2PolicyConfig, sample_hybrid_action
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

MECHANICS_FIX_COMMIT = "0fa27d2fd846b4e8d4b9955a0ad88c2c2af91037"
MECHANICS_EVIDENCE = Path(
    "results/rival2/mechanics_correction/movement_mechanics_parity.json"
)
AUTHORITY = Path("handoff/rival2-acquisition-v1/README.md")

WORLDS = 131_072
CAMPAIGN_SEED = 20_260_827
EVALUATION_WORLDS = 4_096
EVALUATION_SEED = 920_260_827
EVALUATION_INTERVAL = 30
NO_TOUCH_THRESHOLD = 0.01
REQUIRED_CONSECUTIVE = 2
MAX_EVALUATION_DECISIONS = 45 * 30
SCHEMA_VERSION = 1

_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")
_SELF_BOOST_INDEX = OBS_FIELD_NAMES.index("self.boost")
_SELF_ON_GROUND_INDEX = OBS_FIELD_NAMES.index("self.on_ground")
_SELF_HAS_FLIPPED_INDEX = OBS_FIELD_NAMES.index("self.has_flipped")
_SELF_SUPERSONIC_INDEX = OBS_FIELD_NAMES.index("self.is_supersonic")
_RELATIVE_BALL_START = OBS_FIELD_NAMES.index("relative.ball_position.x")
_SELF_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
_NO_TOUCH_AGE_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rival2/acquisition_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    return campaign01._sha256_file(path)


def frozen_configuration() -> dict[str, Any]:
    policy = Rival2PolicyConfig()
    ppo = campaign03.campaign03_ppo_config()
    self_play = Rival2SelfPlayConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY.as_posix(),
        "mechanics_fix_commit": MECHANICS_FIX_COMMIT,
        "campaign_seed": CAMPAIGN_SEED,
        "worlds": WORLDS,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "episode_version": RIVAL2_EPISODE_VERSION,
        "episode_contract_hash": EPISODE_CONTRACT_HASH,
        "episode_mode": (
            "original Rival 2.0 first-goal/15-second-no-touch/45-second-hard-limit intervals"
        ),
        "reward_version": RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        "reward_contract_hash": REWARD_ACQUISITION_V1_CONTRACT_HASH,
        "reward_contract": REWARD_ACQUISITION_V1_CONTRACT,
        "evaluation": {
            "worlds": EVALUATION_WORLDS,
            "seed": EVALUATION_SEED,
            "interval_updates": EVALUATION_INTERVAL,
            "same_short_episode_lifecycle_as_training": True,
            "stochastic_current_policy_self_play": True,
        },
        "gate": {
            "no_touch_fraction_maximum": NO_TOUCH_THRESHOLD,
            "required_consecutive_evaluations": REQUIRED_CONSECUTIVE,
            "sample_cap": None,
        },
        "nexto_training": False,
        "five_minute_matches": False,
    }


def verify_launch(configuration: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", MECHANICS_FIX_COMMIT, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    mechanics = json.loads(MECHANICS_EVIDENCE.read_text(encoding="utf-8"))
    expected = contract_hashes_for_reward(
        RIVAL2_REWARD_ACQUISITION_V1_VERSION, RIVAL2_EPISODE_VERSION
    )
    checks = {
        "mechanics_fix_is_ancestor": True,
        "head_pushed_to_origin_main": head == origin,
        "tracked_worktree_clean": subprocess.run(
            ["git", "diff", "--quiet"]
        ).returncode
        == 0,
        "index_clean": subprocess.run(
            ["git", "diff", "--cached", "--quiet"]
        ).returncode
        == 0,
        "mechanics_gate_green": mechanics["status"] == "PASS",
        "reward_identity_exact": (
            expected[RIVAL2_REWARD_ACQUISITION_V1_VERSION]
            == REWARD_ACQUISITION_V1_CONTRACT_HASH
        ),
        "episode_identity_exact": (
            expected[RIVAL2_EPISODE_VERSION] == EPISODE_CONTRACT_HASH
        ),
        "world_count_exact": configuration["worlds"] == WORLDS,
        "ppo_entropy_zero": configuration["ppo_config"]["entropy_coefficient"]
        == 0.0,
        "no_five_minute_matches": not configuration["five_minute_matches"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "head": head,
        "origin_main": origin,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"acquisition launch gate failed: {checks}")
    return result


def _side_channels(values: torch.Tensor) -> dict[str, dict[str, float]]:
    return {
        side: {
            name: float(values[side_index, channel].item())
            for channel, name in enumerate(ACTION_NAMES)
        }
        for side_index, side in enumerate(("Blue", "Orange"))
    }


@torch.no_grad()
def evaluate(
    *,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    label: str,
) -> dict[str, Any]:
    """Evaluate one original Rival 2.0 episode per held-out world."""

    started = time.perf_counter()
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED
    ) % 5
    env = Rival2Env(
        EVALUATION_WORLDS,
        collision_dir,
        device=device,
        seed=EVALUATION_SEED,
        reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    model = trainer.model
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=env.device).manual_seed(EVALUATION_SEED)
    active = torch.ones(EVALUATION_WORLDS, dtype=torch.bool, device=env.device)
    completed = torch.zeros_like(active)
    episode_decisions = torch.zeros(
        EVALUATION_WORLDS, dtype=torch.int32, device=env.device
    )
    first_touch_latency = torch.full(
        (EVALUATION_WORLDS,), -1, dtype=torch.int32, device=env.device
    )
    side_touch = torch.zeros(2, dtype=torch.float64, device=env.device)
    goals = torch.zeros((), dtype=torch.float64, device=env.device)
    no_touch = torch.zeros((), dtype=torch.float64, device=env.device)
    hard = torch.zeros((), dtype=torch.float64, device=env.device)
    world_decisions = torch.zeros((), dtype=torch.float64, device=env.device)
    action_count = torch.zeros(2, dtype=torch.float64, device=env.device)
    action_sum = torch.zeros((2, 8), dtype=torch.float64, device=env.device)
    action_abs_sum = torch.zeros_like(action_sum)
    boost_level_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_consumed_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_starved = torch.zeros(2, dtype=torch.float64, device=env.device)
    speed_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    max_speed = torch.zeros(2, dtype=torch.float32, device=env.device)
    supersonic = torch.zeros(2, dtype=torch.float64, device=env.device)
    grounded = torch.zeros(2, dtype=torch.float64, device=env.device)
    distance_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    jump_edges = torch.zeros(2, dtype=torch.float64, device=env.device)
    flip_onsets = torch.zeros(2, dtype=torch.float64, device=env.device)
    previous_jump = torch.zeros(
        (EVALUATION_WORLDS, 2), dtype=torch.bool, device=env.device
    )
    position_scale = torch.tensor(
        POSITION_SCALE, dtype=torch.float32, device=env.device
    )
    for decision in range(MAX_EVALUATION_DECISIONS):
        observation = env.observation
        actor, _value = model(observation.reshape(-1, OBS_DIM))
        sample = sample_hybrid_action(
            actor.reshape(EVALUATION_WORLDS, 2, 13),
            generator=generator,
            config=trainer.policy_config,
        )
        action = torch.where(
            active[:, None, None], sample.action, torch.zeros_like(sample.action)
        )
        mask = active[:, None]
        mask3 = mask[..., None]
        mask_float = mask.to(torch.float32)
        action_sum += (action * mask3).sum(dim=0, dtype=torch.float64)
        action_abs_sum += action.abs().sum(dim=0, dtype=torch.float64)
        action_count += active.sum().double()
        current_jump = action[..., 5] > 0.5
        jump_edges += (
            current_jump & ~previous_jump & mask
        ).sum(dim=0, dtype=torch.float64)
        previous_jump.copy_(current_jump & mask)

        boost_before = observation[..., _SELF_BOOST_INDEX]
        boost_level_sum += (boost_before * mask_float).sum(
            dim=0, dtype=torch.float64
        )
        boost_starved += ((boost_before <= 0.01) & mask).sum(
            dim=0, dtype=torch.float64
        )
        velocity = (
            observation[
                ...,
                _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3,
            ]
            * CAR_LINEAR_SPEED_SCALE
        )
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        speed_sum += (speed * mask_float).sum(dim=0, dtype=torch.float64)
        max_speed = torch.maximum(
            max_speed,
            torch.where(mask, speed, torch.zeros_like(speed)).amax(dim=0),
        )
        supersonic += (
            (observation[..., _SELF_SUPERSONIC_INDEX] > 0.5) & mask
        ).sum(dim=0, dtype=torch.float64)
        grounded += (
            (observation[..., _SELF_ON_GROUND_INDEX] > 0.5) & mask
        ).sum(dim=0, dtype=torch.float64)
        relative = (
            observation[..., _RELATIVE_BALL_START : _RELATIVE_BALL_START + 3]
            * position_scale
        )
        distance_sum += (
            torch.linalg.vector_norm(relative, dim=-1) * mask_float
        ).sum(dim=0, dtype=torch.float64)
        flipped_before = observation[..., _SELF_HAS_FLIPPED_INDEX] > 0.5

        transition = env.step(action)
        episode_decisions += active.to(torch.int32)
        world_decisions += active.sum().double()
        transition_observation = transition.transition_observation
        touch = transition_observation[..., _SELF_TOUCH_INDEX] > 0.5
        side_touch += (touch & mask).sum(dim=0, dtype=torch.float64)
        any_touch = (touch & mask).any(dim=1)
        newly_touched = any_touch & (first_touch_latency < 0)
        first_touch_latency.copy_(
            torch.where(
                newly_touched,
                torch.full_like(first_touch_latency, decision + 1),
                first_touch_latency,
            )
        )
        boost_after = transition_observation[..., _SELF_BOOST_INDEX]
        boost_consumed_sum += (
            (boost_before - boost_after).clamp_min(0.0) * 100.0 * mask_float
        ).sum(dim=0, dtype=torch.float64)
        flipped_after = transition_observation[..., _SELF_HAS_FLIPPED_INDEX] > 0.5
        flip_onsets += (
            flipped_after & ~flipped_before & mask
        ).sum(dim=0, dtype=torch.float64)

        done = active & (transition.terminated | transition.truncated)
        goals += (done & transition.terminated).sum().double()
        no_touch_now = (
            done
            & transition.truncated
            & (transition_observation[:, 0, _NO_TOUCH_AGE_INDEX] >= 1.0)
        )
        no_touch += no_touch_now.sum().double()
        hard += (done & transition.truncated & ~no_touch_now).sum().double()
        completed |= done
        active &= ~done
        if not bool(active.any().item()):
            break

    torch.cuda.synchronize(env.device)
    episodes = int(completed.sum().item())
    decisions = float(world_decisions.item())
    simulated_minutes = decisions / 30.0 / 60.0
    contacted = first_touch_latency >= 0
    contact_count = int(contacted.sum().item())
    contacted_latency_seconds = (
        first_touch_latency[contacted].to(torch.float32) / 30.0
    )
    denominator = action_count.clamp_min(1.0)
    action_mean = action_sum / denominator[:, None]
    action_abs_mean = action_abs_sum / denominator[:, None]
    side_names = ("Blue", "Orange")
    movement: dict[str, Any] = {}
    for side_index, side in enumerate(side_names):
        movement[side] = {
            "average_boost_level": float(
                (boost_level_sum[side_index] / denominator[side_index]).item()
                * 100.0
            ),
            "net_observed_boost_consumed": float(
                boost_consumed_sum[side_index].item()
            ),
            "boost_starved_fraction": float(
                (boost_starved[side_index] / denominator[side_index]).item()
            ),
            "mean_speed_uu_per_s": float(
                (speed_sum[side_index] / denominator[side_index]).item()
            ),
            "maximum_speed_uu_per_s": float(max_speed[side_index].item()),
            "supersonic_fraction": float(
                (supersonic[side_index] / denominator[side_index]).item()
            ),
            "grounded_fraction": float(
                (grounded[side_index] / denominator[side_index]).item()
            ),
            "airborne_fraction": float(
                1.0 - (grounded[side_index] / denominator[side_index]).item()
            ),
            "mean_car_ball_distance_uu": float(
                (distance_sum[side_index] / denominator[side_index]).item()
            ),
            "jump_rising_edges": int(jump_edges[side_index].item()),
            "actual_flip_onsets": int(flip_onsets[side_index].item()),
            "flips_per_simulated_minute": float(
                flip_onsets[side_index].item() / simulated_minutes
            ),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": label,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "episode_version": RIVAL2_EPISODE_VERSION,
        "reward_version": RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        "result": {
            "completed_episodes": episodes,
            "goal_terminated_episodes": int(goals.item()),
            "no_touch_truncated_episodes": int(no_touch.item()),
            "hard_truncated_episodes": int(hard.item()),
            "no_touch_truncated_fraction": float(no_touch.item()) / episodes,
            "touches_per_simulated_minute": float(side_touch.sum().item())
            / simulated_minutes,
            "unique_touches": {
                "Blue": int(side_touch[0].item()),
                "Orange": int(side_touch[1].item()),
            },
            "goals_per_simulated_minute": float(goals.item())
            / simulated_minutes,
            "first_touch_latency": {
                "contacted_episodes": contact_count,
                "mean_seconds_contacted": (
                    float(contacted_latency_seconds.mean().item())
                    if contact_count
                    else None
                ),
                "median_seconds_contacted": (
                    float(contacted_latency_seconds.median().item())
                    if contact_count
                    else None
                ),
                "fraction_with_contact_within_seconds": {
                    str(seconds): float(
                        ((first_touch_latency > 0) & (first_touch_latency <= seconds * 30))
                        .sum()
                        .item()
                    )
                    / episodes
                    for seconds in (3, 5, 10, 15)
                },
            },
            "controller_mean": _side_channels(action_mean),
            "controller_mean_absolute": _side_channels(action_abs_mean),
            "movement": movement,
            "simulated_minutes": simulated_minutes,
        },
        "checks": {
            "all_worlds_completed_once": episodes == EVALUATION_WORLDS,
            "done_partition_exact": int(goals.item() + no_touch.item() + hard.item())
            == episodes,
            "no_five_minute_match_lifecycle": True,
            "finite_summary": all(
                np.isfinite(value)
                for value in (
                    float(side_touch.sum().item()),
                    float(goals.item()),
                    simulated_minutes,
                )
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    result["verdict"] = (
        "PASS_GREEN" if all(result["checks"].values()) else "FAIL_RED"
    )
    model.train(was_training)
    del env
    gc.collect()
    torch.cuda.empty_cache()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"acquisition evaluation failed: {result['checks']}")
    return result


def _checkpoint(label: str, trainer: Rival2Trainer, work_dir: Path) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_acquisition_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "label": label,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "reward_version": trainer.env.reward_version,
        "episode_version": trainer.env.episode_version,
        "contract_hashes": dict(trainer.env.contract_hashes),
    }


def publish(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    results = args.results_dir.resolve()
    if results.exists() and any(results.iterdir()):
        raise RuntimeError("results directory must be absent or empty")
    results.mkdir(parents=True, exist_ok=True)
    _write_json(results / "config.json", configuration)
    _write_json(results / "launch_gate.json", launch)
    for name in ("training_curve.jsonl", "evaluation_curve.json", "run_summary.json"):
        shutil.copy2(args.work_dir / name, results / name)
    for path in sorted(args.work_dir.glob("evaluation_update_*.json")):
        shutil.copy2(path, results / path.name)
    source = Path(summary["acquisition_checkpoint"]["path"])
    destination = Path("checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _write_json(
        results / "checkpoint.json",
        {
            "path": destination.as_posix(),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    configuration = frozen_configuration()
    launch = verify_launch(configuration)
    _write_json(args.work_dir / "config.json", configuration)
    _write_json(args.work_dir / "launch_gate.json", launch)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    torch.manual_seed(CAMPAIGN_SEED)
    torch.cuda.manual_seed(CAMPAIGN_SEED)
    kickoff_selector = (
        np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED
    ) % 5
    env = Rival2Env(
        WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03.campaign03_ppo_config(),
        seed=CAMPAIGN_SEED,
    )
    initialization_sha256 = campaign01._state_dict_sha256(
        {
            name: value.detach().cpu().clone()
            for name, value in trainer.model.state_dict().items()
        }
    )
    trainer.add_historical_snapshot()
    consecutive = 0
    evaluations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    ledger = args.work_dir / "training_curve.jsonl"
    started = time.perf_counter()
    while consecutive < REQUIRED_CONSECUTIVE:
        policy_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        env.reset_transfer_counters()
        update_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - update_started
        integrity = campaign01._rollout_integrity(
            trainer,
            rollout,
            metrics,
            policy_version_before=policy_before,
            samples_before=samples_before,
        )
        values = integrity["metrics"]
        point = {
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "wall_seconds": seconds,
            "agent_decisions_per_second": (
                trainer.total_agent_samples - samples_before
            )
            / seconds,
            "terminated_world_intervals": int(
                rollout.terminated[..., 0].sum().item()
            ),
            "truncated_world_intervals": int(
                rollout.truncated[..., 0].sum().item()
            ),
            "metrics": values,
            "integrity": integrity,
            "verdict": integrity["verdict"],
        }
        _append_jsonl(ledger, point)
        print(
            f"acquisition update={trainer.iteration} samples={trainer.total_agent_samples} "
            f"seconds={seconds:.3f} kl={values['approx_kl']:.6f} "
            f"clip={values['clip_fraction']:.6f} verdict={point['verdict']}",
            flush=True,
        )
        if point["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"training integrity failure at {trainer.iteration}")
        del rollout, metrics
        gc.collect()
        if trainer.iteration % EVALUATION_INTERVAL != 0:
            continue
        trainer.add_historical_snapshot()
        label = f"update_{trainer.iteration}"
        checkpoint = _checkpoint(label, trainer, args.work_dir)
        result = evaluate(
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            label=label,
        )
        fraction = result["result"]["no_touch_truncated_fraction"]
        passed = fraction <= NO_TOUCH_THRESHOLD
        consecutive = consecutive + 1 if passed else 0
        result["acquisition_gate"] = {
            "threshold": NO_TOUCH_THRESHOLD,
            "passed": passed,
            "consecutive_passing_evaluations": consecutive,
        }
        _write_json(args.work_dir / f"evaluation_{label}.json", result)
        evaluations.append(result)
        checkpoints.append(checkpoint)
        _write_json(args.work_dir / "evaluation_curve.json", evaluations)
        movement = result["result"]["movement"]
        latency = result["result"]["first_touch_latency"]
        print(
            f"acquisition evaluation={label} no_touch={fraction:.6f} "
            f"touches/min={result['result']['touches_per_simulated_minute']:.6f} "
            f"first_touch_mean={latency['mean_seconds_contacted']} "
            f"boost_blue={movement['Blue']['average_boost_level']:.3f} "
            f"boost_orange={movement['Orange']['average_boost_level']:.3f} "
            f"top_speed_blue={movement['Blue']['maximum_speed_uu_per_s']:.3f} "
            f"top_speed_orange={movement['Orange']['maximum_speed_uu_per_s']:.3f} "
            f"flips_blue={movement['Blue']['actual_flip_onsets']} "
            f"flips_orange={movement['Orange']['actual_flip_onsets']} "
            f"consecutive={consecutive}/{REQUIRED_CONSECUTIVE}",
            flush=True,
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "status": "ACQUISITION_COMPLETE",
        "source_head": launch["head"],
        "initialization_model_sha256": initialization_sha256,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "confirming_no_touch_fractions": [
            item["result"]["no_touch_truncated_fraction"]
            for item in evaluations[-REQUIRED_CONSECUTIVE:]
        ],
        "acquisition_checkpoint": checkpoints[-1],
        "evaluation_count": len(evaluations),
        "wall_seconds_including_evaluations": time.perf_counter() - started,
        "reward_transition_run": False,
        "nexto_training_run": False,
        "five_minute_matches_run": False,
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    publish(args, configuration, launch, summary)
    return summary


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError("work directory must be absent or empty")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise RuntimeError("results directory must be absent or empty")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    campaign01._initialize_runtime(args.device)
    summary = run(args)
    print(
        f"acquisition COMPLETE update={summary['final_iteration']} "
        f"samples={summary['final_agent_decision_samples']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
