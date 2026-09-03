"""Train the validated aerial scorer to continue after a real ground-ball pop.

The source launch primitive is fixed by the ground-ball calibration.  Only the
policy actions after that primitive are optimized, using literal physical state
and contact events.  This file does not modify Rival's production reward.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as aerial_v1  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_ball_pop import (  # noqa: E402
    PrecontactPopConfig,
    PrecontactPopController,
    build_ground_ball_pop_scenarios,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    deterministic_hybrid_action,
    hybrid_log_probability,
    sample_hybrid_action,
)

VERSION = "RIVAL2_GROUND_BALL_AERIAL_CHAIN_V1"
AUTHORITY = ROOT / "results/rival2/ground_ball_aerial_chain_v1/authority.json"
AUTHORITY_SHA256 = "F3DE606FC5E0EA6FDBB8FA9494EA30D20735BEFCAAE0EEB4E5C150D84F4B1BE7"
RESULTS = ROOT / "results/rival2/ground_ball_aerial_chain_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_ball_aerial_chain_v1"
PARENT = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
PARENT_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-ball-aerial-chain-v1")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")


def source_config(authority: dict[str, Any]) -> PrecontactPopConfig:
    return PrecontactPopConfig(**authority["source_launch"]["selected_config"])


def collect_rollout(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    authority: dict[str, Any],
    side: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
    collision_dir: Path,
    handoff_tick: int | None = None,
    phase: int = 0,
) -> tuple[aerial_v1.OptionRollout | None, dict[str, Any]]:
    """Collect post-launch policy experience from ordinary ground balls."""

    del handoff_tick, phase
    initial = build_ground_ball_pop_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed ^ side,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    controller = PrecontactPopController(
        worlds,
        device=device,
        config=source_config(authority),
    )
    tracker = goal_v3.GoalDirectedTrainingTracker(
        worlds,
        attacker_side=side,
        horizon=horizon,
        authority=authority,
    )
    rollout = None if deterministic else aerial_v1.OptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    maximum_ball_height = torch.full((worlds,), 92.75, dtype=torch.float32, device=device)
    launch_started = torch.zeros(worlds, dtype=torch.bool, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            if deterministic:
                learned = deterministic_hybrid_action(actor, model.config)
                pre_tanh = actor[:, :5]
            else:
                sample = sample_hybrid_action(
                    actor,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution,
                )
                learned = sample.action
                pre_tanh = sample.pre_tanh
            old_log_probability = hybrid_log_probability(
                actor,
                learned,
                config=model.config,
                pre_tanh=pre_tanh,
                distribution_override=distribution,
            )
        source = controller.step(learned, observation[:, side])
        learned_active = active_before & source.learned_control
        launch_started |= active_before & source.launch_started
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], source.action, 0.0)
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == side)
        reward, skill_done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=active_before & transition.terminated,
            active=active_before,
        )
        terminal = skill_done | transition.terminated | transition.truncated
        if rollout is not None:
            rollout.observation[tick].copy_(observation[:, side])
            rollout.action[tick].copy_(transition.emitted_action[:, side])
            rollout.pre_tanh[tick].copy_(pre_tanh)
            rollout.old_log_probability[tick].copy_(old_log_probability)
            rollout.reward[tick].copy_(reward)
            rollout.done[tick].copy_(terminal)
            rollout.mask[tick].copy_(learned_active)
        ball_height = (
            transition.transition_observation[:, side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        )
        maximum_ball_height = torch.maximum(maximum_ball_height, ball_height)
        saturation += (
            (transition.emitted_action[:, side, :5].abs() > 0.95) & learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        analog_sum += (transition.emitted_action[:, side, :5] * learned_active[:, None]).sum(
            dim=0, dtype=torch.float64
        )
        button_sum += (transition.emitted_action[:, side, 5:] * learned_active[:, None]).sum(
            dim=0, dtype=torch.float64
        )
        action_count += learned_active.sum(dtype=torch.float64)
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    if rollout is not None and bool(active.any()):
        rollout.done[min(tick, horizon - 1)] |= active
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    telemetry = tracker.telemetry()
    fractions = {
        key: telemetry[name] / worlds
        for key, name in (
            ("pop_touch", "low_pop_touches"),
            ("elevated_follow_touch", "elevated_follow_touches"),
            ("high_follow_touch", "high_follow_touches"),
            ("second_airborne_touch", "second_airborne_touches"),
            ("third_airborne_touch", "third_airborne_touches"),
            ("fourth_airborne_touch", "fourth_airborne_touches"),
            ("fifth_airborne_touch", "fifth_airborne_touches"),
            ("contact_budget_exceeded", "contact_budget_exceeded"),
            ("goal_within_contact_budget", "goals_within_contact_budget"),
            ("goal_over_contact_budget", "goals_over_contact_budget"),
            ("sustained_control", "sustained_control_attempts"),
            ("productive_continuation", "productive_continuation_attempts"),
            ("unassisted_or_ground_goal", "unassisted_or_ground_goals"),
            ("goalward_velocity_contact", "goalward_velocity_contacts"),
        )
    }
    fractions.update(
        {
            "source_launch": float(launch_started.float().mean()),
            "ball_rise_180": float((maximum_ball_height >= 180.0).float().mean()),
            "ball_rise_250": float((maximum_ball_height >= 250.0).float().mean()),
            "ball_rise_400": float((maximum_ball_height >= 400.0).float().mean()),
        }
    )
    metrics = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "source_launch_config": authority["source_launch"]["selected_config"],
        "telemetry": telemetry,
        "fractions": fractions,
        "maximum_ball_height_uu": {
            "p50": float(torch.quantile(maximum_ball_height, 0.5)),
            "p90": float(torch.quantile(maximum_ball_height, 0.9)),
            "maximum": float(maximum_ball_height.max()),
        },
        "reward_per_attempt": telemetry["reward_sum"] / worlds,
        "learned_action_ticks": int(action_count),
        "mean_learned_analog_action": (analog_sum / action_count.clamp_min(1.0)).cpu().tolist(),
        "learned_button_fraction": (button_sum / action_count.clamp_min(1.0)).cpu().tolist(),
        "analog_saturation_fraction": (saturation / action_count.clamp_min(1.0)).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def evaluation_score(rows: list[dict[str, Any]]) -> float:
    minimum = {
        name: min(row["fractions"][name] for row in rows)
        for name in (
            "ball_rise_250",
            "elevated_follow_touch",
            "high_follow_touch",
            "second_airborne_touch",
            "productive_continuation",
            "goal_within_contact_budget",
        )
    }
    return float(
        25.0 * minimum["goal_within_contact_budget"]
        + 8.0 * minimum["elevated_follow_touch"]
        + 6.0 * minimum["second_airborne_touch"]
        + 4.0 * minimum["productive_continuation"]
        + 3.0 * minimum["high_follow_touch"]
        + 0.25 * minimum["ball_rise_250"]
        + 0.05 * sum(row["reward_per_attempt"] for row in rows) / len(rows)
    )


def passes_gate(rows: list[dict[str, Any]], authority: dict[str, Any]) -> bool:
    gate = authority["acceptance"]
    minima = {
        "source_launch": "source_launch_fraction_min",
        "pop_touch": "pop_touch_fraction_min",
        "ball_rise_180": "ball_rise_180_fraction_min",
        "ball_rise_250": "ball_rise_250_fraction_min",
        "elevated_follow_touch": "elevated_follow_touch_fraction_min",
        "high_follow_touch": "high_follow_touch_fraction_min",
        "second_airborne_touch": "second_airborne_touch_fraction_min",
        "productive_continuation": "productive_continuation_fraction_min",
        "goal_within_contact_budget": "goal_within_contact_budget_fraction_min",
    }
    for row in rows:
        fractions = row["fractions"]
        if any(fractions[name] < float(gate[key]) for name, key in minima.items()):
            return False
        if fractions["contact_budget_exceeded"] > float(
            gate["contact_budget_exceeded_fraction_max"]
        ):
            return False
        if fractions["unassisted_or_ground_goal"] > float(
            gate["unassisted_or_ground_goal_fraction_max"]
        ):
            return False
        if not row["finite"] or max(row["analog_saturation_fraction"]) >= float(
            gate["maximum_analog_saturation_fraction"]
        ):
            return False
    return True


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    block: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_GROUND_BALL_AERIAL_CHAIN_V1_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": VERSION,
        "created_utc": goal_v3.utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "controlled_aerial_parent_sha256": PARENT_SHA256,
        "protected_v23_ancestor_sha256": ORANGE_SHA256,
        "canonical_shared_policy": True,
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "production_reward_unchanged": True,
        "ppo_resumable_as_general_policy": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": path.as_posix(),
        "sha256": goal_v3.capability.sha256_file(path),
        "model_tensor_sha256": autonomous.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "block": block,
    }


def configure_goal_runner() -> None:
    """Bind the proven V3 trainer loop to this prospective curriculum."""

    goal_v3.AUTHORITY = AUTHORITY
    goal_v3.AUTHORITY_SHA256 = AUTHORITY_SHA256
    goal_v3.RESULTS = RESULTS
    goal_v3.CHECKPOINTS = CHECKPOINTS
    goal_v3.PARENT = PARENT
    goal_v3.PARENT_SHA256 = PARENT_SHA256
    goal_v3.BLUE = BLUE
    goal_v3.ORANGE = ORANGE
    goal_v3.BLUE_SHA256 = BLUE_SHA256
    goal_v3.ORANGE_SHA256 = ORANGE_SHA256
    goal_v3.DEFAULT_RUN_DIR = DEFAULT_RUN_DIR
    goal_v3.DEFAULT_COLLISION_DIR = DEFAULT_COLLISION_DIR
    goal_v3.GROUND_TO_AIR_GOAL_V3_VERSION = VERSION
    goal_v3.collect_rollout = collect_rollout
    goal_v3.evaluation_score = evaluation_score
    goal_v3.passes_gate = passes_gate
    goal_v3.save_checkpoint = save_checkpoint


def run(args: argparse.Namespace) -> int:
    configure_goal_runner()
    code = goal_v3.run(args)
    if (RESULTS / "preflight.json").exists():
        preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
        preflight.update(
            {
                "format": "RIVAL2_GROUND_BALL_AERIAL_CHAIN_V1_PREFLIGHT",
                "source_launch_calibration_sha256": (
                    "4E4CB2D0A80A90C082FE01D0DE4F89E135608383EB48BD9483B0B9E76BFC943B"
                ),
                "source_launch_candidate": 5,
            }
        )
        goal_v3.write_json(RESULTS / "preflight.json", preflight)
    if not args.preflight_only and (RESULTS / "result.json").exists():
        result = json.loads((RESULTS / "result.json").read_text(encoding="utf-8"))
        result.update(
            {
                "format": "RIVAL2_GROUND_BALL_AERIAL_CHAIN_V1_RESULT",
                "source_launch_calibration_sha256": (
                    "4E4CB2D0A80A90C082FE01D0DE4F89E135608383EB48BD9483B0B9E76BFC943B"
                ),
                "controlled_aerial_parent_sha256": PARENT_SHA256,
                "promoted_into_competitive_policy": False,
            }
        )
        goal_v3.write_json(RESULTS / "result.json", result)
    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-side", type=int, default=1_024)
    parser.add_argument("--evaluation-worlds-per-side", type=int, default=1_024)
    parser.add_argument("--test-worlds-per-side", type=int, default=2_048)
    parser.add_argument("--maximum-blocks", type=int, default=160)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
