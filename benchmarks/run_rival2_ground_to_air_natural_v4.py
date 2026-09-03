"""Train the passing aerial scorer on natural entries and a live defender.

The competitive V23 policies remain immutable.  A separately controlled aerial
option learns from low-bounce, incoming-chip, and matched-dribble training-pack
states.  Half of the training worlds contain a live, frozen V23 defender; the
other half retain an uncontested transfer path.  This is the last isolated
stage before gated full-match integration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as aerial_v1  # noqa: E402
from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_MIXED,
    DEFENDER_PARKED,
    GROUND_TO_AIR_NATURAL_V4_VERSION,
    SETUP_NAMES,
    build_natural_ground_to_air_scenarios,
)
from rivalsim.rival2_ground_to_air_option import (  # noqa: E402
    GroundToAirConfig,
    GroundToAirController,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_log_probability,
    sample_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V4"
AUTHORITY = ROOT / "results/rival2/ground_to_air_natural_v4/authority.json"
AUTHORITY_SHA256 = "17E5EB27B316BB048BD55DB7AA62DE4590A81DFF4753E7D2A624821BA4F9BB5A"
RESULTS = ROOT / "results/rival2/ground_to_air_natural_v4"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_natural_v4"
PARENT = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
PARENT_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-natural-v4")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_authority() -> dict[str, Any]:
    if capability.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("natural ground-to-air authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected natural ground-to-air authority format")
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"natural ground-to-air bound input changed: {path}")
    if float(authority["reward"]["raw_airtime_reward"]) != 0.0:
        raise RuntimeError("raw airtime reward is forbidden")
    if int(authority["integrity"]["optimizer_steps_before_authority_commit"]) != 0:
        raise RuntimeError("natural ground-to-air authority is not prospective")
    return authority


def make_model(payload: dict[str, Any], device: str) -> Rival2ActorCritic:
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.trunk.requires_grad_(True)
    model.actor.requires_grad_(True)
    model.critic.requires_grad_(False)
    return model


def make_optimizer(model: Rival2ActorCritic, authority: dict[str, Any]) -> torch.optim.AdamW:
    training = authority["training"]
    return torch.optim.AdamW(
        [
            {
                "params": model.trunk.parameters(),
                "lr": float(training["trunk_learning_rate"]),
            },
            {
                "params": model.actor.parameters(),
                "lr": float(training["actor_learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )


def distribution_override(authority: dict[str, Any]) -> HybridDistributionOverride:
    exploration = authority["training"]["exploration"]
    return HybridDistributionOverride(
        analog_log_std=float(torch.log(torch.tensor(exploration["analog_sigma"]))),
        button_temperature=float(exploration["button_temperature"]),
    )


def load_defender_policies(device: str) -> dict[int, Rival2ActorCritic]:
    policies: dict[int, Rival2ActorCritic] = {}
    for side, path, expected in (
        (0, BLUE, BLUE_SHA256),
        (1, ORANGE, ORANGE_SHA256),
    ):
        if capability.sha256_file(path) != expected:
            raise RuntimeError(f"protected V23 checkpoint changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        policy = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(
            device
        )
        policy.load_state_dict(payload["model"], strict=True)
        policy.eval().requires_grad_(False)
        policies[side] = policy
    return policies


def collect_rollout(
    model: Rival2ActorCritic,
    defenders: dict[int, Rival2ActorCritic],
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
    setup: int | None,
    defender_mode: str,
    live_defender_fraction: float,
    attacker_boost_range: tuple[float, float],
) -> tuple[aerial_v1.OptionRollout | None, dict[str, Any]]:
    batch = build_natural_ground_to_air_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
        setup=setup,
        defender_mode=defender_mode,
        live_defender_fraction=live_defender_fraction,
        attacker_boost_range=attacker_boost_range,
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed ^ side,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    config = GroundToAirConfig(
        **authority["option_config"],
        learned_after_second_jump=True,
    )
    controller = GroundToAirController(worlds, device=device, config=config)
    tracker = goal_v3.GoalDirectedTrainingTracker(
        worlds,
        attacker_side=side,
        horizon=horizon,
        authority=authority,
    )
    rollout = None if deterministic else aerial_v1.OptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    defender_active = torch.as_tensor(
        batch.defender_active, dtype=torch.bool, device=device
    )
    observation = env.observation
    false = torch.zeros(worlds, dtype=torch.bool, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    other = 1 - side
    defender = defenders[other]
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
            defender_actor, _ = defender(observation[:, other])
            defender_action = deterministic_hybrid_action(
                defender_actor, defender.config
            )
        option = controller.step(
            learned,
            observation[:, side],
            kickoff_active=false,
            match_done=~active_before,
        )
        learned_active = active_before & option.learned_control
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], option.action, 0.0)
        live = active_before & defender_active
        action[:, other] = torch.where(live[:, None], defender_action, 0.0)
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
        saturation += (
            (transition.emitted_action[:, side, :5].abs() > 0.95)
            & learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        analog_sum += (
            transition.emitted_action[:, side, :5] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
        button_sum += (
            transition.emitted_action[:, side, 5:] * learned_active[:, None]
        ).sum(dim=0, dtype=torch.float64)
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
    metrics = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "setup": "mixed" if setup is None else SETUP_NAMES[setup],
        "defender_mode": defender_mode,
        "live_defender_fraction": float(defender_active.float().mean().cpu()),
        "attacker_boost_range": list(attacker_boost_range),
        "telemetry": telemetry,
        "fractions": fractions,
        "reward_per_attempt": telemetry["reward_sum"] / worlds,
        "learned_action_ticks": int(action_count),
        "mean_learned_analog_action": (
            analog_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "learned_button_fraction": (
            button_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def validation_rows(
    model: Rival2ActorCritic,
    defenders: dict[int, Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    authority: dict[str, Any],
    worlds: int,
    seed: int,
    device: str,
    generators: list[torch.Generator],
    distribution: HybridDistributionOverride,
    collision_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon = int(authority["episode"]["horizon_ticks"])
    for setup in range(len(SETUP_NAMES)):
        for defender_mode in (DEFENDER_PARKED, DEFENDER_LIVE):
            for side in (0, 1):
                _unused, metrics = collect_rollout(
                    model,
                    defenders,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=worlds,
                    horizon=horizon,
                    seed=seed + setup * 100_000 + (10_000 if defender_mode == DEFENDER_LIVE else 0),
                    device=device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=True,
                    collision_dir=collision_dir,
                    setup=setup,
                    defender_mode=defender_mode,
                    live_defender_fraction=1.0 if defender_mode == DEFENDER_LIVE else 0.0,
                    attacker_boost_range=tuple(authority["scenario"]["validation_boost_range"]),
                )
                rows.append(metrics)
    return rows


def evaluation_score(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        f = row["fractions"]
        weight = 1.5 if row["defender_mode"] == DEFENDER_LIVE else 1.0
        total += weight * (
            30.0 * f["goal_within_contact_budget"]
            + 5.0 * f["productive_continuation"]
            + 4.0 * f["sustained_control"]
            + 3.0 * f["second_airborne_touch"]
            + 2.0 * f["high_follow_touch"]
            + f["elevated_follow_touch"]
            + 0.25 * f["goalward_velocity_contact"]
            - 10.0 * f["contact_budget_exceeded"]
        )
    return float(total / len(rows))


def passes_gate(rows: list[dict[str, Any]], authority: dict[str, Any]) -> bool:
    acceptance = authority["acceptance"]
    mode_gate = acceptance["per_defender_mode"]
    for row in rows:
        f = row["fractions"]
        gate = mode_gate[row["defender_mode"]]
        for name in (
            "pop_touch",
            "elevated_follow_touch",
            "high_follow_touch",
            "productive_continuation",
            "goal_within_contact_budget",
        ):
            if f[name] < float(gate[f"{name}_fraction_min"]):
                return False
        if f["contact_budget_exceeded"] > float(
            acceptance["contact_budget_exceeded_fraction_max"]
        ):
            return False
        if f["unassisted_or_ground_goal"] > float(
            acceptance["unassisted_or_ground_goal_fraction_max"]
        ):
            return False
        if not row["finite"] or max(row["analog_saturation_fraction"]) >= float(
            acceptance["maximum_analog_saturation_fraction"]
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
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V4_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": GROUND_TO_AIR_NATURAL_V4_VERSION,
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "ground_to_air_goal_v3_parent_sha256": PARENT_SHA256,
        "protected_v23_defenders": {
            "blue": BLUE_SHA256,
            "orange": ORANGE_SHA256,
        },
        "canonical_shared_option": True,
        "accepted_block": block,
        "evaluation": evaluation,
        "critic_frozen": True,
        "natural_setup_families": list(SETUP_NAMES),
        "live_defender_training": True,
        "production_reward_unchanged": True,
        "ppo_resumable_as_general_policy": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": path.as_posix(),
        "sha256": capability.sha256_file(path),
        "model_tensor_sha256": autonomous.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "block": block,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    if capability.sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("controlled aerial scorer parent changed")
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = make_model(source, args.device)
    defenders = load_defender_policies(args.device)
    optimizer = make_optimizer(model, authority)
    critic_hash = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(
            int(authority["seeds"]["optimizer_and_exploration"]) ^ side
        )
        for side in (0, 1)
    ]
    distribution = distribution_override(authority)
    baseline = validation_rows(
        model,
        defenders,
        geometry,
        meshes,
        authority=authority,
        worlds=args.evaluation_worlds_per_row,
        seed=int(authority["seeds"]["validation"]),
        device=args.device,
        generators=generators,
        distribution=distribution,
        collision_dir=args.collision_dir,
    )
    preflight = {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "parent_hash_verified": True,
        "protected_v23_defender_hashes_verified": True,
        "critic_frozen": True,
        "production_reward_unchanged": True,
        "raw_airtime_reward": authority["reward"]["raw_airtime_reward"],
        "optimizer_steps": 0,
        "baseline_validation": baseline,
        "baseline_score": evaluation_score(baseline),
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("natural ground-to-air training requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    best_score = evaluation_score(baseline)
    best_block = 0
    best_evaluation = copy.deepcopy(baseline)
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    stale_boundaries = 0
    consecutive_gate = 0
    stop_reason = "maximum_blocks"
    training = authority["training"]
    compatibility = goal_v3.ppo_compatibility(authority)
    maximum_blocks = min(int(training["maximum_blocks"]), int(args.maximum_blocks))
    interval = int(training["evaluation_interval_blocks"])
    horizon = int(authority["episode"]["horizon_ticks"])
    for block in range(1, maximum_blocks + 1):
        block_model = copy.deepcopy(model.state_dict())
        block_optimizer = copy.deepcopy(optimizer.state_dict())
        sides: list[dict[str, Any]] = []
        try:
            for side in (0, 1):
                rollout, rollout_metrics = collect_rollout(
                    model,
                    defenders,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=args.worlds_per_side,
                    horizon=horizon,
                    seed=int(authority["seeds"]["training"]) + block * 100,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=False,
                    collision_dir=args.collision_dir,
                    setup=None,
                    defender_mode=DEFENDER_MIXED,
                    live_defender_fraction=float(
                        authority["scenario"]["training_live_defender_fraction"]
                    ),
                    attacker_boost_range=tuple(
                        authority["scenario"]["training_boost_range"]
                    ),
                )
                assert rollout is not None
                ppo = aerial_v1.option_ppo_update(
                    model,
                    optimizer,
                    rollout,
                    generator=generators[side],
                    distribution=distribution,
                    authority=compatibility,
                )
                rehearsal = goal_v3.successful_rehearsal_update(
                    model,
                    optimizer,
                    rollout,
                    authority=authority,
                    generator=generators[side],
                )
                goal_rehearsal = goal_v3.successful_rehearsal_update(
                    model,
                    optimizer,
                    rollout,
                    authority=authority,
                    generator=generators[side],
                    rehearsal_key="goal_rehearsal",
                )
                del rollout
                if not all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("nonfinite natural aerial-option parameter")
                sides.append(
                    {
                        "side": side,
                        "rollout": rollout_metrics,
                        "ppo": ppo,
                        "successful_rehearsal": rehearsal,
                        "goal_rehearsal": goal_rehearsal,
                    }
                )
        except (FloatingPointError, RuntimeError) as error:
            model.load_state_dict(block_model, strict=True)
            optimizer.load_state_dict(block_optimizer)
            stop_reason = f"hard_failure:{type(error).__name__}:{error}"
            append_jsonl(
                RESULTS / "hard_failure.jsonl",
                {"block": block, "created_utc": utc_now(), "reason": stop_reason},
            )
            break
        row: dict[str, Any] = {"block": block, "sides": sides}
        if block % interval == 0 or block == maximum_blocks:
            validation = validation_rows(
                model,
                defenders,
                geometry,
                meshes,
                authority=authority,
                worlds=args.evaluation_worlds_per_row,
                seed=int(authority["seeds"]["validation"]),
                device=args.device,
                generators=generators,
                distribution=distribution,
                collision_dir=args.collision_dir,
            )
            score = evaluation_score(validation)
            passed = passes_gate(validation, authority)
            improved = score > best_score + float(training["minimum_score_improvement"])
            if improved:
                best_score = score
                best_block = block
                best_evaluation = copy.deepcopy(validation)
                best_model = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                stale_boundaries = 0
            else:
                stale_boundaries += 1
            consecutive_gate = consecutive_gate + 1 if passed else 0
            row["evaluation"] = {
                "validation": validation,
                "score": score,
                "passed": passed,
                "improved": improved,
                "stale_boundaries": stale_boundaries,
            }
            print(
                json.dumps(
                    {
                        "stage": "ground_to_air_natural_v4",
                        "block": block,
                        "score": score,
                        "best_block": best_block,
                        "passed": passed,
                        "rows": [
                            {
                                "setup": entry["setup"],
                                "defender": entry["defender_mode"],
                                "side": entry["side"],
                                "fractions": entry["fractions"],
                            }
                            for entry in validation
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            save_checkpoint(
                source,
                model,
                optimizer,
                run_dir / "rolling.pt",
                block=block,
                evaluation=row["evaluation"],
            )
            if consecutive_gate >= int(authority["acceptance"]["consecutive_boundaries"]):
                stop_reason = "validation_gate_passed"
                append_jsonl(curve, row)
                break
            if stale_boundaries >= int(training["plateau_patience_boundaries"]):
                stop_reason = "validation_plateau"
                append_jsonl(curve, row)
                break
        append_jsonl(curve, row)

    model.load_state_dict(best_model, strict=True)
    optimizer.load_state_dict(best_optimizer)
    observed_critic = autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in model.critic.state_dict().items()}
    )
    if observed_critic != critic_hash:
        raise RuntimeError("natural aerial option changed the frozen critic")
    validation_pass = passes_gate(best_evaluation, authority)
    test: list[dict[str, Any]] | None = None
    if validation_pass:
        test = validation_rows(
            model,
            defenders,
            geometry,
            meshes,
            authority=authority,
            worlds=args.test_worlds_per_row,
            seed=int(authority["seeds"]["test"]),
            device=args.device,
            generators=generators,
            distribution=distribution,
            collision_dir=args.collision_dir,
        )
    controlled_pass = bool(
        validation_pass and test is not None and passes_gate(test, authority)
    )
    checkpoints: list[dict[str, Any]] = []
    if controlled_pass:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        checkpoints.append(
            save_checkpoint(
                source,
                model,
                optimizer,
                CHECKPOINTS / "rival2_ground_to_air_natural_v4.pt",
                block=best_block,
                evaluation={"validation": best_evaluation, "test": test},
            )
        )
    result = {
        "format": f"{VERSION}_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_validation": baseline,
        "best_block": best_block,
        "best_validation": best_evaluation,
        "best_score": best_score,
        "untouched_test_opened": test is not None,
        "untouched_test": test,
        "stop_reason": stop_reason,
        "controlled_pass": controlled_pass,
        "parent_unchanged": capability.sha256_file(PARENT) == PARENT_SHA256,
        "protected_v23_unchanged": (
            capability.sha256_file(BLUE) == BLUE_SHA256
            and capability.sha256_file(ORANGE) == ORANGE_SHA256
        ),
        "critic_unchanged": True,
        "production_reward_unchanged": True,
        "checkpoints": checkpoints,
        "promoted_into_competitive_policy": False,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if controlled_pass else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-side", type=int, default=1_024)
    parser.add_argument("--evaluation-worlds-per-row", type=int, default=256)
    parser.add_argument("--test-worlds-per-row", type=int, default=512)
    parser.add_argument("--maximum-blocks", type=int, default=96)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
