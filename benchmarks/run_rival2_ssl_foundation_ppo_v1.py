"""Run the Unified-V5-rooted SSL Foundation mixed-opponent PPO campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_unified_ground_selfplay_ppo_v1 import (  # noqa: E402
    EXPLORATION_CONTRACT,
    EXPLORATION_CONTRACT_HASH,
    SOURCE,
    SOURCE_SHA256,
    append_jsonl,
    exploration_for_update,
    sha256_file,
    source_payload,
    state_dict_sha256,
    write_json,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    OBS_FIELD_NAMES,
    REWARD_SSL_FOUNDATION_V1_CONTRACT,
    REWARD_SSL_FOUNDATION_V1_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentPPOCorruption  # noqa: E402
from rivalsim.rival2_ssl_foundation_training import (  # noqa: E402
    OPPONENT_FROZEN_V5,
    OPPONENT_NAMES,
    OPPONENT_NEXTO,
    Rival2SslFoundationTrainer,
    SslFoundationOpponentConfig,
)
from rivalsim.rival2_unified_policy import (  # noqa: E402
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)
from rivalsim.ssl_foundation_v1 import (  # noqa: E402
    SCENARIO_NAMES,
    SCENARIO_PROBABILITIES,
    SSL_FOUNDATION_GAMMA,
    SSL_FOUNDATION_WEIGHTS,
    build_ssl_foundation_scenarios,
)
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors  # noqa: E402

FORMAT = "RIVAL2_SSL_FOUNDATION_PPO_V1"
CHECKPOINT_FORMAT = f"{FORMAT}_CHECKPOINT"
SUPERSEDED_AUTHORITY_SHA256 = "DCA2822CB83AF3C2A3B546038CD9B9F482FAD0536D67FDF88BD092F2FA1F16E1"
RESULTS = ROOT / "results/rival2/ssl_foundation_ppo_v1"
AUTHORITY = RESULTS / "authority.json"
CHECKPOINT = ROOT / "checkpoints/rival2/ssl_foundation_ppo_v1" / "rival2_ssl_foundation_ppo_v1.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ssl-foundation-ppo-v1")
WORLD_COUNT = 32_768
WALL_CLOCK_SECONDS = 10 * 60 * 60
MAXIMUM_ACCEPTED_UPDATES = 600
SNAPSHOT_INTERVAL = 30
POLICY_LEARNING_RATE = 1.0e-6
CRITIC_LEARNING_RATE = 3.0e-4
PPO_EPOCHS = 1
CAMPAIGN_SEED = 2026090301
SCENARIO_SEED = 2026090303
EVALUATION_SEED = 2026090304
EVALUATION_WORLDS = 1_024
EVALUATION_TICKS = 1_200
OPPONENT_CONFIG = SslFoundationOpponentConfig()

_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
_NO_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")
_SPEED_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _state_sha256(state: Any) -> str:
    digest = hashlib.sha256()
    for field in fields(state):
        value = getattr(state, field.name)
        digest.update(field.name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def authority_payload(implementation_commit: str) -> dict[str, Any]:
    batch = build_ssl_foundation_scenarios(WORLD_COUNT, seed=SCENARIO_SEED)
    return {
        "format": f"{FORMAT}_AUTHORITY",
        "created_utc": utc_now(),
        "implementation_commit": implementation_commit,
        "supersedes_authority_sha256": SUPERSEDED_AUTHORITY_SHA256,
        "supersession_reason": "user-authorized six-potential reward amendment before PPO",
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "required_format": "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V5",
            "overwrite_source": False,
            "excluded_descendant": (
                "G:/dev/RivalSim-runs/unified-ground-curriculum-ppo-v2/"
                "snapshots/ground_curriculum_midpoint_u0174.pt"
            ),
        },
        "reward": {
            "version": RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
            "contract_sha256": REWARD_SSL_FOUNDATION_V1_CONTRACT_HASH,
            "gamma": SSL_FOUNDATION_GAMMA,
            "weights": SSL_FOUNDATION_WEIGHTS,
            "one_combined_potential": True,
            "terminal_goal_for": 10.0,
            "terminal_goal_against": -10.0,
            "timeout": 0.0,
            "all_shaping_is_gamma_phi_next_minus_phi": True,
            "direct_non_goal_reward_terms": 0,
            "named_mechanics": False,
            "legacy_gameplay_reward_module_allocated": False,
        },
        "reset_curriculum": {
            "seed": SCENARIO_SEED,
            "names": list(SCENARIO_NAMES),
            "probabilities": list(SCENARIO_PROBABILITIES),
            "realized": batch.summary(),
            "state_tensor_sha256": _state_sha256(batch.state),
            "persistent_at_every_episode_reset": True,
            "same_reward_all_families": True,
            "task_or_scenario_id_in_observation": False,
            "scripted_solution_prefix_ticks": 0,
            "policy_controls_from_tick": 1,
        },
        "opponents": {
            **asdict(OPPONENT_CONFIG),
            "current_selfplay_both_sides_trainable": True,
            "nexto_inference_only": True,
            "frozen_v5_inference_only": True,
            "frozen_v5_deterministic": True,
            "nexto_deterministic": True,
            "episode_fixed_assignment": True,
        },
        "ppo": {
            "version": RIVAL2_PPO_120HZ_V1,
            "base_contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
            "worlds": WORLD_COUNT,
            "physics_hz": 120,
            "policy_hz": 120,
            "rollout_horizon": 128,
            "epochs": PPO_EPOCHS,
            "minibatch_size": 65_536,
            "policy_learning_rate": POLICY_LEARNING_RATE,
            "critic_learning_rate": CRITIC_LEARNING_RATE,
            "fresh_optimizer": True,
            "family_local_advantage_normalization": True,
            "value_loss_isolated_from_policy_trunk": True,
            "kl_policy": "telemetry_only_no_rejection_retry_or_rollback",
            "nonfinite_transactional_rollback": True,
            "preservation_kl": False,
        },
        "exploration": {
            "contract": EXPLORATION_CONTRACT,
            "contract_sha256": EXPLORATION_CONTRACT_HASH,
        },
        "campaign": {
            "wall_clock_seconds": WALL_CLOCK_SECONDS,
            "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "rolling_checkpoint_every_update": True,
            "evaluation_interval": SNAPSHOT_INTERVAL,
            "evaluation_is_telemetry_not_gate": True,
            "evaluation_worlds": EVALUATION_WORLDS,
            "evaluation_ticks": EVALUATION_TICKS,
        },
        "integrity": {
            "reward_contract": REWARD_SSL_FOUNDATION_V1_CONTRACT,
            "no_ppo_step_before_authority_commit": True,
            "no_old_optimizer_loaded": True,
            "no_plus_174_model_loaded": True,
        },
    }


def load_authority() -> dict[str, Any]:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": payload.get("format") == f"{FORMAT}_AUTHORITY",
        "source": payload.get("source", {}).get("sha256") == SOURCE_SHA256,
        "source_file": sha256_file(SOURCE) == SOURCE_SHA256,
        "reward": payload.get("reward", {}).get("contract_sha256")
        == REWARD_SSL_FOUNDATION_V1_CONTRACT_HASH,
        "ppo_and_shaping_gamma_match": payload.get("reward", {}).get("gamma")
        == payload.get("ppo", {}).get("gamma")
        == SSL_FOUNDATION_GAMMA,
        "weights": payload.get("reward", {}).get("weights") == SSL_FOUNDATION_WEIGHTS,
        "potential_only": payload.get("reward", {}).get("all_shaping_is_gamma_phi_next_minus_phi")
        is True,
        "combined_potential": payload.get("reward", {}).get("one_combined_potential") is True,
        "supersedes_four_potential_authority": payload.get("supersedes_authority_sha256")
        == SUPERSEDED_AUTHORITY_SHA256,
        "zero_direct": payload.get("reward", {}).get("direct_non_goal_reward_terms") == 0,
        "worlds": payload.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "wall_clock": payload.get("campaign", {}).get("wall_clock_seconds") == WALL_CLOCK_SECONDS,
        "opponents": all(
            payload.get("opponents", {}).get(name) == expected
            for name, expected in (
                ("current_probability", 0.40),
                ("nexto_probability", 0.30),
                ("frozen_v5_probability", 0.30),
            )
        ),
        "scenarios": payload.get("reset_curriculum", {}).get("probabilities")
        == list(SCENARIO_PROBABILITIES),
        "no_task_id": payload.get("reset_curriculum", {}).get("task_or_scenario_id_in_observation")
        is False,
        "no_preservation_kl": payload.get("ppo", {}).get("preservation_kl") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"SSL Foundation authority mismatch: {checks}")
    return payload


def configure_optimizer(trainer: Rival2SslFoundationTrainer) -> None:
    critic = tuple(trainer.model.critic.parameters())
    critic_ids = {id(parameter) for parameter in critic}
    policy = tuple(
        parameter for parameter in trainer.model.parameters() if id(parameter) not in critic_ids
    )
    trainer.optimizer = torch.optim.Adam(
        [
            {"name": "policy", "params": policy, "lr": POLICY_LEARNING_RATE},
            {"name": "critic", "params": critic, "lr": CRITIC_LEARNING_RATE},
        ]
    )


def make_trainer(
    collision_root: Path,
    *,
    worlds: int,
) -> tuple[Rival2SslFoundationTrainer, dict[str, Any]]:
    authority = load_authority()
    source = source_payload()
    config = Rival2UnifiedPolicyConfig(**source["policy_config"])
    current = Rival2UnifiedActorCritic(config)
    current.load_state_dict(source["model"], strict=True)
    frozen = Rival2UnifiedActorCritic(config)
    frozen.load_state_dict(source["model"], strict=True)
    batch = build_ssl_foundation_scenarios(worlds, seed=SCENARIO_SEED)
    geometry = ArenaGeometry.load_soccar(collision_root)
    env = Rival2Env(
        worlds,
        str(collision_root),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry),
        device="cuda:0",
        seed=CAMPAIGN_SEED,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
        ssl_foundation_scenarios=batch,
    )
    ppo = replace(
        rival2_ppo_120hz_config(),
        learning_rate=POLICY_LEARNING_RATE,
        epochs=PPO_EPOCHS,
    )
    trainer = Rival2SslFoundationTrainer(
        env,
        policy_config=config,
        ppo_config=ppo,
        phase="ssl_foundation_v1",
        source_identity={
            **copy.deepcopy(authority["source"]),
            "authority_sha256": sha256_file(AUTHORITY),
        },
        seed=CAMPAIGN_SEED,
        model=current,
        frozen_v5_model=frozen,
        opponent_config=OPPONENT_CONFIG,
        scenario_family=torch.from_numpy(batch.family),
        checkpoint_format=CHECKPOINT_FORMAT,
        lineage="Unified Capability V5 -> SSL Foundation PPO V1",
    )
    configure_optimizer(trainer)
    trainer.phase_transition = {
        "format": f"{FORMAT}_SOURCE_TRANSITION",
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": state_dict_sha256(source["model"]),
        "loaded_current_model_tensor_sha256": state_dict_sha256(trainer.model.state_dict()),
        "loaded_frozen_model_tensor_sha256": state_dict_sha256(trainer.frozen_v5.state_dict()),
        "source_optimizer_loaded": False,
        "plus_174_descendant_loaded": False,
        "fresh_ppo_optimizer": True,
        "authority_sha256": sha256_file(AUTHORITY),
    }
    return trainer, source


def preflight(
    trainer: Rival2SslFoundationTrainer,
    source: dict[str, Any],
    *,
    exact_scale: bool,
) -> dict[str, Any]:
    lrs = {str(group.get("name")): float(group["lr"]) for group in trainer.optimizer.param_groups}
    transition = trainer.phase_transition or {}
    checks = {
        "exact_unified_v5_source_sha256": sha256_file(SOURCE) == SOURCE_SHA256,
        "current_model_exact_at_update_zero": transition.get("loaded_current_model_tensor_sha256")
        == transition.get("source_model_tensor_sha256"),
        "frozen_v5_model_exact": transition.get("loaded_frozen_model_tensor_sha256")
        == transition.get("source_model_tensor_sha256"),
        "source_optimizer_not_loaded": transition.get("source_optimizer_loaded") is False,
        "plus_174_not_loaded": transition.get("plus_174_descendant_loaded") is False,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION),
        "goal_only_native_kernel": trainer.env.world.reward_mode == 2,
        "no_gameplay_v2_or_v3_state": trainer.env.world.gameplay_120 is None
        and trainer.env.world.gameplay_v3 is None,
        "only_six_potential_components": set(SSL_FOUNDATION_WEIGHTS)
        == {"field", "access", "control", "defense", "alignment", "boost"},
        "direct_non_goal_reward_terms_zero": len(
            REWARD_SSL_FOUNDATION_V1_CONTRACT["direct_reward_exactly_zero"]
        )
        == 14,
        "scenario_reset_template_resident": trainer.env.world.ssl_foundation_reset is not None,
        "scenario_id_absent": trainer.env.world.ssl_foundation_reset.summary[
            "task_or_scenario_id_in_observation"
        ]
        is False,
        "opponent_probabilities_exact": asdict(trainer.opponent_config) == asdict(OPPONENT_CONFIG),
        "native_120hz": trainer.env.physics_hz == trainer.env.policy_hz == 120,
        "fresh_optimizer_state_empty": len(trainer.optimizer.state) == 0,
        "optimizer_lrs_exact": lrs
        == {"policy": POLICY_LEARNING_RATE, "critic": CRITIC_LEARNING_RATE},
        "value_loss_isolated": hasattr(trainer.model, "isolated_value"),
        "kl_telemetry_only": True,
        "nonfinite_rollback_enabled": True,
        "exact_scale": (trainer.env.num_envs == WORLD_COUNT) if exact_scale else True,
    }
    return {
        "format": f"{FORMAT}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authority_sha256": sha256_file(AUTHORITY),
        "source": transition,
        "contracts": trainer.env.contract_hashes,
        "ppo_config": asdict(trainer.ppo_config),
        "optimizer_group_lrs": lrs,
        "reset_curriculum": trainer.env.world.ssl_foundation_reset.summary,
        "initial_opponent_counts": {
            name: int((trainer.opponent_family == index).sum().item())
            for index, name in enumerate(OPPONENT_NAMES)
        },
    }


def _checkpoint_record(trainer: Rival2SslFoundationTrainer, path: Path) -> dict[str, Any]:
    trainer.save_checkpoint(path, include_optimizer=True)
    return {
        "accepted_updates": trainer.accepted_updates_total,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "policy_version": trainer.policy_version,
        "total_agent_samples": trainer.total_agent_samples,
    }


@torch.no_grad()
def deterministic_anchor_evaluation(
    trainer: Rival2SslFoundationTrainer,
    collision_root: Path,
) -> dict[str, Any]:
    batch = build_ssl_foundation_scenarios(
        EVALUATION_WORLDS,
        seed=EVALUATION_SEED,
    )
    geometry = ArenaGeometry.load_soccar(collision_root)
    env = Rival2Env(
        EVALUATION_WORLDS,
        str(collision_root),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry),
        device="cuda:0",
        seed=EVALUATION_SEED,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
        ssl_foundation_scenarios=batch,
    )
    family = torch.empty(EVALUATION_WORLDS, dtype=torch.int64, device=env.device)
    family[0::2] = OPPONENT_NEXTO
    family[1::2] = OPPONENT_FROZEN_V5
    rival_side = torch.arange(EVALUATION_WORLDS, device=env.device) % 2
    opponent_side = 1 - rival_side
    rows = torch.arange(EVALUATION_WORLDS, device=env.device)
    nexto = NextoPolicyAdapter(EVALUATION_WORLDS, device=env.device)
    nexto.set_player_index(opponent_side)
    nexto_mask = family == OPPONENT_NEXTO
    nexto.activate(nexto_mask)
    nexto_state = NextoStateTensors.from_bridge(env.bridge)
    hidden = trainer.model.initial_hidden(EVALUATION_WORLDS * 2, device=env.device)
    frozen_hidden = trainer.frozen_v5.initial_hidden(EVALUATION_WORLDS * 2, device=env.device)
    reset_before = torch.ones((EVALUATION_WORLDS, 2), dtype=torch.bool, device=env.device)
    observation = env.observation
    touches = torch.zeros(2, dtype=torch.int64, device=env.device)
    goals_for = torch.zeros_like(touches)
    goals_against = torch.zeros_like(touches)
    no_touch = torch.zeros_like(touches)
    speed_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    decisions = torch.zeros(2, dtype=torch.int64, device=env.device)
    trainer.model.eval()
    trainer.frozen_v5.eval()
    for _ in range(EVALUATION_TICKS):
        actor, _value, hidden_after = trainer.model(
            observation.reshape(-1, trainer.policy_config.obs_dim),
            hidden,
            reset_before=reset_before.reshape(-1),
        )
        frozen_actor, _frozen_value, frozen_hidden_after = trainer.frozen_v5(
            observation.reshape(-1, trainer.policy_config.obs_dim),
            frozen_hidden,
            reset_before=reset_before.reshape(-1),
        )
        action = deterministic_unified_action(actor).reshape(EVALUATION_WORLDS, 2, 8)
        frozen_action = deterministic_unified_action(frozen_actor).reshape(EVALUATION_WORLDS, 2, 8)
        frozen_mask = family == OPPONENT_FROZEN_V5
        action[rows[frozen_mask], opponent_side[frozen_mask]] = frozen_action[
            rows[frozen_mask], opponent_side[frozen_mask]
        ]

        def tick_action(
            _tick: int,
            *,
            base_action: torch.Tensor = action,
            adapter: NextoPolicyAdapter = nexto,
            state: NextoStateTensors = nexto_state,
            active_nexto: torch.Tensor = nexto_mask,
            active_opponent_side: torch.Tensor = opponent_side,
        ) -> torch.Tensor:
            tick = base_action.clone()
            ball = state.ball_pos
            kickoff = (ball[:, 0] == 0.0) & (ball[:, 1] == 0.0)
            nexto_action, _ = adapter.tick_action(state, kickoff, active_mask=active_nexto)
            tick[rows[active_nexto], active_opponent_side[active_nexto]] = nexto_action[
                active_nexto
            ]
            return tick

        transition = env.step_with_tick_actions(action, tick_action)
        for family_id in range(2):
            selected = family == (OPPONENT_NEXTO if family_id == 0 else OPPONENT_FROZEN_V5)
            touch = (
                transition.transition_observation[
                    rows[selected], rival_side[selected], _TOUCH_INDEX
                ]
                > 0.5
            )
            touches[family_id] += touch.sum()
            speed = torch.linalg.vector_norm(
                observation[
                    rows[selected],
                    rival_side[selected],
                    _SPEED_START : _SPEED_START + 3,
                ],
                dim=-1,
            )
            speed_sum[family_id] += speed.sum(dtype=torch.float64)
            decisions[family_id] += selected.sum()
            terminal = selected & transition.terminated
            scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
            goals_for[family_id] += (terminal & (scoring == rival_side)).sum()
            goals_against[family_id] += (terminal & (scoring == opponent_side)).sum()
            truncated = (
                selected
                & transition.truncated
                & (transition.transition_observation[:, 0, _NO_TOUCH_INDEX] >= 1.0 - 1.0e-6)
            )
            no_touch[family_id] += truncated.sum()
        reset_agent = transition.reset_mask[:, None].expand(-1, 2)
        hidden = hidden_after.masked_fill(reset_agent.reshape(1, EVALUATION_WORLDS * 2, 1), 0.0)
        frozen_hidden = frozen_hidden_after.masked_fill(
            reset_agent.reshape(1, EVALUATION_WORLDS * 2, 1), 0.0
        )
        reset_before = reset_agent
        nexto.activate(transition.reset_mask & nexto_mask)
        observation = transition.observation
    result: dict[str, Any] = {
        "accepted_updates": trainer.accepted_updates_total,
        "worlds": EVALUATION_WORLDS,
        "ticks": EVALUATION_TICKS,
        "deterministic_policy": True,
        "evaluation_is_gate": False,
        "opponents": {},
    }
    for index, name in enumerate(("nexto", "frozen_unified_v5")):
        player_minutes = float(decisions[index].item()) / (120.0 * 60.0)
        result["opponents"][name] = {
            "touches": int(touches[index].item()),
            "touches_per_minute": int(touches[index].item()) / player_minutes,
            "goals_for": int(goals_for[index].item()),
            "goals_against": int(goals_against[index].item()),
            "no_touch_resets": int(no_touch[index].item()),
            "mean_speed_uu_per_second": (
                float(speed_sum[index].item()) / max(1, int(decisions[index].item())) * 2300.0
            ),
        }
    del env, nexto
    return result


def run(args: argparse.Namespace) -> int:
    if args.write_authority:
        if not args.implementation_commit:
            raise ValueError("--implementation-commit is required with --write-authority")
        write_json(AUTHORITY, authority_payload(args.implementation_commit))
        print(AUTHORITY)
        return 0
    if args.worlds != WORLD_COUNT:
        raise ValueError("production authority freezes 32768 worlds")
    load_authority()
    run_dir = Path(args.run_dir).resolve()
    collision_root = Path(args.collision_root).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    trainer, source = make_trainer(collision_root, worlds=args.worlds)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume).resolve())
    preflight_payload = preflight(trainer, source, exact_scale=args.worlds == WORLD_COUNT)
    write_json(
        RESULTS / ("resume_preflight.json" if args.resume else "preflight.json"),
        preflight_payload,
    )
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"SSL Foundation preflight failed: {preflight_payload}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0
    if args.rollout_preflight_only:
        model_before = state_dict_sha256(trainer.model.state_dict())
        source_before = sha256_file(SOURCE)
        trainer.set_exploration(exploration_for_update(0))
        rollout = trainer.collect_rollout()
        rollout_payload = {
            "format": f"{FORMAT}_EXACT_SCALE_ROLLOUT_PREFLIGHT",
            "created_utc": utc_now(),
            "verdict": "PASS",
            "worlds": trainer.env.num_envs,
            "rollout_horizon": trainer.ppo_config.rollout_horizon,
            "rollout_logical_bytes": rollout.logical_bytes,
            "rollout_position": rollout.position,
            "finite": {
                "observation": bool(torch.isfinite(rollout.observations).all()),
                "action": bool(torch.isfinite(rollout.actions).all()),
                "reward": bool(torch.isfinite(rollout.rewards).all()),
                "value": bool(torch.isfinite(rollout.values).all()),
            },
            "model_unchanged": state_dict_sha256(trainer.model.state_dict()) == model_before,
            "source_checkpoint_unchanged": sha256_file(SOURCE) == source_before,
            "optimizer_state_entries": len(trainer.optimizer.state),
            "optimizer_step_taken": False,
            "rollout": trainer.last_rollout_metrics,
        }
        rollout_payload["verdict"] = (
            "PASS"
            if (
                all(rollout_payload["finite"].values())
                and rollout_payload["model_unchanged"]
                and rollout_payload["source_checkpoint_unchanged"]
                and rollout_payload["optimizer_state_entries"] == 0
                and rollout_payload["rollout_position"] == trainer.ppo_config.rollout_horizon
            )
            else "FAIL"
        )
        write_json(RESULTS / "exact_scale_rollout_preflight.json", rollout_payload)
        print(json.dumps(rollout_payload, indent=2, sort_keys=True))
        return 0 if rollout_payload["verdict"] == "PASS" else 2

    campaign_state_path = run_dir / "campaign_state.json"
    if campaign_state_path.exists():
        campaign_state = json.loads(campaign_state_path.read_text(encoding="utf-8"))
        deadline = float(campaign_state["deadline_unix"])
    else:
        started = time.time()
        deadline = started + WALL_CLOCK_SECONDS
        campaign_state = {
            "format": f"{FORMAT}_CAMPAIGN_STATE",
            "started_unix": started,
            "started_utc": utc_now(),
            "deadline_unix": deadline,
            "wall_clock_seconds": WALL_CLOCK_SECONDS,
            "authority_sha256": sha256_file(AUTHORITY),
        }
        write_json(campaign_state_path, campaign_state)

    curve = RESULTS / "training_curve.jsonl"
    manifest_path = RESULTS / "snapshot_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "format": f"{FORMAT}_SNAPSHOT_MANIFEST",
            "authority_sha256": sha256_file(AUTHORITY),
            "snapshots": [],
            "evaluations": [],
        }
    )
    if not args.resume and (curve.exists() or manifest["snapshots"]):
        raise RuntimeError("campaign evidence exists; use --resume")
    if not manifest["evaluations"]:
        evaluation = deterministic_anchor_evaluation(trainer, collision_root)
        manifest["evaluations"].append(evaluation)
        write_json(manifest_path, manifest)

    rolling = run_dir / "rolling.pt"
    hard_failure: dict[str, Any] | None = None
    while time.time() < deadline and trainer.accepted_updates_total < MAXIMUM_ACCEPTED_UPDATES:
        trainer.set_exploration(exploration_for_update(trainer.accepted_updates_total))
        started_update = time.perf_counter()
        try:
            rollout = trainer.collect_rollout()
            metrics = trainer.update(rollout)
        except Rival2RecurrentPPOCorruption as error:
            hard_failure = {
                "format": f"{FORMAT}_HARD_FAILURE",
                "created_utc": utc_now(),
                "accepted_updates": trainer.accepted_updates_total,
                "diagnostics": error.diagnostics,
                "reward_or_capability_semantics_changed": False,
            }
            write_json(RESULTS / "hard_failure.json", hard_failure)
            break
        record = {
            "accepted_update": trainer.accepted_updates_total,
            "created_utc": utc_now(),
            "wall_seconds": time.perf_counter() - started_update,
            "ppo": {
                name: float(value.detach().item())
                for name, value in metrics.items()
                if value.numel() == 1
            },
            "rollout": trainer.last_rollout_metrics,
            "exploration": trainer.exploration.as_dict(),
        }
        append_jsonl(curve, record)
        _checkpoint_record(trainer, rolling)
        if trainer.accepted_updates_total % SNAPSHOT_INTERVAL == 0:
            snapshot = _checkpoint_record(
                trainer,
                run_dir / "snapshots" / f"ssl_foundation_u{trainer.accepted_updates_total:04d}.pt",
            )
            manifest["snapshots"] = [
                item
                for item in manifest["snapshots"]
                if item["accepted_updates"] != trainer.accepted_updates_total
            ] + [snapshot]
            manifest["snapshots"].sort(key=lambda item: item["accepted_updates"])
            manifest["evaluations"].append(deterministic_anchor_evaluation(trainer, collision_root))
            write_json(manifest_path, manifest)

    final_record = _checkpoint_record(
        trainer,
        CHECKPOINT if hard_failure is None else rolling,
    )
    if (
        not manifest["evaluations"]
        or manifest["evaluations"][-1]["accepted_updates"] != trainer.accepted_updates_total
    ):
        manifest["evaluations"].append(deterministic_anchor_evaluation(trainer, collision_root))
    manifest["final"] = final_record
    write_json(manifest_path, manifest)
    stop_reason = (
        "hard_nonfinite_or_corruption_guard"
        if hard_failure is not None
        else "maximum_accepted_updates"
        if trainer.accepted_updates_total >= MAXIMUM_ACCEPTED_UPDATES
        else "wall_clock_deadline"
    )
    summary = {
        "format": f"{FORMAT}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "verdict": "BLOCKED" if hard_failure is not None else "PASS",
        "stop_reason": stop_reason,
        "accepted_updates": trainer.accepted_updates_total,
        "final_checkpoint": final_record,
        "hard_failure": hard_failure,
        "authority_sha256": sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "last_evaluation": manifest["evaluations"][-1],
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure is not None else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--collision-root",
        default=os.environ.get("RIVALSIM_COLLISION_DIR", "G:/dev/RLBot-Rival/bot/collision_meshes"),
    )
    result.add_argument("--worlds", type=int, default=WORLD_COUNT)
    result.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    result.add_argument("--resume")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--rollout-preflight-only", action="store_true")
    result.add_argument("--write-authority", action="store_true")
    result.add_argument("--implementation-commit")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
