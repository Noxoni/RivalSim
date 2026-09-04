"""Bounded fresh-optimizer PPO comparison rooted directly at original Unified V5."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_ppo_v2_amended as amended  # noqa: E402
from benchmarks.run_rival2_unified_ground_selfplay_ppo_v1 import (  # noqa: E402
    exploration_for_update as original_exploration_for_update,
)
from rivalsim.rival2_independent_critic import (  # noqa: E402
    CRITIC_VERSION,
    IndependentCriticActorCritic,
    IndependentCriticPolicyConfig,
    upgrade_state_dict,
)

engine = amended.engine
FORMAT = "RIVAL2_SSL_FOUNDATION_V5_RESTART_V1"
LINEAGE = "Original Unified Capability V5 -> bounded independent-critic PPO restart V1"
RESULTS = ROOT / "results/rival2/ssl_foundation_v5_restart_v1"
AUTHORITY = RESULTS / "authority.json"
LAUNCH = RESULTS / "launch_authority.json"
CHECKPOINT = ROOT / "checkpoints/rival2/ssl_foundation_v5_restart_v1/final.pt"
RUN_DIR = Path("G:/dev/RivalSim-runs/ssl-foundation-v5-restart-v1")
REVIEW_UPDATES = 100
EXPLORATION_OFFSET = 600
SETTINGS_AUTHORITY_SHA256 = "10B6872D38CFAF555DC929D492432197CDD85640DDB224202ACD8ED0E65155BE"
IMPLEMENTATION_PATHS = (
    "benchmarks/run_rival2_ssl_foundation_v5_restart_v1.py",
    "benchmarks/run_rival2_ssl_foundation_ppo_v1.py",
    "rivalsim/rival2_independent_critic.py",
    "rivalsim/rival2_recurrent_ppo.py",
    "rivalsim/rival2_recurrent_training.py",
    "rivalsim/rival2_ssl_foundation_training.py",
    "rivalsim/rival2_unified_policy.py",
    "rivalsim/ssl_foundation_v1.py",
)


def restart_exploration(accepted_update: int):
    if accepted_update < 0:
        raise ValueError("accepted update cannot be negative")
    # Preserve actual exploration used by amended updates 601 onward, not the
    # old low-noise startup ramp. PPO counters remain zero-based in this run.
    return original_exploration_for_update(EXPLORATION_OFFSET + accepted_update)


def authority_payload(implementation_commit: str, created_utc: str) -> dict[str, Any]:
    if engine.sha256_file(amended.AUTHORITY) != SETTINGS_AUTHORITY_SHA256:
        raise ValueError("comparison settings authority changed")
    payload = copy.deepcopy(json.loads(amended.AUTHORITY.read_text(encoding="utf-8")))
    del payload["amendment"]
    payload["format"] = FORMAT + "_AUTHORITY"
    payload["created_utc"] = created_utc
    payload["implementation_commit"] = implementation_commit
    payload["implementation_sha256"] = {
        name: engine.sha256_file(ROOT / name) for name in IMPLEMENTATION_PATHS
    }
    payload["supersedes_authority_sha256"] = SETTINGS_AUTHORITY_SHA256
    payload["supersession_reason"] = (
        "bounded comparison from original V5 weights and fresh PPO Adam, not any SSL descendant"
    )
    payload["campaign"].update(
        maximum_accepted_updates=REVIEW_UPDATES,
        continuation_review_marker=REVIEW_UPDATES,
        continuation_after_marker="stop for user review; no automatic continuation",
        evaluation_ticks=3600,
        evaluation_interval=50,
        snapshot_interval=50,
    )
    payload["restart"] = {
        "version": FORMAT,
        "source_checkpoint_sha256": engine.SOURCE_SHA256,
        "settings_authority_sha256": SETTINGS_AUTHORITY_SHA256,
        "initial_accepted_updates": 0,
        "initial_trainable_samples": 0,
        "initial_resume_checkpoint": None,
        "optimizer": "fresh PPO Adam for every actor and critic parameter; no state restored",
        "critic": {
            "version": CRITIC_VERSION,
            "architecture": [182, 512, 512, 512, 1],
            "activation": "silu",
            "initialization": "copy original V5 trunk and critic head into independent network",
            "initial_actor_value_hidden_parity": "exact",
            "value_gradients_into_actor": False,
        },
        "exploration_schedule_offset": EXPLORATION_OFFSET,
        "exploration_schedule_index": "600 + local accepted PPO updates",
        "effective_analog_sigma": 0.04,
        "effective_button_temperature": 0.25,
        "reward_changed": False,
        "scenario_corpus_changed": False,
        "training_episode_contract_changed": False,
        "forbidden_initialization": "all SSL PPO descendants, including updates 600 and later",
        "selection_basis": "paired deterministic gameplay results; not training loss or KL",
        "comparison": "matched 30-second evaluations at local 0, 50 and 100",
        "resume_semantics": (
            "new simulator episodes and zero hidden; model/Adam/counters/RNG resume"
        ),
    }
    return payload


def launch_payload() -> dict[str, Any]:
    return {
        "format": FORMAT + "_LAUNCH",
        "parent_authority_sha256": engine.sha256_file(AUTHORITY),
        "source_sha256": engine.SOURCE_SHA256,
        "fresh_optimizer": True,
        "initial_accepted_updates": 0,
        "initial_resume_checkpoint": None,
        "maximum_accepted_updates": REVIEW_UPDATES,
        "evaluation_and_snapshot_interval": 50,
        "evaluation_ticks": 3600,
        "exploration_schedule_offset": EXPLORATION_OFFSET,
        "automatic_continuation": False,
    }


def load_authority() -> dict[str, Any]:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    expected = authority_payload(payload["implementation_commit"], payload["created_utc"])
    if payload != expected:
        raise ValueError("restart authority or frozen implementation changed")
    if engine.sha256_file(engine.SOURCE) != engine.SOURCE_SHA256:
        raise ValueError("original Unified V5 source changed")
    return payload


def load_launch_authority() -> dict[str, Any]:
    payload = json.loads(LAUNCH.read_text(encoding="utf-8"))
    if payload != launch_payload():
        raise ValueError("restart launch binding mismatch")
    return payload


def configure_engine() -> None:
    amended.original._configure_engine()
    engine.FORMAT = FORMAT
    engine.CHECKPOINT_FORMAT = FORMAT + "_CHECKPOINT"
    engine.LINEAGE = LINEAGE
    engine.RESULTS = RESULTS
    engine.AUTHORITY = AUTHORITY
    engine.SCHEDULE_AUTHORITY = LAUNCH
    engine.CHECKPOINT = CHECKPOINT
    engine.DEFAULT_RUN_DIR = RUN_DIR
    engine.MAXIMUM_ACCEPTED_UPDATES = REVIEW_UPDATES
    engine.CONTINUATION_REVIEW_MARKER = REVIEW_UPDATES
    engine.POLICY_LEARNING_RATE = 1e-4
    engine.CRITIC_LEARNING_RATE = 3e-4
    engine.PPO_EPOCHS = 2
    engine.EVALUATION_TICKS = 3600
    engine.OPPONENT_CONFIG = amended.OPPONENT_CONFIG
    engine.load_authority = load_authority
    engine.load_schedule_authority = load_launch_authority
    engine.make_trainer = make_trainer
    engine.preflight = preflight
    engine.exploration_for_update = restart_exploration


def build_root_model(source: dict[str, Any]) -> IndependentCriticActorCritic:
    if source.get("format") != "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V5":
        raise ValueError("restart requires original Unified V5, never a PPO descendant")
    legacy_config = engine.Rival2UnifiedPolicyConfig(**source["policy_config"])
    if source["policy_config_sha256"] != legacy_config.content_hash:
        raise ValueError("source policy config hash mismatch")
    model = IndependentCriticActorCritic(IndependentCriticPolicyConfig(**source["policy_config"]))
    model.load_state_dict(upgrade_state_dict(source["model"]), strict=True)
    return model


def make_trainer(collision_root: Path, *, worlds: int):
    torch.manual_seed(engine.CAMPAIGN_SEED)
    trainer, source = amended._base_make_trainer(collision_root, worlds=worlds)
    trainer.model = build_root_model(source).to(trainer.device)
    trainer.policy_config = trainer.model.config
    engine.configure_optimizer(trainer)
    trainer.set_exploration(restart_exploration(0))
    trainer.phase_transition["restart"] = {
        "version": FORMAT,
        "source_checkpoint_sha256": engine.SOURCE_SHA256,
        "authority_sha256": engine.sha256_file(AUTHORITY),
        "initial_accepted_updates": trainer.accepted_updates_total,
        "initial_trainable_samples": trainer.total_agent_samples,
        "optimizer_state_entries_at_initialization": len(trainer.optimizer.state),
        "optimizer_state_loaded": False,
        "critic_architecture": CRITIC_VERSION,
        "augmented_initial_model_sha256": engine.state_dict_sha256(trainer.model.state_dict()),
        "exploration_schedule_offset": EXPLORATION_OFFSET,
    }
    return trainer, source


def preflight(trainer, source, *, exact_scale: bool):
    report = amended._base_preflight(trainer, source, exact_scale=exact_scale)
    report["checks"].update(
        independent_critic=isinstance(trainer.model, IndependentCriticActorCritic),
        two_ppo_passes=trainer.ppo_config.epochs == 2,
        evaluation_30_seconds=engine.EVALUATION_TICKS == 3600,
        bounded_100_updates=engine.CONTINUATION_REVIEW_MARKER == REVIEW_UPDATES,
        critic_does_not_alias_actor=not (
            {p.data_ptr() for p in trainer.model.critic.parameters()}
            & {p.data_ptr() for p in trainer.model.trunk.parameters()}
        ),
        exploration_matches_amended_run=restart_exploration(0).analog_sigma == 0.04
        and restart_exploration(0).button_temperature == 0.25,
    )
    if trainer.accepted_updates_total == 0:
        observation = trainer.env.observation.reshape(-1, trainer.policy_config.obs_dim)[:256]
        with torch.no_grad():
            before = trainer.frozen_v5(observation)
            after = trainer.model(observation)
        report["checks"].update(
            exact_actor_value_hidden_parity=all(
                torch.equal(a, b) for a, b in zip(before, after, strict=True)
            ),
            source_actor_tensors_exact=all(
                torch.equal(value.to(trainer.device), trainer.model.state_dict()[name])
                for name, value in source["model"].items()
                if not name.startswith("critic.")
            ),
            fresh_optimizer=len(trainer.optimizer.state) == 0,
            samples_start_at_zero=trainer.total_agent_samples == 0,
        )
        report["initial_actor_max_abs_difference"] = (before[0] - after[0]).abs().max().item()
        report["initial_value_max_abs_difference"] = (before[1] - after[1]).abs().max().item()
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    return report


def validate_resume_payload(payload: dict[str, Any]) -> None:
    source = payload.get("source", {})
    restart = (payload.get("phase_transition") or {}).get("restart", {})
    if (
        payload.get("format") != FORMAT + "_CHECKPOINT"
        or payload.get("lineage") != LINEAGE
        or source.get("sha256") != engine.SOURCE_SHA256
        or source.get("authority_sha256") != engine.sha256_file(AUTHORITY)
        or restart.get("version") != FORMAT
        or restart.get("optimizer_state_loaded") is not False
        or payload.get("ppo_config") != asdict(amended.new_ppo_config())
        or payload.get("ppo_config_sha256") != amended.new_ppo_config().content_hash
        or payload.get("policy_config", {}).get("critic_architecture") != CRITIC_VERSION
        or payload.get("opponents", {}).get("config") != asdict(amended.OPPONENT_CONFIG)
        or not 0 <= payload.get("accepted_updates_total", -1) <= REVIEW_UPDATES
    ):
        raise ValueError("resume must be from this bounded V5 restart, not a prior lineage")


def parser() -> argparse.ArgumentParser:
    result = engine.parser()
    result.set_defaults(run_dir=str(RUN_DIR), resume=None, continue_after_600=False)
    return result


def run(args: argparse.Namespace) -> int:
    configure_engine()
    if args.continue_after_600:
        raise ValueError("this bounded comparison cannot continue beyond 100 updates")
    if Path(args.run_dir).resolve() != RUN_DIR.resolve():
        raise ValueError("use the isolated restart run directory")
    if args.write_authority:
        if not args.implementation_commit:
            raise ValueError("implementation commit required")
        if AUTHORITY.exists() or LAUNCH.exists():
            raise FileExistsError("preserve the frozen restart authorities")
        engine.write_json(
            AUTHORITY, authority_payload(args.implementation_commit, engine.utc_now())
        )
        engine.write_json(LAUNCH, launch_payload())
        return 0
    load_authority()
    load_launch_authority()
    if args.resume:
        validate_resume_payload(torch.load(args.resume, map_location="cpu", weights_only=False))
    return engine.run(args)


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
