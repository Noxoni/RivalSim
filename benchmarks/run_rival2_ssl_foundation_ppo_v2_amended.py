"""Continue corrected SSL V2 at update 600 under the user's PPO amendment.

The original 600-update evidence remains immutable. This is the same model and
Adam lineage, with separately identified 30-second evaluation evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_ppo_v2 as original  # noqa: E402
from rivalsim.rival2_independent_critic import (  # noqa: E402
    CRITIC_VERSION,
    IndependentCriticActorCritic,
    IndependentCriticPolicyConfig,
    upgrade_state_dict,
)
from rivalsim.rival2_ppo import rival2_ppo_120hz_config  # noqa: E402
from rivalsim.rival2_ssl_foundation_training import SslFoundationOpponentConfig  # noqa: E402

engine = original.engine
_base_make_trainer = engine.make_trainer
_base_preflight = engine.preflight
RESULTS = original.RESULTS / "amendment_v1"
AUTHORITY = RESULTS / "authority.json"
LAUNCH = RESULTS / "launch_authority.json"
RUN_DIR = Path("G:/dev/RivalSim-runs/ssl-foundation-ppo-v2-amendment-v1")
PARENT = original.CHECKPOINT
PARENT_SHA256 = "0A5956ABE92B23CDFA5B2F10CFC610957B5559FA0AFA2EFE51B0BB1B758C94A5"
ORIGINAL_AUTHORITY_SHA256 = "7FD6999C948D8A2EB0AEF3DA5C35B83BBD71D98C761948F8623C43F1274D39E6"
ORIGINAL_LAUNCH_SHA256 = "8E23A6C991F2EB6B018FCF47116F8405C40C48F1587197A7BE38E2565E2EFF37"
AMENDMENT_VERSION = "RIVAL2_SSL_FOUNDATION_PPO_V2_AMENDMENT_V1"
STARTUP = RUN_DIR / "amended_start_u0600.pt"
OPPONENT_CONFIG = SslFoundationOpponentConfig(
    current_probability=0.4, nexto_probability=0.4, frozen_v5_probability=0.2
)


def new_ppo_config():
    return replace(rival2_ppo_120hz_config(), learning_rate=1.0e-4, epochs=2)


def authority_payload() -> dict[str, Any]:
    if engine.sha256_file(original.AUTHORITY) != ORIGINAL_AUTHORITY_SHA256:
        raise ValueError("original corrected V2 authority changed")
    if engine.sha256_file(original.SCHEDULE_AUTHORITY) != ORIGINAL_LAUNCH_SHA256:
        raise ValueError("original corrected V2 launch authority changed")
    payload = json.loads(original.AUTHORITY.read_text(encoding="utf-8"))
    payload["ppo"]["policy_learning_rate"] = 1.0e-4
    payload["ppo"]["epochs"] = 2
    payload["opponents"].update(asdict(OPPONENT_CONFIG))
    payload["campaign"]["evaluation_ticks"] = 3600
    payload["campaign"]["continuation_after_marker"] = "user-authorized amended continuation"
    payload["campaign"]["maximum_accepted_updates"] = None
    payload["amendment"] = {
        "version": AMENDMENT_VERSION,
        "parent_authority_sha256": ORIGINAL_AUTHORITY_SHA256,
        "parent_launch_authority_sha256": ORIGINAL_LAUNCH_SHA256,
        "resume_parent": {
            "path": PARENT.relative_to(ROOT).as_posix(),
            "sha256": PARENT_SHA256,
            "accepted_updates": 600,
        },
        "optimizer_state": (
            "preserve actor and value-head Adam; new critic features start fresh Adam"
        ),
        "architecture_changed": True,
        "critic": {
            "version": CRITIC_VERSION,
            "architecture": [182, 512, 512, 512, 1],
            "activation": "silu",
            "initialization": (
                "copy update-600 trunk and critic head into independent value network"
            ),
            "initial_actor_and_value_parity": "exact",
            "value_gradients_into_actor": False,
            "learning_rate": 3.0e-4,
        },
        "reward_changed": False,
        "exploration_changed": False,
        "training_episode_contract_changed": False,
        "evaluation_seconds": 30,
        "evaluation_comparison": "new 30-second baseline at unchanged update-600 weights",
        "opponent_probabilities_apply_at": "episode assignment; sample shares also reported",
        "continue_until": "user interruption or nonfinite/corruption failure",
        "resume_simulator": "fresh scenario episodes and zero hidden state, as in V2 resumes",
    }
    return payload


def launch_payload() -> dict[str, Any]:
    return {
        "format": AMENDMENT_VERSION + "_LAUNCH",
        "parent_authority_sha256": engine.sha256_file(AUTHORITY),
        "resume_parent_sha256": PARENT_SHA256,
        "resume_parent_update": 600,
        "fresh_optimizer": False,
        "evaluation_and_snapshot_interval": 50,
        "evaluation_ticks": 3600,
        "continue_after_600": True,
    }


def load_authority() -> dict[str, Any]:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if payload != authority_payload():
        raise ValueError("amended authority differs from the permitted changes")
    if engine.sha256_file(engine.SOURCE) != engine.SOURCE_SHA256:
        raise ValueError("original Unified V5 root changed")
    return payload


def load_launch_authority() -> dict[str, Any]:
    payload = json.loads(LAUNCH.read_text(encoding="utf-8"))
    if payload != launch_payload():
        raise ValueError("amended launch authority mismatch")
    return payload


def configure_engine() -> None:
    original._configure_engine()
    engine.RESULTS = RESULTS
    engine.AUTHORITY = AUTHORITY
    engine.SCHEDULE_AUTHORITY = LAUNCH
    engine.CHECKPOINT = ROOT / "checkpoints/rival2/ssl_foundation_ppo_v2/amendment_v1/final.pt"
    engine.DEFAULT_RUN_DIR = RUN_DIR
    engine.POLICY_LEARNING_RATE = 1.0e-4
    engine.PPO_EPOCHS = 2
    engine.EVALUATION_TICKS = 3600
    engine.OPPONENT_CONFIG = OPPONENT_CONFIG
    engine.load_authority = load_authority
    engine.load_schedule_authority = load_launch_authority
    engine.make_trainer = make_trainer
    engine.preflight = preflight


def make_trainer(collision_root: Path, *, worlds: int):
    trainer, source = _base_make_trainer(collision_root, worlds=worlds)
    config = IndependentCriticPolicyConfig(**source["policy_config"])
    model = IndependentCriticActorCritic(config)
    model.load_state_dict(upgrade_state_dict(source["model"]), strict=True)
    trainer.model = model.to(trainer.device)
    trainer.policy_config = config
    engine.configure_optimizer(trainer)
    return trainer, source


def preflight(trainer, source, *, exact_scale: bool):
    report = _base_preflight(trainer, source, exact_scale=exact_scale)
    report["checks"].update(
        {
            "independent_critic": isinstance(trainer.model, IndependentCriticActorCritic),
            "two_ppo_passes": trainer.ppo_config.epochs == 2,
            "thirty_second_evaluation": engine.EVALUATION_TICKS == 3600,
            "critic_layers_do_not_alias_actor": not (
                {p.data_ptr() for p in trainer.model.critic.parameters()}
                & {p.data_ptr() for p in trainer.model.trunk.parameters()}
            ),
        }
    )
    if trainer.accepted_updates_total == 600:
        with torch.random.fork_rng(devices=[trainer.device]):
            parent = torch.load(PARENT, map_location="cpu", weights_only=False)
            parent_config = engine.Rival2UnifiedPolicyConfig(**parent["policy_config"])
            legacy = engine.Rival2UnifiedActorCritic(parent_config).to(trainer.device)
            legacy.load_state_dict(parent["model"], strict=True)
            observation = trainer.env.observation.reshape(-1, parent_config.obs_dim)[:256]
            with torch.no_grad():
                before = legacy(observation)
                after = trainer.model(observation)
            report["checks"]["real_state_actor_value_hidden_parity_at_transition"] = all(
                torch.equal(left, right) for left, right in zip(before, after, strict=True)
            )
            report["initial_actor_max_abs_difference"] = (before[0] - after[0]).abs().max().item()
            report["initial_value_max_abs_difference"] = (before[1] - after[1]).abs().max().item()
            del legacy, parent
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    return report


def validate_parent(payload: dict[str, Any]) -> None:
    expected_config = replace(rival2_ppo_120hz_config(), learning_rate=1.0e-6, epochs=1)
    checks = {
        "format": payload["format"] == original.FORMAT + "_CHECKPOINT",
        "update": payload["accepted_updates_total"] == payload["policy_version"] == 600,
        "ppo": payload["ppo_config"] == asdict(expected_config),
        "ppo_hash": payload["ppo_config_sha256"] == expected_config.content_hash,
        "root": payload["source"]["sha256"] == engine.SOURCE_SHA256,
        "authority": payload["source"]["authority_sha256"] == ORIGINAL_AUTHORITY_SHA256,
        "optimizer": bool(payload["optimizer"]["state"]),
        "cadence": payload["policy_hz"] == payload["physics_hz"] == 120,
    }
    if not all(checks.values()):
        raise ValueError(f"amendment parent rejected: {checks}")


def migrate_payload(parent: dict[str, Any], authority_hash: str, launch_hash: str):
    """Preserve the actor and initialize a value-equivalent independent MLP critic."""
    validate_parent(parent)
    payload = copy.deepcopy(parent)
    policy_config = IndependentCriticPolicyConfig(**parent["policy_config"])
    payload["policy_config"] = asdict(policy_config)
    payload["policy_config_sha256"] = policy_config.content_hash
    payload["model"] = upgrade_state_dict(parent["model"])
    config = new_ppo_config()
    payload["ppo_config"] = asdict(config)
    payload["ppo_config_sha256"] = config.content_hash
    next_parameter_id = 1 + max(
        p for group in payload["optimizer"]["param_groups"] for p in group["params"]
    )
    for group in payload["optimizer"]["param_groups"]:
        if group["name"] == "policy":
            group["lr"] = config.learning_rate
        elif group["name"] != "critic" or group["lr"] != 3.0e-4:
            raise ValueError("unexpected optimizer group")
        else:
            if len(group["params"]) != 2:
                raise ValueError("legacy critic must have exactly a weight and bias")
            feature_count = sum(name.startswith("trunk.") for name in parent["model"])
            group["params"] = (
                list(range(next_parameter_id, next_parameter_id + feature_count)) + group["params"]
            )
    payload["optimizer_group_lrs"] = {"policy": 1.0e-4, "critic": 3.0e-4}
    payload["opponents"]["config"] = asdict(OPPONENT_CONFIG)
    payload["source"]["authority_sha256"] = authority_hash
    payload["source"]["schedule_authority_sha256"] = launch_hash
    payload["phase_transition"]["amendment"] = {
        "version": AMENDMENT_VERSION,
        "authority_sha256": authority_hash,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_accepted_updates": 600,
        "parent_model_tensor_sha256": engine.state_dict_sha256(parent["model"]),
        "actor_tensors_unchanged": True,
        "value_function_initially_identical": True,
        "existing_adam_moments_and_steps_preserved": True,
        "new_critic_feature_adam": "fresh",
        "critic_architecture": CRITIC_VERSION,
    }
    payload["phase_transition"]["authority_sha256"] = authority_hash
    payload["phase_transition"]["schedule_authority_sha256"] = launch_hash
    return payload


def prepare() -> None:
    if engine.sha256_file(PARENT) != PARENT_SHA256:
        raise ValueError("update-600 parent checkpoint hash mismatch")
    for destination in (AUTHORITY, LAUNCH, STARTUP):
        if destination.exists():
            raise FileExistsError(f"preserve existing amendment artifact: {destination}")
    engine.write_json(AUTHORITY, authority_payload())
    engine.write_json(LAUNCH, launch_payload())
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    payload = migrate_payload(parent, engine.sha256_file(AUTHORITY), engine.sha256_file(LAUNCH))
    STARTUP.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, STARTUP)
    engine.write_json(
        RESULTS / "transition.json",
        {
            "version": AMENDMENT_VERSION,
            "parent": str(PARENT),
            "parent_sha256": PARENT_SHA256,
            "startup": str(STARTUP),
            "startup_sha256": engine.sha256_file(STARTUP),
            "model_tensor_sha256": engine.state_dict_sha256(payload["model"]),
            "ppo_config": payload["ppo_config"],
            "optimizer_group_lrs": payload["optimizer_group_lrs"],
            "accepted_updates": payload["accepted_updates_total"],
            "optimizer_step_taken": False,
            "amendment": payload["phase_transition"]["amendment"],
        },
    )


def verify_resume(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = engine.sha256_file(AUTHORITY)
    amendment = (payload.get("phase_transition") or {}).get("amendment", {})
    if (
        amendment.get("authority_sha256") != expected
        or amendment.get("parent_checkpoint_sha256") != PARENT_SHA256
        or payload.get("source", {}).get("authority_sha256") != expected
        or payload.get("ppo_config_sha256") != new_ppo_config().content_hash
        or payload.get("accepted_updates_total", -1) < 600
    ):
        raise ValueError("resume checkpoint is not from this amended V2 lineage")
    if payload["accepted_updates_total"] == 600:
        transition = json.loads((RESULTS / "transition.json").read_text())
        if engine.sha256_file(path) != transition["startup_sha256"]:
            raise ValueError("initial amended checkpoint identity mismatch")


def parser() -> argparse.ArgumentParser:
    result = engine.parser()
    result.add_argument("--prepare-amendment", action="store_true")
    result.set_defaults(run_dir=str(RUN_DIR), resume=str(STARTUP), continue_after_600=True)
    return result


def run(args: argparse.Namespace) -> int:
    configure_engine()
    if args.write_authority:
        raise ValueError("use --prepare-amendment to bind checkpoint and authorities together")
    if args.prepare_amendment:
        prepare()
        return 0
    load_authority()
    load_launch_authority()
    verify_resume(Path(args.resume))
    return engine.run(args)


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
