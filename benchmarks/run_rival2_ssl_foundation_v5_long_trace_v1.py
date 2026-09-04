"""Resume the V5 restart with three-second rollouts and a three-second GAE half-life."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_v5_restart_v1 as prior  # noqa: E402
from rivalsim.rival2_policy import hybrid_log_probability  # noqa: E402
from rivalsim.rival2_recurrent_ppo import _sequence_major  # noqa: E402

engine = prior.engine
FORMAT = "RIVAL2_SSL_FOUNDATION_V5_LONG_TRACE_V1"
LINEAGE = prior.LINEAGE + " -> user-authorized long-trace amendment at update 10"
RESULTS = ROOT / "results/rival2/ssl_foundation_v5_long_trace_v1"
AUTHORITY = RESULTS / "authority.json"
LAUNCH = RESULTS / "launch_authority.json"
RUN_DIR = Path("G:/dev/RivalSim-runs/ssl-foundation-v5-long-trace-v1")
STARTUP = RUN_DIR / "transition_u0010.pt"
CHECKPOINT = ROOT / "checkpoints/rival2/ssl_foundation_v5_long_trace_v1/final.pt"
PARENT = prior.CHECKPOINT
PARENT_SHA256 = "EFDB7725D8DBFAFBFA75A66F4AAF618AC561DE6BDC660B0976B6AFDAE775927B"
PARENT_AUTHORITY_SHA256 = "AB4E37BE9A4D93EDA7578868B3831B5FFD7EF46A44ADA1EA4FC9CF59399A73C5"
PARENT_UPDATE = 10
HORIZON = 360
TRACE_HALF_LIFE_SECONDS = 3.0
IMPLEMENTATION_PATHS = (
    *prior.IMPLEMENTATION_PATHS,
    "benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py",
    "rivalsim/rival2_ppo.py",
)


def new_ppo_config():
    base = prior.amended.new_ppo_config()
    # gamma stays fixed, including in potential shaping. This controls the GAE
    # TD-error trace, not a new reward discount or a uniform retrospective label.
    trace_decay = 0.5 ** (1.0 / (120.0 * TRACE_HALF_LIFE_SECONDS))
    return replace(base, rollout_horizon=HORIZON, gae_lambda=trace_decay / base.gamma)


def tree_sha256(value: Any) -> str:
    """Deterministic identity for tensor/optimizer/RNG trees, independent of torch.save."""
    digest = hashlib.sha256()

    def visit(item):
        digest.update(type(item).__name__.encode() + b":")
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        digest.update(b";")

    visit(value)
    return digest.hexdigest().upper()


def authority_payload(commit: str, created_utc: str):
    if engine.sha256_file(prior.AUTHORITY) != PARENT_AUTHORITY_SHA256:
        raise ValueError("preserved parent authority changed")
    payload = copy.deepcopy(json.loads(prior.AUTHORITY.read_text()))
    payload.update(
        format=FORMAT + "_AUTHORITY",
        created_utc=created_utc,
        implementation_commit=commit,
        implementation_sha256={p: engine.sha256_file(ROOT / p) for p in IMPLEMENTATION_PATHS},
        supersedes_authority_sha256=PARENT_AUTHORITY_SHA256,
        supersession_reason="user requested longer retrospective credit, not new rewards",
    )
    config = new_ppo_config()
    payload["ppo"].update(
        version=FORMAT,
        rollout_horizon=HORIZON,
        gae_lambda=config.gae_lambda,
        effective_config=asdict(config),
        effective_config_sha256=config.content_hash,
        fresh_optimizer=False,
    )
    payload["credit_assignment_amendment"] = {
        "parent_path": PARENT.relative_to(ROOT).as_posix(),
        "parent_sha256": PARENT_SHA256,
        "parent_update": PARENT_UPDATE,
        "parent_authority_sha256": PARENT_AUTHORITY_SHA256,
        "only_ppo_changes": ["rollout_horizon", "gae_lambda"],
        "trace_half_life_seconds": TRACE_HALF_LIFE_SECONDS,
        "trace_formula": "(gamma*gae_lambda)^ticks; lambda=2^(-1/360)/gamma",
        "trace_weights": {str(t): (config.gamma * config.gae_lambda) ** t for t in (120, 240, 360)},
        "rollout_seconds": 3.0,
        "reward_and_shaping_gamma_changed": False,
        "model_and_optimizer_preserved": True,
        "counters_and_rng_preserved": True,
        "resume_physics": "fresh episodes and zero hidden, as in existing resume contract",
        "bound": "stop at total local update 100, not 100 additional updates",
        "evaluations": "unchanged 3600-tick protocol at boundary 10, then 50 and 100",
        "comparison": "report cumulative samples and elapsed time, not just update count",
        "operational_memory_fix": "release completed rollout before collecting next one",
        "limits": "not causal action labels; boundaries cut traces; longer traces raise variance",
    }
    return payload


def launch_payload():
    return {
        "format": FORMAT + "_LAUNCH",
        "parent_authority_sha256": engine.sha256_file(AUTHORITY),
        "source_sha256": engine.SOURCE_SHA256,
        "resume_parent_sha256": PARENT_SHA256,
        "resume_parent_update": PARENT_UPDATE,
        "fresh_optimizer": False,
        "maximum_accepted_updates": prior.REVIEW_UPDATES,
        "evaluation_and_snapshot_interval": 50,
        "evaluation_ticks": 3600,
        "automatic_continuation": False,
    }


def load_authority():
    payload = json.loads(AUTHORITY.read_text())
    if payload != authority_payload(payload["implementation_commit"], payload["created_utc"]):
        raise ValueError("long-trace authority/implementation changed")
    if engine.sha256_file(engine.SOURCE) != engine.SOURCE_SHA256:
        raise ValueError("original V5 root changed")
    return payload


def load_launch_authority():
    payload = json.loads(LAUNCH.read_text())
    if payload != launch_payload():
        raise ValueError("long-trace launch mismatch")
    return payload


def migrate_payload(parent, authority_hash, launch_hash):
    prior.validate_resume_payload(parent)
    if (
        parent["accepted_updates_total"] != PARENT_UPDATE
        or parent["policy_version"] != PARENT_UPDATE
    ):
        raise ValueError("transition requires the exact saved update-10 boundary")
    payload = copy.deepcopy(parent)
    payload.update(
        format=FORMAT + "_CHECKPOINT",
        lineage=LINEAGE,
        ppo_config=asdict(new_ppo_config()),
        ppo_config_sha256=new_ppo_config().content_hash,
    )
    payload["source"].update(authority_sha256=authority_hash, schedule_authority_sha256=launch_hash)
    payload["phase_transition"]["credit_assignment_amendment"] = {
        "version": FORMAT,
        "parent_sha256": PARENT_SHA256,
        "parent_update": PARENT_UPDATE,
        "authority_sha256": authority_hash,
        "parent_model_sha256": tree_sha256(parent["model"]),
        "parent_optimizer_sha256": tree_sha256(parent["optimizer"]),
        "optimizer_reset": False,
    }
    payload["phase_transition"].update(
        authority_sha256=authority_hash,
        schedule_authority_sha256=launch_hash,
    )
    return payload


def validate_resume_payload(payload):
    transition = (payload.get("phase_transition") or {}).get("credit_assignment_amendment", {})
    if (
        payload.get("format") != FORMAT + "_CHECKPOINT"
        or payload.get("lineage") != LINEAGE
        or payload.get("source", {}).get("sha256") != engine.SOURCE_SHA256
        or payload.get("source", {}).get("authority_sha256") != engine.sha256_file(AUTHORITY)
        or payload.get("source", {}).get("schedule_authority_sha256") != engine.sha256_file(LAUNCH)
        or transition.get("parent_sha256") != PARENT_SHA256
        or transition.get("authority_sha256") != engine.sha256_file(AUTHORITY)
        or transition.get("optimizer_reset") is not False
        or payload.get("ppo_config") != asdict(new_ppo_config())
        or payload.get("ppo_config_sha256") != new_ppo_config().content_hash
        or payload.get("policy_config", {}).get("critic_architecture") != prior.CRITIC_VERSION
        or payload.get("opponents", {}).get("config") != asdict(prior.amended.OPPONENT_CONFIG)
        or not payload.get("optimizer", {}).get("state")
        or not PARENT_UPDATE <= payload.get("accepted_updates_total", -1) <= prior.REVIEW_UPDATES
    ):
        raise ValueError("resume must belong to this exact long-trace amendment")


def configure_engine():
    prior.configure_engine()
    for name, value in {
        "FORMAT": FORMAT,
        "CHECKPOINT_FORMAT": FORMAT + "_CHECKPOINT",
        "LINEAGE": LINEAGE,
        "RESULTS": RESULTS,
        "AUTHORITY": AUTHORITY,
        "SCHEDULE_AUTHORITY": LAUNCH,
        "CHECKPOINT": CHECKPOINT,
        "DEFAULT_RUN_DIR": RUN_DIR,
        "load_authority": load_authority,
        "load_schedule_authority": load_launch_authority,
        "make_trainer": make_trainer,
        "preflight": preflight,
    }.items():
        setattr(engine, name, value)


def make_trainer(collision_root, *, worlds):
    trainer, source = prior.make_trainer(collision_root, worlds=worlds)
    trainer.ppo_config = new_ppo_config()
    collect, update = trainer.collect_rollout, trainer.update

    def measured_collect():
        torch.cuda.reset_peak_memory_stats(trainer.device)
        return collect()

    def measured_update(rollout):
        metrics = update(rollout)
        for name, value in {
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(trainer.device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(trainer.device) / 2**30,
            "rollout_logical_gib": rollout.logical_bytes / 2**30,
            "rollout_horizon": HORIZON,
            "gae_lambda": trainer.ppo_config.gae_lambda,
        }.items():
            metrics[name] = torch.tensor(value, dtype=torch.float64, device=trainer.device)
        return metrics

    trainer.collect_rollout, trainer.update = measured_collect, measured_update
    return trainer, source


def preflight(trainer, source, *, exact_scale):
    report = prior.preflight(trainer, source, exact_scale=exact_scale)
    report["checks"].update(
        long_trace_config=trainer.ppo_config == new_ppo_config(),
        gamma_and_reward_unchanged=trainer.ppo_config.gamma == engine.SSL_FOUNDATION_GAMMA,
        optimizer_continued=bool(trainer.optimizer.state),
    )
    if trainer.accepted_updates_total == PARENT_UPDATE:
        transition = json.loads((RESULTS / "transition.json").read_text())
        report["checks"].update(
            model_exact_at_transition=tree_sha256(trainer.model.state_dict())
            == transition["model_sha256"],
            optimizer_exact_at_transition=tree_sha256(trainer.optimizer.state_dict())
            == transition["optimizer_sha256"],
        )
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    return report


def prepare(commit):
    if engine.sha256_file(PARENT) != PARENT_SHA256:
        raise ValueError("paused checkpoint hash mismatch")
    summary = json.loads((prior.RESULTS / "training_summary.json").read_text())
    if summary["accepted_updates"] != PARENT_UPDATE or summary["hard_failure"] is not None:
        raise ValueError("parent did not finish its clean accepted-boundary pause")
    for path in (AUTHORITY, LAUNCH, STARTUP):
        if path.exists():
            raise FileExistsError(f"preserve existing authority/transition: {path}")
    engine.write_json(AUTHORITY, authority_payload(commit, engine.utc_now()))
    engine.write_json(LAUNCH, launch_payload())
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    migrated = migrate_payload(parent, engine.sha256_file(AUTHORITY), engine.sha256_file(LAUNCH))
    STARTUP.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, STARTUP)
    unchanged = {
        key: tree_sha256(parent[key]) == tree_sha256(migrated[key])
        for key in parent
        if key
        not in {
            "format",
            "lineage",
            "ppo_config",
            "ppo_config_sha256",
            "source",
            "phase_transition",
        }
    }
    if not all(unchanged.values()):
        raise AssertionError("checkpoint migration changed protected state")
    engine.write_json(
        RESULTS / "transition.json",
        {
            "parent": str(PARENT),
            "parent_sha256": PARENT_SHA256,
            "startup": str(STARTUP),
            "startup_sha256": engine.sha256_file(STARTUP),
            "model_sha256": tree_sha256(migrated["model"]),
            "optimizer_sha256": tree_sha256(migrated["optimizer"]),
            "unchanged_checkpoint_fields": unchanged,
            "optimizer_step_taken": False,
            "accepted_updates": PARENT_UPDATE,
            "total_agent_samples": parent["total_agent_samples"],
            "ppo_config": migrated["ppo_config"],
            "ppo_config_sha256": migrated["ppo_config_sha256"],
        },
    )
    # Reuse the just-completed deterministic boundary evaluation: identical weights.
    engine.write_json(
        RESULTS / "snapshot_manifest.json",
        {
            "format": FORMAT + "_SNAPSHOT_MANIFEST",
            "authority_sha256": engine.sha256_file(AUTHORITY),
            "schedule_authority_sha256": engine.sha256_file(LAUNCH),
            "snapshots": [],
            "evaluations": [summary["last_evaluation"]],
            "baseline_origin": "preserved pre-amendment evaluation of exact update-10 parent",
        },
    )
    state = json.loads((prior.RUN_DIR / "campaign_state.json").read_text())
    state.update(
        format=FORMAT + "_CAMPAIGN_STATE",
        authority_sha256=engine.sha256_file(AUTHORITY),
        schedule_authority_sha256=engine.sha256_file(LAUNCH),
    )
    engine.write_json(RUN_DIR / "campaign_state.json", state)


def memory_preflight(args):
    """Full-size real rollout + real recurrent backward; no optimizer step or saved learning."""
    trainer, source = make_trainer(Path(args.collision_root), worlds=args.worlds)
    trainer.load_checkpoint(args.resume)
    report = preflight(trainer, source, exact_scale=True)
    if report["verdict"] != "PASS":
        raise RuntimeError(report)
    before_model, before_optimizer = (
        tree_sha256(trainer.model.state_dict()),
        tree_sha256(trainer.optimizer.state_dict()),
    )
    trainer.set_exploration(prior.restart_exploration(trainer.accepted_updates_total))
    rollout = trainer.collect_rollout()
    rollout.compute_gae(trainer.ppo_config)
    count = rollout.sequence_layout(trainer.ppo_config.minibatch_size).sequences_per_minibatch
    obs = _sequence_major(rollout.observations)[:count].contiguous()
    reset = _sequence_major(rollout.reset_before)[:count].contiguous()
    mask = _sequence_major(rollout.train_mask)[:count]
    hidden = rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim)[:, :count]
    actor, values, _ = trainer.model(obs, hidden, reset_before=reset)
    log_prob = hybrid_log_probability(
        actor[mask],
        _sequence_major(rollout.actions)[:count][mask],
        config=trainer.policy_config,
        pre_tanh=_sequence_major(rollout.pre_tanh)[:count][mask],
        distribution_override=trainer.exploration.distribution_override,
    )
    advantage = _sequence_major(rollout.advantages)[:count][mask]
    loss = (
        -(log_prob * advantage.detach()).mean()
        + 0.5 * (values[mask] - _sequence_major(rollout.returns)[:count][mask]).square().mean()
    )
    loss.backward()
    report["checks"].update(
        finite_loss=bool(torch.isfinite(loss)),
        finite_gradients=all(
            bool(torch.isfinite(p.grad).all())
            for p in trainer.model.parameters()
            if p.grad is not None
        ),
        finite_rollout=all(
            bool(torch.isfinite(getattr(rollout, name)).all())
            for name in ("observations", "actions", "rewards", "values", "advantages")
        ),
        model_unchanged=tree_sha256(trainer.model.state_dict()) == before_model,
        optimizer_unchanged=tree_sha256(trainer.optimizer.state_dict()) == before_optimizer,
        no_update_accepted=trainer.accepted_updates_total == PARENT_UPDATE,
    )
    trainer.optimizer.zero_grad(set_to_none=True)
    report.update(
        verdict="PASS" if all(report["checks"].values()) else "FAIL",
        optimizer_step_taken=False,
        source_checkpoint_sha256=engine.sha256_file(PARENT),
        rollout_horizon=rollout.horizon,
        rollout_logical_gib=rollout.logical_bytes / 2**30,
        minibatch_sequences=count,
        minibatch_ticks=count * HORIZON,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(trainer.device) / 2**30,
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(trainer.device) / 2**30,
        scope="one full-size rollout and one full recurrent-minibatch backward, no Adam step",
    )
    engine.write_json(RESULTS / "memory_preflight.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


def parser():
    result = engine.parser()
    result.add_argument("--prepare-amendment", action="store_true")
    result.add_argument("--memory-preflight-only", action="store_true")
    result.set_defaults(run_dir=str(RUN_DIR), resume=str(STARTUP), continue_after_600=False)
    return result


def run(args):
    configure_engine()
    if args.continue_after_600 or args.worlds != 32768 or Path(args.run_dir).resolve() != RUN_DIR:
        raise ValueError("preserve the 32768-world, total-100-update bound and isolated directory")
    if args.prepare_amendment:
        if not args.implementation_commit:
            raise ValueError("implementation commit required before freezing authority")
        prepare(args.implementation_commit)
        return 0
    if args.write_authority or args.rollout_preflight_only:
        raise ValueError("use --prepare-amendment or --memory-preflight-only")
    load_authority()
    load_launch_authority()
    if not args.resume:
        raise ValueError("resume the preserved current policy, do not restart weights")
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    validate_resume_payload(payload)
    if payload["accepted_updates_total"] == PARENT_UPDATE:
        record = json.loads((RESULTS / "transition.json").read_text())
        if engine.sha256_file(args.resume) != record["startup_sha256"]:
            raise ValueError("initial transition checkpoint identity mismatch")
    del payload
    if args.memory_preflight_only:
        return memory_preflight(args)
    if not args.preflight_only:
        evidence = json.loads((RESULTS / "memory_preflight.json").read_text())
        if evidence["verdict"] != "PASS" or evidence["authority_sha256"] != engine.sha256_file(
            AUTHORITY
        ):
            raise ValueError("exact-scale memory preflight required before learning")
    return engine.run(args)


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
