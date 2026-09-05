"""Exploration-only amendment of the existing bounded 120 Hz long-trace run.

No new lineage, optimizer, reward, cadence, opponent mix, or campaign budget.
The original runner stays immutable and rejects this amended authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as c  # noqa: E402
from rivalsim.rival2_exploration import FreshHumanSeedExploration  # noqa: E402
from rivalsim.rival2_policy import hybrid_log_probability, sample_hybrid_action  # noqa: E402
from rivalsim.rival2_recurrent_ppo import _sequence_major, recurrent_minibatch_step  # noqa: E402

EVIDENCE = c.RESULTS / "strong_exploration"
ANCHOR = EVIDENCE / "anchor.json"
OWNER = "strong-exploration-amendment-20260905"
OLD_AUTHORITY = "AEA37884836DD4055BACCD605E3F660A0EC2D70162DF5C3032DF6C43134DFD72"
OLD_LAUNCH = "E8B08BD9A0FA0CBAF81B352F98AC94DECBFBAA79E32EA932103B35B2AE10BB1A"
VERSION = "RIVAL2_SSL_STRONG_EXPLORATION_V1"
CONTRACT = {
    "version": VERSION,
    "analog": {"sigma": 0.30, "space": "pre_tanh", "raw_actor_log_std_bypassed": True},
    "buttons": {"temperature": 1.0, "effective_logits": "learned_logits/temperature"},
    "activation": (
        "first new rollout after preserved accepted-boundary restart; constant thereafter"
    ),
    "coherence": ["sampling", "stored log probability", "PPO log probability", "entropy", "KL"],
    "scope": "current trainable Rival only; opponents and deterministic evaluation unchanged",
    "entropy_loss_coefficient_changed": False,
    "new_action_hold_or_correlated_noise": False,
    "kl_telemetry_only": True,
}
CONTRACT_HASH = (
    hashlib.sha256(json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode())
    .hexdigest()
    .upper()
)


def exploration_for_update(accepted_update):
    if accepted_update < 0:
        raise ValueError("negative accepted update")
    return FreshHumanSeedExploration(
        accepted_update=int(accepted_update),
        normalized_progress=1.0,
        analog_sigma=CONTRACT["analog"]["sigma"],
        analog_log_sigma=math.log(CONTRACT["analog"]["sigma"]),
        button_temperature=CONTRACT["buttons"]["temperature"],
        version=VERSION,
        contract_sha256=CONTRACT_HASH,
    )


def authority_payload(commit, created):
    original = EVIDENCE / "original/authority.json"
    if c.engine.sha256_file(original) != OLD_AUTHORITY:
        raise ValueError("original authority archive changed")
    payload = json.loads(original.read_text())
    # No changes to any pre-existing runtime file are needed by this amendment.
    for path, digest in payload["implementation_sha256"].items():
        if c.engine.sha256_file(ROOT / path) != digest:
            raise ValueError(f"unrelated runtime changed: {path}")
    payload.update(
        implementation_commit=commit,
        created_utc=created,
        supersedes_authority_sha256=OLD_AUTHORITY,
        supersession_reason=(
            "user requested substantially stronger exploration "
            "before any 30Hz/fresh-weight transition"
        ),
    )
    path = Path(__file__).relative_to(ROOT).as_posix()
    payload["implementation_sha256"][path] = c.engine.sha256_file(Path(__file__))
    payload["exploration"] = {"contract": CONTRACT, "contract_sha256": CONTRACT_HASH}
    payload["exploration_amendment"] = {
        "anchor": json.loads(ANCHOR.read_text()),
        "anchor_sha256": c.engine.sha256_file(ANCHOR),
        "model_optimizer_counters_rng_preserved": True,
        "new_rollout_required": True,
        "old_rollout_reused": False,
        "budget_and_deadline_unchanged": True,
        "restart_metadata_is_historical": (
            "restart block describes local zero, not active exploration"
        ),
    }
    return payload


def load_authority():
    payload = json.loads(c.AUTHORITY.read_text())
    if payload != authority_payload(payload["implementation_commit"], payload["created_utc"]):
        raise ValueError("strong exploration authority mismatch")
    if c.engine.sha256_file(c.engine.SOURCE) != c.engine.SOURCE_SHA256:
        raise ValueError("original V5 root changed")
    return payload


def launch_payload():
    return {
        **c.launch_payload(),
        "exploration_contract_sha256": CONTRACT_HASH,
        "exploration_anchor_sha256": c.engine.sha256_file(ANCHOR),
    }


def load_launch():
    payload = json.loads(c.LAUNCH.read_text())
    if payload != launch_payload():
        raise ValueError("strong exploration launch mismatch")
    return payload


def validate_resume(payload):
    c.validate_resume_payload(payload)
    anchor = json.loads(ANCHOR.read_text())
    if payload.get("phase_transition", {}).get("strong_exploration") != anchor:
        raise ValueError("missing/wrong strong exploration checkpoint lineage")
    if payload["accepted_updates_total"] < anchor["accepted_update"]:
        raise ValueError("cannot roll back before exploration transition")
    # The transition checkpoint truthfully retains its last old rollout metadata.
    if (
        payload["accepted_updates_total"] > anchor["accepted_update"]
        and payload["exploration"]
        != exploration_for_update(payload["accepted_updates_total"] - 1).as_dict()
    ):
        raise ValueError("checkpoint trained with unexpected exploration")


def preflight(trainer, source, *, exact_scale):
    trainer.set_exploration(exploration_for_update(trainer.accepted_updates_total))
    report = c.preflight(trainer, source, exact_scale=exact_scale)
    # The inherited check refers to the unchanged local-zero historical schedule.
    report["checks"]["active_strong_exploration"] = (
        trainer.exploration.as_dict()
        == exploration_for_update(trainer.accepted_updates_total).as_dict()
    )
    report["active_exploration"] = trainer.exploration.as_dict()
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    return report


def configure():
    c.configure_engine()
    c.engine.load_authority = load_authority
    c.engine.load_schedule_authority = load_launch
    c.engine.exploration_for_update = exploration_for_update
    c.engine.preflight = preflight


def no_training_process():
    import psutil

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if (
            process.info["pid"] == os.getpid()
            or "python" not in (process.info["name"] or "").lower()
        ):
            continue
        command = " ".join(process.info["cmdline"] or [])
        if (
            "run_rival2_ssl_foundation_v5_long_trace_v1.py" in command
            or "run_rival2_ssl_foundation_strong_exploration.py" in command
        ) and not any(
            flag in command for flag in ("--prepare-exploration", "--no-step-validation")
        ):
            raise RuntimeError(f"training/evaluation worker still active: {process.info['pid']}")


def prepare(args):
    no_training_process()
    c.configure_engine()
    c.load_authority()
    c.load_launch_authority()
    if (
        c.engine.sha256_file(c.AUTHORITY) != OLD_AUTHORITY
        or c.engine.sha256_file(c.LAUNCH) != OLD_LAUNCH
    ):
        raise ValueError("unexpected pre-amendment authority")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        != args.implementation_commit
    ):
        raise ValueError("implementation must be committed before authority")
    stop = json.loads((c.RUN_DIR / "STOP_REQUESTED").read_text())
    if stop.get("owner") != OWNER:
        raise ValueError("not this amendment's owned stop")
    summary = json.loads((c.RESULTS / "training_summary.json").read_text())
    source = Path(args.resume).resolve()
    before = torch.load(source, map_location="cpu", weights_only=False)
    c.validate_resume_payload(before)
    update = before["accepted_updates_total"]
    if not (
        10 < update < 100
        and summary["accepted_updates"] == update
        and summary["hard_failure"] is None
        and summary["stop_reason"] == "user_requested_stop_at_accepted_boundary"
    ):
        raise ValueError("need clean latest accepted boundary below original update100 limit")
    rows = [
        json.loads(line) for line in (c.RESULTS / "training_curve.jsonl").read_text().splitlines()
    ]
    if rows[-1]["accepted_update"] != update:
        raise ValueError("resume is not latest accepted checkpoint")
    archive = EVIDENCE / "original"
    archive.mkdir(parents=True, exist_ok=False)
    for path in (
        c.AUTHORITY,
        c.LAUNCH,
        c.RESULTS / "memory_preflight.json",
        c.RESULTS / "resume_preflight.json",
        c.RESULTS / "training_summary.json",
        c.RESULTS / "snapshot_manifest.json",
        c.RUN_DIR / "campaign_state.json",
    ):
        shutil.copy2(path, archive / path.name)
    checkpoint_dir = c.CHECKPOINT.parent
    preserved = checkpoint_dir / f"pre_strong_exploration_u{update:04d}.pt"
    resume = checkpoint_dir / f"strong_exploration_transition_u{update:04d}.pt"
    if preserved.exists() or resume.exists():
        raise FileExistsError("preserve existing exploration transition")
    shutil.copy2(source, preserved)
    anchor = {
        "version": VERSION,
        "accepted_update": update,
        "first_stronger_update": update + 1,
        "source": preserved.relative_to(ROOT).as_posix(),
        "source_sha256": c.engine.sha256_file(preserved),
        "previous_authority_sha256": OLD_AUTHORITY,
        "contract_sha256": CONTRACT_HASH,
    }
    c.engine.write_json(ANCHOR, anchor)
    c.engine.write_json(
        c.AUTHORITY, authority_payload(args.implementation_commit, c.engine.utc_now())
    )
    c.engine.write_json(c.LAUNCH, launch_payload())
    ah, lh = c.engine.sha256_file(c.AUTHORITY), c.engine.sha256_file(c.LAUNCH)
    after = copy.deepcopy(before)
    after["source"].update(authority_sha256=ah, schedule_authority_sha256=lh)
    after["phase_transition"].update(
        authority_sha256=ah, schedule_authority_sha256=lh, strong_exploration=anchor
    )
    after["phase_transition"]["credit_assignment_amendment"]["authority_sha256"] = ah
    checks = {
        key: c.tree_sha256(before[key]) == c.tree_sha256(after[key])
        for key in before
        if key not in ("source", "phase_transition")
    }
    if not all(checks.values()):
        raise RuntimeError("transition altered learning state")
    validate_resume(after)
    torch.save(after, resume)
    for path in (c.RUN_DIR / "campaign_state.json", c.RESULTS / "snapshot_manifest.json"):
        data = json.loads(path.read_text())
        data.update(authority_sha256=ah, schedule_authority_sha256=lh)
        c.engine.write_json(path, data)
    c.engine.write_json(
        EVIDENCE / "transition.json",
        {
            **anchor,
            "resume": str(resume),
            "resume_sha256": c.engine.sha256_file(resume),
            "authority_sha256": ah,
            "launch_authority_sha256": lh,
            "unchanged_checkpoint_fields": checks,
            "optimizer_step_taken": False,
            "resume_physics": (
                "existing fresh episodes and zero recurrent hidden; saved RNG restored"
            ),
        },
    )
    configure()
    load_authority()
    load_launch()
    print(
        json.dumps({"accepted_update": update, "resume": str(resume), "checks": checks}, indent=2)
    )


def no_step_validation(args):
    no_training_process()
    trainer, source = c.make_trainer(Path(args.collision_root), worlds=32768)
    trainer.load_checkpoint(args.resume)
    report = preflight(trainer, source, exact_scale=True)
    if report["verdict"] != "PASS":
        raise ValueError(report)
    before_model = c.tree_sha256(trainer.model.state_dict())
    before_adam = c.tree_sha256(trainer.optimizer.state_dict())
    accepted = trainer.accepted_updates_total
    rollout = trainer.collect_rollout()
    rollout.compute_gae(trainer.ppo_config)
    mask = _sequence_major(rollout.train_mask)
    raw = _sequence_major(rollout.advantages)
    family = _sequence_major(rollout.opponent_family)
    normalized = torch.zeros_like(raw)
    for family_id in torch.unique(family[mask]).tolist():
        selected = mask & (family == int(family_id))
        values = raw[selected]
        normalized[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    count = rollout.sequence_layout(65536).sequences_per_minibatch
    obs = _sequence_major(rollout.observations)
    hidden = rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim)
    reset = _sequence_major(rollout.reset_before)
    action = _sequence_major(rollout.actions)
    pre_tanh = _sequence_major(rollout.pre_tanh)
    old_logp = _sequence_major(rollout.old_log_probability)
    with torch.no_grad():
        actor, _ = trainer.model.forward_actor(
            obs[:count], hidden[:, :count], reset_before=reset[:count]
        )
        selected_actor = actor[mask[:count]]
        logp = hybrid_log_probability(
            selected_actor,
            action[:count][mask[:count]],
            config=trainer.policy_config,
            pre_tanh=pre_tanh[:count][mask[:count]],
            distribution_override=trainer.exploration.distribution_override,
        )
        error = (logp - old_logp[:count][mask[:count]]).abs()
        logp_max = error.max().item()
        logp_mean = error.mean().item()
        distribution_effect = {}
        for name, exploration in (
            ("old", c.prior.restart_exploration(accepted)),
            ("strong", trainer.exploration),
        ):
            sample = sample_hybrid_action(
                selected_actor,
                generator=torch.Generator(device=trainer.device).manual_seed(39313),
                config=trainer.policy_config,
                distribution_override=exploration.distribution_override,
            )
            distribution_effect[name] = {
                "pre_tanh_noise_std_by_analog_channel": (sample.pre_tanh - selected_actor[:, :5])
                .std(dim=0)
                .tolist(),
                "action_rmse_from_deterministic_by_analog_channel": (
                    sample.action[:, :5] - selected_actor[:, :5].tanh()
                )
                .square()
                .mean(dim=0)
                .sqrt()
                .tolist(),
                "button_disagreement_from_deterministic_by_channel": (
                    sample.action[:, 5:].bool() != (selected_actor[:, 10:] > 0)
                )
                .float()
                .mean(dim=0)
                .tolist(),
            }
    trainer.model.train()
    metrics = recurrent_minibatch_step(
        trainer.model,
        trainer.optimizer,
        trainer.ppo_config,
        trainer.exploration.distribution_override,
        observation=obs,
        initial_hidden=hidden,
        reset_before=reset,
        action=action,
        pre_tanh=pre_tanh,
        old_log_probability=old_logp,
        normalized_advantage=normalized,
        returns=_sequence_major(rollout.returns),
        train_mask=mask,
        sequence_index=torch.arange(count, device=trainer.device),
        sequence_microbatch_size=count,
        optimize_execution=True,
        take_step=False,
    )
    report["checks"].update(
        finite_rollout=all(
            bool(torch.isfinite(part).all())
            for name in ("observations", "actions", "rewards", "values", "advantages")
            for part in getattr(rollout, name).split(16)
        ),
        finite_backward=all(bool(torch.isfinite(v).all()) for v in metrics.values()),
        finite_gradients=all(
            bool(torch.isfinite(p.grad).all())
            for p in trainer.model.parameters()
            if p.grad is not None
        ),
        sampling_ppo_logp_agree=logp_max < 0.005,
        model_unchanged=c.tree_sha256(trainer.model.state_dict()) == before_model,
        adam_unchanged=c.tree_sha256(trainer.optimizer.state_dict()) == before_adam,
        accepted_counter_unchanged=trainer.accepted_updates_total == accepted,
        gpu_memory_headroom=torch.cuda.max_memory_allocated() < 28 * 2**30,
    )
    report.update(
        verdict="PASS" if all(report["checks"].values()) else "FAIL",
        source_checkpoint_sha256=c.engine.sha256_file(Path(args.resume)),
        no_optimizer_step=True,
        worlds=32768,
        horizon=360,
        minibatch_sequences=count,
        log_probability_max_abs_error=logp_max,
        log_probability_mean_abs_error=logp_mean,
        matched_actor_distribution_effect=distribution_effect,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        rollout=trainer.last_rollout_metrics,
        interpretation="execution/distribution validation, not learned gameplay improvement",
    )
    c.engine.write_json(EVIDENCE / "no_step_validation.json", report)
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        raise RuntimeError("strong exploration validation failed")


def main():
    torch.set_num_threads(2)
    parser = c.engine.parser()
    parser.add_argument("--prepare-exploration", action="store_true")
    parser.add_argument("--no-step-validation", action="store_true")
    parser.set_defaults(run_dir=str(c.RUN_DIR), continue_after_600=False)
    args = parser.parse_args()
    if args.worlds != 32768 or Path(args.run_dir).resolve() != c.RUN_DIR or args.continue_after_600:
        raise ValueError("preserve exact world count, existing directory and total100 bound")
    if args.write_authority or args.rollout_preflight_only or not args.resume:
        raise ValueError("use owned preparation and the latest accepted checkpoint")
    if args.prepare_exploration:
        prepare(args)
        return 0
    configure()
    load_authority()
    load_launch()
    validate_resume(torch.load(args.resume, map_location="cpu", weights_only=False))
    if args.no_step_validation:
        no_step_validation(args)
        return 0
    evidence = json.loads((EVIDENCE / "no_step_validation.json").read_text())
    if evidence["verdict"] != "PASS" or evidence["authority_sha256"] != c.engine.sha256_file(
        c.AUTHORITY
    ):
        raise ValueError("prospective distribution validation required")
    return c.engine.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
