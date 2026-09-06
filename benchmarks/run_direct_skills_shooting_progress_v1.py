"""Bounded +650 -> +675 curriculum-only shooting-pressure continuation."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks import run_rival2_direct_skills_v1 as base
from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    append_json,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from rivalsim.direct_skills_exploration_v1 import TEMPERATURE, TrainingExplorationPolicy
from benchmarks import run_direct_skills_exploration_followup_v1 as prior
from rivalsim.direct_skills_shooting_curriculum_v1 import (
    scenarios as training_scenarios, build as build_training_scenarios, curriculum_authority,
)
from rivalsim.fresh_ground_30hz import content_hash, ppo_config, scenario_hash

VERSION = "RIVAL2_DIRECT_SKILLS_SHOOTING_PROGRESS_V1"
PRIOR_AMENDMENT_SHA = "2E6DC900CFEFE85710FF6111F2AAA6692F1B66C15FE8C52BB3B8CD4945385843"

OUT = base.RESULTS / "shooting_progress_v1"
SOURCE = base.CHECKPOINTS / "plus_000650.pt"
SOURCE_SHA = "4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F"
START, REVIEW = 650, 675
SOURCES = (
    "benchmarks/run_direct_skills_shooting_progress_v1.py",
    "rivalsim/direct_skills_shooting_curriculum_v1.py",
    "tests/test_direct_skills_shooting_progress.py",
)


def amendment():
    return dict(
        version=VERSION, parent=SOURCE.relative_to(ROOT).as_posix(), parent_sha256=SOURCE_SHA,
        start_offset=START, review_boundary=REVIEW,
        root_authority_sha256=content_hash(base.authority()),
        root_package_sha256=content_hash(json.loads((base.RESULTS / "package.json").read_text())),
        prior_amendment_sha256=PRIOR_AMENDMENT_SHA,
        sole_learning_change="Replace half of finishing source states with physically valid "
        "intermediate recovering-to-set-keeper starts; about10percent of full source bank",
        curriculum=curriculum_authority(),
        decision_evidence="shooting_diagnostic_000650/RESULTS.md: matched650 scores49/64 "
        "initially open,50/64 recovering,5/64 established; firstRivalcontact64/64 in all modes",
        training_temperature=TEMPERATURE, ppo=asdict(ppo_config()),
        unchanged="Exact650 actor/critic/entity/GRU weights, Adam moments/counters and fourRNG. "
        "All rewards,32768worlds,30Hz/120Hz,four-tickhold,50/50Nexto/selfplay worlds, "
        "one-thirdNexto learner decisions, original family mix and remaining90percent source starts",
        initialization="Same latest accepted650, no fresh model or optimizer. New scenario bank "
        "with fresh episodes and zero hidden at resume, as already declared",
        architecture_change=False, reward_change=False, critic_change=False,
        inference="Original raw argmax, original fixed64cases/family and ten regulation matches",
        checkpoints="Every accepted update rolling; pre-step entry650, first accepted651, permanent675",
        evaluations="Unchanged original skill and corrected full-match method at675; compare650 "
        "and600. Do not substitute easier training starts for evaluation or select older model",
        end="Pause at675 for actual scoring/full-match review, not cancellation of user's goal",
        safety="Finite model/gradient/Adam and whole-update corruption rollback; KL telemetry ONLY",
    )


def verify(published=True):
    root, _ = prior.verify(published=published)
    assert content_hash(prior.amendment()) == PRIOR_AMENDMENT_SHA
    package = json.loads((OUT / "package.json").read_text())
    assert json.loads((OUT / "authority.json").read_text()) == amendment()
    assert package["authority_sha256"] == content_hash(amendment())
    assert sha(SOURCE) == SOURCE_SHA
    for path, digest in package["sources"].items():
        assert base.text_sha(ROOT / path) == digest, path
    for name, digest in package["evidence"].items():
        assert base.text_sha(OUT / name) == digest, name
    if published:
        paths = [
            *package["sources"],
            *[
                (OUT / name).relative_to(ROOT).as_posix()
                for name in ("authority.json", "package.json", *package["evidence"])
            ],
        ]
        for path in paths:
            remote = subprocess.check_output(["git", "show", "origin/main:" + path], cwd=ROOT)
            assert remote.replace(b"\r\n", b"\n") == (ROOT / path).read_bytes().replace(
                b"\r\n", b"\n"
            ), path
    return root, package


def load(payload):
    model = TrainingExplorationPolicy().cuda()
    model.load_state_dict(payload["model"], strict=True)
    assert model.config.content_hash == payload["policy_config_sha256"]
    assert ppo_config().content_hash == payload["ppo_config_sha256"]
    optimizer = base.fresh_entity_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    assert {int(s["step"]) for s in optimizer.state.values()} == {
        payload["cumulative_optimizer_steps"]
    }
    assert base.finite_model_and_optimizer(model, optimizer)
    assert tensor_hash(model.state_dict()) == tensor_hash(payload["model"])
    return model, optimizer


def adam_hash(optimizer):
    return tensor_hash(
        {
            f"{i}.{k}": v
            for i, state in optimizer.state_dict()["state"].items()
            for k, v in state.items()
        }
    )


def preflight():
    base.verify()
    assert sha(SOURCE) == SOURCE_SHA
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model, optimizer = load(payload)
    model_before, adam_before = tensor_hash(model.state_dict()), adam_hash(optimizer)
    env = base.DirectSkillsEnv(
        1024,
        base.COLLISION,
        device="cuda:0",
        seed=base.SEED,
        ssl_foundation_scenarios=training_scenarios(1024),
    )
    collector = base.DirectSkillCollector(env, model, seed=base.SEED)
    nexto_before = tensor_hash(collector.nexto.actor.state_dict())
    rollout = collector.collect()
    data = base.mixed_sequence_data(rollout, ppo_config())
    index = data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()
    loss, _ = base.joint_sequence_loss(model, data, index, ppo_config())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.isolated_value(data["observations"][index]).sum().backward()
    isolated = all(
        p.grad is None or not bool(p.grad.any())
        for n, p in model.named_parameters()
        if not n.startswith("critic.")
    )
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    original = base.EntityJointControlActorCritic().cuda().eval()
    original.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        args = (data["observations"][index], data["initial_hidden"][:, index])
        kw = dict(reset_before=data["reset_before"][index])
        soft, sv, sh = model(*args, **kw)
        raw, rv, rh = original(*args, **kw)
        logp, entropy = base.categorical_statistics(soft, data["action_indices"][index])
        _, raw_entropy = base.categorical_statistics(raw, data["action_indices"][index])
        mask = data["train_mask"][index]
    checks = dict(
        strict_parent_model_and_adam_load=True,
        parent_file_unchanged=sha(SOURCE) == SOURCE_SHA,
        entry_actor_argmax_exact=torch.equal(soft.argmax(-1), raw.argmax(-1)),
        exact_logits_scaling=torch.equal(soft, raw / TEMPERATURE),
        value_and_hidden_unchanged=torch.equal(sv, rv) and torch.equal(sh, rh),
        finite_gradient=bool(torch.isfinite(grad_norm)),
        critic_isolated=isolated,
        no_optimizer_steps={int(s["step"]) for s in optimizer.state.values()} == {144648},
        model_unchanged=model_before == tensor_hash(model.state_dict()),
        adam_unchanged=adam_before == adam_hash(optimizer),
        nexto_unchanged=nexto_before == tensor_hash(collector.nexto.actor.state_dict()),
        exact_actions=torch.equal(
            model.action_table[rollout.action_indices][rollout.train_mask],
            rollout.actions[rollout.train_mask],
        ),
        nexto_sample_third=collector.last_metrics["nexto_training_sample_count"] * 3
        == collector.last_metrics["trainable_agent_samples"],
        finite_model_adam=base.finite_model_and_optimizer(model, optimizer),
        native_reward_path_unchanged=env.world.gameplay_v3 is None
        and env.world.gameplay_120 is None,
    )
    result = dict(
        utc=utc(),
        worlds=1024,
        rollout_decisions=90,
        checks=checks,
        authority_sha256=content_hash(amendment()),
        optimizer_steps=0,
        parent_sha256=SOURCE_SHA,
        gradient_norm=float(grad_norm),
        same_visited_state_entropy_T1=float(raw_entropy[mask].mean()),
        same_visited_state_entropy_T2=float(entropy[mask].mean()),
        replay_max_logp_error=float(
            (logp[mask] - data["old_log_probability"][index][mask]).abs().max()
        ),
        replay_error_is_telemetry_not_a_new_guard=True,
        collection=collector.last_metrics,
    )
    write_json(OUT / "preflight.json", result)
    assert all(checks.values()), checks
    print(json.dumps({k: v for k, v in result.items() if k != "collection"}), flush=True)


def prepare():
    assert not (OUT / "authority.json").exists(), "Never overwrite frozen amendment"
    assert all(json.loads((OUT / "preflight.json").read_text())["checks"].values())
    training_bank, audit = build_training_scenarios(32768)
    write_json(OUT / "training_bank_audit.json", audit)
    write_json(OUT / "authority.json", amendment())
    write_json(
        OUT / "package.json",
        dict(
            authority_sha256=content_hash(amendment()),
            sources={p: base.text_sha(ROOT / p) for p in SOURCES},
            scenario_sha256=scenario_hash(training_bank),
            evidence={n: base.text_sha(OUT / n) for n in ("preflight.json", "tests.xml", "training_bank_audit.json")},
        ),
    )


def validate_latest(path, expected, *, external=base.EXTERNAL):
    """Check under the GPU lease. Never clear STOP or silently choose a parent."""
    if (external / "STOP").exists():
        raise RuntimeError("Respect STOP; do not clear it in a recovery launcher")
    latest = json.loads((external / "latest.json").read_text())
    state = json.loads((external / "campaign_state.json").read_text())
    offset = latest["accepted_updates"]
    if type(offset) is not int or not START <= offset <= REVIEW:
        raise RuntimeError("Latest outside this prospective block")
    if state["accepted_updates"] not in (offset, offset - 1):
        raise RuntimeError("State/latest discrepancy requires operational audit")
    if offset == REVIEW and state.get("status") == "stopped_for_curriculum_review":
        raise RuntimeError("Completed review boundary is not a crash")
    allowed = {content_hash(amendment())}
    # Entry and first publication can carry either the prior review status or
    # the new run's offset650 status. No later state can claim the prior block.
    if state["accepted_updates"] == START:
        allowed.add(PRIOR_AMENDMENT_SHA)
    if state.get("exploration_amendment_sha256") not in allowed:
        raise RuntimeError("Wrong exploration authority")
    if not expected or expected.upper() != latest["sha256"].upper():
        raise RuntimeError("Explicit resume must identify latest accepted checkpoint")
    if sha(Path(path)) != expected.upper():
        raise RuntimeError("Checkpoint hash mismatch; no fallback")
    return latest


def run(args):
    root_package, package = verify()
    validate_latest(args.resume, args.resume_sha256)
    source = Path(args.resume)
    assert args.resume_sha256 and sha(source) == args.resume_sha256.upper()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    offset = payload["accepted_updates"]
    assert START <= offset <= REVIEW
    assert payload["parent_sha256"] == base.PARENT_SHA
    assert payload["authority_sha256"] == content_hash(base.authority())
    if payload["exploration_amendment_sha256"] == PRIOR_AMENDMENT_SHA:
        assert offset == START
        assert sha(source) == SOURCE_SHA
    else:
        assert payload["exploration_amendment_sha256"] == content_hash(amendment())
        assert payload["exploration_parent_sha256"] == SOURCE_SHA
    model, optimizer = load(payload)
    samples, ticks, adam_steps = (
        payload[k]
        for k in (
            "direct_skill_samples",
            "direct_skill_physics_ticks",
            "cumulative_optimizer_steps",
        )
    )
    bank = training_scenarios(32768)
    assert scenario_hash(bank) == package["scenario_sha256"]
    assert scenario_hash(bank) != root_package["scenario_sha256"]
    env = base.DirectSkillsEnv(
        32768, base.COLLISION, device="cuda:0", seed=base.SEED, ssl_foundation_scenarios=bank
    )
    collector = base.DirectSkillCollector(env, model, seed=base.SEED)
    collector.generator.set_state(payload["policy_generator_state"].cpu())
    shuffle = torch.Generator(device="cuda:0")
    shuffle.set_state(payload["shuffle_generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu())
    latest = None

    def state(status, **kwargs):
        write_json(
            base.EXTERNAL / "campaign_state.json",
            dict(
                utc=utc(),
                pid=os.getpid(),
                status=status,
                accepted_updates=offset,
                cumulative_optimizer_steps=adam_steps,
                latest_checkpoint=latest,
                exploration_amendment=VERSION,
                exploration_amendment_sha256=content_hash(amendment()),
                review_boundary=REVIEW,
                **kwargs,
            ),
        )

    def save(path):
        p = dict(payload)
        p.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            package=root_package,
            exploration_amendment_sha256=content_hash(amendment()),
            exploration_package=package,
            exploration_parent_sha256=SOURCE_SHA,
            prior_exploration_amendment_sha256=PRIOR_AMENDMENT_SHA,
            training_distribution=dict(
                temperature=TEMPERATURE,
                on_policy_sampling_and_likelihood=True,
                deterministic_evaluation="raw argmax",
            ),
            accepted_updates=offset,
            direct_skill_samples=samples,
            direct_skill_physics_ticks=ticks,
            new_agent_samples=1676472162 + samples,
            new_physics_ticks=3456368640 + ticks,
            cumulative_optimizer_steps=adam_steps,
            fresh_optimizer=False,
            runtime_contract_hashes=env.contract_hashes,
            policy_generator_state=collector.generator.get_state(),
            shuffle_generator_state=shuffle.get_state(),
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(),
            last_training_metrics=collector.last_metrics,
            resume_count=payload.get("resume_count", 0) + 1,
            scenario_reset_note=amendment()["initialization"],
            training_curriculum_authority_sha256=content_hash(curriculum_authority()),
            effective_training_scenario_sha256=package["scenario_sha256"],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        with tmp.open("wb") as f:
            torch.save(p, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return dict(path=str(path), sha256=sha(path), accepted_updates=offset)

    def checkpoint():
        identity = save(base.EXTERNAL / f"rolling_{offset % 2}.pt")
        write_json(base.EXTERNAL / "latest.json", identity)
        return identity

    def evaluate():
        state("evaluating")
        frozen = base.CHECKPOINTS / f"plus_{offset:06d}.pt"
        if frozen.exists():
            stored = torch.load(frozen, map_location="cpu", weights_only=False)
            assert stored["exploration_amendment_sha256"] == content_hash(amendment())
            assert tensor_hash(stored["model"]) == tensor_hash(model.state_dict())
            del stored
            identity = dict(path=str(frozen), sha256=sha(frozen), accepted_updates=offset)
        else:
            identity = save(frozen)
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        common = dict(
            utc=utc(),
            accepted_updates=offset,
            checkpoint=identity,
            optimizer_steps=0,
            authority_sha256=content_hash(base.authority()),
            exploration_amendment_sha256=content_hash(amendment()),
        )
        skill_path = base.RESULTS / f"evaluation_{offset:06d}.json"
        if not skill_path.exists():
            write_json(skill_path, dict(common, skills=base.skill_evaluation(model)))
        match_path = base.RESULTS / f"full_match_{offset:06d}.json"
        if not match_path.exists():
            with base.owned_match_stream():
                runner = base.CandidateMatchRunner(frozen, identity["sha256"], entity=True)
                elapsed = runner.run_ticks(base.REGULATION_TICKS).seconds
                for _ in range(base.OVERTIME_CAP_TICKS // 600):
                    if bool(runner.phase_status()["done"].all()):
                        break
                    elapsed += runner.run_ticks(600).seconds
                raw = runner.export()["raw"]
                assert not bool(raw["goal_overflow"].any())
                assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
                assert sha(frozen) == identity["sha256"]
                write_json(
                    match_path,
                    dict(
                        common,
                        match_reset_version=base.MATCH_RESET_VERSION,
                        runtime_package_sha256=content_hash(root_package),
                        evaluation_runtime_note="Original evaluation sources unchanged; "
                        "training-only amendment explicit",
                        summary=base.summarize(raw),
                        raw={k: v.tolist() for k, v in raw.items()},
                        wall_seconds=elapsed,
                        hidden_resets=runner.hidden_reset_count.cpu().tolist(),
                        model_unchanged=True,
                        checkpoint_unchanged=True,
                    ),
                )
                print("MATCH_EVAL " + json.dumps(base.summarize(raw)), flush=True)
                del runner
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        if offset == START:
            entry_path = base.CHECKPOINTS / "shooting_entry_000650.pt"
            if not entry_path.exists():
                save(entry_path)
        latest = checkpoint()
        while offset < REVIEW and not (base.EXTERNAL / "STOP").exists():
            state("rollout")
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            rollout = collector.collect()
            rollout_seconds = time.monotonic() - started
            state("optimizing")
            started = time.monotonic()
            ppo = base.mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic() - started
            offset += 1
            adam_steps += ppo["optimizer_steps"]
            samples += collector.last_metrics["trainable_agent_samples"]
            ticks += collector.last_metrics["physical_physics_ticks"]
            del rollout
            latest = checkpoint()
            if offset == START + 1:
                first_path = base.CHECKPOINTS / "shooting_resume_000651.pt"
                if not first_path.exists():
                    save(first_path)
            row = dict(
                utc=utc(),
                accepted_updates=offset,
                cumulative_optimizer_steps=adam_steps,
                samples=samples,
                physical_ticks=ticks,
                ppo=ppo,
                training=collector.last_metrics,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                exploration_amendment_sha256=content_hash(amendment()),
                training_temperature=TEMPERATURE,
            )
            append_json(base.EXTERNAL / "training_curve.jsonl", row)
            print("ACCEPTED " + json.dumps(row), flush=True)
        if offset == REVIEW:
            evaluate()
            state("stopped_for_curriculum_review")
            # Deliberate review boundary, not cancellation/nonfinite/KL rejection.
            marker = base.EXTERNAL / "STOP"
            if not marker.exists():
                marker.write_text(
                    "Agent review boundary675: shooting-pressure curriculum block complete. "
                    "Review newly completed skill/fullmatch evaluations before continuation.\n"
                )
        else:
            latest = save(base.CHECKPOINTS / f"paused_{offset:06d}_exploration.pt")
            state("stopped_at_accepted_boundary")
    except Exception as exc:
        failure = dict(
            utc=utc(),
            accepted_updates=offset,
            latest_checkpoint=latest,
            exception=repr(exc),
            traceback=traceback.format_exc(),
            automatic_retry=False,
            exploration_amendment=VERSION,
        )
        write_json(base.EXTERNAL / "failure_shooting_progress.json", failure)
        state("failed", failure=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "verify", "run"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    if args.command == "run" and (args.resume is None or not args.resume_sha256):
        parser.error("run requires explicit --resume and --resume-sha256 for latest accepted state")
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    else:
        with base.gpu_lease():
            preflight() if args.command == "preflight" else run(args)
