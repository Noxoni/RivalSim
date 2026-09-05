"""Isolated exact-scale no-step execution comparison; not a training entrypoint."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as campaign  # noqa: E402
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402
from rivalsim.rival2_recurrent_ppo import _sequence_major, recurrent_minibatch_step  # noqa: E402

OLD_AUTHORITY = "BE82C618296A6124858BFCCE99EB66E779074B524108476690D2E227F6B8CB4C"
CANDIDATE_PATHS = {
    "rivalsim/recurrent_execution.py",
    "rivalsim/rival2_unified_policy.py",
    "rivalsim/rival2_recurrent_ppo.py",
    "rivalsim/rival2_recurrent_training.py",
    "rivalsim/rival2_ssl_foundation_training.py",
    "benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py",
}


def timed(function, repetitions=5):
    function()  # warmup, never an optimizer step
    values = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return {"seconds": values, "median_seconds": statistics.median(values)}


def close(a, b):
    return bool(torch.allclose(a, b, atol=3e-6, rtol=1e-4))


def run(args):
    torch.set_num_threads(2)
    campaign.configure_engine()
    authority = Path(args.authority)
    assert campaign.engine.sha256_file(authority) == OLD_AUTHORITY
    frozen = json.loads(authority.read_text())
    for path, sha in frozen["implementation_sha256"].items():
        if path not in CANDIDATE_PATHS:
            assert campaign.engine.sha256_file(ROOT / path) == sha, path
    # Baseline frozen learning/environment authority is used ONLY in this isolated
    # no-step comparison. The production loader remains strict and fail-closed.
    campaign.engine.load_authority = lambda: frozen
    trainer, source = campaign.make_trainer(Path(args.collision_root), worlds=32768)
    trainer.load_checkpoint(args.resume)
    trainer.optimize_execution = True
    preflight = campaign.preflight(trainer, source, exact_scale=True)
    assert preflight["verdict"] == "PASS", preflight
    model_hash = campaign.tree_sha256(trainer.model.state_dict())
    optimizer_hash = campaign.tree_sha256(trainer.optimizer.state_dict())
    accepted = trainer.accepted_updates_total
    source_hash = campaign.engine.sha256_file(Path(args.resume))
    trainer.set_exploration(campaign.prior.restart_exploration(accepted))
    start = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize()
    collect_seconds = time.perf_counter() - start
    checks = {}
    component = {}
    with torch.no_grad():
        obs = rollout.observations[-1]
        flat = obs.reshape(-1, 182)
        h = trainer._flat_hidden()
        fh = trainer._flat_frozen_hidden()
        resets = trainer.reset_before
        actor, value, hidden = trainer.model(flat, h)
        fast_actor, fast_hidden = trainer.model.forward_actor(flat, h)
        fast_value = trainer.model.isolated_value(flat)
        checks.update(
            actor_close=close(actor, fast_actor),
            value_exact=torch.equal(value, fast_value),
            hidden_close=close(hidden, fast_hidden),
        )
        full, _, full_hidden = trainer.frozen_v5(flat, fh, reset_before=resets.reshape(-1))
        active, active_hidden = trainer.active_frozen_forward(obs, resets)
        rows = trainer.world_rows[trainer.opponent_family == 2]
        idx = rows * 2 + 1 - trainer.rival_side[rows]
        checks.update(
            active_actor_close=close(full[idx], active[idx]),
            active_hidden_close=close(full_hidden[:, idx], active_hidden[:, idx]),
            active_buttons_exact=torch.equal(
                deterministic_hybrid_action(full[idx])[:, 5:],
                deterministic_hybrid_action(active[idx])[:, 5:],
            ),
        )
        errors = {
            "actor_max_abs": float((actor - fast_actor).abs().max()),
            "active_actor_max_abs": float((full[idx] - active[idx]).abs().max()),
            "active_hidden_max_abs": float(
                (full_hidden[:, idx] - active_hidden[:, idx]).abs().max()
            ),
        }
        for name, baseline, candidate in (
            (
                "next_value",
                lambda: trainer.model(flat, h),
                lambda: trainer.model.isolated_value(flat),
            ),
            (
                "frozen_opponent",
                lambda: trainer.frozen_v5(flat, fh, reset_before=resets.reshape(-1)),
                lambda: trainer.active_frozen_forward(obs, resets),
            ),
        ):
            component[name] = {"baseline": timed(baseline, 10), "candidate": timed(candidate, 10)}
        del (
            actor,
            value,
            hidden,
            fast_actor,
            fast_value,
            fast_hidden,
            full,
            full_hidden,
            active,
            active_hidden,
        )
    rollout.compute_gae(trainer.ppo_config)
    mask = _sequence_major(rollout.train_mask)
    raw = _sequence_major(rollout.advantages)
    family = _sequence_major(rollout.opponent_family)
    normalized = torch.zeros_like(raw)
    for family_id in torch.unique(family[mask]).tolist():
        selected = mask & (family == int(family_id))
        values = raw[selected]
        normalized[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    kwargs = dict(
        observation=_sequence_major(rollout.observations),
        initial_hidden=rollout.initial_hidden.reshape(1, -1, 256),
        reset_before=_sequence_major(rollout.reset_before),
        action=_sequence_major(rollout.actions),
        pre_tanh=_sequence_major(rollout.pre_tanh),
        old_log_probability=_sequence_major(rollout.old_log_probability),
        normalized_advantage=normalized,
        returns=_sequence_major(rollout.returns),
        train_mask=mask,
        sequence_microbatch_size=182,
        take_step=False,
    )
    minibatches = []
    for offset in (0, 182, 910, 3640, 8190):
        kwargs["sequence_index"] = torch.arange(offset, offset + 182, device=trainer.device)
        row = {"sequence_offset": offset}
        reference = None
        for optimized in (False, True):

            def step(optimized=optimized):
                return recurrent_minibatch_step(
                    trainer.model,
                    trainer.optimizer,
                    trainer.ppo_config,
                    trainer.exploration.distribution_override,
                    **kwargs,
                    optimize_execution=optimized,
                )

            timing = timed(step, 5)
            metrics = step()
            grads = {
                n: p.grad.detach().cpu().clone()
                for n, p in trainer.model.named_parameters()
                if p.grad is not None
            }
            if reference is None:
                reference = grads, {n: v.detach().cpu().clone() for n, v in metrics.items()}
            else:
                row["gradient_close"] = all(close(g, reference[0][n]) for n, g in grads.items())
                row["max_gradient_abs_error"] = max(
                    float((g - reference[0][n]).abs().max()) for n, g in grads.items()
                )
                row["metrics_close"] = all(
                    close(v.cpu(), reference[1][n]) for n, v in metrics.items()
                )
            row["candidate" if optimized else "baseline"] = timing
        minibatches.append(row)
        print(json.dumps(row), flush=True)
    checks.update(
        finite_rollout=all(
            bool(torch.isfinite(chunk).all())
            for name in ("observations", "actions", "rewards", "values", "advantages")
            for chunk in getattr(rollout, name).split(16)
        ),
        gradients_close=all(r["gradient_close"] for r in minibatches),
        metrics_close=all(r["metrics_close"] for r in minibatches),
        model_unchanged=model_hash == campaign.tree_sha256(trainer.model.state_dict()),
        optimizer_unchanged=optimizer_hash == campaign.tree_sha256(trainer.optimizer.state_dict()),
        no_accepted_update=accepted == trainer.accepted_updates_total,
        source_unchanged=source_hash == campaign.engine.sha256_file(Path(args.resume)),
        physical_tensor_headroom=torch.cuda.max_memory_allocated() < 28 * 2**30,
    )
    report = dict(
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        errors=errors,
        source=str(Path(args.resume).resolve()),
        source_sha256=source_hash,
        accepted_update=accepted,
        authority_sha256=OLD_AUTHORITY,
        no_optimizer_step=True,
        worlds=32768,
        horizon=360,
        minibatch_sequences=182,
        rollout_logical_gib=rollout.logical_bytes / 2**30,
        optimized_collect_seconds=collect_seconds,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        component_timings=component,
        minibatches=minibatches,
        implementation_sha256={
            p: campaign.engine.sha256_file(ROOT / p) for p in sorted(CANDIDATE_PATHS)
        },
        scope="Matched real-input no-step timings; not an end-to-end update speed claim",
    )
    report["production_preflight"] = preflight
    campaign.engine.write_json(Path(args.output), report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"component_timings", "minibatches"}},
            indent=2,
        )
    )
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--collision-root", default="G:/dev/RLBot-Rival/bot/collision_meshes")
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
