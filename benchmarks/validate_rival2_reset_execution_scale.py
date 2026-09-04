"""Real 32768-world/360-tick rollout and no-step PPO execution comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as campaign  # noqa: E402
from benchmarks.benchmark_rival2_reset_execution import CANDIDATES  # noqa: E402
from rivalsim.rival2_recurrent_ppo import _sequence_major, recurrent_minibatch_step  # noqa: E402


def run(args):
    torch.set_num_threads(2)
    campaign.configure_engine()
    # Isolated NO-STEP candidate comparison, not the production launch path.
    # Retain the frozen environment/config authority while comparing execution
    # implementations. Production must separately rebind and commit its hashes.
    authority_path = Path(args.authority)
    frozen = json.loads(authority_path.read_text())
    if campaign.engine.sha256_file(authority_path) != (
        "6DB1F9F8EE1C564853CE0C731DC8DAC24FBA0DF26D5513EC9B308F4B57E1ADAE"
    ):
        raise ValueError("comparison requires the archived pre-repair authority")
    for name, digest in frozen["implementation_sha256"].items():
        if (
            name != "rivalsim/rival2_unified_policy.py"
            and name != ("benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py")
            and campaign.engine.sha256_file(ROOT / name) != digest
        ):
            raise ValueError(f"noncandidate implementation changed: {name}")
    campaign.engine.load_authority = lambda: frozen
    trainer, _ = campaign.make_trainer(Path(args.collision_root), worlds=32768)
    trainer.load_checkpoint(args.resume)
    before = {
        k: campaign.tree_sha256(v)
        for k, v in (
            ("model", trainer.model.state_dict()),
            ("optimizer", trainer.optimizer.state_dict()),
        )
    }
    trainer.set_exploration(campaign.prior.restart_exploration(trainer.accepted_updates_total))
    started = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize()
    collect_seconds = time.perf_counter() - started
    rollout.compute_gae(trainer.ppo_config)
    mask = _sequence_major(rollout.train_mask)
    raw = _sequence_major(rollout.advantages)
    family = _sequence_major(rollout.opponent_family)
    normalized = torch.zeros_like(raw)
    for family_id in torch.unique(family[mask]).tolist():
        selected = mask & (family == int(family_id))
        values = raw[selected]
        normalized[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    size = rollout.sequence_layout(trainer.ppo_config.minibatch_size).sequences_per_minibatch
    kwargs = dict(
        observation=_sequence_major(rollout.observations),
        initial_hidden=rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim),
        reset_before=_sequence_major(rollout.reset_before),
        action=_sequence_major(rollout.actions),
        pre_tanh=_sequence_major(rollout.pre_tanh),
        old_log_probability=_sequence_major(rollout.old_log_probability),
        normalized_advantage=normalized,
        returns=_sequence_major(rollout.returns),
        train_mask=mask,
        sequence_index=torch.arange(size, device=trainer.device),
        take_step=False,
    )
    original_method = trainer.model._context_with_resets
    rows = []
    reference_grad = None
    reference_outputs = None
    options = [("legacy", 32), ("spans", 32), ("packed_episodes", 32), ("packed_fp32", 32)]
    options += [("spans", chunk) for chunk in (64, 91, 182)]
    if args.stress:
        # Deliberately worst-case reset density, for allocation safety only.
        # This is not a valid training rollout and no optimizer step is taken.
        kwargs["reset_before"] = kwargs["reset_before"].clone()
        kwargs["reset_before"][:size] = True
        options = [("spans", chunk) for chunk in (32, 64, 91, 182)]
    for method, chunk in options:
        function = CANDIDATES[method]
        trainer.model._context_with_resets = types.MethodType(
            lambda self, x, h, r, fn=function: fn(self.context_gru, x, h, r), trainer.model
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _repeat in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = recurrent_minibatch_step(
                trainer.model,
                trainer.optimizer,
                trainer.ppo_config,
                trainer.exploration.distribution_override,
                sequence_microbatch_size=chunk,
                **kwargs,
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        gradients = {
            n: p.grad.detach().cpu().clone()
            for n, p in trainer.model.named_parameters()
            if p.grad is not None
        }
        with torch.no_grad():
            outputs = [
                x.cpu()
                for x in trainer.model(
                    kwargs["observation"][:32],
                    kwargs["initial_hidden"][:, :32],
                    reset_before=kwargs["reset_before"][:32],
                )
            ]
        if reference_grad is None:
            reference_grad, reference_outputs = gradients, outputs
        errors = {
            n: float((value - reference_grad[n]).abs().max()) for n, value in gradients.items()
        }
        # Frozen numerical acceptance: standard FP32 tolerance, not a KL guard.
        gradient_close = all(
            torch.allclose(value, reference_grad[n], atol=3e-6, rtol=1e-4)
            for n, value in gradients.items()
        )
        output_close = all(
            torch.allclose(a, b, atol=3e-6, rtol=1e-4)
            for a, b in zip(outputs, reference_outputs, strict=True)
        )
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        row = dict(
            method=method,
            microbatch=chunk,
            measurements=times,
            median_seconds=statistics.median(times),
            max_gradient_absolute_error=max(errors.values()),
            gradient_close=gradient_close,
            output_close=output_close,
            output_max_absolute_errors=[
                float((a - b).abs().max()) for a, b in zip(outputs, reference_outputs, strict=True)
            ],
            peak_allocated_gib=allocated,
            peak_reserved_gib=reserved,
            physical_tensor_headroom=allocated < 28.0,
            metrics={k: float(v.item()) for k, v in metrics.items()},
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model._context_with_resets = original_method
    unchanged = {
        k: before[k] == campaign.tree_sha256(v)
        for k, v in (
            ("model", trainer.model.state_dict()),
            ("optimizer", trainer.optimizer.state_dict()),
        )
    }
    report = dict(
        source_checkpoint=str(Path(args.resume).resolve()),
        source_checkpoint_sha256=campaign.engine.sha256_file(Path(args.resume)),
        frozen_authority_sha256=campaign.engine.sha256_file(authority_path),
        worlds=32768,
        horizon=360,
        effective_minibatch_sequences=size,
        effective_minibatch_ticks=size * 360,
        collect_seconds=collect_seconds,
        rollout_logical_gib=rollout.logical_bytes / 2**30,
        reset_count=int(rollout.reset_before.sum()),
        reset_count_after_first=int(rollout.reset_before[1:].sum()),
        no_optimizer_step=True,
        worst_case_reset_stress=args.stress,
        protected_state_unchanged=unchanged,
        rows=rows,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    assert all(unchanged.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority",
        default=str(campaign.RESULTS / "performance_repair/original/authority.json"),
    )
    parser.add_argument("--resume", default=str(campaign.STARTUP))
    parser.add_argument("--collision-root", default="G:/dev/RLBot-Rival/bot/collision_meshes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
