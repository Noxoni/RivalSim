"""No-step native rollout audit of policy versus entropy gradients after the pilot."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_ssl_exploration_comparison import (
    ARMS,
    CHECKPOINTS,
    EXTERNAL,
    RESULTS,
    make,
    restore,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from rivalsim.fresh_ground_30hz import policy_config
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic
from rivalsim.rival2_policy import hybrid_entropy, hybrid_log_probability
from rivalsim.rival2_recurrent_ppo import _sequence_major


def grouped(named, gradients):
    output = {k: [] for k in ("mean_heads", "log_std_heads", "button_heads", "actor_features")}
    for (name, _), gradient in zip(named, gradients, strict=True):
        if gradient is None:
            continue
        if name.startswith(("actor.", "context_actor.")):
            output["mean_heads"].append(gradient[:5].reshape(-1))
            output["log_std_heads"].append(gradient[5:10].reshape(-1))
            output["button_heads"].append(gradient[10:].reshape(-1))
        else:
            output["actor_features"].append(gradient.reshape(-1))
    return {k: torch.cat(v) for k, v in output.items()}


def main():
    for arm in ARMS:
        assert (
            json.loads((EXTERNAL / arm / "campaign_state.json").read_text())["status"]
            == "completed"
        )
    reports = {}
    for arm in ARMS:
        path = CHECKPOINTS / arm / "plus_030.pt"
        digest = sha(path)
        trainer, _ = make(arm, "G:/dev/RLBot-Rival/bot/collision_meshes")
        restore(trainer, arm, path)
        before = tensor_hash(trainer.model.state_dict())
        optimizer_steps = [float(s["step"]) for s in trainer.optimizer.state.values()]
        rollout = trainer.collect_rollout()
        rollout.compute_gae(trainer.ppo_config)
        observations = _sequence_major(rollout.observations)
        initial = rollout.initial_hidden.reshape(1, -1, trainer.policy_config.hidden_dim)
        resets = _sequence_major(rollout.reset_before)
        advantages = _sequence_major(rollout.advantages)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-8
        )
        trainer.model.train()
        named = [(n, p) for n, p in trainer.model.named_parameters() if not n.startswith("critic.")]
        batches = []
        for start in (0, 10000, 40000):
            sl = slice(start, start + 728)
            with torch.no_grad():
                raw, _, _ = IndependentCriticActorCritic._forward(
                    trainer.model,
                    observations[sl],
                    initial[:, sl],
                    reset_before=resets[sl],
                    include_value=False,
                )
                base = policy_config()
                clamp_stats = dict(
                    below_floor_fraction=float((raw[..., 5:10] <= base.log_std_min).float().mean()),
                    above_ceiling_fraction=float(
                        (raw[..., 5:10] >= base.log_std_max).float().mean()
                    ),
                )
            actor, _ = trainer.model.forward_actor(
                observations[sl], initial[:, sl], reset_before=resets[sl]
            )
            logp = hybrid_log_probability(
                actor,
                _sequence_major(rollout.actions)[sl],
                config=trainer.policy_config,
                pre_tanh=_sequence_major(rollout.pre_tanh)[sl],
            )
            old = _sequence_major(rollout.old_log_probability)[sl]
            policy = -(torch.exp(logp - old) * advantages[sl]).mean()
            entropy = (
                -trainer.ppo_config.entropy_coefficient
                * hybrid_entropy(actor, trainer.policy_config).mean()
            )
            p = grouped(
                named,
                torch.autograd.grad(
                    policy, [v for _, v in named], retain_graph=True, allow_unused=True
                ),
            )
            e = grouped(
                named, torch.autograd.grad(entropy, [v for _, v in named], allow_unused=True)
            )
            stats = {}
            for key in p:
                pn, en = p[key].norm(), e[key].norm()
                stats[key] = dict(
                    policy_norm=float(pn),
                    entropy_norm=float(en),
                    entropy_to_policy_norm_ratio=float(en / pn.clamp_min(1e-20)),
                    cosine=float((p[key] * e[key]).sum() / (pn * en).clamp_min(1e-20)),
                )
            batches.append(
                dict(
                    sequence_start=start,
                    sequence_count=728,
                    old_logp_max_error=float((logp - old).abs().max().detach()),
                    raw_log_std_clamp=clamp_stats,
                    gradients=stats,
                )
            )
            del actor, logp, policy, entropy, p, e
        assert before == tensor_hash(trainer.model.state_dict()) and sha(path) == digest
        assert optimizer_steps == [float(s["step"]) for s in trainer.optimizer.state.values()]
        reports[arm] = dict(
            checkpoint_sha256=digest,
            batches=batches,
            native_rollout=trainer.last_rollout_metrics,
            optimizer_steps_taken=0,
            model_and_optimizer_unchanged=True,
            caveat="Three predetermined minibatches from fresh scenario episodes; descriptive gradient evidence, not a causal learning ablation or all-state claim.",
        )
        del trainer, rollout, observations, initial, resets, advantages, named
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        RESULTS / "diagnostics" / "entropy_gradient_audit.json", dict(utc=utc(), arms=reports)
    )
    print(json.dumps({a: r["batches"] for a, r in reports.items()}))


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
