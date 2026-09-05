"""Full native 32,768-world entity/categorical preflight; no optimizer step."""
# ruff: noqa: E402 -- Direct script invocation must establish repository imports.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    check_package,
    sha,
    tensor_hash,
    utc,
    write_json,
)
from benchmarks.run_rival2_ssl_exploration_comparison import PARENT, PARENT_SHA
from rivalsim.fresh_ground_30hz import SEED, FreshGroundEnv, ppo_config, scenario_hash, scenarios
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic, entity_schema
from rivalsim.ssl_entity_training import (
    EntityRolloutCollector,
    fresh_entity_optimizer,
    joint_sequence_loss,
    sequence_data,
)
from rivalsim.ssl_joint_control_policy import JointControlActorCritic, categorical_statistics

RESULTS = ROOT / "results/rival2/ssl_entity_joint_control_v1"
COLLISION = "G:/dev/RLBot-Rival/bot/collision_meshes"


def main():
    torch.set_num_threads(8)
    check_package(require_preflight=True)
    assert sha(PARENT) == PARENT_SHA
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    torch.manual_seed(SEED + 701)
    model = EntityJointControlActorCritic()
    model.initialize_from_hybrid(parent["model"])
    before = tensor_hash(model.state_dict())
    bank = scenarios(32768)
    env = FreshGroundEnv(
        32768, COLLISION, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    collector = EntityRolloutCollector(env, model)
    collector.generator.set_state(parent["policy_generator_state"].cpu())
    optimizer = fresh_entity_optimizer(model)
    initial_obs = env.observation.clone()
    # Compare identical CPU-projected initialization. CPU versus CUDA GEMM
    # projection has rounding differences unrelated to the zero entity branch.
    joint = JointControlActorCritic()
    joint.initialize_from_hybrid(parent["model"])
    joint = joint.to("cuda:0")
    with torch.no_grad():
        small = initial_obs.reshape(-1, 182)[:1024]
        a, v, h = model(small)
        a0, v0, h0 = joint(small)
        parity = all(torch.equal(x, y) for x, y in [(a, a0), (v, v0), (h, h0)])
        tokens = model.entities.tokens(small)
        _, attention = model.entities.attend(tokens, weights=True)
        attention_error = float((attention.sum(-1) - 1).abs().max())
    del joint, a, v, h, a0, v0, h0, tokens, attention
    torch.cuda.synchronize()
    start = time.monotonic()
    rollout = collector.collect()
    torch.cuda.synchronize()
    rollout_seconds = time.monotonic() - start
    print(json.dumps(dict(stage="native_rollout_complete", seconds=rollout_seconds)), flush=True)
    data = sequence_data(rollout, ppo_config())
    index = torch.arange(728, device="cuda:0")
    with torch.no_grad():
        logits, value, _ = model(
            data["observations"][index],
            data["initial_hidden"][:, index],
            reset_before=data["reset_before"][index],
        )
        logp, _ = categorical_statistics(logits, data["action_indices"][index])
        logp_error = float((logp - data["old_log_probability"][index]).abs().max())
        value_error = float((value - data["values"][index]).abs().max())
    torch.cuda.reset_peak_memory_stats()
    model.train()  # cuDNN needs a training forward to retain RNN backward workspace.
    torch.cuda.synchronize()
    start = time.monotonic()
    loss, metrics = joint_sequence_loss(model, data, index, ppo_config())
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated()
    checks = dict(
        parent_hash=sha(PARENT) == PARENT_SHA,
        model_unchanged=tensor_hash(model.state_dict()) == before,
        no_optimizer_steps=len(optimizer.state) == 0,
        observation_not_mutated=torch.equal(initial_obs, rollout.observations[0]),
        exact_initial_joint_control_parity=parity,
        normalized_attention=attention_error < 1e-6,
        stored_actions_exact=torch.equal(
            model.action_table[rollout.action_indices], rollout.actions
        ),
        likelihood_recomputed=logp_error < 1e-5,
        values_recomputed=value_error < 1e-3,
        finite_backward=bool(torch.isfinite(loss))
        and all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        ),
        entity_actor_gradient_present=bool(model.entity_actor.weight.grad.abs().sum() > 0),
        entity_context_gradient_present=bool(model.entity_context.weight.grad.abs().sum() > 0),
        no_mechanic_hot_path=env.world.gameplay_v3 is None and env.world.gameplay_120 is None,
    )
    report = dict(
        utc=utc(),
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        worlds=32768,
        policy_hz=30,
        physics_hz=120,
        horizon=90,
        optimizer_steps=0,
        initialization_sha256=before,
        parent_sha256=PARENT_SHA,
        scenario_sha256=scenario_hash(bank),
        policy_config_sha256=model.config.content_hash,
        entity_schema=entity_schema(),
        action_table_sha256=tensor_hash({"action_table": model.action_table}),
        maximum_old_logprob_error=logp_error,
        maximum_value_error=value_error,
        rollout_seconds=rollout_seconds,
        complete_728_sequence_forward_backward_seconds=backward_seconds,
        peak_allocated_bytes=peak,
        rollout_logical_bytes=rollout.logical_bytes,
        training_capability_claim=False,
        rollout_metrics=collector.last_metrics,
        loss_metrics={k: float(v) for k, v in metrics.items()},
        note="Native rollout and backward only, not a trained model or learning-speed "
        "claim. Entity encoders initially have zero downstream gradient until zero residual "
        "projections first learn; output projections have nonzero gradients.",
    )
    write_json(RESULTS / "native_preflight.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verdict",
                    "checks",
                    "rollout_seconds",
                    "complete_728_sequence_forward_backward_seconds",
                    "peak_allocated_bytes",
                )
            }
        ),
        flush=True,
    )
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
