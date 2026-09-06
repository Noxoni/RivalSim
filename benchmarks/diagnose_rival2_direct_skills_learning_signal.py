"""Prospectively bounded native PPO-signal diagnostic. Never steps an optimizer."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.direct_skills_eval_stream import owned_match_stream  # noqa: E402
from benchmarks.run_rival2_fresh_ground_30hz_v1 import (  # noqa: E402
    sha,
    tensor_hash,
    utc,
    write_json,
)
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease  # noqa: E402
from rivalsim.direct_skills_training import DirectSkillCollector  # noqa: E402
from rivalsim.direct_skills_v1 import EVENTS, NAMES, DirectSkillsEnv, scenarios  # noqa: E402
from rivalsim.fresh_ground_30hz import ppo_config, scenario_hash  # noqa: E402
from rivalsim.rival2_recurrent_ppo import _sequence_major  # noqa: E402
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data  # noqa: E402
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic  # noqa: E402
from rivalsim.ssl_joint_control_policy import categorical_statistics  # noqa: E402

RESULTS = ROOT / "results/rival2/direct_skills_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-v1")
CHECKPOINT = ROOT / "checkpoints/rival2/direct_skills_v1/plus_000550.pt"
EXPECTED = "C8564101BEAE09219026A11ED5F5433E6CEC460EFAE808326F7B8D7FB5DB33FE"
PLAN = "results/rival2/direct_skills_v1/ADVANTAGE_DIAGNOSTIC_PLAN_000550.md"
PLAN_COMMIT = "e4e9a05a63a87a06ef4395ee205ea50cceb716af"
SEED, WORLDS, ROLLOUTS = 2026090655, 1024, 4
BANK_SHA = "D46A5DACAA3E742BF64E4EA06EB3642F3F370D4AEA2F73FF2791369018A9A73B"
OUTPUT = RESULTS / "advantage_diagnostic_000550"


def cpu(value):
    return value.detach().cpu().numpy().copy()


def independent_gae(reward, value, next_value, terminated, truncated, gamma, lam):
    """Independent NumPy recurrence, including true-terminal/truncation semantics."""
    result = np.zeros_like(reward)
    carry = np.zeros_like(reward[0])
    g, trace = np.float32(gamma), np.float32(gamma * lam)
    for tick in range(len(reward) - 1, -1, -1):
        bootstrap = g * next_value[tick] * (~terminated[tick]).astype(np.float32)
        delta = reward[tick] + bootstrap - value[tick]
        carry = delta + trace * (~(terminated[tick] | truncated[tick])).astype(np.float32) * carry
        result[tick] = carry
    return result


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    assert np.isfinite(values).all()
    return dict(
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        min=float(values.min()),
        q05=float(np.quantile(values, 0.05)),
        median=float(np.median(values)),
        q95=float(np.quantile(values, 0.95)),
        max=float(values.max()),
        positive_fraction=float((values > 0).mean()),
    )


def precedes_event(event, done, maximum_lag):
    """Next 1..lag decisions in this buffer, never across an episode boundary."""
    result = np.zeros_like(event, dtype=bool)
    reachable = np.ones_like(event, dtype=bool)
    for lag in range(1, min(maximum_lag + 1, len(event))):
        reachable[:-lag] &= ~done[lag - 1 : -1]
        result[:-lag] |= reachable[:-lag] & event[lag:]
    return result


def role_summary(arrays):
    result = {}
    terminal = arrays["terminated"]
    winner = arrays["scoring_team"][..., None] == np.arange(2)
    events = dict(
        goal=terminal & winner,
        concede=terminal & ~winner,
        touch=arrays["touch_count"] > 0,
        truncation=arrays["truncated"],
    )
    for name in ("goal", "concede"):
        for lag in (30, 89):
            events[f"before_{name}_within_{lag}_decisions"] = precedes_event(
                events[name], terminal | arrays["truncated"], lag
            )
    events.update({name: arrays["events"][..., i] for i, name in enumerate(EVENTS)})
    for role, name in enumerate(NAMES):
        for opponent, label in enumerate(("selfplay", "nexto")):
            mask = (
                arrays["train_mask"]
                & (arrays["roles"] == role)
                & (arrays["opponent_family"] == opponent)
            )
            value, returns = arrays["values"][mask], arrays["returns"][mask]
            variance = float(returns.var()) if returns.size else 0.0
            entry = dict(
                samples=int(mask.sum()),
                value=distribution(value),
                returns=distribution(returns),
                critic_rmse=float(np.sqrt(np.square(returns - value).mean()))
                if returns.size
                else None,
                critic_explained_variance=1 - float((returns - value).var()) / variance
                if variance > 1e-10
                else None,
                entropy=distribution(arrays["entropy"][mask]),
                maximum_action_probability=distribution(arrays["maximum_probability"][mask]),
                sampled_action_probability=distribution(
                    np.exp(arrays["old_log_probability"][mask])
                ),
                advantage=distribution(arrays["advantages"][mask]),
                normalized_advantage=distribution(arrays["normalized_advantage"][mask]),
                event_summaries={},
            )
            for event, selected in events.items():
                selected = selected & mask
                entry["event_summaries"][event] = dict(
                    reward=distribution(arrays["rewards"][selected]),
                    advantage=distribution(arrays["advantages"][selected]),
                    normalized_advantage=distribution(arrays["normalized_advantage"][selected]),
                )
            result[f"{name}_{label}"] = entry
    return result


@torch.no_grad()
def stepwise_replay(model, data):
    """Replay all original perspectives at the original collection batch size."""
    hidden = data["initial_hidden"].clone()
    logps, values = [], []
    for tick in range(data["observations"].shape[1]):
        logits, value, hidden = model(
            data["observations"][:, tick].contiguous(),
            hidden,
            reset_before=data["reset_before"][:, tick].contiguous(),
        )
        logp, _ = categorical_statistics(logits, data["action_indices"][:, tick])
        logps.append(logp)
        values.append(value)
    replay = torch.stack(logps, 1)
    value = torch.stack(values, 1)
    assert torch.isfinite(replay).all() and torch.isfinite(value).all()
    mask = data["train_mask"]
    return replay, dict(
        perspectives=data["observations"].shape[0],
        decisions=data["observations"].shape[1],
        collected_log_probability_exactly_equal=bool(
            torch.equal(replay, data["old_log_probability"])
        ),
        absolute_log_probability_error=distribution(
            cpu((replay - data["old_log_probability"])[mask].abs())
        ),
        absolute_value_error=distribution(cpu((value - data["values"])[mask].abs())),
        interpretation="Same full batch and reset history; measurements are diagnostic, "
        "not a newly invented PPO acceptance gate.",
    )


def gradients(model, data, roles, stepwise_logp):
    """Entry-of-update policy gradients, same normalization and full sequences."""
    # Match production PPO forward mode: cuDNN GRU backward requires train().
    # No dropout/batchnorm exists in this policy; state_dict parity is checked.
    model.train()
    config = ppo_config()
    eligible = data["train_mask"].any(1).nonzero().flatten()
    index = eligible[: config.minibatch_size // config.rollout_horizon]
    logits, value, _ = model(
        data["observations"][index],
        data["initial_hidden"][:, index],
        reset_before=data["reset_before"][index],
    )
    logp, entropy = categorical_statistics(logits, data["action_indices"][index])
    mask = data["train_mask"][index]
    log_ratio = logp - data["old_log_probability"][index]
    assert torch.isfinite(log_ratio).all()
    maximum_replay_error = float(log_ratio[mask].abs().max().detach())
    replay_comparison = dict(
        original_helper_1e_minus_3_assertion_would_pass=maximum_replay_error <= 1e-3,
        batch_vs_collection_absolute_error=distribution(cpu(log_ratio[mask].abs())),
        batch_vs_stepwise_absolute_error=distribution(
            cpu((logp - stepwise_logp[index])[mask].abs())
        ),
        batch_vs_collection_ratio=distribution(cpu(log_ratio[mask].exp())),
        ratio_outside_production_clip_count=int(
            ((log_ratio[mask].exp() - 1).abs() > config.clip_range).sum()
        ),
        note="Original failed helper assertion is preserved in attempt0. This reports "
        "replay error rather than silently raising its threshold. No production guard changed.",
    )
    ratio = log_ratio.exp()
    advantages = data["normalized_advantage"][index]
    per_item = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantages
    )
    parameters = [
        (name, p) for name, p in model.named_parameters() if not name.startswith("critic.")
    ]
    actor = [p for _, p in parameters]
    vectors, reports = {}, {}
    role_ids = _sequence_major(roles)[index]
    family = data["opponent_family"][index]
    for role, name in enumerate(NAMES):
        for opponent, label in enumerate(("selfplay", "nexto")):
            selected = mask & (role_ids == role) & (family == opponent)
            count = int(selected.sum())
            if not count:
                continue
            loss = per_item[selected].mean()
            grads = torch.autograd.grad(loss, actor, retain_graph=True, allow_unused=True)
            flat = torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                    for p, g in zip(actor, grads, strict=True)
                ]
            )
            assert torch.isfinite(flat).all()
            key = f"{name}_{label}"
            vectors[key] = flat.cpu()
            by_component = {}
            for (parameter_name, _), grad in zip(parameters, grads, strict=True):
                component = parameter_name.split(".")[0]
                by_component[component] = by_component.get(component, 0.0) + (
                    0.0 if grad is None else float(grad.detach().square().sum())
                )
            reports[key] = dict(
                samples=count,
                loss=float(loss.detach()),
                gradient_norm=float(flat.norm()),
                component_gradient_norm={k: v**0.5 for k, v in by_component.items()},
            )
    value_loss = 0.5 * (value[mask] - data["returns"][index][mask]).square().mean()
    all_parameters = list(model.parameters())
    critic_grad = torch.autograd.grad(
        value_loss, all_parameters, retain_graph=True, allow_unused=True
    )
    actor_ids = {id(p) for p in actor}
    assert all(
        g is None or not bool(g.any())
        for p, g in zip(all_parameters, critic_grad, strict=True)
        if id(p) in actor_ids
    )
    critic_norm = sum(float(g.detach().square().sum()) for g in critic_grad if g is not None) ** 0.5
    total_loss = (
        per_item[mask].mean()
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * entropy[mask].mean()
    )
    combined = torch.autograd.grad(total_loss, all_parameters, allow_unused=True)
    total_norm = sum(float(g.detach().square().sum()) for g in combined if g is not None) ** 0.5
    assert all(g is None or bool(torch.isfinite(g).all()) for g in combined)
    cosine = {}
    for name, a in vectors.items():
        cosine[name] = {}
        for other, b in vectors.items():
            norm = float(a.norm() * b.norm())
            cosine[name][other] = float(torch.dot(a, b)) / norm if norm > 0 else None
    model.zero_grad(set_to_none=True)
    model.eval()
    return dict(
        sequences=len(index),
        trainable_samples=int(mask.sum()),
        collection_replay_max_log_probability_error=maximum_replay_error,
        replay_comparison=replay_comparison,
        per_role=reports,
        pairwise_actor_gradient_cosine=cosine,
        critic_gradient_norm=critic_norm,
        value_loss_actor_gradient_exactly_zero=True,
        combined_preclip_gradient_norm=total_norm,
        production_clip_multiplier_if_stepped=min(
            1.0, config.max_gradient_norm / (total_norm + 1e-6)
        ),
        optimizer_steps=0,
        interpretation="Local same-buffer gradients, not causal proof of a plateau.",
    )


def validate_sources():
    package = json.loads((RESULTS / "package.json").read_text())
    for path, expected in package["sources"].items():
        actual = (
            hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()
        )
        assert actual == expected, path
    assert subprocess.check_output(["git", "show", f"{PLAN_COMMIT}:{PLAN}"], cwd=ROOT).replace(
        b"\r\n", b"\n"
    ) == (ROOT / PLAN).read_bytes().replace(b"\r\n", b"\n")
    assert sha(CHECKPOINT) == EXPECTED
    assert (
        sha(ROOT / "checkpoints/rival2/direct_skills_v1/paused_000554.pt")
        == "CF5C9D022FFB4EF18DBD728206D78A3601AB8C9128BFA0F0E9ADE31BCF961BAD"
    )
    return package


def run():
    package = validate_sources()
    state = json.loads((EXTERNAL / "campaign_state.json").read_text())
    assert state["status"] == "stopped_at_accepted_boundary" and state["accepted_updates"] == 554
    assert (EXTERNAL / "STOP").exists()
    assert not OUTPUT.exists(), (
        "Never replace a prior diagnostic; inspect any partial artifacts first"
    )
    bank = scenarios(WORLDS, seed=SEED)
    assert scenario_hash(bank) == BANK_SHA
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    source_hash = sha(Path(__file__))
    report = dict(
        schema="RIVAL2_DIRECT_SKILLS_LEARNING_SIGNAL_DIAGNOSTIC_V1",
        utc=utc(),
        plan_commit=PLAN_COMMIT,
        plan_sha256=sha(ROOT / PLAN),
        source_sha256=source_hash,
        checkpoint_sha256=EXPECTED,
        scenario_sha256=BANK_SHA,
        seed=SEED,
        worlds=WORLDS,
        rollouts=[],
        optimizer_steps=0,
        frozen_runtime_sources=len(package["sources"]),
    )
    with gpu_lease(), owned_match_stream():
        policy = EntityJointControlActorCritic().cuda().eval()
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["model"], strict=True)
        model_hash = tensor_hash(policy.state_dict())
        assert ppo_config().content_hash == payload["ppo_config_sha256"]
        del payload
        env = DirectSkillsEnv(
            WORLDS,
            Path("G:/dev/RLBot-Rival/bot/collision_meshes"),
            device="cuda:0",
            seed=SEED,
            ssl_foundation_scenarios=bank,
        )
        collector = DirectSkillCollector(env, policy, seed=SEED)
        original_step, original_sample = env._step_impl, policy.sample
        taps = {}

        def step(*args, **kwargs):
            paid_before = env.events.paid.clone()
            result = original_step(*args, **kwargs)
            values = {key: env.last_skill[key] for key in ("roles", "events", "bonus", "approach")}
            values.update(
                {
                    key: env.last_native[key]
                    for key in ("touch_count", "scoring_team", "episode_ticks", "first_goal_tick")
                }
            )
            values.update({f"component.{key}": value for key, value in env.last_components.items()})
            values["paid_before"] = paid_before
            for key, value in values.items():
                taps.setdefault(key, []).append(value.detach().clone())
            return result

        def sample(logits, generator):
            result = original_sample(logits, generator)
            probs = logits.softmax(-1)
            entropy = -(probs * logits.log_softmax(-1)).sum(-1)
            taps.setdefault("entropy", []).append(entropy.reshape(WORLDS, 2).detach())
            taps.setdefault("maximum_probability", []).append(
                probs.amax(-1).reshape(WORLDS, 2).detach()
            )
            return result

        env._step_impl, policy.sample = step, sample
        OUTPUT.mkdir()
        for iteration in range(ROLLOUTS):
            taps.clear()
            rollout = collector.collect()
            assert len(taps["roles"]) == 90 and len(taps["entropy"]) == 90
            data = mixed_sequence_data(rollout, ppo_config())
            stacked = {key: torch.stack(values) for key, values in taps.items()}
            arrays = {key: cpu(value) for key, value in stacked.items()}
            for key in (
                "observations",
                "actions",
                "action_indices",
                "old_log_probability",
                "values",
                "next_values",
                "rewards",
                "terminated",
                "truncated",
                "train_mask",
                "opponent_family",
                "reset_before",
                "advantages",
                "returns",
                "initial_hidden",
            ):
                arrays[key] = cpu(getattr(rollout, key))
            arrays["normalized_advantage"] = cpu(
                data["normalized_advantage"].reshape(WORLDS, 2, 90).permute(2, 0, 1)
            )
            # Preserve the collected evidence before any diagnostic assertion/autograd.
            archive = OUTPUT / f"rollout_{iteration:02d}.npz"
            np.savez_compressed(archive, **arrays)
            write_json(
                OUTPUT / "progress.json",
                dict(
                    rollout=iteration,
                    phase="collected_archive_saved",
                    optimizer_steps=0,
                    archive_sha256=sha(archive),
                    source_sha256=source_hash,
                ),
            )
            assert all(np.isfinite(v).all() for v in arrays.values())
            config = ppo_config()
            reference = independent_gae(
                arrays["rewards"],
                arrays["values"],
                arrays["next_values"],
                arrays["terminated"],
                arrays["truncated"],
                config.gamma,
                config.gae_lambda,
            )
            gae_error = float(np.abs(reference - arrays["advantages"]).max())
            assert gae_error <= 2e-5, gae_error
            terminal = arrays["terminated"]
            terminal_return_error = (
                float(np.abs(arrays["returns"][terminal] - arrays["rewards"][terminal]).max())
                if terminal.any()
                else None
            )
            assert terminal_return_error is None or terminal_return_error <= 2e-5
            replay_logp, replay_report = stepwise_replay(policy, data)
            write_json(OUTPUT / f"rollout_{iteration:02d}_replay.json", replay_report)
            grad_report = gradients(policy, data, stacked["roles"], replay_logp)
            result = dict(
                rollout=iteration,
                fresh_initial_rollout=iteration == 0,
                native_ticks_per_world=360,
                trainable_samples=int(arrays["train_mask"].sum()),
                archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,
                independent_gae_max_error=gae_error,
                terminal_return_max_error=terminal_return_error,
                same_batch_stepwise_replay=replay_report,
                roles=role_summary(arrays),
                gradients=grad_report,
                collection_metrics=collector.last_metrics,
            )
            write_json(OUTPUT / f"rollout_{iteration:02d}.json", result)
            report["rollouts"].append(result)
            assert tensor_hash(policy.state_dict()) == model_hash
            print(
                json.dumps(
                    dict(
                        rollout=iteration,
                        gae_error=gae_error,
                        replay_error=grad_report["collection_replay_max_log_probability_error"],
                        archive_bytes=archive.stat().st_size,
                    )
                ),
                flush=True,
            )
            del arrays, rollout, data, stacked, reference, replay_logp
            gc.collect()
        env._step_impl, policy.sample = original_step, original_sample
        report.update(
            model_unchanged=tensor_hash(policy.state_dict()) == model_hash,
            checkpoint_unchanged=sha(CHECKPOINT) == EXPECTED,
            native_ticks_per_world=1440,
            total_trainable_samples=sum(r["trainable_samples"] for r in report["rollouts"]),
        )
    assert sha(Path(__file__)) == source_hash
    validate_sources()
    write_json(OUTPUT / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rollouts"}), flush=True)


if __name__ == "__main__":
    run()
