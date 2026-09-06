"""CPU-only check of a preserved accepted exploration checkpoint and curve prefix."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_direct_skills_exploration_v1 import OUT, SOURCE, SOURCE_SHA, START, amendment
from benchmarks.run_rival2_direct_skills_v1 import EXTERNAL, authority
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, utc, write_json
from rivalsim.fresh_ground_30hz import content_hash, ppo_config


def audit(path):
    expected = sha(path)
    p = torch.load(path, map_location="cpu", weights_only=False)
    parent = torch.load(SOURCE, map_location="cpu", weights_only=False)
    offset = p["accepted_updates"]
    assert START < offset <= 600
    rows = [
        json.loads(s)
        for s in (EXTERNAL / "training_curve.jsonl").read_text().splitlines()
        if s.strip()
    ]
    rows = [r for r in rows if START < r["accepted_updates"] <= offset]
    checks = dict(
        source_unchanged=sha(SOURCE) == SOURCE_SHA,
        stable_snapshot_read=sha(path) == expected,
        root_authority=p["authority_sha256"] == content_hash(authority()),
        amendment=p["exploration_amendment_sha256"] == content_hash(amendment()),
        exploration_parent=p["exploration_parent_sha256"] == SOURCE_SHA,
        distribution=p["training_distribution"]["temperature"] == 2.0
        and p["training_distribution"]["on_policy_sampling_and_likelihood"],
        same_ppo=p["ppo_config_sha256"] == ppo_config().content_hash,
        contiguous_updates=[r["accepted_updates"] for r in rows]
        == list(range(START + 1, offset + 1)),
        curve_authority=all(
            r["training_temperature"] == 2
            and r["exploration_amendment_sha256"] == content_hash(amendment())
            for r in rows
        ),
        exact_sample_count=p["direct_skill_samples"]
        == parent["direct_skill_samples"] + (offset - START) * 4423680,
        exact_physics_count=p["direct_skill_physics_ticks"]
        == parent["direct_skill_physics_ticks"] + (offset - START) * 11796480,
        cumulative_adam=p["cumulative_optimizer_steps"]
        == 131038 + sum(r["ppo"]["optimizer_steps"] for r in rows),
        optimizer_counter_uniform={int(s["step"]) for s in p["optimizer"]["state"].values()}
        == {p["cumulative_optimizer_steps"]},
        optimizer_groups_unchanged=p["optimizer"]["param_groups"]
        == parent["optimizer"]["param_groups"],
        model_schema=p["model"].keys() == parent["model"].keys()
        and all(v.shape == parent["model"][k].shape for k, v in p["model"].items()),
        finite_model=all(bool(torch.isfinite(v).all()) for v in p["model"].values()),
        finite_adam=all(
            bool(torch.isfinite(v).all())
            for s in p["optimizer"]["state"].values()
            for v in s.values()
            if torch.is_tensor(v)
        ),
        all_four_rng=all(
            torch.is_tensor(p[k]) and p[k].dtype == torch.uint8 and p[k].numel() > 0
            for k in (
                "policy_generator_state",
                "shuffle_generator_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
            )
        ),
        exact_nexto_third=all(
            r["training"]["nexto_training_sample_count"] * 3
            == r["training"]["trainable_agent_samples"]
            == 4423680
            for r in rows
        ),
        finite_ppo=all(
            math.isfinite(v) for r in rows for v in r["ppo"].values() if isinstance(v, (int, float))
        ),
        no_kl_rejections=all(r["ppo"]["kl_rejections"] == 0 for r in rows),
        source_rewards_and_contracts=p["runtime_contract_hashes"]
        == parent["runtime_contract_hashes"]
        and p["reward_authority"] == parent["reward_authority"]
        and p["package"] == parent["package"],
    )
    curve = OUT / f"training_curve_000555_to_{offset:06d}.jsonl"
    assert not curve.exists(), "Never overwrite an immutable curve prefix"
    curve.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    result = dict(
        utc=utc(),
        checkpoint=str(path),
        checkpoint_sha256=expected,
        accepted_updates=offset,
        checks=checks,
        direct_skill_samples=p["direct_skill_samples"],
        cumulative_optimizer_steps=p["cumulative_optimizer_steps"],
        curve_path=str(curve.relative_to(ROOT)),
        curve_sha256=sha(curve),
        max_completed_update_mean_kl=max(r["ppo"]["completed_update_mean_kl"] for r in rows),
        mean_entropies=[r["training"]["mean_entropy"] for r in rows],
        maximum_allocated_cuda_bytes=max(r["cuda_peak_allocated_bytes"] for r in rows),
        entry_observation="Pre-first-step snapshot window was missed; direct live-entry byte "
        "parity not claimed. Strict loader and prospective native preflight establish source "
        "model/Adam correctness.",
        interpretation="Training health/continuity only; no new deterministic evaluation yet",
    )
    write_json(OUT / f"resume_accepted_{offset:06d}.json", result)
    assert all(checks.values()), checks
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    audit(parser.parse_args().checkpoint)
