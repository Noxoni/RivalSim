"""Read-only CPU reduction of closed corrected-Nexto evaluations and log prefixes.

Creates immutable report artifacts, never an environment, learner or optimizer.
It neither selects checkpoints nor changes the already frozen campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.report_rival2_ssl_entity_match_followup import reduce

OUT = ROOT / "results/rival2/direct_skills_native_nexto_v1"
RUN = Path("G:/dev/RivalSim-runs/direct-skills-native-nexto-v1")
METRICS = ("wins", "losses", "goals_for", "goals_against", "touches",
           "touches_per_minute", "kickoff_first_touches", "matches_without_rival_touch")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest().upper()


def read(path):
    return json.loads(path.read_text())


def compare_records(parent, child):
    """Compare the same fixed development starts, not independent random trials."""
    for key in ("evaluation_spec_sha256", "nexto_controller", "nexto_sampling_mode", "nexto_seed"):
        if parent[key] != child[key]:
            raise ValueError("Evaluation domains differ: " + key)
    for record in (parent, child):
        if record["optimizer_steps"] != 0 or not all(record[k] for k in (
                "model_unchanged", "checkpoint_unchanged", "nexto_unchanged")):
            raise ValueError("Not an immutable inference-only evaluation")
        if record["summary"]["unresolved"] or not all(record["raw"]["match.done"]):
            raise ValueError("Incomplete evaluation")
    for key in ("match.rival_side", "match.starting_layout"):
        if parent["raw"][key] != child["raw"][key]:
            raise ValueError("Development starts differ: " + key)

    paired = []
    for world, side in enumerate(parent["raw"]["match.rival_side"]):
        def metrics(record):
            raw = record["raw"]
            scores = (raw["match.blue_score"][world], raw["match.orange_score"][world])
            return dict(goals_for=scores[side], goals_against=scores[1-side],
                        goal_difference=scores[side]-scores[1-side],
                        contacts=raw["touch_count"][world][side],
                        seconds=raw["match.total_ticks"][world]/120)
        before, after = metrics(parent), metrics(child)
        paired.append(dict(world=world, rival_side=side,
            starting_layout=parent["raw"]["match.starting_layout"][world],
            before=before, after=after,
            delta={k: after[k]-before[k] for k in before}))
    return dict(
        summary={k: dict(before=parent["summary"][k], after=child["summary"][k],
                        delta=child["summary"][k]-parent["summary"][k]) for k in METRICS},
        paired_worlds=paired,
        goal_difference_worlds=dict(improved=sum(p["delta"]["goal_difference"] > 0 for p in paired),
            unchanged=sum(p["delta"]["goal_difference"] == 0 for p in paired),
            worsened=sum(p["delta"]["goal_difference"] < 0 for p in paired)),
        inference="Descriptive fixed development comparison; ten matches do not establish statistical significance, native parity or SSL.")


def training_prefix(path, count):
    """Read only complete immutable leading records, never a live partial tail."""
    lines, rows = [], []
    with path.open("rb") as stream:
        for offset in range(1, count+1):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("Training prefix is not complete")
            row = json.loads(line)
            if row["branch_updates"] != offset:
                raise ValueError("Training prefix has a gap or duplicate")
            lines.append(line); rows.append(row)
    return b"".join(lines), rows


def save_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Existing immutable report differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(data)


def checkpoint_checks(parent_identity, child_identity, authority, rows):
    import torch
    from benchmarks.audit_direct_skills_native_nexto_entry import equal

    parent = torch.load(ROOT / parent_identity["path"], map_location="cpu", weights_only=False)
    child = torch.load(ROOT / child_identity["path"], map_location="cpu", weights_only=False)
    steps = sum(row["ppo"]["optimizer_steps"] for row in rows)
    samples = sum(row["training"]["trainable_agent_samples"] for row in rows)
    ticks = sum(row["training"]["physical_physics_ticks"] for row in rows)
    checks = dict(
        authority=child["native_nexto_authority_sha256"] == canonical(authority),
        parent=child["native_nexto_parent_sha256"] == parent_identity["sha256"],
        accepted_offset=child["native_nexto_branch_updates"] == len(rows),
        ancestry=child["accepted_updates"] == parent["accepted_updates"] + len(rows),
        optimizer_steps=child["cumulative_optimizer_steps"] == parent["cumulative_optimizer_steps"] + steps,
        uniform_adam_steps={int(s["step"]) for s in child["optimizer"]["state"].values()} == {child["cumulative_optimizer_steps"]},
        learner_decisions=child["direct_skill_samples"] == parent["direct_skill_samples"] + samples,
        physical_ticks=child["direct_skill_physics_ticks"] == parent["direct_skill_physics_ticks"] + ticks,
        finite_model=all(bool(torch.isfinite(t).all()) for t in child["model"].values()),
        finite_adam=all(bool(torch.isfinite(t).all()) for s in child["optimizer"]["state"].values() for t in s.values()),
        model_changed=not equal(parent["model"], child["model"]),
        reward_preserved=child["reward_authority"] == parent["reward_authority"] == authority["reward"],
        ppo_preserved=child["ppo_config_sha256"] == parent["ppo_config_sha256"],
        policy_preserved=child["policy_config_sha256"] == parent["policy_config_sha256"],
        rng_saved=all(k in child for k in ("policy_generator_state", "shuffle_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_state")),
        native_controller=child["opponent_state"]["native_nexto"]["version"] == authority["opponents"]["controller"],
        native_mode=child["opponent_state"]["native_nexto"]["sampling_mode"] == authority["opponents"]["sampling_mode"],
    )
    assert all(checks.values()), checks
    return checks


def report(offset):
    authority = read(OUT / "training_authority.json")
    if offset not in authority["evaluation_boundaries"]:
        raise ValueError("Not a frozen evaluation boundary")
    parent_path = OUT / (authority["parent_choice"] + ".json")
    target_path = OUT / f"child_{offset:06d}.json"
    parent, target = read(parent_path), read(target_path)
    # The runner writes the completed integrity artifact after the result file.
    target_integrity = read(target_path.with_suffix(".integrity.json"))
    reductions = [reduce(parent_path), reduce(target_path)]
    assert reductions[1] == target_integrity
    assert parent["checkpoint"]["sha256"] == authority["parent"]["sha256"]
    assert target["context"] == dict(native_nexto_authority_sha256=canonical(authority), branch_updates=offset)
    assert target["evaluation_spec_sha256"] == authority["evaluation_spec_sha256"]
    for record in (parent, target):
        assert sha(ROOT / record["checkpoint"]["path"]) == record["checkpoint"]["sha256"]
    prefix, rows = training_prefix(RUN / "training_curve.jsonl", offset)
    for row in rows:
        assert row["native_nexto_authority_sha256"] == canonical(authority)
        assert row["accepted_updates"] == authority["parent"]["accepted_updates"] + row["branch_updates"]
        assert row["training"]["trainable_agent_samples"] == 4423680
        assert row["training"]["nexto_training_sample_count"] == 1474560
        assert row["ppo"]["kl_rejections"] == 0
    paired = compare_records(parent, target)
    checks = checkpoint_checks(parent["checkpoint"], target["checkpoint"], authority, rows)
    metric_names = ("physical_goals", "resets", "movement_speed", "touches_per_minute",
                   "mean_first_touch_seconds_if_touched", "no_touch", "time_limit")
    result = dict(
        version="RIVAL2_DIRECT_SKILLS_NATIVE_NEXTO_PROGRESS_V1", branch_updates=offset,
        authority_sha256=canonical(authority), parent=parent["checkpoint"], selected_for_evaluation=target["checkpoint"],
        checkpoint_integrity=checks,
        sources={p.name: sha(p) for p in (parent_path, target_path, target_path.with_suffix(".integrity.json"))},
        comparison=paired, contacts_before=reductions[0]["contacts"], contacts_after=reductions[1]["contacts"],
        training=dict(accepted_updates=offset, optimizer_steps=sum(r["ppo"]["optimizer_steps"] for r in rows),
            learner_decisions=sum(r["training"]["trainable_agent_samples"] for r in rows),
            physics_world_ticks=sum(r["training"]["physical_physics_ticks"] for r in rows),
            kl_rejections=sum(r["ppo"]["kl_rejections"] for r in rows),
            max_completed_update_mean_kl=max(r["ppo"]["completed_update_mean_kl"] for r in rows),
            max_sample_kl_telemetry=max(r["ppo"]["completed_update_sample_kl_max"] for r in rows),
            rollout_seconds=sum(r["rollout_seconds"] for r in rows),
            ppo_seconds=sum(r["ppo_seconds"] for r in rows),
            peak_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in rows),
            first_last={k: [rows[0]["training"][k], rows[-1]["training"][k]] for k in metric_names}),
        training_prefix_sha256=hashlib.sha256(prefix).hexdigest().upper(),
        optimizer_steps_in_report=0,
        interpretation="Inference integrity passed. Gameplay outcome is reported without an automatic promotion verdict.",
        semantics=dict(followup="Next distinct contacting player, not continuous possession or mechanic intent",
            kickoff="First contacts only, not kickoff wins or possession",
            no_touch="No no-touch resets by protocol; use matches without Rival contact",
            training="Stochastic T2 scenario telemetry, not deterministic full-match results"))
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    save_once(OUT / f"through_{offset:06d}.jsonl", prefix)
    save_once(OUT / f"progress_{offset:06d}.json", encoded)
    print(json.dumps(dict(offset=offset, comparison=paired["summary"],
                          paired_goal_difference=paired["goal_difference_worlds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("offset", type=int)
    report(parser.parse_args().offset)
