"""CPU-only audit of immutable entry/first checkpoints; no optimizer construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/direct_skills_native_nexto_v1"
CKPTS = ROOT / "checkpoints/rival2/direct_skills_native_nexto_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-native-nexto-v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest().upper()


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def audit():
    spec = json.loads((OUT / "training_authority.json").read_text())
    parent_path = ROOT / spec["parent"]["path"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    entry_path, first_path = CKPTS / "entry_000000.pt", CKPTS / "first_000001.pt"
    entry = torch.load(entry_path, map_location="cpu", weights_only=False)
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    # The first flushed record is immutable while later updates append. Do not
    # race a partially written final line of the live training log.
    with (EXTERNAL / "training_curve.jsonl").open() as f:
        row = json.loads(f.readline())
    assert row["branch_updates"] == 1
    rng = ("policy_generator_state", "shuffle_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_state")
    checks = dict(
        source_file_exact=sha(parent_path) == spec["parent"]["sha256"],
        entry_model_exact=equal(entry["model"], parent["model"]),
        entry_optimizer_exact=equal(entry["optimizer"], parent["optimizer"]),
        entry_four_rng_exact=all(equal(entry[k], parent[k]) for k in rng),
        entry_counters_exact=all(entry[k] == parent[k] for k in ("accepted_updates", "direct_skill_samples", "direct_skill_physics_ticks", "cumulative_optimizer_steps")),
        both_new_authority=all(p["native_nexto_authority_sha256"] == canonical(spec) for p in (entry, first)),
        both_parent_bound=all(p["native_nexto_parent_sha256"] == spec["parent"]["sha256"] for p in (entry, first)),
        first_offset=first["native_nexto_branch_updates"] == 1 and first["accepted_updates"] == parent["accepted_updates"]+1,
        first_weights_changed=not equal(first["model"], entry["model"]),
        first_finite_model=all(bool(torch.isfinite(t).all()) for t in first["model"].values()),
        first_finite_adam=all(bool(torch.isfinite(t).all()) for s in first["optimizer"]["state"].values() for t in s.values()),
        adam_step_delta=first["cumulative_optimizer_steps"]-entry["cumulative_optimizer_steps"] == row["ppo"]["optimizer_steps"] > 0,
        uniform_adam_steps={int(s["step"]) for s in first["optimizer"]["state"].values()} == {first["cumulative_optimizer_steps"]},
        first_sample_delta=first["direct_skill_samples"]-entry["direct_skill_samples"] == row["training"]["trainable_agent_samples"] == 4423680,
        first_tick_delta=first["direct_skill_physics_ticks"]-entry["direct_skill_physics_ticks"] == 11796480,
        nexto_learner_masks=row["training"]["nexto_training_sample_count"]*3 == row["training"]["trainable_agent_samples"],
        no_kl_rejection=row["ppo"]["kl_rejections"] == 0,
        preserved_ppo_contract=first["ppo_config_sha256"] == parent["ppo_config_sha256"],
        preserved_policy_config=first["policy_config_sha256"] == parent["policy_config_sha256"],
        preserved_finishing_reward=first["reward_authority"] == parent["reward_authority"] == spec["reward"],
        real_native_opponent=all(p["opponent_state"]["native_nexto"]["version"] == spec["opponents"]["controller"]
            and p["opponent_state"]["native_nexto"]["sampling_mode"] == "native_v5" for p in (entry,first)),
    )
    assert all(checks.values()), checks
    result = dict(checks=checks, optimizer_steps_in_audit=0, audit_device="cpu",
        entry=dict(path=entry_path.relative_to(ROOT).as_posix(), sha256=sha(entry_path)),
        first=dict(path=first_path.relative_to(ROOT).as_posix(), sha256=sha(first_path)),
        parent=spec["parent"], authority_sha256=canonical(spec),
        first_model_sha256=tensor_hash(first["model"]),
        first_update=row, interpretation="Actual accepted learning and resume integrity, not gameplay improvement")
    destination = OUT / "first_update_audit.json"
    with destination.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    print(json.dumps(dict(checks=checks, first_sha256=result["first"]["sha256"])))


if __name__ == "__main__": audit()
