"""Read-only CPU audit of a completed entity-continuation checkpoint boundary.

Writes evidence only. Does not import the simulator, construct a policy/optimizer,
run inference, change guards, or select a checkpoint. Live append logs are read
only through the requested completed boundary; partial trailing writes are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/ssl_entity_continuation_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/ssl-entity-continuation-v1")
BUFFERS = ("action_table", "entities.pad_positions", "entities.pad_large", "entities.type_indices")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest().upper()


def json_hash(data):
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_prefix(path, target):
    rows, lines = [], []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["accepted_updates"] > target:
                break
            rows.append(row)
            lines.append(line.replace(b"\r\n", b"\n"))
            if row["accepted_updates"] == target:
                break
    return rows, b"".join(lines)


def ledger(rows, target, parent_steps, capacity, ticks):
    require(
        [r["accepted_updates"] for r in rows] == list(range(101, target + 1)),
        "noncontiguous_or_duplicate_update_prefix",
    )
    samples = nexto = physical = 0
    steps = parent_steps
    for row in rows:
        training, ppo = row["training"], row["ppo"]
        n = training["trainable_agent_samples"]
        require(0 < n <= capacity, "invalid_trainable_sample_count")
        require(
            capacity - n == training["nexto_training_sample_count"],
            "ignored_nexto_slots_not_conserved",
        )
        require(sum(training["action_index_counts"]) == n, "action_count_not_learner_only")
        require(training["physical_physics_ticks"] == ticks, "wrong_physics_exposure")
        require(ppo["optimizer_steps"] > 0, "no_accepted_optimizer_steps")
        require(ppo["kl_rejections"] == 0, "unexpected_kl_rejection")
        require(all(math.isfinite(v) for v in ppo.values()), "nonfinite_ppo_metrics")
        steps += ppo["optimizer_steps"]
        require(row["cumulative_optimizer_steps"] == steps, "adam_step_counter_discontinuity")
        samples += n
        nexto += training["nexto_training_sample_count"]
        physical += ticks
    return dict(
        additional_samples=samples,
        nexto_current_agent_samples=nexto,
        additional_physical_ticks=physical,
        cumulative_optimizer_steps=steps,
    )


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return math.isfinite(value) if isinstance(value, float) else True


def audit(target):
    require(target >= 150 and target % 50 == 0, "not_a_scheduled_boundary")
    authority = json.loads((RESULTS / "authority.json").read_text())
    package = json.loads((RESULTS / "package.json").read_text())
    evaluation = json.loads((RESULTS / f"evaluation_{target:06d}.json").read_text())
    parent_path = ROOT / authority["parent_path"]
    path = ROOT / f"checkpoints/rival2/ssl_entity_continuation_v1/plus_{target:06d}.pt"
    parent_bytes, candidate_bytes = parent_path.read_bytes(), path.read_bytes()
    require(
        digest(parent_bytes) == authority["parent_sha256"] == package["parent_sha256"],
        "parent_hash_mismatch",
    )
    require(
        digest(candidate_bytes) == evaluation["checkpoint"]["sha256"], "candidate_hash_mismatch"
    )
    parent = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=False)
    candidate = torch.load(io.BytesIO(candidate_bytes), map_location="cpu", weights_only=False)
    require(
        candidate["accepted_updates"] == target == evaluation["accepted_updates"],
        "checkpoint_evaluation_boundary_mismatch",
    )
    require(
        candidate["format"] == "RIVAL2_SSL_ENTITY_CONTINUATION_V1_CHECKPOINT",
        "wrong_checkpoint_format",
    )
    require(candidate["continuation_parent_sha256"] == digest(parent_bytes), "wrong_parent")
    require(
        candidate["continuation_authority_sha256"]
        == json_hash(authority)
        == package["authority_sha256"]
        == evaluation["authority_sha256"],
        "wrong_authority",
    )
    require(candidate["continuation_package"] == package, "wrong_package")
    for source, expected in package["sources"].items():
        require(
            digest((ROOT / source).read_bytes().replace(b"\r\n", b"\n")) == expected,
            f"runtime_source_changed:{source}",
        )
    for key in (
        "policy_config_sha256",
        "ppo_config_sha256",
        "observation_schema_sha256",
        "action_contract",
        "entity_schema",
    ):
        require(candidate[key] == parent[key], f"contract_changed:{key}")
    require(candidate["model"].keys() == parent["model"].keys(), "model_schema_changed")
    for key in BUFFERS:
        require(
            torch.equal(candidate["model"][key], parent["model"][key]),
            f"immutable_buffer_changed:{key}",
        )
    require(
        finite_tree(candidate["model"]) and finite_tree(candidate["optimizer"]),
        "nonfinite_model_or_adam",
    )
    require(
        candidate["optimizer"]["param_groups"] == parent["optimizer"]["param_groups"],
        "optimizer_groups_or_hyperparameters_changed",
    )
    require(
        candidate["optimizer"]["state"].keys() == parent["optimizer"]["state"].keys(),
        "optimizer_state_schema_changed",
    )
    parent_step_set = {int(s["step"]) for s in parent["optimizer"]["state"].values()}
    require(len(parent_step_set) == 1, "inconsistent_parent_adam_steps")
    rows, prefix = read_prefix(EXTERNAL / "training_curve.jsonl", target)
    worlds, horizon = authority["worlds"], authority["ppo"]["rollout_horizon"]
    counts = ledger(
        rows,
        target,
        parent_step_set.pop(),
        worlds * horizon * 2,
        worlds * horizon * authority["hold_ticks"],
    )
    steps = counts["cumulative_optimizer_steps"]
    require(
        all(int(s["step"]) == steps for s in candidate["optimizer"]["state"].values())
        and candidate["cumulative_optimizer_steps"] == steps,
        "checkpoint_adam_steps_mismatch",
    )
    require(
        candidate["new_agent_samples"]
        == parent["new_agent_samples"] + counts["additional_samples"],
        "checkpoint_sample_counter_mismatch",
    )
    require(
        candidate["new_physics_ticks"]
        == parent["new_physics_ticks"] + counts["additional_physical_ticks"],
        "checkpoint_physics_counter_mismatch",
    )
    require(
        candidate["opponent_state"]["nexto_probability"] == evaluation["nexto_probability"],
        "checkpoint_opponent_schedule_mismatch",
    )
    prefix_path = RESULTS / f"training_curve_through_{target:06d}.jsonl"
    if prefix_path.exists():
        require(
            prefix_path.read_bytes().replace(b"\r\n", b"\n") == prefix,
            "existing_frozen_prefix_mismatch",
        )
    else:
        prefix_path.write_bytes(prefix)
    report = dict(
        verdict="PASS",
        meaning="Saved checkpoint/ledger integrity, not gameplay capability",
        optimizer_steps_in_audit=0,
        device="cpu",
        target=target,
        checkpoint=path.relative_to(ROOT).as_posix(),
        checkpoint_sha256=digest(candidate_bytes),
        curve_prefix_sha256=digest(prefix),
        authority_sha256=json_hash(authority),
        counts=counts,
        cumulative_entity_samples=candidate["new_agent_samples"],
        cumulative_entity_physics_world_ticks=candidate["new_physics_ticks"],
        maximum_completed_mean_kl=max(r["ppo"]["completed_update_mean_kl"] for r in rows),
        maximum_sample_kl=max(r["ppo"]["completed_update_sample_kl_max"] for r in rows),
        note="Adam topology/steps accounted; exact resume-state equality is audited separately.",
    )
    (RESULTS / f"boundary_{target:06d}_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit(args.update), indent=2))
