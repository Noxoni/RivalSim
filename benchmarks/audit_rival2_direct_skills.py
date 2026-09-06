"""Read-only CPU checkpoint/evaluation audit; never constructs an optimizer/model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/direct_skills_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/direct_skills_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/direct-skills-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def audit(offset):
    parent = CHECKPOINTS / "parent_entity_000293.pt"
    candidate = CHECKPOINTS / f"plus_{offset:06d}.pt"
    authority = json.loads((RESULTS / "authority.json").read_text())
    package = json.loads((RESULTS / "package.json").read_text())
    assert digest(parent) == authority["parent_sha256"]
    p = torch.load(parent, map_location="cpu", weights_only=False)
    q = torch.load(candidate, map_location="cpu", weights_only=False)
    assert q["format"] == "RIVAL2_DIRECT_SKILLS_V1_CHECKPOINT"
    assert q["accepted_updates"] == offset
    assert q["authority_sha256"] == package["authority_sha256"]
    assert q["parent_sha256"] == digest(parent)
    checks = dict(
        parent_hash=True,
        authority=True,
        unchanged_action_contract=q["action_contract"] == p["action_contract"],
        unchanged_observation_contract=q["observation_schema_sha256"]
        == p["observation_schema_sha256"],
        unchanged_ppo_contract=q["ppo_config_sha256"] == p["ppo_config_sha256"],
        model_finite=all(bool(torch.isfinite(v).all()) for v in q["model"].values()),
        adam_finite=all(
            bool(torch.isfinite(v).all())
            for s in q["optimizer"]["state"].values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        ),
    )
    if offset == 0:
        checks.update(
            model_exact_parent=equal(p["model"], q["model"]),
            adam_exact_parent=equal(p["optimizer"], q["optimizer"]),
            zero_new_samples=q["direct_skill_samples"] == 0,
            zero_new_optimizer_steps=q["cumulative_optimizer_steps"] == 52150,
        )
    else:
        lines = []
        for line in (EXTERNAL / "training_curve.jsonl").read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["accepted_updates"] > offset:
                break
            lines.append(line.replace(b"\r\n", b"\n"))
        rows = [json.loads(line) for line in lines]
        checks["contiguous_steps"] = [r["accepted_updates"] for r in rows] == list(
            range(1, offset + 1)
        )
        checks["exact_samples"] = q["direct_skill_samples"] == offset * 4423680
        checks["exact_adam_counter"] = q["cumulative_optimizer_steps"] == 52150 + sum(
            r["ppo"]["optimizer_steps"] for r in rows
        )
        checks["no_kl_rejection"] = all(r["ppo"]["kl_rejections"] == 0 for r in rows)
        checks["nexto_one_third"] = all(
            r["training"]["nexto_training_sample_count"] * 3 == 4423680 for r in rows
        )
        prefix = RESULTS / f"training_curve_through_{offset:06d}.jsonl"
        if prefix.exists():
            assert prefix.read_bytes().replace(b"\r\n", b"\n") == b"".join(lines)
        else:
            prefix.write_bytes(b"".join(lines))
    evaluation = json.loads((RESULTS / f"evaluation_{offset:06d}.json").read_text())
    checks["evaluation_checkpoint"] = evaluation["checkpoint"]["sha256"] == digest(candidate)
    checks["evaluation_zero_steps"] = evaluation["optimizer_steps"] == 0
    assert all(checks.values()), checks
    result = dict(
        verdict="PASS",
        checks=checks,
        checkpoint_sha256=digest(candidate),
        accepted_updates=offset,
        optimizer_steps=q["cumulative_optimizer_steps"],
        new_samples=q["direct_skill_samples"],
        audit_optimizer_steps=0,
        capability_verdict="Not determined by this integrity audit",
    )
    output = RESULTS / f"integrity_{offset:06d}.json"
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def audit_start():
    """Freeze the first accepted update without running another evaluation."""
    path = CHECKPOINTS / "first_accepted_000001.pt"
    q = torch.load(path, map_location="cpu", weights_only=False)
    p = torch.load(CHECKPOINTS / "parent_entity_000293.pt", map_location="cpu", weights_only=False)
    row = json.loads((EXTERNAL / "training_curve.jsonl").read_text().splitlines()[0])
    assert q["accepted_updates"] == row["accepted_updates"] == 1
    assert q["direct_skill_samples"] == row["samples"] == 4423680
    assert q["cumulative_optimizer_steps"] == 52150 + row["ppo"]["optimizer_steps"]
    assert {int(s["step"]) for s in q["optimizer"]["state"].values()} == {
        q["cumulative_optimizer_steps"]
    }
    assert all(bool(torch.isfinite(t).all()) for t in q["model"].values())
    assert all(
        bool(torch.isfinite(t).all())
        for s in q["optimizer"]["state"].values()
        for t in s.values()
        if isinstance(t, torch.Tensor)
    )
    assert not equal(q["model"], p["model"]) and not equal(q["optimizer"], p["optimizer"])
    assert row["ppo"]["kl_rejections"] == 0
    assert row["training"]["nexto_training_sample_count"] == 1474560
    result = dict(
        verdict="PASS",
        meaning="Real accepted learning and checkpoint integrity, not capability improvement",
        checkpoint=str(path.relative_to(ROOT)),
        checkpoint_sha256=digest(path),
        accepted_updates=1,
        model_and_adam_changed=True,
        model_and_adam_finite=True,
        audit_optimizer_steps=0,
        first_training_row=row,
    )
    output = RESULTS / "first_accepted_update.json"
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "first_training_row"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--update", type=int)
    mode.add_argument("--startup", action="store_true")
    args = parser.parse_args()
    audit_start() if args.startup else audit(args.update)
