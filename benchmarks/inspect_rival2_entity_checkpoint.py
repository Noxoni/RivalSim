"""Read-only model audit plus immutable evidence copy of an accepted checkpoint."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, utc, write_json
from benchmarks.run_rival2_ssl_entity_joint_control import (
    CHECKPOINTS,
    EXTERNAL,
    INITIAL,
    RESULTS,
    verify,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    verify(published=True)
    latest = (
        json.loads((EXTERNAL / "latest.json").read_text())
        if args.checkpoint is None
        else {
            "path": str(args.checkpoint),
            "sha256": args.sha256,
        }
    )
    assert latest["sha256"], "Explicit checkpoint requires --sha256"
    raw = Path(latest["path"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    assert digest == latest["sha256"], (
        "Rolling changed during inspection; inspect next stable snapshot"
    )
    p = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)
    assert p["accepted_updates"] > 0
    if args.checkpoint is None:
        assert p["accepted_updates"] == latest["accepted_updates"]
    source = Path(latest["path"]).resolve()
    path = (
        source
        if source.parent == CHECKPOINTS.resolve() and source.name.startswith("plus_")
        else CHECKPOINTS / f"audited_plus_{p['accepted_updates']:03d}.pt"
    )
    if path.exists():
        assert sha(path) == digest
    else:
        path.write_bytes(raw)
    names = (
        "trunk.",
        "context_encoder.",
        "context_gru.",
        "actor.",
        "context_actor.",
        "entities.",
        "entity_actor.",
        "entity_context.",
        "critic.",
    )
    groups = {}
    for prefix in names:
        deltas = [
            (name, float((value - initial["model"][name]).abs().max()))
            for name, value in p["model"].items()
            if name.startswith(prefix) and value.is_floating_point()
        ]
        groups[prefix] = dict(
            changed_tensors=sum(v > 0 for _, v in deltas),
            maximum_parameter_change=max(v for _, v in deltas),
        )
    checks = dict(
        all_model_finite=all(bool(torch.isfinite(v).all()) for v in p["model"].values()),
        all_adam_finite=all(
            bool(torch.isfinite(v).all())
            for s in p["optimizer"]["state"].values()
            for v in s.values()
            if torch.is_tensor(v)
        ),
        all_step_counters_exact=all(
            float(s["step"]) == p["accepted_updates"] * 182
            for s in p["optimizer"]["state"].values()
        ),
        all_expected_parameter_groups_updated=all(
            g["changed_tensors"] > 0 for g in groups.values()
        ),
        entity_map_and_action_table_unchanged=all(
            torch.equal(p["model"][name], initial["model"][name])
            for name in (
                "action_table",
                "entities.pad_positions",
                "entities.pad_large",
                "entities.type_indices",
            )
        ),
        parent_preserved=p["parent_sha256"] == initial["parent_sha256"],
    )
    curve = [
        json.loads(s) for s in (EXTERNAL / "training_curve.jsonl").read_text().splitlines() if s
    ]
    row = next(r for r in curve if r["accepted_updates"] == p["accepted_updates"])
    result = dict(
        utc=utc(),
        verdict="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        accepted_updates=p["accepted_updates"],
        checkpoint=str(path.relative_to(ROOT)),
        sha256=digest,
        changed_groups=groups,
        optimizer_state_tensors=len(p["optimizer"]["state"]),
        sampled_training_row=row,
        capability_claim=False,
        note="This proves parameters actually updated finitely, not learned gameplay. "
        "Use scheduled deterministic evaluations.",
    )
    write_json(RESULTS / f"accepted_{p['accepted_updates']:03d}_integrity.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "verdict",
                    "checks",
                    "accepted_updates",
                    "checkpoint",
                    "sha256",
                    "changed_groups",
                )
            }
        )
    )
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
