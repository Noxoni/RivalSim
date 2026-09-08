"""CPU-only publication of a closed accepted-update prefix, never live log staging."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.run_sustained_acquisition_v1 import OUT,CKPTS,EXTERNAL,load_parent,parent_identity,authority,nested_equal
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json
from rivalsim.sustained_acquisition_v1 import VERSION
from rivalsim.sustained_gameplay_v1 import POLICY_VERSION
from rivalsim.fresh_ground_30hz import content_hash


def report(offset):
    parent=load_parent();path=CKPTS/f"plus_{offset:06d}.pt"
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    rows=[]
    if offset>27:
        for line in (EXTERNAL/"training_curve.jsonl").read_text().splitlines():
            row=json.loads(line)
            if row["accepted_updates"]>offset:break
            rows.append(row)
    expected=list(range(28,offset+1))
    checks=dict(parent=checkpoint["parent"]==parent_identity(),
        original_root=checkpoint["initialized_checkpoint_sha256"]==parent["initialized_checkpoint_sha256"],
        format=checkpoint["format"]==VERSION+"_CHECKPOINT",architecture=checkpoint["policy_version"]==POLICY_VERSION,
        authority=checkpoint["authority_sha256"]==content_hash(authority()),
        contracts_unchanged=checkpoint["runtime_contract_hashes"]==parent["runtime_contract_hashes"],
        offset=checkpoint["accepted_updates"]==offset and checkpoint["additional_updates"]==offset-27,
        curve_contiguous=[r["accepted_updates"] for r in rows]==expected,
        decisions=checkpoint["learner_decisions"]==offset*4423680,
        physical_ticks=checkpoint["world_physics_ticks"]==offset*11796480,
        adam_counters={int(s["step"]) for s in checkpoint["optimizer"]["state"].values()}=={checkpoint["optimizer_steps"]},
        optimizer_steps=checkpoint["optimizer_steps"]==parent["optimizer_steps"]+sum(r["ppo"]["optimizer_steps"] for r in rows),
        model_finite=all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values()),
        adam_finite=all(bool(torch.isfinite(v).all()) for s in checkpoint["optimizer"]["state"].values() for v in s.values() if torch.is_tensor(v)),
        kl_only_telemetry=all(r["ppo"]["kl_rejections"]==0 for r in rows),
        nexto_mode=checkpoint["opponent_state"]["native_nexto"]["sampling_mode"]=="native_v5")
    if offset==27:
        checks["entry_model_exact"]=nested_equal(checkpoint["model"],parent["model"])
        checks["entry_adam_exact"]=nested_equal(checkpoint["optimizer"],parent["optimizer"])
    assert all(checks.values()),checks
    digest=sha(path)
    result=dict(utc=utc(),checks=checks,checkpoint=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest),
        accepted_updates=offset,additional_updates=offset-27,learner_decisions=checkpoint["learner_decisions"],
        optimizer_steps=checkpoint["optimizer_steps"],model_sha256=tensor_hash(checkpoint["model"]),
        retirement_at_snapshot=checkpoint["retirement"],last_training=rows[-1] if rows else None,
        interpretation="Numerical/lineage audit, not demonstrated acquisition or match improvement")
    write_json(OUT/f"through_{offset:06d}.json",dict(rows=rows))
    write_json(OUT/f"audit_{offset:06d}.json",result)
    print(json.dumps(dict(accepted_updates=offset,checks=checks,checkpoint=result["checkpoint"])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("offset",type=int);args=p.parse_args()
    torch.set_num_threads(4);report(args.offset)
