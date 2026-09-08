"""CPU-only immutable publication from an accepted permanent snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.run_sustained_gameplay_v1 import OUT,CKPTS,EXTERNAL,INITIAL,authority
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json
from rivalsim.fresh_ground_30hz import content_hash
from rivalsim.sustained_gameplay_v1 import VERSION,POLICY_VERSION


def report(offset):
    path=CKPTS/f"plus_{offset:06d}.pt"
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    steps={int(s["step"]) for s in checkpoint["optimizer"]["state"].values()}
    rows=[]
    if offset:
        with (EXTERNAL/"training_curve.jsonl").open() as f:
            for line in f:
                if not line.endswith("\n"):break
                row=json.loads(line)
                if row["accepted_updates"]>offset:break
                rows.append(row)
        assert [r["accepted_updates"] for r in rows]==list(range(1,offset+1))
    finite_model=all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values())
    finite_adam=all(bool(torch.isfinite(v).all()) for s in checkpoint["optimizer"]["state"].values()
                    for v in s.values() if torch.is_tensor(v))
    checks=dict(fresh_parent=checkpoint["parent"] is None,
        initialized_exact=checkpoint["initialized_checkpoint_sha256"]==sha(INITIAL),
        format_exact=checkpoint["format"]==VERSION+"_CHECKPOINT",
        architecture_exact=checkpoint["policy_version"]==POLICY_VERSION,
        authority_exact=checkpoint["authority_sha256"]==content_hash(authority()),
        offset_exact=checkpoint["accepted_updates"]==offset,
        finite_model=finite_model,finite_adam=finite_adam,
        adam_counter_exact=steps==({checkpoint["optimizer_steps"]} if offset else set()),
        learner_samples_exact=checkpoint["learner_decisions"]==4423680*offset,
        world_ticks_exact=checkpoint["world_physics_ticks"]==11796480*offset,
        native_nexto_frozen=checkpoint["opponent_state"]["native_nexto"]["sampling_mode"]=="native_v5",
        no_kl_rejections=all(r["ppo"]["kl_rejections"]==0 for r in rows),
        episode_identity=VERSION+"_EPISODES" in checkpoint["runtime_contract_hashes"],
        reward_identity=VERSION+"_REWARD" in checkpoint["runtime_contract_hashes"])
    assert all(checks.values()),checks
    digest=sha(path)
    evaluation_path=OUT/f"eval_{offset:06d}.json"
    evaluation=None
    if evaluation_path.exists():
        evaluation=json.loads(evaluation_path.read_text())
        assert evaluation["checkpoint"]["sha256"]==digest
        from benchmarks.report_rival2_ssl_entity_match_followup import reduce
        assert all(reduce(evaluation_path)["integrity"].values())
    evidence=dict(utc=utc(),checks=checks,checkpoint=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest),
        accepted_updates=offset,optimizer_steps=checkpoint["optimizer_steps"],
        learner_decisions=checkpoint["learner_decisions"],world_physics_ticks=checkpoint["world_physics_ticks"],
        initial_model_sha256=tensor_hash(initial["model"]),model_sha256=tensor_hash(checkpoint["model"]),
        actor_changed=any(not torch.equal(v,initial["model"][n]) for n,v in checkpoint["model"].items() if not n.startswith("critic.")),
        critic_changed=any(not torch.equal(v,initial["model"][n]) for n,v in checkpoint["model"].items() if n.startswith("critic.")),
        last_training=rows[-1] if rows else None,evaluation=evaluation["summary"] if evaluation else None,
        interpretation="Accepted numerical training boundary, not proof of gameplay improvement. Compare complete development matches.")
    write_json(OUT/f"through_{offset:06d}.json",dict(rows=rows))
    write_json(OUT/f"audit_{offset:06d}.json",evidence)
    print(json.dumps(dict(accepted_updates=offset,checks=checks,checkpoint=evidence["checkpoint"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("offset",type=int)
    args=parser.parse_args();torch.set_num_threads(4);report(args.offset)
