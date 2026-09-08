"""Frozen deterministic development matches, no training and no candidate selection."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, summarize
from benchmarks.report_rival2_ssl_entity_match_followup import reduce
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json
from rivalsim.fresh_ground_30hz import content_hash
from rivalsim.sustained_gameplay_v1 import VERSION,SustainedPolicy,SEED
from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter,VERSION as NEXTOVERSION


def specification():
    return dict(version=VERSION+"_DEVELOPMENT_EVAL",matches=10,
        starts="Five standard standing kickoff layouts, both sides; no training-state assistance",
        seconds=300,overtime_cap_seconds=120,physics_seed=2026090573,
        nexto_controller=NEXTOVERSION,nexto_seed=SEED+4,nexto_sampling_mode="native_v5",
        rival="Raw deterministic joint90 argmax; actor hidden persists until native goal reset",
        timeouts="No training inactivity rule in evaluation. Regulation/overtime match clock only.",
        interpretation="Repeated development telemetry, not held-out acceptance or SSL/ranked proof",
        optimizer_steps=0)


def evaluate(checkpoint,expected,output):
    import json
    checkpoint,output=Path(checkpoint),Path(output)
    if output.exists():
        result=json.loads(output.read_text())
        assert result["checkpoint"]["sha256"]==expected==sha(checkpoint)
        assert result["evaluation_spec_sha256"]==content_hash(specification())
        assert all(reduce(output)["integrity"].values())
        return result
    receipt=output.with_suffix(".started.json")
    if receipt.exists():raise RuntimeError("Interrupted evaluation needs operational audit: "+str(receipt))
    write_json(receipt,dict(utc=utc(),checkpoint=str(checkpoint),sha256=expected))
    with owned_match_stream():
        runner=CandidateMatchRunner(checkpoint,expected,entity=True,policy_factory=SustainedPolicy)
        runner.nexto=NextoNativeV5PolicyAdapter(runner.num_worlds,device=runner.device,
            sampling_mode="native_v5",seed=SEED+4)
        runner.nexto.set_player_index(runner.nexto_side)
        nexto_before=tensor_hash(runner.nexto.actor.state_dict())
        elapsed=0.
        for _ in range(15):
            elapsed+=runner.run_ticks(2400).seconds
            print("EVAL_PROGRESS "+json.dumps(dict(ticks=runner.host_tick,seconds=elapsed)),flush=True)
        for _ in range(24):
            if bool(runner.phase_status()["done"].all()):break
            elapsed+=runner.run_ticks(600).seconds
        raw=runner.export()["raw"]
        assert not raw["goal_overflow"].any()
        assert tensor_hash(runner.rival_policy.state_dict())==runner.model_hash_before
        assert tensor_hash(runner.nexto.actor.state_dict())==nexto_before
        assert sha(checkpoint)==expected
        result=dict(utc=utc(),checkpoint=dict(path=checkpoint.relative_to(ROOT).as_posix(),sha256=expected),
            evaluation_spec_sha256=content_hash(specification()),summary=summarize(raw),
            raw={k:v.tolist() for k,v in raw.items()},wall_seconds=elapsed,
            hidden_resets=runner.hidden_reset_count.cpu().tolist(),optimizer_steps=0,
            model_unchanged=True,checkpoint_unchanged=True,nexto_unchanged=True)
        write_json(output,result)
        del runner
        gc.collect();torch.cuda.empty_cache()
    integrity=reduce(output)
    assert all(integrity["integrity"].values())
    write_json(output.with_suffix(".integrity.json"),integrity)
    print("EVAL_COMPLETE "+json.dumps(result["summary"]),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--sha256",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.checkpoint,args.sha256.upper(),args.output)
