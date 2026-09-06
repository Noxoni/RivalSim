"""Fixed full-regulation follow-up to the completed 20s integration intervention.

Two arms only, unchanged Rival and physics. Not training, deployment, checkpoint
selection, or a claim that all native Nexto observation/RNG semantics now match.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.compare_nexto_native_integration import CHECKPOINT, CHECKPOINT_SHA, sha, write

OUT = ROOT/"results/rival2/nexto_native_full_match_v1"
SHORT = ROOT/"results/rival2/nexto_native_causal_v1"
ARMS = (("baseline", False), ("native_timing_and_table", True))
TICKS = 36000
SOURCES = ("benchmarks/compare_nexto_native_full_match.py",
           "benchmarks/probe_nexto_native_timing.py",
           "benchmarks/compare_nexto_native_integration.py",
           "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
           "benchmarks/direct_skills_eval_stream.py", "third_party/nexto/adapter.py",
           "rivalsim/full_match.py", "rivalsim/rival2_env.py",
           "results/rival2/native_nexto_integration_audit_v1/kickoff_tables.npz",
           "results/rival2/nexto_native_causal_v1/results.json",
           "results/rival2/nexto_native_causal_v1/baseline.npz",
           "results/rival2/nexto_native_causal_v1/table_and_timing.npz")


def prepare():
    assert not OUT.exists()
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    OUT.mkdir()
    write(OUT/"protocol.json", dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=CHECKPOINT_SHA, arms=ARMS, worlds=10, ticks=TICKS,
        physics_hz=120, rival_policy_hz=30, optimizer_steps=0,
        start="Same fixed five layouts on each side and seed as the completed 20s comparison",
        intervention="Exact same diagnostic timing/table arm as completed20s; no new opponent settings",
        prior_trace_check="First2400ticks all Rival and Nexto controls and600pose rows must match respective prior arms exactly",
        stop="Full300seconds of simulator regulation; no overtime extension, ties explicitly unresolved",
        telemetry="Existing full-match score, touches and physical outcomes. No reward modification or candidate selection",
        limitations="Neural compute/emission and kickoff yaw only. Other Nexto admission, observation construction, argmax/RNG, native countdown timing and physical differences remain unchanged. Not full-native parity or SSL proof.",
        sources={p:sha(ROOT/p) for p in SOURCES}))


def run():
    import numpy as np
    import torch
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, summarize
    from benchmarks.probe_nexto_native_timing import make_timing_adapter
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
    from third_party.nexto.adapter import NextoPolicyAdapter
    protocol=json.loads((OUT/"protocol.json").read_text())
    for p,h in protocol["sources"].items():assert sha(ROOT/p)==h,p
    for p in (*SOURCES,"results/rival2/nexto_native_full_match_v1/protocol.json"):
        committed=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        assert committed.replace(b"\r\n",b"\n")== (ROOT/p).read_bytes().replace(b"\r\n",b"\n"),p
    assert not (OUT/"results.json").exists()
    table=np.load(ROOT/"results/rival2/native_nexto_integration_audit_v1/kickoff_tables.npz")["actual_vendored_upstream"]
    class Runner(CandidateMatchRunner):
        def __init__(self, changed):
            super().__init__(CHECKPOINT,CHECKPOINT_SHA,entity=True)
            if changed:
                self.nexto=make_timing_adapter(NextoPolicyAdapter)(self.num_worlds,device=self.device)
                self.nexto.set_player_index(self.nexto_side)
                self.nexto.kickoff_sequence.copy_(torch.from_numpy(table).to(self.device))
            self.actions_recorded,self.poses_recorded=[],[]

        def tick(self):
            super().tick()
            self.actions_recorded.append(self.actions.detach().cpu().numpy().copy())
            if self.host_tick%4==0:
                self.poses_recorded.append(torch.cat((self.nexto_state.car_pos.flatten(1),self.nexto_state.ball_pos),dim=1).cpu().numpy().copy())

    torch.set_num_threads(2)
    with gpu_lease(),owned_match_stream():
        results={}
        for name,changed in ARMS:
            assert not (OUT/(name+".json")).exists(), "Never silently rerun a finished arm"
            runner=Runner(changed)
            wall=0.
            for block in range(TICKS//2400):
                wall+=runner.run_ticks(2400).seconds
                print(json.dumps(dict(arm=name,ticks=runner.host_tick,wall_seconds=wall)),flush=True)
            raw=runner.export()["raw"]
            actions,poses=np.stack(runner.actions_recorded),np.stack(runner.poses_recorded)
            old=np.load(SHORT/("table_and_timing.npz" if changed else "baseline.npz"))
            np.testing.assert_array_equal(actions[:2400],old["actions"])
            np.testing.assert_array_equal(poses[:600],old["poses"])
            assert np.isfinite(actions).all() and np.isfinite(poses).all()
            assert not bool(raw["goal_overflow"].any())
            assert tensor_hash(runner.rival_policy.state_dict())==runner.model_hash_before
            assert sha(CHECKPOINT)==CHECKPOINT_SHA
            np.savez_compressed(OUT/(name+".npz"),actions=actions,poses=poses)
            result=dict(arm=name,summary=summarize(raw),wall_seconds=wall,
                raw={k:v.tolist() for k,v in raw.items()},checkpoint_unchanged=True,
                original20s_trace_exact=True,protocol_sha256=sha(OUT/"protocol.json"),
                trace_sha256=sha(OUT/(name+".npz")),optimizer_steps=0)
            write(OUT/(name+".json"),result)
            results[name]=result["summary"]
            print(json.dumps(dict(arm=name,summary=result["summary"])),flush=True)
            del runner
            gc.collect();torch.cuda.empty_cache()
        for p,h in protocol["sources"].items():assert sha(ROOT/p)==h,p
        write(OUT/"results.json",dict(protocol_sha256=sha(OUT/"protocol.json"),
             summaries=results,checkpoint_sha256=sha(CHECKPOINT),optimizer_steps=0,
             production_modified=False,full_regulation=True,overtime_run=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","run"))
    (prepare if p.parse_args().mode=="prepare" else run)()
