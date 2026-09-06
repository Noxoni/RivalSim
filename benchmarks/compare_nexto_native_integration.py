"""Frozen 2x2 native Nexto timing/table intervention; no training or deployment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"results/rival2/nexto_native_causal_v1"
CHECKPOINT = ROOT/"checkpoints/rival2/direct_skills_v1/plus_000600.pt"
CHECKPOINT_SHA = "8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2"
TICKS = 2400
ARMS = (("baseline", False, False), ("table_only", True, False),
        ("timing_only", False, True), ("table_and_timing", True, True))
SOURCES = ("benchmarks/compare_nexto_native_integration.py",
           "benchmarks/probe_nexto_native_timing.py",
           "benchmarks/evaluate_rival2_ssl_entity_full_match.py",
           "benchmarks/direct_skills_eval_stream.py", "third_party/nexto/adapter.py",
           "rivalsim/full_match.py", "rivalsim/rival2_env.py",
           "results/rival2/native_nexto_integration_audit_v1/audit.json",
           "results/rival2/native_nexto_integration_audit_v1/source_binding.json",
           "results/rival2/native_nexto_integration_audit_v1/kickoff_tables.npz",
           "results/rival2/entity_native_bridge_v1/reference600_native_sequences.npz",
           "tests/test_nexto_native_timing_probe.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path, obj):
    assert not path.exists()
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",newline="\n")


def prepare():
    assert not OUT.exists()
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    OUT.mkdir()
    write(OUT/"protocol.json", dict(
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=CHECKPOINT_SHA,
        worlds=10, physics_ticks_per_arm=TICKS, physics_hz=120, rival_policy_hz=30,
        arms=ARMS, model_selection=False, optimizer_steps=0,
        method="2x2 intervention on kickoff yaw rows44:60 and neural compute/emission clock. Other Nexto inputs, deterministic argmax, kickoff admission and script lifecycle remain baseline. Not a full native Nexto replacement.",
        starts="Existing fixed five standard layouts on both sides; identical seed/reset authority in every arm",
        baseline_check="All6000 Rival decisions must exactly match existing reference600_native_sequences source_action",
        scope="20 seconds per world, not complete matches. Report goals/touches/kickoffs and exact action/state differences. No native game, reward change, PPO, promotion or checkpoint selection.",
        limit="Stop after all four arms; never tune from their results or claim all native timing/observation semantics are matched",
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
    protocol = json.loads((OUT/"protocol.json").read_text())
    for p,h in protocol["sources"].items(): assert sha(ROOT/p) == h, p
    for p in (*SOURCES, "results/rival2/nexto_native_causal_v1/protocol.json"):
        remote = subprocess.check_output(["git", "show", "origin/main:"+p],cwd=ROOT)
        assert remote.replace(b"\r\n",b"\n") == (ROOT/p).read_bytes().replace(b"\r\n",b"\n"),p
    assert not (OUT/"results.json").exists()
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    exact_table = np.load(ROOT/"results/rival2/native_nexto_integration_audit_v1/kickoff_tables.npz")["actual_vendored_upstream"]
    class Runner(CandidateMatchRunner):
        def __init__(self, table, timing):
            super().__init__(CHECKPOINT, CHECKPOINT_SHA, entity=True)
            if timing:
                self.nexto = make_timing_adapter(NextoPolicyAdapter)(self.num_worlds, device=self.device)
                self.nexto.set_player_index(self.nexto_side)
            if table:
                self.nexto.kickoff_sequence.copy_(torch.from_numpy(exact_table).to(self.device))
            self.action_records, self.pose_records = [], []

        def tick(self):
            super().tick()
            self.action_records.append(self.actions.detach().cpu().numpy().copy())
            if self.host_tick % 4 == 0:
                self.pose_records.append(torch.cat((self.nexto_state.car_pos.flatten(1),
                                                    self.nexto_state.ball_pos),dim=1).cpu().numpy().copy())

    torch.set_num_threads(2)
    with gpu_lease(), owned_match_stream():
        results = {}
        baseline_actions, baseline_pose = None, None
        for name,table,timing in ARMS:
            assert not (OUT/(name+".json")).exists(), "Do not silently repeat a completed arm"
            runner = Runner(table,timing)
            wall = runner.run_ticks(TICKS).seconds
            raw = runner.export()["raw"]
            actions,poses = np.stack(runner.action_records),np.stack(runner.pose_records)
            assert np.isfinite(actions).all() and np.isfinite(poses).all()
            assert not bool(raw["goal_overflow"].any())
            assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
            assert sha(CHECKPOINT) == CHECKPOINT_SHA
            sides = raw["match.rival_side"].astype(np.int64)
            rival_actions = actions[np.arange(TICKS)[:,None],np.arange(10)[None,:],sides[None,:]]
            if name == "baseline":
                baseline_actions, baseline_pose = actions, poses
                prior = np.load(ROOT/"results/rival2/entity_native_bridge_v1/reference600_native_sequences.npz")
                np.testing.assert_array_equal(rival_actions[::4],prior["source_action"])
            np.savez_compressed(OUT/(name+".npz"), actions=actions,poses=poses)
            result = dict(arm=name, summary=summarize(raw), wall_seconds=wall,
                          raw={k:v.tolist() for k,v in raw.items()}, model_unchanged=True,
                          action_world_ticks_different_from_baseline=int(np.any(actions!=baseline_actions,axis=(2,3)).sum()),
                          pose_max_abs_difference_from_baseline=float(np.abs(poses-baseline_pose).max()),
                          trace_sha256=sha(OUT/(name+".npz")), protocol_sha256=sha(OUT/"protocol.json"))
            if timing:
                result["native_clock_compute_frames"] = runner.nexto.compute_frames
                result["native_clock_emission_frames"] = runner.nexto.emission_frames
            results[name] = result
            write(OUT/(name+".json"),result)
            print(json.dumps(dict(arm=name,summary=result["summary"],action_world_ticks_changed=result["action_world_ticks_different_from_baseline"],wall_seconds=wall)),flush=True)
            del runner
            gc.collect();torch.cuda.empty_cache()
        for p,h in protocol["sources"].items(): assert sha(ROOT/p)==h,p
        write(OUT/"results.json",dict(protocol_sha256=sha(OUT/"protocol.json"),
                                     summaries={k:v["summary"] for k,v in results.items()},
                                     checkpoint_sha256=sha(CHECKPOINT), optimizer_steps=0,
                                     production_modified=False, baseline_6000_actions_exact=True,
                                     completed_regulation=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","run"))
    (prepare if p.parse_args().mode=="prepare" else run)()
