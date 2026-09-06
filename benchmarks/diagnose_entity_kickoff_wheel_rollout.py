"""One-factor 20-second kickoff input probe; not a runtime fix or training."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.evaluate_direct_skills_handbrake_v3 import verify
from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner,summarize
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash

OUT=ROOT/"results/rival2/entity_native_gap_v1"
CHECKPOINT=ROOT/"checkpoints/rival2/direct_skills_v1/plus_000600.pt"
SHA="8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2"
TICKS=2400


class ProbeRunner(CandidateMatchRunner):
    def __init__(self,override):
        super().__init__(CHECKPOINT,SHA,entity=True)
        self.override=override
        self.prior_reset=torch.full_like(self.hidden_reset_count,-1)
        self.initial_rows=[]
        self.action_rows=[]

    @torch.inference_mode()
    def _update_rival_action(self):
        new=self.hidden_reset_count!=self.prior_reset
        original=self.rival_observation
        if self.override:
            changed=original.clone()
            for index in (35,36,37,38,74,75,76,77):
                changed[self.batch_index[new],self.rival_side[new],index]=1
            self.rival_observation=changed
        try:
            super()._update_rival_action()
        finally:
            self.rival_observation=original
        if bool(new.any()):
            self.initial_rows.append(dict(tick=self.host_tick,worlds=self.batch_index[new].tolist(),
                actions=self.rival_action[new].tolist(),reset_count=self.hidden_reset_count[new].tolist()))
        self.action_rows.append(self.rival_action.cpu().numpy().copy())
        self.prior_reset=self.hidden_reset_count.clone()


def run():
    verify()
    target=OUT/"wheel_rollout.json"
    assert not target.exists(),"Preserve previous ablation"
    result={};traces={}
    with gpu_lease(),owned_match_stream():
        for label,override in (("unchanged",False),("native_style_first_wheels",True)):
            runner=ProbeRunner(override)
            seconds=runner.run_ticks(TICKS).seconds
            raw=runner.export()["raw"]
            assert tensor_hash(runner.rival_policy.state_dict())==runner.model_hash_before
            assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest().upper()==SHA
            actions=np.stack(runner.action_rows)
            if not override:
                with np.load(ROOT/"results/rival2/entity_native_bridge_v1/reference600_native_sequences.npz") as captured:
                    np.testing.assert_array_equal(actions,captured["source_action"])
            result[label]=dict(summary=summarize(raw),seconds=seconds,initial_actions=runner.initial_rows,
                raw={k:v.tolist() for k,v in raw.items()},unchanged_checkpoint=True)
            traces[label+"_actions"]=actions
            print(json.dumps(dict(arm=label,summary=summarize(raw))),flush=True)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
    np.savez_compressed(OUT/"wheel_rollout_actions.npz",**traces)
    report=dict(version="RIVAL2_KICKOFF_FIRST_INPUT_WHEEL_ABLATION_V1",checkpoint_sha256=SHA,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
        action_trace_sha256=hashlib.sha256((OUT/"wheel_rollout_actions.npz").read_bytes()).hexdigest().upper(),
        physics_ticks=TICKS,worlds=10,regulation_completed=False,results=result,optimizer_steps=0,
        interpretation="Exactly one-factor counterfactual: at the first Rival decision of each kickoff only, set self/opponent wheel inputs to1 as the qualified native packet runtime does on flat ground. Actual simulator state, physics, Nexto, all other inputs, model and resets unchanged. Both arms 20seconds on all five layouts and both teams, not full matches or checkpoint selection. Overrides confined to this diagnostic subclass; not a proposed deployment masking fix.")
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",newline="\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
