"""Read-only native kickoff input sensitivity; never a deployment transform."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
from report_entity_native_comparison import DECISION, OUT as CAPTURE, packets, read_records

OUT=ROOT/"results/rival2/entity_native_gap_v1"
WHEELS=np.r_[35:39,74:78]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def quaternion(rotator):
    """RL pitch-up/roll convention to right-handed xyzw, state export only."""
    sr,cr=math.sin(-rotator.roll/2),math.cos(-rotator.roll/2)
    sp,cp=math.sin(-rotator.pitch/2),math.cos(-rotator.pitch/2)
    sy,cy=math.sin(rotator.yaw/2),math.cos(rotator.yaw/2)
    return np.array([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy],np.float32)


def wheel_probe(observation, value, indices=WHEELS):
    result=np.array(observation,copy=True)
    result[...,indices]=value
    return result


def run():
    folder=CAPTURE/"reference600_blue"
    ready=json.loads((folder/"ready.json").read_text())
    ds=read_records(folder/"decisions.bin.gz",DECISION)
    sim_path=ROOT/"results/rival2/entity_native_bridge_v1/reference600_native_sequences.npz"
    sim=np.load(sim_path)
    artifact=ROOT/ready["artifact"]["path"]
    assert sha(artifact)==ready["artifact"]["sha256"]
    assert sha(ROOT/ready["source"]["path"])==ready["source"]["sha256"]
    actor=torch.jit.load(str(artifact),map_location="cpu").eval()
    manifest=json.loads((CAPTURE/"reference600_manifest.json").read_text())
    arrays={k:[] for k in ("car_pos","car_vel","car_ang_vel","car_quat","car_boost",
        "car_on_ground","native_input","ball_pos","ball_vel","ball_ang_vel","kickoff","seconds_remaining")}
    channels=("throttle","steer","pitch","yaw","roll","jump","boost","handbrake")
    def vec(v):return [v.x,v.y,v.z]
    for i,p in enumerate(packets(folder/"decision_packets.bin.gz")):
        assert i<len(ds) and int(p.match_info.frame_num)==int(ds[i]["frame"])
        players=sorted(p.players,key=lambda q:int(q.team))
        assert [int(q.team) for q in players]==[0,1]
        assert int(players[0].player_id)==int(ready["player_id"])
        for k,field in (("car_pos","location"),("car_vel","velocity"),("car_ang_vel","angular_velocity")):
            arrays[k].append([vec(getattr(q.physics,field)) for q in players])
        arrays["car_quat"].append([quaternion(q.physics.rotation) for q in players])
        arrays["car_boost"].append([q.boost for q in players])
        arrays["car_on_ground"].append([str(q.air_state).split(".")[-1]=="OnGround" for q in players])
        arrays["native_input"].append([[float(getattr(q.last_input,k)) for k in channels] for q in players])
        for k,field in (("ball_pos","location"),("ball_vel","velocity"),("ball_ang_vel","angular_velocity")):
            arrays[k].append(vec(getattr(p.balls[0].physics,field)))
        arrays["kickoff"].append(str(p.match_info.match_phase).split(".")[-1]=="Kickoff")
        arrays["seconds_remaining"].append(p.match_info.game_time_remaining)
    assert len(arrays["car_pos"])==len(ds)
    arrays={k:np.array(v,np.float32) for k,v in arrays.items()}
    arrays.update(frame=ds["frame"].copy(),resets=ds["resets"].copy(),action=ds["action"].copy())
    starts=np.flatnonzero(np.r_[True,ds["resets"][1:]!=ds["resets"][:-1]])
    rows=[];field_differences=[]
    with torch.inference_mode():
        for start in starts:
            obs=ds["observation"][start].copy()
            assert arrays["kickoff"][start]
            physical=np.r_[9:12,48:51]
            # Both-car geometry, Blue perspective. Not an exact hidden-state match.
            choices=sim["observation"][0,:5]
            nearest=int(np.argmin(np.linalg.norm(choices[:,physical]-obs[physical],axis=1)))
            pair=choices[nearest].copy()
            probes={"native":obs,"native_zero_self_wheels":wheel_probe(obs,0,np.arange(35,39)),
                "native_zero_opponent_wheels":wheel_probe(obs,0,np.arange(74,78)),
                "native_zero_all_wheels":wheel_probe(obs,0),"sim_initial":pair,
                "sim_initial_all_wheels_one":wheel_probe(pair,1)}
            results={}
            for name,value in probes.items():
                action,hidden,logits=actor(torch.from_numpy(value[None]),torch.zeros(1,1,256))
                assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
                results[name]=dict(action=action[0].tolist(),argmax=int(logits.argmax()))
            assert results["native"]["action"]==ds["action"][start].tolist()
            difference=np.abs(obs-pair)
            field_differences.append(difference)
            row=dict(frame=int(ds["frame"][start]),reset=int(ds["resets"][start]),
                nearest_standard_layout=nearest,normalized_both_car_position_distance=float(np.linalg.norm(obs[physical]-pair[physical])),
                native_wheels=obs[WHEELS].tolist(),sim_wheels=pair[WHEELS].tolist(),probes=results)
            rows.append(row)
    diff=np.stack(field_differences)
    fields=[dict(index=i,field=manifest["fields"][i]["field"],mean_abs=float(diff[:,i].mean()),
                 max_abs=float(diff[:,i].max())) for i in range(182)]
    count=lambda pred:sum(pred(r) for r in rows)
    summary=dict(kickoffs=len(rows),native_initial_boost=count(lambda r:bool(r["probes"]["native"]["action"][6])),
        all_wheels_zero_probe_boost=count(lambda r:bool(r["probes"]["native_zero_all_wheels"]["action"][6])),
        all_wheels_zero_probe_changed_actions=count(lambda r:r["probes"]["native"]["action"]!=r["probes"]["native_zero_all_wheels"]["action"]),
        maximum_both_car_position_distance=max(r["normalized_both_car_position_distance"] for r in rows))
    assert not OUT.exists(),"Preserve existing diagnostic"
    OUT.mkdir()
    np.savez_compressed(OUT/"native_physical.npz",**arrays)
    result=dict(version="RIVAL2_NATIVE_KICKOFF_GAP_DIAGNOSTIC_V1",summary=summary,rows=rows,fields=fields,
        source=ready["source"],artifact=ready["artifact"],
        hashes={str(p.relative_to(ROOT).as_posix()):sha(p) for p in
            (folder/"decisions.bin.gz",folder/"decision_packets.bin.gz",sim_path,OUT/"native_physical.npz",Path(__file__))},
        optimizer_steps=0,production_changes=False,
        interpretation="Offline first-input ablations only, with zero recurrent state. Zeroing wheel contacts is NOT installed or asserted physically true. Nearest five standard layouts are comparison states, not exact native internal states. Action sensitivity is not proof this alone caused the 0-37 outcome or that masking would fix gameplay.")
    (OUT/"kickoff_input_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps(summary))


if __name__=="__main__":
    torch.set_num_threads(1)
    run()
