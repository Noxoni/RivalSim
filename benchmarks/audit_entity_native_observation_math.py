"""Two-interpreter, read-only observation math comparison on a closed native case.

Export decodes native packets with native Python. Compare uses the real production
Rival2TensorBridge observation methods on CPU tensor views, without a simulator,
GPU, training, or a second hand-transcribed observation reference. Reconstructed
timers/wheels/lifecycle/pads are shared inputs, NOT independently verified native
measurements. This isolates formatting from unobservable state and dynamics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/rival2/entity_native_observation_math_v1"
SOURCE = ROOT/"results/rival2/entity_native_reset_v2/reference600_blue"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path, obj):
    assert not path.exists()
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",newline="\n")


def export():
    sys.path.insert(0,str(ROOT/"benchmarks"))
    import report_entity_native_reset_v2 as report
    from report_entity_native_comparison import packets, CHANNELS
    from diagnose_entity_native_kickoff_gap import quaternion
    assert not OUT.exists()
    assert json.loads((SOURCE/"bot_state.json").read_text())["closed"]
    manifest=json.loads((ROOT/"results/rival2/entity_native_packet_v1/reference600_manifest.json").read_text())
    spec=manifest["observation"]
    fields=[r["field"] for r in manifest["fields"]]
    ds=report.read_records(SOURCE/"decisions.bin.gz",report.schema.DECISION)
    n=len(ds); obs=ds["observation"]
    assert np.all(ds["team"]==0),"This frozen audit is complete Blue only"
    values={}
    raw={k:[] for k in ("car_pos","car_vel","car_quat","car_ang_vel","boost",
        "on_ground","has_jumped","is_jumping","has_double_jumped","has_flipped",
        "is_flipping","is_supersonic","car_is_demoed","demo_respawn_timer","flip_time",
        "ball_pos","ball_vel","ball_ang_vel","rival2.previous_action")}
    vec=lambda v:[v.x,v.y,v.z]
    for i,p in enumerate(packets(SOURCE/"decision_packets.bin.gz")):
        assert i<n and int(p.match_info.frame_num)==int(ds["frame"][i])
        cars=sorted(p.players,key=lambda c:int(c.team))
        assert [int(c.team) for c in cars]==[0,1]
        for k,member in (("car_pos","location"),("car_vel","velocity"),("car_ang_vel","angular_velocity")):
            raw[k].append([vec(getattr(c.physics,member)) for c in cars])
        raw["car_quat"].append([quaternion(c.physics.rotation) for c in cars])
        raw["boost"].append([float(c.boost) for c in cars])
        phases=[str(c.air_state).split(".")[-1] for c in cars]
        raw["on_ground"].append([s=="OnGround" for s in phases])
        raw["is_jumping"].append([s=="Jumping" for s in phases])
        raw["is_flipping"].append([s=="Dodging" for s in phases])
        for k,member in (("has_jumped","has_jumped"),("has_double_jumped","has_double_jumped"),
                         ("has_flipped","has_dodged"),("is_supersonic","is_supersonic")):
            raw[k].append([bool(getattr(c,member)) for c in cars])
        raw["car_is_demoed"].append([float(c.demolished_timeout)>0 for c in cars])
        raw["demo_respawn_timer"].append([max(0,float(c.demolished_timeout)) for c in cars])
        raw["flip_time"].append([max(0,float(c.dodge_elapsed)) if c.has_dodged and s!="OnGround" else 0 for c,s in zip(cars,phases)])
        for k,member in (("ball_pos","location"),("ball_vel","velocity"),("ball_ang_vel","angular_velocity")):
            raw[k].append(vec(getattr(p.balls[0].physics,member)))
        raw["rival2.previous_action"].append([[float(getattr(c.last_input,key)) for key in CHANNELS] for c in cars])
    assert len(raw["car_pos"])==n
    values.update({k:np.asarray(v,np.float32) for k,v in raw.items()})
    # Use existing reconstructions identically in both formatting paths; not a
    # second measurement of native hidden state. Clipped timers stay clipped.
    copied=[]
    for key,scale_key in (("jump_time","jump_time_scale"),("air_time","air_time_scale"),
        ("air_time_since_jump","air_time_scale"),("boosting_time","boosting_time_scale"),
        ("time_since_boosted","time_since_boosted_scale"),("supersonic_time","supersonic_time_scale"),
        ("sticky_ticks","sticky_tick_scale")):
        indices=[fields.index(prefix+"."+key) for prefix in ("self","opponent")]
        values[key]=obs[:,indices]*np.float32(spec[scale_key]);copied.extend(indices)
    wheel_indices=np.r_[35:39,74:78]
    values["wheel_contact"]=obs[:,wheel_indices].reshape(n,2,4).copy();copied.extend(wheel_indices.tolist())
    durations=np.asarray(spec["canonical_boost_pad_durations"],np.float32)
    values["pad_cooldown"]=obs[:,100:167:2]*durations;copied.extend(range(99,167))
    values["rival2.kickoff_indicator"]=obs[:,175].copy()
    values["rival2.touch_count"]=obs[:,176:178].copy()
    values["rival2.demoed_event"]=obs[:,178:180].copy()
    values["rival2.episode_ticks"]=obs[:,180]*np.float32(spec["episode_age_scale_ticks"])
    values["rival2.no_touch_ticks"]=obs[:,181]*np.float32(spec["no_touch_age_scale_ticks"])
    copied.extend(range(175,182))
    OUT.mkdir()
    np.savez_compressed(OUT/"inputs.npz",**values,recorded_observation=obs,
                        recorded_action=ds["action"],resets=ds["resets"],frame=ds["frame"])
    write(OUT/"export.json",dict(case="reference600_blue",samples=n,
        fields=fields,shared_reconstruction_indices=sorted(set(copied)),
        raw_packet_view_names=list(raw),source=manifest["source"],artifact=manifest["artifact"],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE/"decisions.bin.gz",SOURCE/"decision_packets.bin.gz",OUT/"inputs.npz",Path(__file__))},
        qualification="Packet vectors/native flags are independently decoded. Shared wheel/timer/pad/lifecycle values isolate observation math only; cannot establish their physical correctness. Stored first-reset wheel projection preserved. Native previous control is decoded, not manufactured."))
    print(json.dumps(dict(exported=n,shared_reconstruction_fields=len(set(copied)))))


def compare():
    sys.path.insert(0,str(ROOT))
    import torch
    from rivalsim.rival2_env import Rival2TensorBridge
    from rivalsim.rival2_contracts import POSITION_SCALE,ORANGE_PAD_REMAP
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(),"CPU-only diagnostic"
    meta=json.loads((OUT/"export.json").read_text())
    for p,h in meta["hashes"].items():assert sha(ROOT/p)==h,p
    data=np.load(OUT/"inputs.npz")
    skip={"recorded_observation","recorded_action","resets","frame"}
    bridge=object.__new__(Rival2TensorBridge)
    bridge.num_envs=len(data["frame"]);bridge.device=torch.device("cpu")
    bridge.views={k:torch.from_numpy(data[k]) for k in data.files if k not in skip}
    bridge.position_scale=torch.tensor(POSITION_SCALE,dtype=torch.float32)
    bridge.pad_durations=torch.tensor([10.]*6+[4.]*28,dtype=torch.float32)
    bridge.orange_pad_remap=torch.tensor(ORANGE_PAD_REMAP,dtype=torch.long)
    bridge.blue_pad_remap=torch.arange(34)
    bridge.team_signs=torch.tensor(((1,1,1),(-1,-1,1)),dtype=torch.float32)
    rebuilt=bridge.observation()[:,0].numpy()
    recorded=data["recorded_observation"]
    assert np.isfinite(rebuilt).all()
    error=np.abs(rebuilt-recorded)
    fields=[dict(index=i,field=f,max_abs=float(error[:,i].max()),mean_abs=float(error[:,i].mean()),
        above_1e5=int(np.count_nonzero(error[:,i]>1e-5)),shared_reconstruction=i in meta["shared_reconstruction_indices"])
        for i,f in enumerate(meta["fields"])]
    artifact=ROOT/meta["artifact"]["path"]
    assert sha(artifact)==meta["artifact"]["sha256"]
    actor=torch.jit.load(str(artifact),map_location="cpu").eval()
    hidden=[torch.zeros(1,1,256),torch.zeros(1,1,256)]
    previous_reset=None;actions=[[],[]]
    with torch.inference_mode():
        for i,reset in enumerate(data["resets"]):
            if reset!=previous_reset:
                hidden=[torch.zeros_like(h) for h in hidden];previous_reset=reset
            for arm,observations in enumerate((recorded,rebuilt)):
                act,hidden[arm],logits=actor(torch.from_numpy(observations[i:i+1].copy()),hidden[arm])
                assert torch.isfinite(logits).all() and torch.isfinite(hidden[arm]).all()
                actions[arm].append(act[0].numpy())
    actions=np.asarray(actions)
    assert np.array_equal(actions[0],data["recorded_action"]),"Original recorded action replay failed"
    changed=np.any(actions[0]!=actions[1],axis=1)
    np.savez_compressed(OUT/"comparison.npz",rebuilt=rebuilt,recorded_action=actions[0],rebuilt_action=actions[1])
    assert not torch.cuda.is_initialized()
    assert sha(artifact)==meta["artifact"]["sha256"]
    result=dict(samples=len(recorded),fields=fields,maximum_absolute_difference=float(error.max()),
        rows_any_difference_above_1e5=int(np.any(error>1e-5,axis=1).sum()),
        recorded_action_replay_exact=True,rebuilt_action_changed_rows=int(changed.sum()),
        first_changed_rows=np.flatnonzero(changed)[:20].tolist(),
        cpu_only=True,optimizer_steps=0,production_modified=False,source_and_export_unchanged=True,
        source=meta["source"],artifact=meta["artifact"],
        hashes={str(p.relative_to(ROOT)):sha(p) for p in (OUT/"export.json",OUT/"inputs.npz",OUT/"comparison.npz",ROOT/"rivalsim/rival2_env.py",ROOT/"rivalsim/rival2_contracts.py",Path(__file__))},
        interpretation="Actual production observation methods called on CPU views of recorded/shared-reconstructed state, with full sequential exact-actor replay. Not proof the shared inferred timers/wheels/pads equal unobserved engine state, nor a physics-equivalent trajectory.")
    write(OUT/"audit.json",result)
    print(json.dumps({k:result[k] for k in ("samples","maximum_absolute_difference","rows_any_difference_above_1e5","recorded_action_replay_exact","rebuilt_action_changed_rows")}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("export","compare"))
    (export if p.parse_args().mode=="export" else compare)()
