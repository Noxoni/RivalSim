"""Reduce closed complete/partial native cases; no learning or policy changes."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import torch
import rlbot.flat as flat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"deployment/entity_native_v1"))
from bot import DECISION, TICK

EXTERNAL=Path("G:/dev/RivalSim-runs/entity-native-comparison-v1")
OUT=ROOT/"results/rival2/entity_native_packet_v1"
CHANNELS=("throttle","steer","pitch","yaw","roll","jump","boost","handbrake")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def read_records(path,dtype):
    with gzip.open(path,"rb") as stream:
        raw=stream.read()
    assert len(raw)%dtype.itemsize==0,"Partial binary record"
    return np.frombuffer(raw,dtype=dtype)


def packets(path):
    with gzip.open(path,"rb") as f:
        while True:
            size=f.read(4)
            if not size:
                return
            assert len(size)==4
            n=struct.unpack("<I",size)[0]
            assert 0<n<1000000
            raw=f.read(n)
            assert len(raw)==n
            yield flat.GamePacket.unpack(raw)


def compare_delivered_history(ds, native_inputs):
    """Compare packet last_input before this decision with the prior command.

    This cannot establish delivery on the intervening unrecorded packets, but
    distinguishes a control-delivery mismatch at recorded decision endpoints
    from a bad decision made by an otherwise correctly connected policy.
    """
    native_inputs=np.asarray(native_inputs,dtype=np.float32)
    assert native_inputs.shape==(len(ds),8) and len(ds)>0
    assert np.isfinite(native_inputs).all()
    same_episode=np.zeros(len(ds),dtype=bool)
    same_episode[1:]=ds["resets"][1:]==ds["resets"][:-1]
    eligible=same_episode & (ds["missed"]==0)
    assert eligible.any(),"No uninterrupted nonboundary decisions"
    prior=np.zeros((len(ds),8),dtype=np.float32)
    prior[1:]=ds["action"][:-1]
    history=ds["observation"][:,167:175]
    native_difference=np.abs(native_inputs[eligible]-history[eligible])
    history_difference=np.abs(history[eligible]-prior[eligible])
    return dict(
        recorded_decisions=len(ds),eligible_decisions=int(eligible.sum()),
        excluded_reset_boundaries=int((~same_episode).sum()),
        excluded_gap_decisions=int((same_episode & (ds["missed"]!=0)).sum()),
        native_vs_previous_observation_mismatch_rows=int(np.any(native_difference!=0,axis=1).sum()),
        observation_vs_previous_issued_mismatch_rows=int(np.any(history_difference!=0,axis=1).sum()),
        per_channel_native_mismatches=dict(zip(CHANNELS,(native_difference!=0).sum(axis=0).tolist())),
        maximum_absolute_difference=float(native_difference.max()),
        interpretation="Native packet last_input is compared with the prior emitted command, not the command just being calculated. Excludes reset boundaries and missing-packet intervals. Decision endpoints only: not proof of every intervening 120Hz applied input or observation-domain equivalence.")


def report_delivery(case):
    folder=OUT/case
    assert json.loads((folder/"bot_state.json").read_text())["closed"]
    ready=json.loads((folder/"ready.json").read_text())
    ds=read_records(folder/"decisions.bin.gz",DECISION)
    inputs=[]
    for i,packet in enumerate(packets(folder/"decision_packets.bin.gz")):
        assert i<len(ds) and int(packet.match_info.frame_num)==int(ds[i]["frame"])
        own=[p for p in packet.players if int(p.player_id)==int(ready["player_id"])]
        assert len(own)==1 and int(own[0].team)==int(ds[i]["team"])
        inputs.append([float(getattr(own[0].last_input,key)) for key in CHANNELS])
    result=compare_delivered_history(ds,inputs)
    result.update(case=case,optimizer_steps=0,
        evidence_sha256={name:sha(folder/name) for name in ("decisions.bin.gz","decision_packets.bin.gz","ready.json","bot_state.json")})
    target=folder/"delivery_audit.json"
    assert not target.exists(),"Preserve previous delivery audit"
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result))


def report(case):
    folder=EXTERNAL/case
    state=json.loads((folder/"bot_state.json").read_text())
    assert state["closed"],"Wait for completed/closed native actor files"
    completed_result=folder/"match_result.json"
    partial_state=None
    result=json.loads(completed_result.read_text()) if completed_result.exists() else None
    assert not (folder/"failure.json").exists(),"Preserve failure separately; not a healthy match"
    ready=json.loads((folder/"ready.json").read_text())
    if result is not None:
        assert ready==result["readiness"]
    else:
        partial_state=json.loads((EXTERNAL/"campaign_state.json").read_text())
        assert partial_state["status"]=="stopped" and partial_state["case"]==case
    artifact=ROOT/ready["artifact"]["path"]
    source=ROOT/ready["source"]["path"]
    assert sha(artifact)==ready["artifact"]["sha256"]
    assert sha(source)==ready["source"]["sha256"]
    ds=read_records(folder/"decisions.bin.gz",DECISION)
    ticks=read_records(folder/"ticks.bin.gz",TICK)
    assert len(ds)>0 and len(ds)==int(ticks["decision"].sum())
    assert len(ds)==state["runtime"]["scheduler"]["decisions"]
    np.testing.assert_array_equal(ticks["frame"][ticks["decision"]!=0],ds["frame"])
    np.testing.assert_array_equal(ticks["action"][ticks["decision"]!=0],ds["action"])
    # Every active held/delivered packet must retain the last actual decision.
    current=np.zeros(8,np.float32)
    for tick in ticks:
        if tick["decision"]:
            current=tick["action"].copy()
        if tick["active"]:
            np.testing.assert_array_equal(tick["action"],current)
    actor=torch.jit.load(str(artifact),map_location="cpu").eval()
    hidden=torch.zeros(1,1,256)
    resets=None
    replay_mismatches=0
    with torch.inference_mode():
        for row in ds:
            if resets!=int(row["resets"]):
                hidden=torch.zeros_like(hidden)
                resets=int(row["resets"])
            action,hidden,logits=actor(torch.from_numpy(row["observation"].copy()[None]),hidden)
            assert torch.isfinite(logits).all() and torch.isfinite(hidden).all()
            replay_mismatches+=int(not np.array_equal(action[0].numpy(),row["action"]))
    assert replay_mismatches==0,"Recorded native policy execution did not reproduce"
    side=int(ready["team"])
    latest_touch={}
    touches=[0,0]
    speed=[];height=[];dist=[];ball_progress=[];ground=[];supersonic=[]
    packet_count=0
    last_packet=None
    for i,p in enumerate(packets(folder/"decision_packets.bin.gz")):
        assert i<len(ds)
        assert int(p.match_info.frame_num)==int(ds[i]["frame"])
        own=[q for q in p.players if int(q.player_id)==int(ready["player_id"])]
        assert len(own)==1 and int(own[0].team)==side
        for player in p.players:
            if player.latest_touch is not None:
                key=(float(player.latest_touch.game_seconds),int(player.latest_touch.ball_index))
                pid=int(player.player_id)
                if latest_touch.get(pid)!=key:
                    touches[int(player.team)]+=1
                    latest_touch[pid]=key
        a=own[0].physics
        b=p.balls[0].physics
        speed.append(float(np.linalg.norm([a.velocity.x,a.velocity.y,a.velocity.z])))
        height.append(float(a.location.z))
        dist.append(float(np.linalg.norm([a.location.x-b.location.x,a.location.y-b.location.y,a.location.z-b.location.z])))
        ball_progress.append(float(b.location.y)*(1 if side==0 else -1))
        ground.append(str(own[0].air_state).split(".")[-1]=="OnGround")
        supersonic.append(bool(own[0].is_supersonic))
        packet_count+=1
        last_packet=p
    assert packet_count==len(ds)
    score=result["native_result"]["score"] if result is not None else state["score"]
    audit=dict(case=case,candidate=ready["candidate"],source=ready["source"],artifact=ready["artifact"],
        completed_match=result is not None and not result["unresolved"],goals_for=score[side],goals_against=score[1-side],
        partial_stop_reason=None if result is not None else partial_state["error"],
        seconds_remaining_at_last_recorded_decision=float(last_packet.match_info.game_time_remaining),
        own_team=side,decisions=len(ds),delivered_packets=len(ticks),
        native_touches_observed=touches[side],nexto_touches_observed=touches[1-side],
        touch_count_scope="Distinct per-player latest_touch timestamps seen in decision packets; lower bound if multiple touches between recorded decisions. Not inferred possession.",
        motion=dict(speed_mean=float(np.mean(speed)),speed_p95=float(np.percentile(speed,95)),
                    speed_below100_fraction=float(np.mean(np.asarray(speed)<100)),
                    grounded_fraction=float(np.mean(ground)),car_height_above300_fraction=float(np.mean(np.asarray(height)>300)),
                    supersonic_fraction=float(np.mean(supersonic)),ball_distance_mean=float(np.mean(dist)),
                    opponent_half_ball_fraction=float(np.mean(np.asarray(ball_progress)>0))),
        actions=dict(throttle_forward=float(np.mean(ds["action"][:,0]>0)),throttle_reverse=float(np.mean(ds["action"][:,0]<0)),
                     jump=float(np.mean(ds["action"][:,5])),boost=float(np.mean(ds["action"][:,6])),handbrake=float(np.mean(ds["action"][:,7]))),
        runtime=state["runtime"],
        exact_recorded_action_replay=True,action_replay_mismatches=replay_mismatches,held_controls_exact=True,
        source_and_export_unchanged=True,optimizer_steps=0,
        observation_domain_exact=False,
        record_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        interpretation="Actual native local-game outcome, not ranked SSL proof. Compare each candidate on both sides; do not equate simulator scoring rates with native games.")
    target=OUT/case
    assert not target.exists(),"Preserve previous reduction"
    target.mkdir()
    for p in folder.iterdir():
        if p.is_file():
            shutil.copyfile(p,target/p.name)
    if partial_state is not None:
        (target/"partial_campaign_state.json").write_text(json.dumps(partial_state,indent=2,sort_keys=True)+"\n")
    (target/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:audit[k] for k in ("case","goals_for","goals_against","native_touches_observed","decisions","motion","actions","exact_recorded_action_replay")}))


if __name__=="__main__":
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument("case")
    p.add_argument("--delivery-only",action="store_true")
    args=p.parse_args()
    (report_delivery if args.delivery_only else report)(args.case)
