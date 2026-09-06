"""Coordinate/player mapping fixtures derived from captured simulator states.

Not native Rocket League telemetry: only fields representable in a packet are
round-tripped. Internal timers/wheels are neither seeded nor declared exact.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/entity_native_packet_v1"
sys.path.insert(0,str(ROOT/"deployment/entity_native_v1"))
from packet_observation import Rival2LiveAdapter


def vec(v):
    return NS(x=float(v[0]),y=float(v[1]),z=float(v[2]))


def packet_from_observation(obs, side, spec, frame):
    sign=np.array((1.,1.,1.) if side==0 else (-1.,-1.,1.),np.float32)
    pos=np.asarray(spec["position_scale"],np.float32)
    players=[]
    for own,start in ((True,9),(False,48)):
        car=obs[start:start+39]
        fwd,up=car[6:9]*sign,car[9:12]*sign
        pitch=float(np.arctan2(fwd[2],np.hypot(fwd[0],fwd[1])))
        yaw=float(np.arctan2(fwd[1],fwd[0]))
        # Dot against the zero-roll up/right basis, not up.z/cos(pitch),
        # which is ill-conditioned near a vertical captured car orientation.
        zero_up=np.asarray((-np.cos(yaw)*np.sin(pitch),-np.sin(yaw)*np.sin(pitch),np.cos(pitch)))
        right=np.asarray((-np.sin(yaw),np.cos(yaw),0.))
        roll=float(np.arctan2(np.dot(up,right),np.dot(up,zero_up)))
        phase="OnGround" if car[16]>0 else "Jumping" if car[18]>0 else "Dodging" if car[21]>0 else "InAir"
        players.append(NS(team=side if own else 1-side, player_id=17 if own else 29,
            physics=NS(location=vec(car[:3]*pos*sign),velocity=vec(car[3:6]*2300*sign),
                angular_velocity=vec(car[12:15]*6*sign),rotation=NS(pitch=pitch,yaw=yaw,roll=roll)),
            boost=float(car[15]*100), air_state="AirState."+phase, has_jumped=bool(car[17]),
            has_double_jumped=bool(car[19]),has_dodged=bool(car[20]),
            dodge_elapsed=float(car[33]*.95), demolished_timeout=float(car[25]*3) if car[24]>0 else -1.,
            is_supersonic=bool(car[36]),latest_touch=None,
            last_input=NS(throttle=0,steer=0,pitch=0,yaw=0,roll=0,jump=False,boost=False,handbrake=False)))
    # Reverse actual array order on half the frames; team/player IDs determine mapping.
    if frame%8:
        players.reverse()
    pads=[None]*34
    remap=list(range(34)) if side==0 else spec["orange_pad_remap"]
    for canonical,physical in enumerate(remap):
        active=bool(obs[99+2*canonical])
        duration=spec["canonical_boost_pad_durations"][physical]
        pads[physical]=NS(is_active=active,timer=0. if active else duration*(1-float(obs[100+2*canonical])))
    return NS(players=players,teams=[NS(score=0),NS(score=0)],boost_pads=pads,
        balls=[NS(physics=NS(location=vec(obs[:3]*pos*sign),velocity=vec(obs[3:6]*6000*sign),angular_velocity=vec(obs[6:9]*6*sign)))],
        match_info=NS(frame_num=frame,seconds_elapsed=frame/120.,match_phase="MatchPhase.Active"))


def run():
    reports=[]
    for label in ("reference600","finishing650"):
        m=json.loads((OUT/f"{label}_manifest.json").read_text())
        fields=m["fields"]
        shared=np.asarray([f["classification"] in ("direct","derived") and not f["field"].startswith("previous_action.") for f in fields])
        field=NS(boost_pads=[NS(location=vec(p)) for p in m["observation"]["canonical_boost_pad_positions"]])
        a=Rival2LiveAdapter(m,field)
        with np.load(ROOT/f"results/rival2/entity_native_bridge_v1/{label}_native_sequences.npz") as z:
            observations=z["observation"]; sides=z["rival_side"]; frames=z["frame"]
        maxima=np.zeros(182)
        for row, frame in zip(observations,frames):
            for obs, side in zip(row,sides):
                p=packet_from_observation(obs,int(side),m["observation"],int(frame))
                a.reset(p)
                actual=a.observation(p)[0,int(side)]
                maxima=np.maximum(maxima,np.abs(actual-obs))
        assert maxima[shared].max()<2e-5, [(fields[i]["field"],maxima[i]) for i in np.flatnonzero(shared & (maxima>2e-5))]
        reports.append(dict(candidate=label,states=6000,shared_fields=int(shared.sum()),
            shared_max_abs=float(maxima[shared].max()),
            per_field=[dict(**f,max_abs=float(maxima[i]),round_trip_asserted=bool(shared[i])) for i,f in enumerate(fields)]))
    result=dict(result=reports, live_packet_parity=False, optimizer_steps=0,
        scope="Packet-shaped fixtures built from actual captured simulator physical states. Direct/derived geometry, kinematics, flags, pad mapping tested across both sides and shuffled player order. No timers/wheels injected as if observed. Report all differences, assert only representable fields. Previous action is unseeded in this stateless mapping test; sequential storage checked separately.")
    path=OUT/"captured_state_mapping.json"
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k!="per_field"} for r in reports]))


if __name__=="__main__":
    run()
