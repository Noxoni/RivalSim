"""Bounded open-loop kickoff motion comparison, no policy or runtime edits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import warp as wp
from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from rivalsim.state import StateSnapshot
from rivalsim.static_world import CompleteWorldSim

OUT=ROOT/"results/rival2/entity_native_gap_v1"
CHANNELS=("throttle","steer","pitch","yaw","roll","jump","boost","handbrake")


def run():
    path=OUT/"native_physical.npz"
    data=np.load(path)
    starts=np.flatnonzero(np.r_[True,data["resets"][1:]!=data["resets"][:-1]])
    # Only complete, uninterrupted first 40 ticks. No rollout after ball contact.
    selected=[]
    for start in starts:
        stop=start+11
        if stop<=len(data["frame"]) and np.all(data["resets"][start:stop]==data["resets"][start]) and np.all(np.diff(data["frame"][start:stop])==4):
            selected.append(start)
    selected=np.asarray(selected)
    assert len(selected)>0
    target=OUT/"ground_motion.json"
    assert not target.exists(),"Preserve previous diagnostic"
    results=[];traces={}
    with gpu_lease(),owned_match_stream():
        for delay in (0,1,2):
            initial=StateSnapshot.empty(len(selected))
            for key in ("car_pos","car_vel","car_ang_vel","car_quat","ball_pos","ball_vel","ball_ang_vel"):
                getattr(initial,key)[:]=data[key][selected]
            initial.boost[:]=data["car_boost"][selected]
            initial.on_ground[:]=data["car_on_ground"][selected]
            for i,key in enumerate(CHANNELS):
                getattr(initial,"prev_"+key)[:]=data["native_input"][selected,:,i]
            world=CompleteWorldSim(len(selected),"G:/dev/RLBot-Rival/bot/collision_meshes",initial=initial,auto_kickoff=False,seed=202609061)
            controls={k:wp.to_torch(getattr(world.controls,k)) for k in CHANNELS}
            position=wp.to_torch(world.state.car_pos).reshape(-1,2,3)
            velocity=wp.to_torch(world.state.car_vel).reshape(-1,2,3)
            positions=[position.cpu().numpy().copy()];velocities=[velocity.cpu().numpy().copy()]
            for tick in range(40):
                rows=selected+max(0,tick-delay)//4
                # Focal actions are the actual recorded issued commands. Before
                # the assumed delay, retain native last_input from the start.
                action=data["native_input"][rows].copy()
                action[:,0]=data["action"][rows] if tick>=delay else data["native_input"][selected,0]
                for i,key in enumerate(CHANNELS):
                    controls[key].copy_(torch.from_numpy(action[...,i].copy()).to(device="cuda",dtype=controls[key].dtype).flatten())
                world.step(1)
                if (tick+1)%4==0:
                    positions.append(position.cpu().numpy().copy())
                    velocities.append(velocity.cpu().numpy().copy())
            predicted=np.stack(positions);predicted_v=np.stack(velocities)
            native=data["car_pos"][selected[None,:]+np.arange(11)[:,None]]
            native_v=data["car_vel"][selected[None,:]+np.arange(11)[:,None]]
            pos_err=np.linalg.norm(predicted[:,:,0]-native[:,:,0],axis=-1)
            vel_err=np.linalg.norm(predicted_v[:,:,0]-native_v[:,:,0],axis=-1)
            assert np.isfinite(predicted).all() and np.isfinite(predicted_v).all()
            results.append(dict(assumed_control_delay_physics_ticks=delay,
                final_position_error_uu_mean=float(pos_err[-1].mean()),final_position_error_uu_max=float(pos_err[-1].max()),
                final_velocity_error_uu_s_mean=float(vel_err[-1].mean()),final_velocity_error_uu_s_max=float(vel_err[-1].max()),
                per_kickoff_final_position_error=pos_err[-1].tolist(),per_kickoff_final_velocity_error=vel_err[-1].tolist()))
            traces[f"delay{delay}_car_pos"]=predicted
            traces[f"delay{delay}_car_vel"]=predicted_v
            del world
    traces.update(native_car_pos=native,native_car_vel=native_v,start_frames=data["frame"][selected])
    np.savez_compressed(OUT/"ground_motion_traces.npz",**traces)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest().upper()
    report=dict(version="RIVAL2_NATIVE_GROUND_MOTION_DIAGNOSTIC_V1",kickoffs=len(selected),physics_ticks=40,
        compared_car="Blue Rival only; no car-ball or car-car encounter in this short initial segment",
        source_sha256=sha(path),script_sha256=sha(Path(__file__)),trace_sha256=sha(OUT/"ground_motion_traces.npz"),results=results,
        optimizer_steps=0,production_changes=False,
        qualification="Closest-state replay, not exact Rocket League internal-state restoration. Native pose/velocities/boost/aggregate ground state and observed prior inputs copied. Unexposed solver caches and wheel history use default simulator initialization. Opponent controls are only sampled endpoints and are not a Nexto fidelity measurement. Three fixed delay hypotheses bound command-onset uncertainty; none is installed or selected as a production timing correction. Short ground motion does not establish collision/aerial/general physics parity.")
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps(dict(kickoffs=len(selected),results=[{k:v for k,v in r.items() if not k.startswith("per_")} for r in results])))


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
