"""Bounded read-only trajectory diagnostic; never trains.

Uses the production environment with only CUDA-stream activation replaced by
a CPU device assertion. CPU trajectories are not GPU bitwise-parity evidence
and do not replace the frozen production evaluation or transition criterion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import (
    CHECKPOINTS, RESULTS, check_package, sha, utc, write_json,
)
from rivalsim.fresh_ground_30hz import FreshGroundEnv, SEED, scenarios, scenario_hash, CHECKPOINT_FORMAT
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic, IndependentCriticPolicyConfig
from rivalsim.rival2_unified_policy import deterministic_unified_action
from rivalsim.rival2_contracts import OBS_FIELD_NAMES, POSITION_SCALE


class CPUInspectionEnv(FreshGroundEnv):
    def _activate_torch_stream(self):
        assert self.device.type == "cpu", "This diagnostic must not compete with GPU training"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, default=250)
    parser.add_argument("--decisions", type=int, default=480)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.decisions <= 900:
        raise ValueError("Bounded to at most 30 simulated seconds")
    torch.set_num_threads(2)
    check_package(require_preflight=True)
    path = args.checkpoint or CHECKPOINTS / f"u{args.update:06d}.pt"
    digest = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == CHECKPOINT_FORMAT
    args.update = payload["accepted_updates_total"]
    model = IndependentCriticActorCritic(IndependentCriticPolicyConfig(**payload["policy_config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(args.device).eval().requires_grad_(False)
    bank = scenarios(64, SEED+100, family_only=2)
    started = time.monotonic()
    env_type = CPUInspectionEnv if args.device == "cpu" else FreshGroundEnv
    env = env_type(64, "G:/dev/RLBot-Rival/bot/collision_meshes", device=args.device,
                           seed=SEED+100, ssl_foundation_scenarios=bank)
    rows = torch.arange(64, device=args.device)
    side = torch.tensor(bank.focal_side.astype("int64"), device=args.device)
    hidden = model.initial_hidden(128, device=args.device)
    reset = torch.ones(128, dtype=torch.bool, device=args.device)
    alive = torch.ones(64, dtype=torch.bool, device=args.device)
    history = {k: [] for k in ("active_before", "car_position", "ball_position", "car_velocity",
        "ball_velocity", "car_quaternion", "action", "speed", "forward_speed", "closing_speed",
        "nose_ball_cosine", "ball_distance", "on_ground", "is_flipping", "boost_fraction",
        "touches", "terminated", "truncated", "scoring_team")}
    initial_distance = None
    final_distance = torch.full((64,), float("nan"), device=args.device)
    position_scale = torch.tensor(POSITION_SCALE, device=args.device)
    relative_ball_start = OBS_FIELD_NAMES.index("relative.ball_position.x")
    for tick in range(args.decisions):
        v = env.bridge.views
        cp = v["car_pos"].reshape(64, 2, 3)[rows, side].clone()
        cv = v["car_vel"].reshape(64, 2, 3)[rows, side].clone()
        cq = v["car_quat"].reshape(64, 2, 4)[rows, side].clone()
        bp, bv = v["ball_pos"].clone(), v["ball_vel"].clone()
        delta = bp-cp
        distance = delta.norm(dim=-1)
        unit = delta / distance.clamp_min(1e-6)[:, None]
        forward, _ = env.bridge._basis(cq)
        actor, hidden = model.forward_actor(env.observation.reshape(-1,182), hidden, reset_before=reset)
        action = deterministic_unified_action(actor).reshape(64,2,8)
        record = dict(active_before=alive.clone(), car_position=cp, ball_position=bp,
            car_velocity=cv, ball_velocity=bv, car_quaternion=cq, action=action[rows, side].clone(),
            speed=cv.norm(dim=-1), forward_speed=(cv*forward).sum(-1),
            closing_speed=((cv-bv)*unit).sum(-1), nose_ball_cosine=(forward*unit).sum(-1),
            ball_distance=distance, on_ground=v["on_ground"].reshape(64,2)[rows,side].clone(),
            is_flipping=v["is_flipping"].reshape(64,2)[rows,side].clone(),
            boost_fraction=v["boost"].reshape(64,2)[rows,side].clone()/100.)
        if initial_distance is None:
            initial_distance = distance.clone()
        tr = env.step(action)
        # Final observation is pre-reset; physical backing buffers are already reset.
        final_distance = torch.where(alive,
            (tr.transition_observation[rows,side,relative_ball_start:relative_ball_start+3]
             *position_scale).norm(dim=-1),
            final_distance)
        record.update(touches=env.last_native["touch_count"][rows,side].clone(),
            terminated=tr.terminated.clone(), truncated=tr.truncated.clone(),
            scoring_team=env.last_native["scoring_team"].clone())
        for k,value in record.items():
            history[k].append(value.cpu().numpy())
        alive &= ~tr.reset_mask
        reset = tr.reset_mask[:,None].expand(-1,2).reshape(-1)
        hidden = hidden.masked_fill(reset[None,:,None],0)
        if tick % 90 == 0:
            print(f"{args.device} trajectory {tick/30:.1f}s / {args.decisions/30:.1f}s; {int(alive.sum())} original episodes active",flush=True)
        if not bool(alive.any()):
            break
    arrays = {k: np.stack(value) for k,value in history.items()}
    arrays["focal_side"] = side.cpu().numpy()
    arrays["final_ball_distance"] = final_distance.cpu().numpy()
    valid = arrays["active_before"]
    contact = (arrays["touches"]*valid).sum(0)
    min_distance = np.where(valid, arrays["ball_distance"],np.inf).min(0)
    min_tick = np.where(valid, arrays["ball_distance"],np.inf).argmin(0)
    def summary(mask):
        n = int(mask.sum())
        if n == 0:
            return {"samples":0}
        def mean(x): return float(x[mask].mean())
        return dict(samples=n,mean_speed_uu_s=mean(arrays["speed"]),
            nearly_stationary_fraction_speed_lt100=mean(arrays["speed"]<100),
            backward_fraction_forward_speed_lt_minus100=mean(arrays["forward_speed"] < -100),
            closing_fraction_speed_gt100=mean(arrays["closing_speed"]>100),
            receding_fraction_speed_lt_minus100=mean(arrays["closing_speed"] < -100),
            mean_ball_distance=mean(arrays["ball_distance"]),
            mean_nose_ball_cosine=mean(arrays["nose_ball_cosine"]),
            airborne_fraction=mean(arrays["on_ground"]==0),
            native_flip_active_fraction=mean(arrays["is_flipping"]!=0),
            mean_throttle=mean(arrays["action"][...,0]),
            jump_held_fraction=mean(arrays["action"][...,5]),
            boost_requested_fraction=mean(arrays["action"][...,6]),
            handbrake_held_fraction=mean(arrays["action"][...,7]),
            mean_boost_fraction=mean(arrays["boost_fraction"]))
    bins = {}
    t = np.arange(len(valid))[:,None]/30
    for lo,hi in ((0,1),(1,3),(3,5),(5,10),(10,16)):
        bins[f"{lo}_{hi}_seconds"] = summary(valid & (t>=lo) & (t<hi))
    report = dict(utc=utc(),update=args.update,checkpoint=dict(path=str(path),sha256=digest),
        scenario_sha256=scenario_hash(bank),worlds=64,device=args.device,optimizer_steps=0,
        method="Production FreshGroundEnv; CPU mode overrides stream activation. CUDA mode must run only while the learner is paused. Descriptive trajectory diagnostic, not acceptance authority.",
        thresholds="Speed thresholds +/-100 uu/s are descriptive reporting bins, not rewards or mechanic detectors.",
        decisions=len(valid),maximum_seconds=args.decisions/30,wall_seconds=time.monotonic()-started,
        total_contacts=int(contact.sum()),cases_with_touch=int((contact>0).sum()),
        surviving_original_episodes=int(alive.sum()),whole_trajectory=summary(valid),time_bins=bins,
        per_case=[dict(world=i,side=int(side[i]),initial_distance=float(initial_distance[i]),
            minimum_distance=float(min_distance[i]),minimum_distance_seconds=float(min_tick[i]/30),
            final_distance=float(final_distance[i]),contacts=int(contact[i]),
            summary=summary(valid & (np.arange(64)[None,:]==i))) for i in range(64)])
    assert sha(path)==digest
    if args.device == "cpu":
        assert torch.cuda.memory_allocated()==0
    check_package(require_preflight=True)
    report["checkpoint_unchanged"] = True
    report["frozen_training_sources_unchanged"] = True
    report["torch_cuda_bytes_allocated"] = torch.cuda.memory_allocated()
    directory = args.output_dir or RESULTS / "monitoring"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"trajectory_{args.device.replace(':', '_')}_u{args.update:06d}"
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    report["raw_trace"] = dict(path=str(out.with_suffix(".npz")),sha256=sha(out.with_suffix(".npz")))
    write_json(out.with_suffix(".json"),report)
    print({k:v for k,v in report.items() if k not in ("per_case","time_bins")},flush=True)
    print(bins,flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(RESULTS / "monitoring" / "cpu_trajectory_diagnostic_failure.json", dict(
            utc=utc(),verdict="BLOCKED_DIAGNOSTIC_ONLY",error=str(error),
            exception_type=type(error).__name__,traceback=traceback.format_exc(),
            command=sys.argv,script_sha256=sha(Path(__file__)),
            optimizer_steps=0,torch_cuda_bytes_allocated=torch.cuda.memory_allocated(),
            campaign_source_changes=False,campaign_process_interrupted=False,
            interpretation="No completed trajectory evidence. This separate diagnostic failure is not evidence of a learner failure."))
        raise
