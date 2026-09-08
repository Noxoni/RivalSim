"""Fresh random sustained-gameplay campaign. STOP is honored at accepted boundaries.

No imports from a past campaign's authority/selection function. Old checkpoints
and stopped jobs remain untouched. Main campaign requires pushed preflight.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json,append_json
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from benchmarks.evaluate_sustained_gameplay_v1 import specification as evaluation_spec
from rivalsim.direct_skills_kickoff_race_v1 import scenarios,curriculum_authority
from rivalsim.fresh_ground_30hz import scenario_hash,content_hash
from rivalsim.sustained_gameplay_v1 import VERSION,POLICY_VERSION,SEED,SustainedEnv,fresh_model,ppo_config,reward_authority
from rivalsim.sustained_gameplay_training_v1 import SustainedCollector
from rivalsim.ssl_entity_training import fresh_entity_optimizer,joint_sequence_loss,finite_model_and_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data,mixed_joint_ppo_update

OUT=ROOT/"results/rival2/sustained_gameplay_v1"
CKPTS=ROOT/"checkpoints/rival2/sustained_gameplay_v1"
EXTERNAL=Path("G:/dev/RivalSim-runs/sustained-gameplay-v1")
COLLISION="G:/dev/RLBot-Rival/bot/collision_meshes"
INITIAL=CKPTS/"initialized.pt"
SOURCES=(
    "rivalsim/sustained_gameplay_v1.py","rivalsim/sustained_gameplay_training_v1.py",
    "benchmarks/run_sustained_gameplay_v1.py","benchmarks/evaluate_sustained_gameplay_v1.py",
    "tests/test_sustained_gameplay_v1.py","rivalsim/ssl_entity_mixed_training.py",
    "rivalsim/ssl_entity_training.py","rivalsim/ssl_entity_policy.py","rivalsim/ssl_joint_control_policy.py",
    "rivalsim/recurrent_execution.py","rivalsim/rival2_recurrent_ppo.py","rivalsim/rival2_ppo.py",
    "rivalsim/rival2_unified_policy.py","rivalsim/rival2_independent_critic.py",
    "rivalsim/rival2_env.py","rivalsim/kernels/rival2.py","rivalsim/rival2_contracts.py",
    "rivalsim/fresh_ground_30hz.py","rivalsim/ssl_foundation_v1.py",
    "rivalsim/direct_skills_v1.py","rivalsim/direct_skills_shooting_curriculum_v1.py",
    "rivalsim/direct_skills_kickoff_race_v1.py","third_party/nexto/native_v5.py","third_party/nexto/adapter.py",
    "benchmarks/evaluate_rival2_ssl_entity_full_match.py","benchmarks/direct_skills_eval_stream.py",
    "benchmarks/report_rival2_ssl_entity_match_followup.py","rivalsim/full_match.py",
)


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest().upper()


def authority():
    return dict(version=VERSION,policy_version=POLICY_VERSION,worlds=32768,seed=SEED,
        parent=None,initialization="New random actor, entities, actor GRU and independent critic MLP/GRU; fresh Adam. No prior policy or optimizer loaded.",
        physics_hz=120,policy_hz=30,hold_ticks=4,ppo=asdict(ppo_config()),
        critic="Independent 182->512->512->512->256 SiLU features -> GRU256 -> scalar value; no value gradient into actor. Packed hidden banks actor/critic, each1xBx256.",
        recurrent_training="Complete90-decision sequences; carry both hidden states between rollouts, detach only BPTT boundary; reset only ended rows. Successor bootstrap uses critic history AFTER current observation, without advancing stored state twice.",
        exploration="Categorical joint90 with temperature1, entropy0.001. No inherited temperature2 uniform-barrier intervention intended for saturated old weights.",
        reward=reward_authority(),scenario_bank=curriculum_authority(),scenario_seed=2026090601,
        scenario_semantics="Exactly preserve previous 32768-state start bank including challenge/finishing/defense/standing and momentum-assisted kickoff starts. Replace ALL task rewards/short timeouts with the same sustained-gameplay reward and goal/45s-contact-inactivity lifecycle. No scenario ID in policy input.",
        opponents=dict(nexto_world_fraction=.5,selfplay_world_fraction=.5,
            nexto_learner_sample_fraction=1/3,native_nexto_seed=SEED+3,mode="native_v5",
            frozen_old_rival=0,current_selfplay_both_sides_trainable=True),
        evaluation=evaluation_spec(),evaluation_updates=[0,10,20,50],evaluation_every_after_50=50,
        checkpoints="Initialized0, first1,10,20 and every50 permanent. Alternating atomic rolling each accepted update.",
        budget="Continue until user STOP; no automatic reconfiguration or silent numerical retry. Evaluation failures pause. Numerical/corruption faults roll back and stop; KL is telemetry only.",
        resume="Only this lineage with exact latest checkpoint hash. Model/Adam/all learner and Nexto RNG preserved. Physical episodes restart explicitly and both memories clear on process resume; unfinished episodes are NOT assigned synthetic rewards or appended to GAE. In-process evaluations/checkpoint saves do not reset training worlds or memories.")


def atomic_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".pt.tmp")
    with tmp.open("wb") as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(path=str(path),sha256=sha(path),accepted_updates=payload["accepted_updates"])


def preflight(worlds):
    if (EXTERNAL/"latest.json").exists():raise RuntimeError("Campaign exists; no reinitialization")
    started=time.perf_counter()
    model=fresh_model().cuda()
    optimizer=fresh_entity_optimizer(model)
    before=tensor_hash(model.state_dict())
    if not INITIAL.exists():
        atomic_checkpoint(INITIAL,dict(format=VERSION+"_CHECKPOINT",policy_version=POLICY_VERSION,
            parent=None,seed=SEED,accepted_updates=0,model=model.state_dict(),model_config=asdict(model.config),
            authority_sha256=content_hash(authority()),optimizer=optimizer.state_dict()))
    else:
        original=torch.load(INITIAL,map_location="cpu",weights_only=False)
        assert original["accepted_updates"]==0 and original["parent"] is None
        assert original["authority_sha256"]==content_hash(authority())
        assert before==tensor_hash(original["model"])
    env=SustainedEnv(worlds,COLLISION,device="cuda:0",seed=SEED,ssl_foundation_scenarios=scenarios(worlds))
    collector=SustainedCollector(env,model)
    torch.cuda.reset_peak_memory_stats()
    rollout=collector.collect()
    data=mixed_sequence_data(rollout,ppo_config())
    ix=data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train();loss,_=joint_sequence_loss(model,data,ix,ppo_config());loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.critic(data["observations"][ix],data["initial_hidden"][1:,ix],
                 data["reset_before"][ix])[0].square().mean().backward()
    isolated=all(p.grad is None for n,p in model.named_parameters() if not n.startswith("critic."))
    optimizer.zero_grad(set_to_none=True)
    checks=dict(model_unchanged=before==tensor_hash(model.state_dict()),fresh_optimizer_no_steps=not optimizer.state,
        finite_loss_gradient=bool(torch.isfinite(loss)&torch.isfinite(norm)),critic_gradient_isolated=isolated,
        recurrent_actor_and_critic=collector.hidden.shape==(2,worlds,2,256),
        nexto_learner_mask_exact=collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"],
        action_targets_exact=torch.equal(model.action_table[rollout.action_indices][rollout.train_mask],rollout.actions[rollout.train_mask]),
        no_mechanics_hotpath=env.world.gameplay_v3 is None and env.world.gameplay_120 is None,
        no_direct_task_reward_state=not hasattr(env,"events") and not hasattr(env,"race"),
        finite_model_optimizer=finite_model_and_optimizer(model,optimizer))
    assert all(checks.values()),checks
    report=dict(utc=utc(),worlds=worlds,decisions=90,optimizer_steps=0,checks=checks,
        peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),rollout_bytes=rollout.logical_bytes,
        seconds=time.perf_counter()-started,model_sha256=before,initialized=sha(INITIAL),
        gradient_norm=float(norm),runtime_contract_hashes=env.contract_hashes,training=collector.last_metrics)
    write_json(OUT/f"preflight_{worlds}.json",report)
    print("PREFLIGHT "+json.dumps(report),flush=True)


def freeze():
    if (EXTERNAL/"campaign_state.json").exists():raise RuntimeError("Campaign already entered; freeze is closed")
    if (OUT/"authority.json").exists():
        assert json.loads((OUT/"authority.json").read_text())==authority(), "Cannot retune frozen authority"
    pre=json.loads((OUT/"preflight_32768.json").read_text())
    assert all(pre["checks"].values()) and pre["optimizer_steps"]==0
    suites=ET.parse(OUT/"tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    spec=authority();write_json(OUT/"authority.json",spec)
    write_json(OUT/"package.json",dict(authority_sha256=content_hash(spec),
        initialized_checkpoint_sha256=sha(INITIAL),scenario_sha256=scenario_hash(scenarios(32768)),
        sources={p:text_sha(ROOT/p) for p in SOURCES},
        evidence={name:sha(OUT/name) for name in ("tests.xml","preflight_32768.json")}))


def verify():
    package=json.loads((OUT/"package.json").read_text())
    assert json.loads((OUT/"authority.json").read_text())==authority()
    assert package["authority_sha256"]==content_hash(authority())
    assert sha(INITIAL)==package["initialized_checkpoint_sha256"]
    for p,h in package["sources"].items():assert text_sha(ROOT/p)==h,p
    for p,h in package["evidence"].items():assert sha(OUT/p)==h,p
    paths=(*SOURCES,INITIAL.relative_to(ROOT).as_posix(),
        *("results/rival2/sustained_gameplay_v1/"+n for n in ("package.json","authority.json",*package["evidence"])))
    for p in paths:
        remote=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        local=(ROOT/p).read_bytes()
        assert (remote==local if p.endswith(".pt") else remote.replace(b"\r\n",b"\n")==local.replace(b"\r\n",b"\n")),p
    return package


def run(args):
    package=verify();spec=authority()
    EXTERNAL.mkdir(parents=True,exist_ok=True)
    if (EXTERNAL/"STOP").exists():raise RuntimeError("Respect STOP")
    latest_path=EXTERNAL/"latest.json"
    source=None
    if args.resume:
        latest=json.loads(latest_path.read_text())
        if str(Path(args.resume).resolve())!=str(Path(latest["path"]).resolve()):raise RuntimeError("Only latest accepted resume")
        if not args.resume_sha256 or sha(args.resume)!=args.resume_sha256.upper() or latest["sha256"]!=args.resume_sha256.upper():raise RuntimeError("Resume hash mismatch")
        source=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert source["format"]==VERSION+"_CHECKPOINT" and source["authority_sha256"]==content_hash(spec)
        assert source["parent"] is None and source["policy_version"]==POLICY_VERSION
    elif latest_path.exists() or (EXTERNAL/"campaign_state.json").exists():
        raise RuntimeError("Existing campaign requires explicit resume")
    model=fresh_model().cuda();optimizer=fresh_entity_optimizer(model)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    assert tensor_hash(model.state_dict())==tensor_hash(initial["model"])
    del initial
    step=samples=ticks=optimizer_steps=0
    if source is not None:
        model.load_state_dict(source["model"],strict=True);optimizer.load_state_dict(source["optimizer"])
        step=source["accepted_updates"];samples=source["learner_decisions"];ticks=source["world_physics_ticks"]
        optimizer_steps=source["optimizer_steps"]
    bank=scenarios(32768)
    assert scenario_hash(bank)==package["scenario_sha256"]
    env=SustainedEnv(32768,COLLISION,device="cuda:0",seed=SEED,ssl_foundation_scenarios=bank)
    collector=SustainedCollector(env,model)
    shuffle=torch.Generator(device=env.device).manual_seed(SEED+2)
    if source is not None:
        collector.nexto.load_checkpoint_state(source["opponent_state"]["native_nexto"])
        collector.nexto.activate(torch.ones_like(collector.is_nexto));collector.assign(torch.ones_like(collector.is_nexto))
        collector.opponent_generator.set_state(source["opponent_state"]["generator_state"].cpu())
        collector.generator.set_state(source["policy_rng"].cpu());shuffle.set_state(source["shuffle_rng"].cpu())
        torch.set_rng_state(source["cpu_rng"].cpu());torch.cuda.set_rng_state(source["cuda_rng"].cpu())
    latest=None
    def state(status,**extra):
        write_json(EXTERNAL/"campaign_state.json",dict(utc=utc(),pid=os.getpid(),status=status,
            accepted_updates=step,learner_decisions=samples,world_physics_ticks=ticks,
            optimizer_steps=optimizer_steps,latest_checkpoint=latest,authority_sha256=content_hash(spec),**extra))
    def save(path):
        payload=dict(format=VERSION+"_CHECKPOINT",policy_version=POLICY_VERSION,seed=SEED,parent=None,
            initialized_checkpoint_sha256=sha(INITIAL),authority_sha256=content_hash(spec),package=package,
            model=model.state_dict(),model_config=asdict(model.config),optimizer=optimizer.state_dict(),
            accepted_updates=step,learner_decisions=samples,world_physics_ticks=ticks,
            learner_physics_exposure=samples*4,optimizer_steps=optimizer_steps,
            policy_rng=collector.generator.get_state(),shuffle_rng=shuffle.get_state(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(),runtime_contract_hashes=env.contract_hashes,
            recurrent_state_audit=dict(shape=list(collector.hidden.shape),
                sha256=tensor_hash({"hidden":collector.hidden}),
                semantics="Physical worlds are not serialized; no hidden replay on process resume"),
            last_metrics=collector.last_metrics,resume_semantics=spec["resume"])
        return atomic_checkpoint(path,payload)
    def evaluate_boundary():
        snapshot=CKPTS/f"plus_{step:06d}.pt"
        receipt=save(snapshot)
        output=OUT/f"eval_{step:06d}.json"
        state("evaluating",evaluation_checkpoint=receipt)
        write_json(EXTERNAL/"evaluation_request.json",dict(accepted_updates=step,checkpoint=receipt,
                   output=str(output)))
        # Separate process keeps evaluation RNG, streams and allocator independent.
        result=subprocess.run([sys.executable,"benchmarks/evaluate_sustained_gameplay_v1.py",
            "--checkpoint",str(snapshot),"--sha256",receipt["sha256"],"--output",str(output)],cwd=ROOT)
        if result.returncode:raise RuntimeError("Evaluation failure: inspect output before resume")
        report=json.loads(output.read_text());assert report["checkpoint"]["sha256"]==receipt["sha256"]
        write_json(OUT/"latest_evaluation.json",dict(accepted_updates=step,checkpoint=receipt,
            evaluation_file=output.relative_to(ROOT).as_posix(),evaluation_sha256=sha(output),summary=report["summary"]))
    try:
        latest=save(EXTERNAL/f"rolling_{step%2}.pt");write_json(latest_path,latest)
        state("initialized" if not step else "resumed_fresh_physical_episodes")
        if step in (0,10,20) or step%50==0:evaluate_boundary()
        while not (EXTERNAL/"STOP").exists():
            state("collecting")
            start=time.perf_counter();rollout=collector.collect();collected=time.perf_counter()
            state("optimizing",proposed_update=step+1)
            report=mixed_joint_ppo_update(model,optimizer,rollout,ppo_config(),shuffle)
            step+=1;samples+=collector.last_metrics["trainable_agent_samples"]
            ticks+=collector.last_metrics["physical_physics_ticks"];optimizer_steps+=report["optimizer_steps"]
            row=dict(utc=utc(),accepted_updates=step,learner_decisions=samples,world_physics_ticks=ticks,
                optimizer_steps=optimizer_steps,collect_seconds=collected-start,
                optimize_seconds=time.perf_counter()-collected,ppo=report,training=collector.last_metrics)
            del rollout;gc.collect()
            latest=save(EXTERNAL/f"rolling_{step%2}.pt");write_json(latest_path,latest)
            append_json(EXTERNAL/"training_curve.jsonl",row)
            write_json(EXTERNAL/"progress.json",row);state("accepted")
            print("ACCEPTED "+json.dumps(dict(update=step,collect_seconds=row["collect_seconds"],
                optimize_seconds=row["optimize_seconds"],mean_kl=report["completed_update_mean_kl"],
                goals=collector.last_metrics["physical_goals"],touches_min=collector.last_metrics["touches_per_minute"])),flush=True)
            if step==1:save(CKPTS/"plus_000001.pt")
            if step in (10,20) or step%50==0:
                if (EXTERNAL/"STOP").exists():break
                torch.cuda.empty_cache();evaluate_boundary()
        state("stopped_at_accepted_boundary")
    except BaseException as exc:
        failure=dict(utc=utc(),accepted_updates=step,exception=repr(exc),traceback=traceback.format_exc(),
            latest_checkpoint=latest,automatic_retry=False)
        write_json(EXTERNAL/"failure.json",failure);state("failure",failure=repr(exc))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("preflight","freeze","verify","run"))
    parser.add_argument("--worlds",type=int,default=32768)
    parser.add_argument("--resume");parser.add_argument("--resume-sha256")
    args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=="freeze":freeze()
    elif args.mode=="verify":print(json.dumps(verify()))
    else:
        with gpu_lease():
            if args.mode=="preflight":preflight(args.worlds)
            else:run(args)
