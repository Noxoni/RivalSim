"""Same sustained policy/Adam at +27; temporary acquisition reset-bank amendment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
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
from benchmarks.run_sustained_gameplay_v1 import (
    SOURCES as OLD_SOURCES, INITIAL, COLLISION, atomic_checkpoint, text_sha,
    authority as original_authority,
)
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json,append_json
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from rivalsim.sustained_acquisition_v1 import (
    VERSION, AcquisitionEnv, AcquisitionCollector, Retirement, specification,
    training_starts, acquisition_starts, SEED as CURRICULUM_SEED,
)
from rivalsim.sustained_gameplay_v1 import SustainedPolicy,POLICY_VERSION,VERSION as REWARD_VERSION,SEED,ppo_config
from rivalsim.fresh_ground_30hz import content_hash,scenario_hash
from rivalsim.ssl_entity_training import fresh_entity_optimizer,joint_sequence_loss,finite_model_and_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data,mixed_joint_ppo_update

OUT=ROOT/"results/rival2/sustained_acquisition_v1"
CKPTS=ROOT/"checkpoints/rival2/sustained_acquisition_v1"
EXTERNAL=Path("G:/dev/RivalSim-runs/sustained-acquisition-v1")
PARENT=ROOT/"checkpoints/rival2/sustained_gameplay_v1/plus_000027.pt"
PARENT_SHA="C2532D3D347A13C23919D52CA5A1E30409BBA0E5102577CB5DE2FDC9888D8DAF"
SOURCES=(*OLD_SOURCES,"rivalsim/sustained_acquisition_v1.py",
    "benchmarks/run_sustained_acquisition_v1.py","benchmarks/evaluate_sustained_acquisition_v1.py",
    "benchmarks/report_sustained_acquisition_v1.py",
    "tests/test_sustained_acquisition_v1.py")


def parent_identity():
    return dict(path=PARENT.relative_to(ROOT).as_posix(),sha256=PARENT_SHA,accepted_updates=27)


def authority():
    return dict(version=VERSION,parent=parent_identity(),
        original_authority_sha256=content_hash(original_authority()),
        original_authority_path="results/rival2/sustained_gameplay_v1/authority.json",
        original_initialization_sha256=sha(INITIAL),
        continuation="Exact accepted+27 model, Adam and optimizer counters, learner/Nexto RNG. No new random lineage. Fresh physical episodes and cleared recurrent memory on process entry, as already documented for parent resume.",
        unchanged="All parent reward/episode/observation/action/architecture/PPO/exploration and opponent semantics remain unchanged. Only reset-bank curriculum and its read-only contact diagnostics change.",
        curriculum=specification(),worlds=32768,ppo=asdict(ppo_config()),
        checks="Acquisition probe at entry27 then every10 additional updates while active. After retirement every50 total updates. Full original10-match Nexto check every50 total and whenever two acquisition probes pass and removal is considered. No gameplay claims from probes alone.",
        checkpoints="Atomic alternating rolling every accepted update, permanent at entry, first amendment update, all diagnostic boundaries and every50 total. Preserve retirement state, original counters, model and Adam.",
        budget="Continue until user STOP, no automatic reward/hyperparameter changes. KL telemetry only; nonfinite/corruption still stops at last accepted checkpoint.")


def nested_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def load_parent():
    assert sha(PARENT)==PARENT_SHA
    source=torch.load(PARENT,map_location="cpu",weights_only=False)
    assert source["accepted_updates"]==27 and source["parent"] is None
    assert source["policy_version"]==POLICY_VERSION
    assert source["authority_sha256"]==content_hash(original_authority())
    return source


def restore_model(source):
    model=SustainedPolicy().cuda()
    model.load_state_dict(source["model"],strict=True)
    optimizer=fresh_entity_optimizer(model)
    optimizer.load_state_dict(source["optimizer"])
    assert nested_equal(model.state_dict(),source["model"])
    assert nested_equal(optimizer.state_dict(),source["optimizer"])
    assert finite_model_and_optimizer(model,optimizer)
    return model,optimizer


def preflight(worlds):
    assert not (EXTERNAL/"campaign_state.json").exists()
    source=load_parent();model,optimizer=restore_model(source)
    env=AcquisitionEnv(worlds,COLLISION,device="cuda:0",seed=SEED,ssl_foundation_scenarios=training_starts(worlds))
    collector=AcquisitionCollector(env,model)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    rollout=collector.collect();data=mixed_sequence_data(rollout,ppo_config())
    ix=data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()
    loss,_=joint_sequence_loss(model,data,ix,ppo_config());loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    checks=dict(model_preserved=nested_equal(model.state_dict(),source["model"]),
        adam_moments_and_counters_preserved=nested_equal(optimizer.state_dict(),source["optimizer"]),
        finite=finite_model_and_optimizer(model,optimizer) and bool(torch.isfinite(loss)&torch.isfinite(norm)),
        exact_parent=sha(PARENT)==PARENT_SHA,
        reward_and_contracts_unchanged=env.contract_hashes==source["runtime_contract_hashes"],
        actor_critic_memory=collector.hidden.shape==(2,worlds,2,256),
        acquisition_exposed=any(v["world_seconds"]>0 for k,v in collector.last_metrics["by_start_family_and_opponent"].items() if k.startswith("ball_acquisition")),
        nexto_mask=collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"],
        no_direct_task_or_mechanics_state=not hasattr(env,"events") and not hasattr(env,"race") and env.world.gameplay_v3 is None,
        exact_actions=torch.equal(model.action_table[rollout.action_indices][rollout.train_mask],rollout.actions[rollout.train_mask]))
    assert all(checks.values()),checks
    result=dict(utc=utc(),worlds=worlds,optimizer_steps=0,parent=parent_identity(),checks=checks,
        elapsed_seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm),
        rollout_bytes=rollout.logical_bytes,training=collector.last_metrics)
    write_json(OUT/f"preflight_{worlds}.json",result)
    print("PREFLIGHT "+json.dumps(result),flush=True)


def freeze():
    assert not (EXTERNAL/"campaign_state.json").exists()
    spec=authority()
    if (OUT/"authority.json").exists():assert json.loads((OUT/"authority.json").read_text())==spec
    pre=json.loads((OUT/"preflight_32768.json").read_text())
    assert all(pre["checks"].values()) and pre["optimizer_steps"]==0
    suites=ET.parse(OUT/"tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    probes=[json.loads((OUT/f"probe_preflight_{k}.json").read_text()) for k in ("a","b")]
    assert {k:v for k,v in probes[0].items() if k!="utc"}=={k:v for k,v in probes[1].items() if k!="utc"}
    assert probes[0]["checkpoint_sha256"]==PARENT_SHA and probes[0]["optimizer_steps"]==0
    write_json(OUT/"probe_determinism.json",dict(checks=dict(all_first_contact_ticks_exact=True,
        summary_exact=True,parent_unchanged=True,optimizer_steps_zero=True),
        a_sha256=sha(OUT/"probe_preflight_a.json"),b_sha256=sha(OUT/"probe_preflight_b.json")))
    write_json(OUT/"authority.json",spec)
    write_json(OUT/"package.json",dict(authority_sha256=content_hash(spec),parent=parent_identity(),
        sources={p:text_sha(ROOT/p) for p in SOURCES},
        scenario_sha256={"active":scenario_hash(training_starts(32768)),"retired":scenario_hash(training_starts(32768,True)),
            "probe":scenario_hash(acquisition_starts(1024,CURRICULUM_SEED+100))},
        evidence={p:sha(OUT/p) for p in ("preflight_32768.json","tests.xml","probe_determinism.json",
            "probe_preflight_a.json","probe_preflight_b.json")}))


def verify():
    package=json.loads((OUT/"package.json").read_text())
    assert package["authority_sha256"]==content_hash(authority())
    assert json.loads((OUT/"authority.json").read_text())==authority()
    assert package["parent"]==parent_identity() and sha(PARENT)==PARENT_SHA
    for p,h in package["sources"].items():assert text_sha(ROOT/p)==h,p
    for p,h in package["evidence"].items():assert sha(OUT/p)==h,p
    prefix=OUT.relative_to(ROOT).as_posix()+"/"
    paths=(*SOURCES,parent_identity()["path"],*(prefix+p for p in ("authority.json","package.json",*package["evidence"])))
    for p in paths:
        remote=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        local=(ROOT/p).read_bytes()
        assert (remote==local if p.endswith(".pt") else remote.replace(b"\r\n",b"\n")==local.replace(b"\r\n",b"\n")),p
    return package


def run(args):
    package=verify();spec=authority();EXTERNAL.mkdir(parents=True,exist_ok=True)
    if (EXTERNAL/"STOP").exists():raise RuntimeError("Respect STOP")
    if args.resume:
        latest=json.loads((EXTERNAL/"latest.json").read_text())
        assert Path(args.resume).resolve()==Path(latest["path"]).resolve()
        assert args.resume_sha256 and latest["sha256"]==args.resume_sha256.upper()==sha(args.resume)
        source=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert source["format"]==VERSION+"_CHECKPOINT" and source["authority_sha256"]==content_hash(spec)
        assert source["parent"]==parent_identity()
        retirement=Retirement(**source["retirement"])
    else:
        if (EXTERNAL/"campaign_state.json").exists():raise RuntimeError("Existing run requires explicit resume")
        source=load_parent();retirement=Retirement()
    model,optimizer=restore_model(source)
    bank=training_starts(32768,retirement.retired)
    assert scenario_hash(bank)==package["scenario_sha256"]["retired" if retirement.retired else "active"]
    env=AcquisitionEnv(32768,COLLISION,device="cuda:0",seed=SEED,ssl_foundation_scenarios=bank)
    assert env.contract_hashes==source["runtime_contract_hashes"]
    collector=AcquisitionCollector(env,model)
    collector.nexto.load_checkpoint_state(source["opponent_state"]["native_nexto"])
    collector.nexto.activate(torch.ones_like(collector.is_nexto));collector.assign(torch.ones_like(collector.is_nexto))
    collector.opponent_generator.set_state(source["opponent_state"]["generator_state"].cpu())
    collector.generator.set_state(source["policy_rng"].cpu())
    shuffle=torch.Generator(device=env.device);shuffle.set_state(source["shuffle_rng"].cpu())
    torch.set_rng_state(source["cpu_rng"].cpu());torch.cuda.set_rng_state(source["cuda_rng"].cpu())
    step=source["accepted_updates"];samples=source["learner_decisions"];ticks=source["world_physics_ticks"]
    optimizer_steps=source["optimizer_steps"];latest=None
    entry=dict(utc=utc(),source_sha256=sha(args.resume) if args.resume else PARENT_SHA,
        accepted_updates=step,model_exact=nested_equal(model.state_dict(),source["model"]),
        adam_exact=nested_equal(optimizer.state_dict(),source["optimizer"]),
        policy_rng_exact=torch.equal(collector.generator.get_state().cpu(),source["policy_rng"].cpu()),
        shuffle_rng_exact=torch.equal(shuffle.get_state().cpu(),source["shuffle_rng"].cpu()),
        cpu_rng_exact=torch.equal(torch.get_rng_state().cpu(),source["cpu_rng"].cpu()),
        cuda_rng_exact=torch.equal(torch.cuda.get_rng_state().cpu(),source["cuda_rng"].cpu()),
        fresh_physical_episodes=True,both_memories_zero=not bool(collector.hidden.any()),retirement=asdict(retirement))
    assert all(entry[k] for k in ("model_exact","adam_exact","policy_rng_exact","shuffle_rng_exact","cpu_rng_exact","cuda_rng_exact","both_memories_zero"))
    write_json(EXTERNAL/f"entry_{step:06d}_{time.time_ns()}.json",entry)
    del source;gc.collect()
    def state(status,**extra):
        write_json(EXTERNAL/"campaign_state.json",dict(utc=utc(),pid=os.getpid(),status=status,
            accepted_updates=step,additional_updates=step-27,learner_decisions=samples,
            optimizer_steps=optimizer_steps,latest_checkpoint=latest,retirement=asdict(retirement),**extra))
    def save(path):
        return atomic_checkpoint(path,dict(format=VERSION+"_CHECKPOINT",policy_version=POLICY_VERSION,
            parent=parent_identity(),initialized_checkpoint_sha256=sha(INITIAL),seed=SEED,
            authority_sha256=content_hash(spec),package=package,model=model.state_dict(),model_config=asdict(model.config),
            optimizer=optimizer.state_dict(),accepted_updates=step,additional_updates=step-27,
            learner_decisions=samples,world_physics_ticks=ticks,learner_physics_exposure=samples*4,optimizer_steps=optimizer_steps,
            policy_rng=collector.generator.get_state(),shuffle_rng=shuffle.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),opponent_state=collector.opponent_checkpoint_state(),
            runtime_contract_hashes=env.contract_hashes,retirement=asdict(retirement),last_metrics=collector.last_metrics,
            recurrent_state_audit=dict(shape=list(collector.hidden.shape),sha256=tensor_hash({"hidden":collector.hidden}),
                semantics="Fresh physics and cleared memories on process resume; carry in-process")))
    def rolling():
        nonlocal latest
        latest=save(EXTERNAL/f"rolling_{step%2}.pt");write_json(EXTERNAL/"latest.json",latest)
    def diagnostics():
        snapshot=CKPTS/f"plus_{step:06d}.pt"
        # Once an evaluation starts its snapshot must never be reserialized.
        receipt=(dict(path=str(snapshot),sha256=sha(snapshot),accepted_updates=step) if snapshot.exists() else save(snapshot))
        saved=torch.load(snapshot,map_location="cpu",weights_only=False)
        assert saved["accepted_updates"]==step and nested_equal(saved["model"],model.state_dict())
        del saved
        probe_due=retirement.last_probe_update<step and ((not retirement.retired and (step-27)%10==0) or (retirement.retired and step%50==0))
        wants_match=False;probe_result=None
        def call(script,output):
            completed=subprocess.run([sys.executable,script,"--checkpoint",str(snapshot),"--sha256",receipt["sha256"],"--output",str(output)],cwd=ROOT)
            if completed.returncode:raise RuntimeError("Diagnostic failed; inspect receipt before resume")
            return json.loads(output.read_text())
        if probe_due:
            state("evaluating_acquisition",checkpoint=receipt)
            probe_result=call("benchmarks/evaluate_sustained_acquisition_v1.py",OUT/f"acquisition_{step:06d}.json")
            wants_match=retirement.observe(step,probe_result)
        match=None
        if step%50==0 or wants_match:
            state("evaluating_nexto",checkpoint=receipt)
            match=call("benchmarks/evaluate_sustained_gameplay_v1.py",OUT/f"eval_{step:06d}.json")
            retirement.last_match_update=step
            write_json(OUT/"latest_evaluation.json",dict(accepted_updates=step,checkpoint=receipt,
                evaluation_file=f"eval_{step:06d}.json",summary=match["summary"]))
        if wants_match and retirement.confirm_match(step,match["summary"]):
            bank=training_starts(32768,True)
            assert scenario_hash(bank)==package["scenario_sha256"]["retired"]
            env.retire_starts(bank)
            write_json(OUT/"retirement.json",dict(utc=utc(),accepted_updates=step,checkpoint=receipt,
                rule=specification()["retirement"],probe=probe_result,match=match["summary"],
                future_resets_only=True,retirement=asdict(retirement)))
        rolling()  # Persist streak/removal before any subsequent optimizer step.
        state("accepted",last_diagnostic_update=step)
    try:
        rolling();state("resumed_with_acquisition_curriculum")
        if (step==27 or ((step-27)%10==0 and retirement.last_probe_update<step and not retirement.retired)
                or (step%50==0 and retirement.last_match_update<step)):
            diagnostics()
        while not (EXTERNAL/"STOP").exists():
            state("collecting");start=time.perf_counter();rollout=collector.collect();collected=time.perf_counter()
            state("optimizing",proposed_update=step+1)
            report=mixed_joint_ppo_update(model,optimizer,rollout,ppo_config(),shuffle)
            step+=1;samples+=collector.last_metrics["trainable_agent_samples"]
            ticks+=collector.last_metrics["physical_physics_ticks"];optimizer_steps+=report["optimizer_steps"]
            row=dict(utc=utc(),accepted_updates=step,additional_updates=step-27,learner_decisions=samples,
                world_physics_ticks=ticks,optimizer_steps=optimizer_steps,collect_seconds=collected-start,
                optimize_seconds=time.perf_counter()-collected,ppo=report,training=collector.last_metrics,retirement=asdict(retirement))
            del rollout;gc.collect();rolling()
            append_json(EXTERNAL/"training_curve.jsonl",row);write_json(EXTERNAL/"progress.json",row)
            state("accepted");print("ACCEPTED "+json.dumps(dict(update=step,touches_min=collector.last_metrics["touches_per_minute"],retired=retirement.retired)),flush=True)
            if step==28:save(CKPTS/f"plus_{step:06d}.pt")
            if (EXTERNAL/"STOP").exists():break
            if step%50==0 or (not retirement.retired and (step-27)%10==0):
                torch.cuda.empty_cache();diagnostics()
        state("stopped_at_accepted_boundary")
    except BaseException as exc:
        failure=dict(utc=utc(),accepted_updates=step,latest_checkpoint=latest,exception=repr(exc),traceback=traceback.format_exc(),automatic_retry=False)
        write_json(EXTERNAL/"failure.json",failure);state("failure",failure=repr(exc));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("preflight","freeze","verify","run"))
    p.add_argument("--worlds",type=int,default=32768);p.add_argument("--resume");p.add_argument("--resume-sha256")
    args=p.parse_args();torch.set_num_threads(4)
    if args.mode=="freeze":freeze()
    elif args.mode=="verify":print(json.dumps(verify()))
    else:
        with gpu_lease():
            preflight(args.worlds) if args.mode=="preflight" else run(args)
