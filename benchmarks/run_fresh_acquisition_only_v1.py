"""Bounded fresh-model acquisition-only experiment; shared PPO/reward left intact."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.run_sustained_gameplay_v1 import SOURCES as BASE_SOURCES, atomic_checkpoint, text_sha, COLLISION
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json, append_json
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from rivalsim.fresh_acquisition_only_v1 import VERSION, SEED, MAX_UPDATES, WORLDS, training_bank, new_model, authority
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, AcquisitionCollector, FAMILY, acquisition_starts, SEED as BANK_SEED
from rivalsim.sustained_gameplay_v1 import POLICY_VERSION, ppo_config
from rivalsim.ssl_entity_training import fresh_entity_optimizer, finite_model_and_optimizer, joint_sequence_loss
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data, mixed_joint_ppo_update
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash

OUT = ROOT / "results/rival2/fresh_acquisition_only_v1"
CKPTS = ROOT / "checkpoints/rival2/fresh_acquisition_only_v1"
EXTERNAL = Path("G:/dev/RivalSim-runs/fresh-acquisition-only-v1")
INITIAL = CKPTS / "initialized.pt"
SOURCES = (*BASE_SOURCES, "rivalsim/sustained_acquisition_v1.py",
    "benchmarks/evaluate_sustained_acquisition_v1.py", "rivalsim/fresh_acquisition_only_v1.py",
    "benchmarks/run_fresh_acquisition_only_v1.py", "tests/test_fresh_acquisition_only_v1.py")


def preflight():
    if INITIAL.exists() or (EXTERNAL / "campaign_state.json").exists():
        raise RuntimeError("Do not overwrite an existing initialization/campaign")
    model = new_model().cuda()
    optimizer = fresh_entity_optimizer(model)
    before = tensor_hash(model.state_dict())
    env = AcquisitionEnv(WORLDS, COLLISION, device="cuda:0", seed=SEED,
                         ssl_foundation_scenarios=training_bank())
    collector = AcquisitionCollector(env, model, seed=SEED)
    torch.cuda.reset_peak_memory_stats()
    rollout = collector.collect()
    data = mixed_sequence_data(rollout, ppo_config())
    index = data["train_mask"].any(1).nonzero().flatten()[:728]
    model.train()
    loss, _ = joint_sequence_loss(model, data, index, ppo_config())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.critic(data["observations"][index], data["initial_hidden"][1:,index],
                 data["reset_before"][index])[0].square().mean().backward()
    isolated = all(p.grad is None for n,p in model.named_parameters() if not n.startswith("critic."))
    optimizer.zero_grad(set_to_none=True)
    exposure = collector.last_metrics["by_start_family_and_opponent"]
    checks = dict(model_unchanged=before==tensor_hash(model.state_dict()), optimizer_empty=not optimizer.state,
        finite=finite_model_and_optimizer(model,optimizer) and bool(torch.isfinite(loss)&torch.isfinite(norm)),
        critic_isolated=isolated, all_live_acquisition=bool((env.family==FAMILY).all()),
        no_other_family_exposure=all(v["world_seconds"]==0 for k,v in exposure.items() if not k.startswith("ball_acquisition_")),
        both_memories=collector.hidden.shape==(2,WORLDS,2,256),
        masks=collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"],
        exact_actions=torch.equal(model.action_table[rollout.action_indices][rollout.train_mask],rollout.actions[rollout.train_mask]),
        no_mechanic_or_task_state=env.world.gameplay_v3 is None and env.world.gameplay_120 is None and not hasattr(env,"events") and not hasattr(env,"race"))
    assert all(checks.values()), checks
    initial = atomic_checkpoint(INITIAL, dict(format=VERSION+"_CHECKPOINT",policy_version=POLICY_VERSION,
        accepted_updates=0,parent=None,seed=SEED,authority_sha256=content_hash(authority()),
        model=model.state_dict(),model_config=asdict(model.config),optimizer=optimizer.state_dict(),
        runtime_contract_hashes=env.contract_hashes))
    report=dict(utc=utc(),checks=checks,worlds=WORLDS,optimizer_steps=0,initialized=initial,
        model_sha256=before,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm),
        runtime_contract_hashes=env.contract_hashes,training=collector.last_metrics)
    write_json(OUT/"preflight_32768.json",report)
    print(json.dumps(dict(checks=checks,initial=initial)),flush=True)


def freeze():
    if (OUT/"authority.json").exists() or (EXTERNAL/"campaign_state.json").exists():
        raise RuntimeError("Prospective authority already frozen")
    pre=json.loads((OUT/"preflight_32768.json").read_text())
    assert all(pre["checks"].values()) and pre["optimizer_steps"]==0
    suites=ET.parse(OUT/"tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    write_json(OUT/"authority.json",authority())
    write_json(OUT/"package.json",dict(authority_sha256=content_hash(authority()),initialized_sha256=sha(INITIAL),
        scenario_sha256=scenario_hash(training_bank()),probe_sha256=scenario_hash(acquisition_starts(1024,BANK_SEED+100)),
        sources={p:text_sha(ROOT/p) for p in SOURCES},
        evidence={p:sha(OUT/p) for p in ("tests.xml","preflight_32768.json")}))


def verify():
    package=json.loads((OUT/"package.json").read_text())
    assert json.loads((OUT/"authority.json").read_text())==authority()
    assert package["authority_sha256"]==content_hash(authority()) and package["initialized_sha256"]==sha(INITIAL)
    for p,h in package["sources"].items(): assert text_sha(ROOT/p)==h,p
    for p,h in package["evidence"].items(): assert sha(OUT/p)==h,p
    prefix=OUT.relative_to(ROOT).as_posix()+"/"
    paths=(*SOURCES,INITIAL.relative_to(ROOT).as_posix(),*(prefix+p for p in ("authority.json","package.json",*package["evidence"])))
    for p in paths:
        remote=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        local=(ROOT/p).read_bytes()
        assert (remote==local if p.endswith(".pt") else remote.replace(b"\r\n",b"\n")==local.replace(b"\r\n",b"\n")),p
    return package


def run(args):
    package=verify(); spec=authority()
    EXTERNAL.mkdir(parents=True,exist_ok=True)
    if (EXTERNAL/"STOP").exists(): raise RuntimeError("Respect STOP")
    source=None
    if args.resume:
        latest=json.loads((EXTERNAL/"latest.json").read_text())
        assert Path(args.resume).resolve()==Path(latest["path"]).resolve()
        assert args.resume_sha256 and sha(args.resume)==latest["sha256"]==args.resume_sha256.upper()
        source=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert source["parent"] is None and source["format"]==VERSION+"_CHECKPOINT"
        assert source["authority_sha256"]==content_hash(spec) and source["initialized_checkpoint_sha256"]==sha(INITIAL)
    elif (EXTERNAL/"campaign_state.json").exists() or (EXTERNAL/"latest.json").exists():
        raise RuntimeError("Existing campaign requires explicit same-lineage resume")
    model=new_model().cuda();optimizer=fresh_entity_optimizer(model)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    assert tensor_hash(model.state_dict())==tensor_hash(initial["model"]) and not optimizer.state
    steps=samples=ticks=optimizer_steps=0;last_probe=-1
    if source is not None:
        model.load_state_dict(source["model"],strict=True);optimizer.load_state_dict(source["optimizer"])
        steps=source["accepted_updates"];samples=source["learner_decisions"];ticks=source["world_physics_ticks"]
        optimizer_steps=source["optimizer_steps"];last_probe=source["last_probe_update"]
    bank=training_bank();assert scenario_hash(bank)==package["scenario_sha256"]
    env=AcquisitionEnv(WORLDS,COLLISION,device="cuda:0",seed=SEED,ssl_foundation_scenarios=bank)
    assert env.contract_hashes==initial["runtime_contract_hashes"]
    collector=AcquisitionCollector(env,model,seed=SEED)
    shuffle=torch.Generator(device=env.device).manual_seed(SEED+2)
    if source is not None:
        collector.nexto.load_checkpoint_state(source["opponent_state"]["native_nexto"])
        collector.nexto.activate(torch.ones_like(collector.is_nexto));collector.assign(torch.ones_like(collector.is_nexto))
        collector.opponent_generator.set_state(source["opponent_state"]["generator_state"].cpu())
        collector.generator.set_state(source["policy_rng"].cpu());shuffle.set_state(source["shuffle_rng"].cpu())
        torch.set_rng_state(source["cpu_rng"].cpu());torch.cuda.set_rng_state(source["cuda_rng"].cpu())
    del source,initial;gc.collect()
    latest=None
    def state(status,**extra):
        write_json(EXTERNAL/"campaign_state.json",dict(utc=utc(),pid=os.getpid(),status=status,
            accepted_updates=steps,learner_decisions=samples,optimizer_steps=optimizer_steps,
            last_probe_update=last_probe,latest_checkpoint=latest,authority_sha256=content_hash(spec),**extra))
    def save(path):
        return atomic_checkpoint(path,dict(format=VERSION+"_CHECKPOINT",policy_version=POLICY_VERSION,parent=None,seed=SEED,
            initialized_checkpoint_sha256=sha(INITIAL),authority_sha256=content_hash(spec),package=package,
            model=model.state_dict(),model_config=asdict(model.config),optimizer=optimizer.state_dict(),
            accepted_updates=steps,learner_decisions=samples,world_physics_ticks=ticks,learner_physics_exposure=samples*4,
            optimizer_steps=optimizer_steps,last_probe_update=last_probe,policy_rng=collector.generator.get_state(),
            shuffle_rng=shuffle.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(),runtime_contract_hashes=env.contract_hashes,
            last_metrics=collector.last_metrics,recurrent_state_audit=dict(shape=list(collector.hidden.shape),
            sha256=tensor_hash({"hidden":collector.hidden}),semantics=spec["resume"])))
    def rolling():
        nonlocal latest
        latest=save(EXTERNAL/f"rolling_{steps%2}.pt");write_json(EXTERNAL/"latest.json",latest)
    def probe():
        nonlocal last_probe
        snapshot=CKPTS/f"plus_{steps:06d}.pt"
        if snapshot.exists():
            saved=torch.load(snapshot,map_location="cpu",weights_only=False)
            assert saved["accepted_updates"]==steps and tensor_hash(saved["model"])==tensor_hash(model.state_dict())
            receipt=dict(path=str(snapshot),sha256=sha(snapshot),accepted_updates=steps)
            del saved
        else:receipt=save(snapshot)
        output=OUT/f"acquisition_{steps:06d}.json";state("evaluating_acquisition",checkpoint=receipt)
        result=subprocess.run([sys.executable,"benchmarks/evaluate_sustained_acquisition_v1.py",
            "--checkpoint",str(snapshot),"--sha256",receipt["sha256"],"--output",str(output)],cwd=ROOT)
        if result.returncode:raise RuntimeError("Diagnostic failed; inspect receipt before resuming")
        d=json.loads(output.read_text());assert d["checkpoint_sha256"]==receipt["sha256"] and d["optimizer_steps"]==0
        last_probe=steps;rolling();state("accepted",completed_probe=str(output))
        write_json(EXTERNAL/"latest_evaluation.json",dict(accepted_updates=steps,path=str(output),sha256=sha(output)))
    try:
        rolling();state("initialized" if steps==0 else "resumed_fresh_physical_episodes")
        write_json(EXTERNAL/f"entry_{steps:06d}_{time.time_ns()}.json",dict(utc=utc(),accepted_updates=steps,
            fresh_random_initialization=not args.resume,parent=None,model_sha256=tensor_hash(model.state_dict()),
            optimizer_empty=not optimizer.state,both_memories_zero=not bool(collector.hidden.any()),
            runtime_contract_hashes=env.contract_hashes,reset_bank_sha256=scenario_hash(bank)))
        if steps%10==0 and last_probe<steps:probe()
        while steps<MAX_UPDATES and not (EXTERNAL/"STOP").exists():
            state("collecting");start=time.perf_counter();rollout=collector.collect();collected=time.perf_counter()
            if not all(v["world_seconds"]==0 for k,v in collector.last_metrics["by_start_family_and_opponent"].items() if not k.startswith("ball_acquisition_")):
                raise RuntimeError("Non-acquisition scenario contamination")
            state("optimizing",proposed_update=steps+1)
            report=mixed_joint_ppo_update(model,optimizer,rollout,ppo_config(),shuffle)
            steps+=1;samples+=collector.last_metrics["trainable_agent_samples"]
            ticks+=collector.last_metrics["physical_physics_ticks"];optimizer_steps+=report["optimizer_steps"]
            row=dict(utc=utc(),accepted_updates=steps,learner_decisions=samples,world_physics_ticks=ticks,
                optimizer_steps=optimizer_steps,collect_seconds=collected-start,optimize_seconds=time.perf_counter()-collected,
                ppo=report,training=collector.last_metrics)
            del rollout;gc.collect();rolling();append_json(EXTERNAL/"training_curve.jsonl",row)
            write_json(EXTERNAL/"progress.json",row);state("accepted")
            print("ACCEPTED "+json.dumps(dict(update=steps,touches_min=collector.last_metrics["touches_per_minute"])),flush=True)
            if steps==1:save(CKPTS/"plus_000001.pt")
            if (EXTERNAL/"STOP").exists():break
            if steps%10==0:torch.cuda.empty_cache();probe()
        state("completed_bounded_diagnostic" if steps==MAX_UPDATES else "stopped_at_accepted_boundary")
    except BaseException as exc:
        write_json(EXTERNAL/"failure.json",dict(utc=utc(),accepted_updates=steps,latest_checkpoint=latest,
            exception=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
        state("failure",exception=repr(exc));raise


def report(offset):
    ckpath=CKPTS/f"plus_{offset:06d}.pt";ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    path=OUT/f"acquisition_{offset:06d}.json";probe=json.loads(path.read_text())
    rows=[json.loads(s) for s in (EXTERNAL/"training_curve.jsonl").read_text().splitlines()] if offset else []
    rows=[r for r in rows if r["accepted_updates"]<=offset]
    counts={int(s["step"]) for s in ck["optimizer"]["state"].values()}
    checks=dict(parent_none=ck["parent"] is None,authority=ck["authority_sha256"]==content_hash(authority()),
        format=ck["format"]==VERSION+"_CHECKPOINT",initial_identity=ck["initialized_checkpoint_sha256"]==sha(INITIAL),
        contracts=ck["runtime_contract_hashes"]==initial["runtime_contract_hashes"],
        contiguous=[r["accepted_updates"] for r in rows]==list(range(1,offset+1)),
        decisions=ck["learner_decisions"]==offset*4423680,physics_ticks=ck["world_physics_ticks"]==offset*11796480,
        counters=ck["optimizer_steps"]==sum(r["ppo"]["optimizer_steps"] for r in rows),
        adam_counters=counts==({ck["optimizer_steps"]} if offset else set()),
        no_kl_rejection=all(r["ppo"]["kl_rejections"]==0 for r in rows),
        model_finite=all(bool(torch.isfinite(v).all()) for v in ck["model"].values()),
        adam_finite=all(bool(torch.isfinite(v).all()) for s in ck["optimizer"]["state"].values() for v in s.values() if torch.is_tensor(v)),
        acquisition_only=all(v["world_seconds"]==0 for r in rows for k,v in r["training"]["by_start_family_and_opponent"].items() if not k.startswith("ball_acquisition_")),
        evaluation_identity=probe["checkpoint_sha256"]==sha(ckpath),
        evaluation_read_only=probe["optimizer_steps"]==0 and all(probe[k] for k in ("checkpoint_unchanged","model_unchanged","nexto_unchanged")))
    assert all(checks.values()),checks
    times=torch.tensor(probe["first_contact_ticks"]);assert len(times)==1024
    for i,name,deadline in ((0,"easy",600),(1,"varied",960)):
        t=times[i::2];good=(t>0)&(t<=deadline)
        assert int(good.sum())==probe[name]["success_count"] and float(good.double().mean())==probe[name]["success_fraction"]
    write_json(OUT/f"audit_{offset:06d}.json",dict(utc=utc(),checks=checks,accepted_updates=offset,
        checkpoint=dict(path=ckpath.relative_to(ROOT).as_posix(),sha256=sha(ckpath)),probe_sha256=sha(path)))
    write_json(OUT/f"through_{offset:06d}.json",dict(rows=rows))
    print(json.dumps(dict(checks=checks,easy=probe["easy"],varied=probe["varied"])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("preflight","freeze","verify","run","report"))
    p.add_argument("--resume");p.add_argument("--resume-sha256");p.add_argument("--offset",type=int)
    args=p.parse_args();torch.set_num_threads(4)
    if args.mode=="freeze":freeze()
    elif args.mode=="verify":print(json.dumps(verify()))
    elif args.mode=="report":report(args.offset)
    else:
        with gpu_lease():
            preflight() if args.mode=="preflight" else run(args)
