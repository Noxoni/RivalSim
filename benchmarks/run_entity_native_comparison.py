"""Bounded offline native RLBot comparison, zero training or state setting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import rlbot.config
import rlbot.flat as flat
import rlbot.managers
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/entity_native_packet_v1"
EXTERNAL=Path("G:/dev/RivalSim-runs/entity-native-comparison-v1")
NEXT=Path("C:/Users/patri/AppData/Local/RLBot5/bots/bob_build_x86_64-windows/Nexto/bot.toml")
CASES=("reference600_blue","finishing650_orange","finishing650_blue","reference600_orange")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def write(p, obj):
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    tmp.replace(p)


def match(case):
    side=0 if case.endswith("blue") else 1
    rival=rlbot.config.load_player_config(ROOT/f"deployment/entity_native_v1/{case}.toml",side)
    nexto=rlbot.config.load_player_config(NEXT,1-side)
    return flat.MatchConfiguration(launcher=flat.Launcher.Steam,auto_start_agents=True,wait_for_agents=True,
        game_map_upk="Stadium_P",player_configurations=[rival,nexto] if side==0 else [nexto,rival],
        script_configurations=[],game_mode=flat.GameMode.Soccar,skip_replays=True,instant_start=False,
        mutators=flat.MutatorSettings(match_length=flat.MatchLengthMutator.FiveMinutes,
            max_score=flat.MaxScoreMutator.Unlimited,overtime=flat.OvertimeMutator.Unlimited,
            game_speed=flat.GameSpeedMutator.Default,boost_amount=flat.BoostAmountMutator.NormalBoost,
            boost_strength=flat.BoostStrengthMutator.One,gravity=flat.GravityMutator.Default,
            demolish=flat.DemolishMutator.Default),
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
        enable_rendering=flat.DebugRendering.AlwaysOff,enable_state_setting=False,
        auto_save_replay=False,freeplay=False,performance_monitor=flat.PerformanceMonitor.NeverShow)


def freeze():
    assert not (OUT/"native_authority.json").exists()
    for case in CASES:
        match(case)  # actual installed RLBot config/schema validation, no launch
    paths=["benchmarks/run_entity_native_comparison.py","deployment/entity_native_v1/bot.py",
           "deployment/entity_native_v1/runtime.py","deployment/entity_native_v1/packet_observation.py",
           "rivalsim/entity_native_bridge.py"]
    paths += [f"deployment/entity_native_v1/{case}.toml" for case in CASES]
    paths += [f"results/rival2/entity_native_packet_v1/{name}_manifest.json" for name in ("reference600","finishing650")]
    paths += ["results/rival2/entity_native_packet_v1/captured_state_mapping.json","results/rival2/entity_native_packet_v1/tests.xml"]
    authority=dict(version="RIVAL2_ENTITY_NATIVE_LOCAL_COMPARISON_V1",cases=list(CASES),
        method="Four standard five-minute native local RLBot Soccar matches, each exact candidate on each side. No ranking claim, no optimizer, no checkpoint selection during testing.",
        observation_qualification="39direct104derived29approximate10unavailable; explicit wheel/sticky proxies, no feature masking. Gaps logged; packet-derived inputs not claimed simulator-exact.",
        cold_warmup=True,physics_hz=120,policy_hz=30,hold_ticks=4,deterministic=True,
        nexto=dict(config=str(NEXT),config_sha256=sha(NEXT),executable_sha256=sha(NEXT.parent/"x86_64-pc-windows-msvc/nexto.exe")),
        sources={p:sha(ROOT/p) for p in paths},
        limits=dict(case_wall_seconds=600,initial_start_seconds=150,overtime_seconds=120),
        stop="User STOP, native runtime nonfinite/error or stalled packet stream. Do not retrain/modify weights or reward. Completed native match evidence is separate from simulator results.",
        installation="Separate bot definitions/run commands in RivalSim repo; old installed Rival/Nexto sources/configs unchanged.")
    write(OUT/"native_authority.json",authority)
    print("Native comparison authority frozen. No game launched.")


def run():
    authority=json.loads((OUT/"native_authority.json").read_text())
    for p,digest in authority["sources"].items():
        assert sha(ROOT/p)==digest,p
        remote=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        assert remote.replace(b"\r\n",b"\n")==(ROOT/p).read_bytes().replace(b"\r\n",b"\n"),p
    remote=subprocess.check_output(["git","show","origin/main:results/rival2/entity_native_packet_v1/native_authority.json"],cwd=ROOT)
    assert json.loads(remote)==authority
    assert sha(NEXT)==authority["nexto"]["config_sha256"]
    assert sha(NEXT.parent/"x86_64-pc-windows-msvc/nexto.exe")==authority["nexto"]["executable_sha256"]
    assert not any(p.name().lower()=="rocketleague.exe" for p in psutil.process_iter()), "Do not interrupt an existing user game"
    assert not (EXTERNAL/"campaign_state.json").exists(), "Never silently restart a previous native comparison"
    EXTERNAL.mkdir(parents=True,exist_ok=True)
    assert not (EXTERNAL/"STOP").exists()
    manager=rlbot.managers.MatchManager()
    state=dict(status="starting",authority_sha256=sha(OUT/"native_authority.json"),pid=__import__("os").getpid(),completed=[])
    write(EXTERNAL/"campaign_state.json",state)
    try:
        for case in CASES:
            state.update(status="running",case=case)
            write(EXTERNAL/"campaign_state.json",state)
            begin=time.monotonic()
            manager.packet=None
            manager.start_match(match(case),wait_for_start=False,ensure_server_started=True)
            last_frame=None
            changed=time.monotonic()
            overtime_begin=None
            latest=None
            while True:
                now=time.monotonic()
                if (EXTERNAL/"STOP").exists():
                    raise RuntimeError("User STOP requested")
                if (EXTERNAL/case/"failure.json").exists():
                    raise RuntimeError("Native actor failed: "+(EXTERNAL/case/"failure.json").read_text())
                if now-begin>600:
                    raise RuntimeError("Native case wall-clock timeout")
                packet=manager.packet
                if packet is None:
                    if now-begin>150:
                        raise RuntimeError("No first native packet within150seconds")
                    time.sleep(.1)
                    continue
                if not any(p.name.startswith("Rival Entity "+case) for p in packet.players):
                    if now-begin>150:
                        raise RuntimeError("No packet for the requested native case within150seconds")
                    time.sleep(.1)
                    continue
                frame=int(packet.match_info.frame_num)
                if frame!=last_frame:
                    changed=now
                    last_frame=frame
                elif now-changed>30:
                    raise RuntimeError("Native packet stream stopped advancing")
                phase=str(packet.match_info.match_phase).split(".")[-1]
                latest=dict(frame=frame,phase=phase,score=[int(t.score) for t in packet.teams],
                    seconds_remaining=float(packet.match_info.game_time_remaining),is_overtime=bool(packet.match_info.is_overtime),
                    players=[dict(name=p.name,team=int(p.team),player_id=int(p.player_id)) for p in packet.players])
                if packet.match_info.is_overtime and overtime_begin is None:
                    overtime_begin=now
                if phase=="Ended" or (overtime_begin is not None and now-overtime_begin>=120):
                    ready=EXTERNAL/case/"ready.json"
                    if not ready.exists():
                        raise RuntimeError("Match ended without bound Rival readiness")
                    result=dict(case=case,native_result=latest,wall_seconds=now-begin,
                                unresolved=phase!="Ended",readiness=json.loads(ready.read_text()),optimizer_steps=0)
                    write(EXTERNAL/case/"match_result.json",result)
                    state["completed"].append(case)
                    write(EXTERNAL/"campaign_state.json",state)
                    print(json.dumps(result),flush=True)
                    break
                if int(now-begin)%5==0:
                    state["latest"]=latest
                    write(EXTERNAL/"campaign_state.json",state)
                time.sleep(.1)
            # Allow Ended packet to reach bot before disconnection closes files.
            time.sleep(1)
            manager.stop_match()
            time.sleep(1)
        state["status"]="complete_review"
    except Exception as exc:
        state.update(status="stopped",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write(EXTERNAL/"campaign_state.json",state)
        manager.stop_match()
        manager.disconnect()


if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("mode",choices=("freeze","run"))
    freeze() if a.parse_args().mode=="freeze" else run()
