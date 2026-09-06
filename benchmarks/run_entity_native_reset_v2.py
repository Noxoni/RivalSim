"""Separate bounded native reset-alignment test; exact reference600, no learning."""
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
OUT=ROOT/"results/rival2/entity_native_reset_v2"
EXTERNAL=Path("G:/dev/RivalSim-runs/entity-native-reset-v2")
NEXT=Path("C:/Users/patri/AppData/Local/RLBot5/bots/bob_build_x86_64-windows/Nexto/bot.toml")
CASES=("reference600_blue","reference600_orange")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def write(p, obj):
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    tmp.replace(p)


def match(case):
    side=0 if case.endswith("blue") else 1
    rival=rlbot.config.load_player_config(ROOT/f"deployment/entity_native_v2/{case}.toml",side)
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


def owned_game_identity():
    games=[p for p in psutil.process_iter() if p.name().lower()=="rocketleague.exe"]
    if not games:
        return None
    assert len(games)==1 and games[0].pid==29304 and games[0].ppid()==42896, "Do not interrupt another game"
    old=json.loads(Path("G:/dev/RivalSim-runs/entity-native-comparison-v1/campaign_state.json").read_text())
    assert old["status"]=="stopped"
    assert not any(p.name().lower() in {"nexto.exe"} for p in psutil.process_iter())
    game=games[0]
    return dict(pid=game.pid,created=game.create_time(),parent=game.ppid(),command=game.cmdline())


def verify_idle_owned_game(identity):
    games=[p for p in psutil.process_iter() if p.name().lower()=="rocketleague.exe"]
    if not games:
        return
    assert identity is not None and len(games)==1, "Existing game not owned by this test"
    game=games[0]
    assert game.pid==identity["pid"] and game.create_time()==identity["created"] and game.cmdline()==identity["command"]
    m=rlbot.managers.MatchManager()
    try:
        m.connect_and_run(wants_match_communications=False,wants_ball_predictions=False,
                          rlbot_server_port=23234,background_thread=True)
        m.rlbot_interface.send_msg(flat.InitComplete())
        until=time.monotonic()+5
        while time.monotonic()<until:
            packet=m.packet
            if packet is not None:
                phase=str(packet.match_info.match_phase).split(".")[-1]
                assert phase in {"Inactive","Ended"}, "Existing game is active; do not interrupt"
            time.sleep(.1)
    finally:
        m.disconnect()


def freeze():
    assert not (OUT/"native_authority.json").exists()
    OUT.mkdir(parents=True,exist_ok=True)
    base_path=ROOT/"results/rival2/entity_native_packet_v1/reference600_manifest.json"
    base=json.loads(base_path.read_text())
    manifest=dict(format="RIVAL2_ENTITY_NATIVE_RESET_CACHE_V2",
        base_manifest=dict(path=base_path.relative_to(ROOT).as_posix(),sha256=sha(base_path)),
        source=base["source"],artifact=base["artifact"],
        projection="First Kickoff decision following observed Countdown only; eight unavailable wheel-cache fields0; on_ground and all other inputs unchanged.",
        deployed=False)
    assert not (OUT/"manifest.json").exists()
    write(OUT/"manifest.json",manifest)
    for case in CASES:
        match(case)
    paths=["benchmarks/run_entity_native_reset_v2.py",
        "deployment/entity_native_v2/bot.py","deployment/entity_native_v2/reset_runtime.py",
        "deployment/entity_native_v1/runtime.py","deployment/entity_native_v1/packet_observation.py",
        "rivalsim/entity_native_bridge.py",
        "benchmarks/report_entity_native_comparison.py","benchmarks/report_entity_native_reset_v2.py",
        "tests/test_entity_native_reset_runtime.py","tests/test_entity_native_reset_report.py",
        "results/rival2/entity_native_reset_v2/manifest.json",
        "results/rival2/entity_native_reset_v2/tests.xml",
        "results/rival2/entity_native_packet_v1/reference600_manifest.json"]
    paths += [f"deployment/entity_native_v2/{case}.toml" for case in CASES]
    authority=dict(version="RIVAL2_ENTITY_NATIVE_RESET_COMPARISON_V2",cases=list(CASES),
        method="Exact reference600 on both teams, two standard5minute native local Soccar matches. No learning or selection. Separate protocol, not resumed V1.",
        comparison="Prior V1 partial0-37 is diagnostic context, not a completed matched-seed baseline. This tests an explicit empty-cache compatibility projection, not physically measured absent wheel contacts.",
        limits=dict(case_wall_seconds=900,initial_start_seconds=150,overtime_seconds=120),
        projection=manifest["projection"],physics_hz=120,policy_hz=30,hold_ticks=4,deterministic=True,
        nexto=dict(config=str(NEXT),config_sha256=sha(NEXT),executable_sha256=sha(NEXT.parent/"x86_64-pc-windows-msvc/nexto.exe")),
        source=base["source"],artifact=base["artifact"],sources={p:sha(ROOT/p) for p in paths},
        owned_existing_game=owned_game_identity(),
        existing_game_safety="Only reuse the exact previous test-owned process identity after5second read-only observer sees no active packet. No unrelated game termination. Otherwise require no game.",
        stop="User STOP, nonfinite/error, no projection exercised within120decisions, packet stall, or prospective time bound. No guard/reward/weights change.",
        record="Both base and final observation, unavailable quality mask, projection flag, phase per delivered packet, actual controls and raw decision packets.")
    write(OUT/"native_authority.json",authority)
    print("Separate V2 native authority frozen; no match started.")


def run():
    authority=json.loads((OUT/"native_authority.json").read_text())
    for p,digest in authority["sources"].items():
        assert sha(ROOT/p)==digest,p
        remote=subprocess.check_output(["git","show","origin/main:"+p],cwd=ROOT)
        assert remote.replace(b"\r\n",b"\n")==(ROOT/p).read_bytes().replace(b"\r\n",b"\n"),p
    remote=subprocess.check_output(["git","show","origin/main:results/rival2/entity_native_reset_v2/native_authority.json"],cwd=ROOT)
    assert json.loads(remote)==authority
    assert sha(NEXT)==authority["nexto"]["config_sha256"]
    assert sha(NEXT.parent/"x86_64-pc-windows-msvc/nexto.exe")==authority["nexto"]["executable_sha256"]
    verify_idle_owned_game(authority["owned_existing_game"])
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
            state.pop("latest",None)
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
                if now-begin>authority["limits"]["case_wall_seconds"]:
                    raise RuntimeError("Native case wall-clock timeout")
                bot_state=EXTERNAL/case/"bot_state.json"
                if bot_state.exists():
                    r=json.loads(bot_state.read_text())["runtime"]
                    if r["scheduler"]["decisions"]>=120 and r["reset_projection_count"]==0:
                        raise RuntimeError("Reset projection was not exercised")
                packet=manager.packet
                if packet is None:
                    if now-begin>authority["limits"]["initial_start_seconds"]:
                        raise RuntimeError("No first native packet within150seconds")
                    time.sleep(.1)
                    continue
                if not any(p.name.startswith("Rival Reset "+case) for p in packet.players):
                    if now-begin>authority["limits"]["initial_start_seconds"]:
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
