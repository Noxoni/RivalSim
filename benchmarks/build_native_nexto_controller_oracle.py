"""Execute installed native controller methods on bounded synthetic packet traces.

Reads the installed archive as a file, never process memory; no model inference,
game connection or training. Native Python3.12 required. Outputs feed simulator
Python tests without importing incompatible native bytecode there.
"""
from __future__ import annotations

import base64
import builtins
import json
import marshal
from pathlib import Path
import subprocess
import types

import numpy as np
import rlbot.flat as flat

import audit_native_nexto_integration as audit

ROOT=audit.ROOT
OUT=ROOT/"results/rival2/nexto_native_controller_v1"
PUBLIC="0bdb6b49072f6f3829319e68bd6210a0ca4b24a2"
N,T=8,512


def run():
    assert not OUT.exists()
    archive=audit.CArchiveReader(str(audit.EXE))
    native=marshal.loads(archive.extract("bot"))
    response=json.loads(subprocess.check_output(["gh","api",f"repos/VirxEC/NectoFamily/contents/nexto/bot.py?ref={PUBLIC}"]))
    source=base64.b64decode(response["content"])
    public=compile(source,"public.py","exec",dont_inherit=True)
    checks={k:getattr(native,k)==getattr(public,k) for k in ("co_code","co_names","co_varnames","co_flags","co_exceptiontable")}
    checks["module_literals"]=all(type(a)==type(b) and (isinstance(a,types.CodeType) or a==b)
                                  for a,b in zip(native.co_consts,public.co_consts,strict=True))
    assert all(checks.values()),checks
    # Bind the actual global kickoff literal, not merely the three method bodies.
    # Restricted AST parser consumes source whose module code/literals match.
    import ast
    tree=ast.parse(source)
    literal=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="KICKOFF_CONTROLS" for x in n.targets))
    namespace={"ControllerState":flat.ControllerState}
    exec(compile(ast.Module(body=[literal],type_ignores=[]),"bound_literal","exec"),namespace)
    table=np.asarray([[float(getattr(c,k)) for k in audit.CHANNELS] for c in namespace["KICKOFF_CONTROLS"]],np.float32)
    assert table.shape==(168,8)
    globals_={"__builtins__":builtins.__dict__,"np":np,"MatchPhase":flat.MatchPhase,"GameMode":flat.GameMode,"KICKOFF_NUMPY":table}
    get=types.FunctionType(audit.find_code(native,"get_output"),globals_)
    kickoff=types.FunctionType(audit.find_code(native,"maybe_do_kickoff"),globals_)
    controls=types.FunctionType(audit.find_code(native,"update_controls"),globals_)
    agent_code=archive.open_embedded_archive("PYZ.pyz").extract("agent")
    lookup=types.FunctionType(audit.find_code(agent_code,"make_lookup_table"),{"np":np})()
    rng=np.random.default_rng(2026090681)
    chosen=rng.integers(0,len(lookup),size=(T,N))
    actions=lookup[chosen].astype(np.float32)
    active=np.ones((T,N),bool)
    active[20:32,2]=False;active[100:145,1]=False;active[280:330,5]=False
    activate=np.zeros_like(active);activate[0]=True
    activate[73,3]=True;activate[145,1]=True;activate[303,0]=True;activate[330,5]=True
    notify=np.zeros_like(active);notify[256,6]=True;notify[344,4]=True
    phases=np.zeros_like(active)
    phases[:211,:2]=True;phases[40:221,2:4]=True;phases[:60,4:6]=True
    phases[200:401,4:6]=True;phases[300:,6:]=True
    side=np.tile(np.arange(N)%2,(T,1));side[73:,3]=0
    pos=np.zeros((T,N,2,3),np.float32)
    pos[:,:,0,1]=-4608;pos[:,:,1,1]=4608;pos[:,:,:,2]=17
    pos[:,7,side[0,7],1]=6000;pos[:,7,1-side[0,7],1]=-1000 # denied taker
    ball=np.zeros((T,N,3),np.float32);ball[:,:,2]=93
    ball[45:55,4:6,1]=10 # phase remains kickoff; y guard disables script
    ball[~phases,1]=500
    emitted=np.zeros((T,N,8),np.float32);pending=np.zeros_like(emitted)
    previous=np.full_like(emitted,np.nan)
    ticks=np.zeros((T,N),np.int64);updates=np.zeros((T,N),bool)
    idx=np.zeros((T,N),np.int64);computed=np.zeros((T,N),bool)
    beta=np.full((T,N),np.nan,np.float32)
    ns=types.SimpleNamespace
    bots=[None]*N;local_frames=np.zeros(N,np.int64)
    current=[0]
    for step in range(T):
        current[0]=step
        for world in range(N):
            if activate[step,world]:
                own=ns(team_num=0);other=ns(team_num=1)
                game=ns(players=[own,other])
                def update(packet,game=game,own=own,other=other):game.players=[own,other]
                game.update=update
                def observe(player,state,action,world=world):
                    previous[current[0],world]=action
                def predict(obs,temperature,world=world):
                    beta[current[0],world]=temperature
                    computed[current[0],world]=True
                    return actions[current[0],world].copy(),None
                b=ns(is_toxic=False,prev_tick=int(local_frames[world]),ticks=8,tick_skip=8,
                     update_action=True,index=int(side[step,world]),team=int(side[step,world]),
                     gamemode=flat.GameMode.Soccar,beta=1,stochastic_kickoffs=True,render=False,
                     hardcoded_kickoffs=True,kickoff_index=-1,action=np.zeros(8),
                     controls=flat.ControllerState(),game_state=game,
                     obs_builder=ns(build_obs=observe),agent=ns(act=predict))
                b.update_controls=types.MethodType(controls,b)
                b.maybe_do_kickoff=types.MethodType(kickoff,b)
                bots[world]=b
            b=bots[world]
            if notify[step,world]:b.kickoff_index=-1 # explicit unseen between-episode phase exit
            if active[step,world]:
                local_frames[world]+=1
                players=[ns(team=j,physics=ns(location=ns(x=float(p[0]),y=float(p[1]),z=float(p[2])))) for j,p in enumerate(pos[step,world])]
                p=ns(players=players,balls=[ns(physics=ns(location=ns(x=float(ball[step,world,0]),y=float(ball[step,world,1]),z=93.)))],
                     match_info=ns(frame_num=int(local_frames[world]),match_phase=flat.MatchPhase.Kickoff if phases[step,world] else flat.MatchPhase.Active))
                get(b,p)
            emitted[step,world]=[float(getattr(b.controls,k)) for k in audit.CHANNELS]
            pending[step,world]=b.action
            ticks[step,world]=b.ticks;updates[step,world]=b.update_action;idx[step,world]=b.kickoff_index
    OUT.mkdir()
    np.savez_compressed(OUT/"oracle.npz",candidate_actions=actions,active=active,activate=activate,
        notify=notify,kickoff=phases,side=side,car_pos=pos,ball_pos=ball,emitted=emitted,
        pending=pending,previous=previous,ticks=ticks,updates=updates,kickoff_index=idx,
        computed=computed,beta=beta,kickoff_table=table,lookup=lookup.astype(np.float32))
    result=dict(worlds=N,ticks=T,installed_executable_sha256=audit.sha(audit.EXE.read_bytes()),
        public_commit=PUBLIC,public_source_sha256=audit.sha(source),module_code_and_literal_checks=checks,
        installed_bot_bytecode_sha256=audit.sha(archive.extract("bot")),
        oracle_sha256=audit.sha((OUT/"oracle.npz").read_bytes()),
        generator_source_sha256=audit.sha(Path(__file__).read_bytes()),
        called_installed_methods=["get_output","maybe_do_kickoff","update_controls","Agent.make_lookup_table"],
        scope="Exact controller methods with deterministic neural-action substitutes, not native neural-output or physics parity. Inactive-world suspension uses per-world virtual frame clocks; activate represents a fresh controller. notify represents phase exit without inventing countdown ticks.",
        beta_selection="Source beta1 ordinary; beta.5 kickoff recorded. Stub action generation is deterministic; separate distribution/RNG tests required.",
        optimizer_steps=0,game_connection=False,production_changed=False)
    (OUT/"oracle.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",newline="\n")
    print(json.dumps(result))


if __name__=="__main__":run()
