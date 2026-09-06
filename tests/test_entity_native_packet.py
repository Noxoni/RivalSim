"""Packet lifecycle fixtures; captured-state checks are a separate evidence run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployment/entity_native_v1"
sys.path.insert(0, str(DEPLOY))
spec = importlib.util.spec_from_file_location("entity_packet_test_runtime", DEPLOY/"runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def manifest():
    return json.loads((ROOT/"results/rival2/entity_native_packet_v1/reference600_manifest.json").read_text())


def field_info(m):
    return NS(boost_pads=[NS(location=NS(x=x,y=y,z=z)) for x,y,z in m["observation"]["canonical_boost_pad_positions"]])


def packet(frame=0, phase="Kickoff", score=(0,0), own_id=17, touch=None, order=False):
    def vec(x=0,y=0,z=0):
        return NS(x=x,y=y,z=z)
    players=[]
    for team in (0,1):
        p=NS(team=team, player_id=own_id if team==0 else 29,
            physics=NS(location=vec(0, -4608 if team==0 else 4608,17), velocity=vec(),
                angular_velocity=vec(), rotation=NS(pitch=0.,yaw=np.pi/2 if team==0 else -np.pi/2,roll=0.)),
            boost=33.3, air_state="AirState.OnGround", has_jumped=False, has_double_jumped=False,
            has_dodged=False, dodge_elapsed=0., demolished_timeout=-1., is_supersonic=False,
            last_input=NS(throttle=1.,steer=0.,pitch=0.,yaw=0.,roll=0.,jump=False,boost=False,handbrake=False),
            latest_touch=NS(game_seconds=touch,ball_index=0) if team==0 and touch is not None else None)
        players.append(p)
    if order:
        players.reverse()
    return NS(players=players, teams=[NS(score=s) for s in score],
        match_info=NS(frame_num=frame, match_phase="MatchPhase."+phase,seconds_elapsed=frame/120.),
        balls=[NS(physics=NS(location=vec(0,0,93),velocity=vec(),angular_velocity=vec()))],
        boost_pads=[NS(is_active=True,timer=0.) for _ in range(34)])


class Counter:
    def __call__(self, obs, h):
        action=torch.zeros(1,8)
        action[0,0]=1
        return action,h+1,torch.zeros(1,90)


def make(model=None):
    m=manifest()
    return runtime.EntityPacketRuntime(m,field_info(m),model=model or Counter())


def test_warmup_cannot_become_initial_hidden():
    r=make()
    assert not r.scheduler.hidden.any() and r.scheduler.stats["decisions"]==0
    r.step(packet(),17)
    assert float(r.scheduler.hidden.max())==1


def test_player_identity_not_array_index_or_fixed_team():
    r=make()
    r.step(packet(order=True),17)
    assert r.last_decision["team"]==0
    r2=make()
    r2.step(packet(),29)
    assert r2.last_decision["team"]==1
    np.testing.assert_allclose(r.last_decision["observation"][:25],r2.last_decision["observation"][:25],atol=2e-7)


def test_interval_events_aggregate_until_decision_and_previous_action_is_held():
    r=make()
    r.step(packet(0),17)
    for frame in (1,2,3):
        r.step(packet(frame,touch=1/120),17)
        assert r.last_decision is None
    r.step(packet(4,touch=1/120),17)
    assert r.last_decision["observation"][176]==1
    assert r.last_decision["observation"][167]==1
    assert not r.adapter.memory.touch_event.any()
    for frame in (5,6,7,8):
        r.step(packet(frame,touch=1/120),17)
    assert r.last_decision["observation"][176]==0


def test_goal_and_countdown_reset_but_normal_pause_preserves_memory():
    r=make()
    for f in range(6):
        r.step(packet(f,"Active"),17)
    h=r.scheduler.hidden.clone()
    age=r.adapter.memory.episode_ticks.copy()
    for f in (6,30,80):
        assert not r.step(packet(f,"Paused"),17).any()
    assert torch.equal(r.scheduler.hidden,h)
    assert np.array_equal(r.adapter.memory.episode_ticks,age)
    r.step(packet(81,"Active"),17)
    assert r.scheduler.stats["resets"]==1
    r.step(packet(82,"GoalScored",(1,0)),17)
    r.step(packet(83,"Countdown",(1,0)),17)
    r.step(packet(84,"Kickoff",(1,0)),17)
    assert r.scheduler.stats["resets"]==2
    assert r.last_decision["observation"][175]==1
    assert not r.last_decision["observation"][167:175].any()
    assert float(r.scheduler.hidden.max())==1


def test_duplicate_and_backward_packets_do_not_rewind_game_history():
    r=make()
    r.step(packet(20),17)
    r.step(packet(20),17)
    r.step(packet(19),17)
    assert r.scheduler.stats["decisions"]==1
    assert r.scheduler.stats["duplicates"]==r.scheduler.stats["out_of_order"]==1


def test_missing_packet_quality_is_explicit_without_hidden_interpolation():
    r=make()
    r.step(packet(0),17)
    r.step(packet(16),17)
    assert r.scheduler.stats["decisions"]==2
    assert r.scheduler.stats["missed_ticks"]==15
    assert r.last_decision["history_degraded"]
    assert (r.last_decision["quality"][167:175]==2).all()
    assert (r.last_decision["quality"]==3).sum()==10


def test_rebind_resets_and_invalid_identity_refuses():
    r=make()
    r.step(packet(0),17)
    r.step(packet(4,own_id=52),52)
    assert r.scheduler.stats["resets"]==2
    with pytest.raises(RuntimeError,match="uniquely bound"):
        r.step(packet(8,own_id=52),17)


def test_manifest_wrong_120hz_or_old_schema_refuses():
    m=manifest()
    m["contracts"]["policy_hz"]=120
    with pytest.raises(RuntimeError,match="contract mismatch"):
        runtime.EntityPacketRuntime(m,field_info(m),model=Counter())


def test_loaded_real_actor_runtime_finite_without_mutating_source():
    m=manifest()
    r=runtime.EntityPacketRuntime(m,field_info(m))
    for f in range(20):
        action=r.step(packet(f),17)
        assert np.isfinite(action).all()
    assert r.scheduler.stats["decisions"]==5
    assert r.summary()["packet_domain_exact"] is False
