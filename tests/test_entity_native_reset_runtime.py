from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
sys.path.insert(0,str(ROOT/"deployment/entity_native_v2"))
from test_entity_native_packet import Counter,field_info,make,manifest,packet
from reset_runtime import ResetAlignedPacketRuntime,VERSION,sha


def make_v2(*,real=False):
    base=manifest()
    path=ROOT/"results/rival2/entity_native_packet_v1/reference600_manifest.json"
    m=dict(format=VERSION,base_manifest=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)),source=base["source"],artifact=base["artifact"])
    return ResetAlignedPacketRuntime(m,field_info(base),model=None if real else Counter())


def test_only_eight_fields_change_after_actual_countdown():
    old,new=make(),make_v2()
    for r in (old,new):
        r.step(packet(0,"Countdown"),17)
        r.step(packet(1,"Kickoff"),17)
    a,b=old.last_decision,new.last_decision
    wheels=np.r_[35:39,74:78];rest=np.setdiff1d(np.arange(182),wheels)
    np.testing.assert_array_equal(a["observation"][rest],b["observation"][rest])
    np.testing.assert_array_equal(a["quality"],b["quality"])
    assert np.all(b["observation"][wheels]==0) and np.all(b["quality"][wheels]==3)
    assert b["reset_projection"] and new.projection_count==1
    for f in range(2,6):
        old.step(packet(f),17);new.step(packet(f),17)
    np.testing.assert_array_equal(old.last_decision["observation"],new.last_decision["observation"])
    assert not new.last_decision["reset_projection"]


def test_midplay_or_midkickoff_attach_cannot_project():
    for phase in ("Active","Kickoff"):
        r=make_v2();r.step(packet(40,phase),17)
        assert not r.last_decision["reset_projection"]
        r.step(packet(44,phase,own_id=52),52)
        assert not r.last_decision["reset_projection"]


def test_duplicate_backward_pause_do_not_rearm_projection():
    r=make_v2();r.step(packet(0,"Countdown"),17);r.step(packet(1),17)
    r.step(packet(1),17);r.step(packet(0),17)
    r.step(packet(2,"Paused"),17);r.step(packet(60,"Paused"),17)
    r.step(packet(61),17)
    decisions=[]
    for f in range(62,70):
        r.step(packet(f),17)
        if r.last_decision is not None:decisions.append(r.last_decision)
    assert decisions and all(not d["reset_projection"] for d in decisions)
    assert r.projection_count==1


def test_new_goal_countdown_projects_again_once():
    r=make_v2();r.step(packet(0,"Countdown"),17);r.step(packet(1),17)
    r.step(packet(4,"Active"),17)
    r.step(packet(8,"GoalScored",(1,0)),17)
    r.step(packet(9,"Countdown",(1,0)),17)
    r.step(packet(10,"Kickoff",(1,0)),17)
    assert r.last_decision["reset_projection"] and r.projection_count==2


def test_skipped_kickoff_or_rebind_drops_pending_projection():
    r=make_v2();r.step(packet(0,"Countdown"),17);r.step(packet(40,"Active"),17)
    assert not r.last_decision["reset_projection"]
    r.step(packet(41,"Countdown"),17);r.step(packet(42,"Kickoff",own_id=52),52)
    assert not r.last_decision["reset_projection"]


def test_exact_export_runs_with_projected_input_and_finite_state():
    r=make_v2(real=True)
    r.step(packet(0,"Countdown"),17)
    action=r.step(packet(1),17)
    assert r.last_decision["reset_projection"] and np.isfinite(action).all()
    for f in range(2,14):assert np.isfinite(r.step(packet(f),17)).all()
    assert r.projection_count==1
