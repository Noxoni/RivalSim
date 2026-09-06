import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_direct_skills_shooting_progress_v1 as current
from benchmarks import run_direct_skills_exploration_followup_v1 as prior
from rivalsim.direct_skills_shooting_curriculum_v1 import build, curriculum_authority


@pytest.mark.parametrize('worlds',[64,1024,32768])
def test_only_half_finishing_opponent_geometry_changes(worlds):
    a = prior.base.scenarios(worlds)
    b,audit = build(worlds)
    assert audit['changed_count'] == int((a.family==2).sum())//2
    changed = np.zeros(worlds,dtype=bool);changed[audit['changed_rows']]=True
    assert (a.family[changed]==2).all()
    assert np.array_equal(a.family,b.family) and np.array_equal(a.focal_side,b.focal_side)
    for name in ('kickoff_indicator','kickoff_layout','wall_aerial_variant'):
        assert np.array_equal(getattr(a,name),getattr(b,name))
    rows = np.arange(worlds);side = a.focal_side
    for name in a.state.__dataclass_fields__:
        old,new = getattr(a.state,name),getattr(b.state,name)
        if name in ('car_pos','car_vel','car_quat'):
            assert np.array_equal(old[~changed],new[~changed]),name
            assert np.array_equal(old[rows,side],new[rows,side]),name
        else:
            assert np.array_equal(old,new),name
    b.state.validate()
    selected=np.array(audit['changed_rows']);opponent=1-side[selected]
    pos=b.state.car_pos[selected,opponent]
    assert (np.linalg.norm(pos-b.state.ball_pos[selected],axis=-1)>=219.99).all()
    assert (np.linalg.norm(pos-b.state.car_pos[selected,side[selected]],axis=-1)>=239.99).all()
    assert (np.abs(pos[:,:2])<np.array([3600,4900])).all()
    assert np.allclose(np.linalg.norm(b.state.car_vel[selected,opponent],axis=-1),600)
    assert min(audit['alpha'])>=.35 and max(audit['alpha'])<=.85


def test_curriculum_deterministic_and_no_reward_or_task_id():
    a,x=build(1024);b,y=build(1024)
    assert x==y and current.scenario_hash(a)==current.scenario_hash(b)
    spec=curriculum_authority()
    assert not spec['reward_change'] and not spec['ppo_change'] and not spec['task_id_in_observation']
    assert not spec['scripted_actions'] and not spec['passive_or_disabled_opponent']


def functions(module):
    return {n.name:n for n in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(n,ast.FunctionDef)}


def test_original_loader_and_complete_evaluation_body_preserved():
    a,b=functions(prior),functions(current)
    assert ast.unparse(a['load'])==ast.unparse(b['load'])
    ea=next(n for n in a['run'].body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    eb=next(n for n in b['run'].body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    assert ast.unparse(ea)==ast.unparse(eb)


def test_same_collector_ppo_config_and_actual_call_sites():
    assert current.TrainingExplorationPolicy is prior.TrainingExplorationPolicy
    assert current.base.DirectSkillCollector is prior.base.DirectSkillCollector
    assert current.base.DirectSkillsEnv is prior.base.DirectSkillsEnv
    assert current.base.mixed_joint_ppo_update is prior.base.mixed_joint_ppo_update
    assert current.ppo_config()==prior.ppo_config() and current.TEMPERATURE==2
    def calls(module):
        return [ast.unparse(n) for n in ast.walk(functions(module)['run'])
                if isinstance(n,ast.Call) and ast.unparse(n.func) in
                ('collector.collect','base.mixed_joint_ppo_update')]
    assert len(calls(current))==2 and calls(current)==calls(prior)
    assert current.START==650 and current.REVIEW==675
    assert current.PRIOR_AMENDMENT_SHA==current.content_hash(prior.amendment())
    assert not current.amendment()['reward_change']


def test_actual_training_uses_new_bank_and_original_evaluation_bank():
    run=ast.unparse(functions(current)['run'])
    assert 'bank = training_scenarios(32768)' in run
    assert "scenario_hash(bank) == package['scenario_sha256']" in run
    assert 'base.skill_evaluation(model)' in run
    assert 'training_curriculum_authority_sha256' in run
    assert 'effective_training_scenario_sha256' in run
    assert 'shooting_entry_000650.pt' in run and 'shooting_resume_000651.pt' in run


@pytest.fixture
def saved(tmp_path):
    p=tmp_path/'latest.pt';p.write_bytes(b'opaque loader fixture')
    digest=hashlib.sha256(p.read_bytes()).hexdigest().upper()
    latest=dict(accepted_updates=650,sha256=digest)
    state=dict(accepted_updates=650,status='stopped_for_exploration_review',
               exploration_amendment_sha256=current.PRIOR_AMENDMENT_SHA)
    def save():
        (tmp_path/'latest.json').write_text(json.dumps(latest))
        (tmp_path/'campaign_state.json').write_text(json.dumps(state))
    save()
    return p,digest,latest,state,save


def test_entry_accepts_exact650_not_old_checkpoint(saved,tmp_path):
    p,digest,latest,_,_=saved
    assert current.validate_latest(p,digest,external=tmp_path)==latest
    with pytest.raises(RuntimeError,match='latest accepted'):
        current.validate_latest(p,'wrong SHA',external=tmp_path)


@pytest.mark.parametrize('offset,state_offset,ancestry',[
    (650,650,True),(650,650,False),(651,650,True),(651,650,False),
    (664,663,False),(675,674,False)])
def test_published_recovery_windows(saved,tmp_path,offset,state_offset,ancestry):
    p,digest,latest,state,save=saved
    latest['accepted_updates']=offset
    state.update(accepted_updates=state_offset,status='failed',
                 exploration_amendment_sha256=current.PRIOR_AMENDMENT_SHA if ancestry
                 else current.content_hash(current.amendment()))
    save()
    assert current.validate_latest(p,digest,external=tmp_path)==latest


@pytest.mark.parametrize('failure',['user_stop','wrong_authority','stale_state','completed','out_of_range','corruption'])
def test_no_unsafe_resume(saved,tmp_path,failure):
    p,digest,latest,state,save=saved
    latest['accepted_updates']=state['accepted_updates']=674
    state.update(status='failed',exploration_amendment_sha256=current.content_hash(current.amendment()))
    if failure=='user_stop':
        (tmp_path/'STOP').write_text('user stop')
    elif failure=='wrong_authority':
        state['exploration_amendment_sha256']=current.PRIOR_AMENDMENT_SHA
    elif failure=='stale_state':
        state['accepted_updates']=660
    elif failure=='completed':
        latest['accepted_updates']=state['accepted_updates']=675
        state['status']='stopped_for_curriculum_review'
    elif failure=='out_of_range':
        latest['accepted_updates']=676
    else:
        p.write_bytes(b'corrupt')
    save()
    with pytest.raises(RuntimeError):
        current.validate_latest(p,digest,external=tmp_path)
    if failure=='user_stop':
        assert (tmp_path/'STOP').read_text()=='user stop'
