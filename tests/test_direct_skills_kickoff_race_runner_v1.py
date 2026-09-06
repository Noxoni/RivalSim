import json

import pytest

from benchmarks import run_direct_skills_kickoff_race_v1 as run
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from rivalsim.direct_skills_kickoff_race_v1 import KickoffRaceEnv
from rivalsim.fresh_ground_30hz import content_hash


def test_frozen_budget_parent_and_other_learning_settings():
    a=run.authority()
    assert a['parent']['accepted_updates']==650
    assert a['parent']['sha256']=='939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D'
    assert a['child_updates']==15 and a['evaluation_boundaries']==[5,15]
    assert a['worlds']==32768 and a['policy_hz']==30 and a['physics_hz']==120
    assert a['training_temperature']==2
    assert a['ppo']['learning_rate']==1e-4 and a['ppo']['epochs']==2
    assert a['ppo']['rollout_horizon']==90
    assert a['curriculum']['family_distribution']['finishing']==.2
    assert a['opponents']['sampling_mode']=='native_v5'
    assert a['opponents']['nexto_world_fraction']==.5
    assert a['architecture_change'] is False and a['fresh_optimizer'] is False


def test_optin_binding_restores_historical_engine_even_on_error():
    keys=('OUT','EXTERNAL','VERSION','LIMIT','FinishingGoalEnv','authority','validate_resume')
    original={k:getattr(engine,k) for k in keys}
    with pytest.raises(RuntimeError,match='test body'):
        with run.configured_engine():
            assert engine.FinishingGoalEnv is KickoffRaceEnv
            assert engine.EXTERNAL==run.EXTERNAL and engine.OUT==run.OUT
            assert engine.LIMIT==15 and engine.EVALUATIONS==(5,15)
            assert engine.authority()==run.authority()
            raise RuntimeError('test body')
    for k,v in original.items():
        assert getattr(engine,k) is v


def test_resume_binds_new_directory_not_old_default(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    spec=run.authority()
    with run.configured_engine():
        assert engine.validate_resume(None,None,spec)==(run.ROOT/run.PARENT['path'],run.PARENT['sha256'],0)
        (tmp_path/'STOP').write_text('user stop')
        with pytest.raises(RuntimeError,match='STOP'):
            engine.validate_resume(None,None,spec)


def test_same_lineage_resume_and_completed_block_refusal(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    spec=run.authority()
    checkpoint=tmp_path/'rolling_1.pt';checkpoint.write_bytes(b'metadata fixture only')
    digest=run.sha(checkpoint)
    (tmp_path/'latest.json').write_text(json.dumps(dict(branch_updates=5,path=str(checkpoint),sha256=digest)))
    state=dict(branch_updates=5,status='rollout',native_nexto_authority_sha256=content_hash(spec))
    (tmp_path/'campaign_state.json').write_text(json.dumps(state))
    with run.configured_engine():
        assert engine.validate_resume(checkpoint,digest,spec)[2]==5
        state['status']='complete_review'
        (tmp_path/'campaign_state.json').write_text(json.dumps(state))
        with pytest.raises(RuntimeError,match='requires explicit audit'):
            engine.validate_resume(checkpoint,digest,spec)
