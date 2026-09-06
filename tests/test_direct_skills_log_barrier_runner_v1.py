import json

import pytest

from benchmarks import run_direct_skills_log_barrier_v1 as run
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from rivalsim import ssl_entity_mixed_training as mixed
from rivalsim.fresh_ground_30hz import content_hash


def test_prospective_only_exploration_change_and_boundaries():
    spec=run.authority()
    old=run.prior.authority()
    for key in ('parent','ppo','reward','reward_sha256','curriculum','opponents','architecture_change','fresh_optimizer'):
        assert spec[key]==old[key]
    assert spec['child_updates']==30 and spec['evaluation_boundaries']==[5,15,30]
    assert spec['exploration_objective']['coefficient']==.01
    assert spec['curriculum']['family_distribution']['finishing']==.2


def test_runtime_loss_and_preflight_loss_both_bound_and_restored():
    before=(mixed.joint_sequence_loss,engine.base.joint_sequence_loss,engine.VERSION,engine.EXTERNAL,engine.validate_resume)
    with pytest.raises(RuntimeError,match='body'):
        with run.configured_engine():
            assert mixed.joint_sequence_loss is run.frozen_loss
            assert engine.base.joint_sequence_loss is run.frozen_loss
            assert engine.VERSION==run.VERSION and engine.EXTERNAL==run.EXTERNAL
            assert engine.LIMIT==30
            raise RuntimeError('body')
    assert before==(mixed.joint_sequence_loss,engine.base.joint_sequence_loss,engine.VERSION,engine.EXTERNAL,engine.validate_resume)


def test_resume_uses_new_directory_and_refuses_complete_or_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    spec=run.authority()
    with run.configured_engine():
        assert engine.validate_resume(None,None,spec)[2]==0
        checkpoint=tmp_path/'rolling_1.pt';checkpoint.write_bytes(b'fixture')
        digest=run.sha(checkpoint)
        (tmp_path/'latest.json').write_text(json.dumps(dict(branch_updates=15,path=str(checkpoint),sha256=digest)))
        state=dict(branch_updates=15,status='rollout',native_nexto_authority_sha256=content_hash(spec))
        (tmp_path/'campaign_state.json').write_text(json.dumps(state))
        assert engine.validate_resume(checkpoint,digest,spec)[2]==15
        state['status']='complete_review';(tmp_path/'campaign_state.json').write_text(json.dumps(state))
        with pytest.raises(RuntimeError):engine.validate_resume(checkpoint,digest,spec)
        (tmp_path/'STOP').write_text('user stop')
        with pytest.raises(RuntimeError,match='STOP'):engine.validate_resume(checkpoint,digest,spec)
