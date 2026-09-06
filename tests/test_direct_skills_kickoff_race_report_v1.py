import json

import pytest

from benchmarks import report_direct_skills_kickoff_race_v1 as report


def test_actual_first_step_identity_and_frozen_reward_transition():
    data=json.loads((report.OUT/'progress_000001.json').read_text())
    assert len(data['checks'])==26 and all(data['checks'].values())
    assert data['optimizer_steps_in_audit']==0 and data['audit_device']=='cpu'
    for key in ('parent','entry','checkpoint'):
        item=data[key]
        assert report.sha(report.ROOT/item['path'])==item['sha256']
    assert data['training']['learner_decisions']==4423680
    assert data['training']['physics_ticks']==11796480
    assert data['training']['optimizer_steps']>0
    assert data['training']['kl_rejections']==0
    spec=json.loads((report.OUT/'training_authority.json').read_text())
    assert data['authority_sha256']==report.canonical(spec)


def test_deterministic_cpu_report_rebuild():
    path=report.OUT/'progress_000001.json'
    before=path.read_bytes()
    report.audit(1)
    assert before==path.read_bytes()


def test_report_rejects_unfrozen_boundary():
    with pytest.raises(ValueError,match='boundary'):
        report.audit(3)
