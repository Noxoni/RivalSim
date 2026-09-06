import copy
import json

import pytest
import torch

from benchmarks import run_direct_skills_exploration_ablation_v1 as run
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.report_direct_skills_exploration_ablation_v1 import objective_checks,validate_rows
from rivalsim import ssl_entity_mixed_training as mixed
from rivalsim.fresh_ground_30hz import content_hash,ppo_config
from rivalsim.ssl_entity_training import joint_sequence_loss


def fixture():
    torch.manual_seed(42)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(5,7)
            self.critic = torch.nn.Linear(5,1)

        def forward(self,obs,hidden,reset_before):
            return self.actor(obs),self.critic(obs).squeeze(-1),hidden
    model = Policy()
    data = dict(observations=torch.randn(3,4,5),initial_hidden=torch.zeros(1,3,2),
        action_indices=torch.randint(0,7,(3,4)),reset_before=torch.zeros(3,4,dtype=torch.bool),
        train_mask=torch.tensor([[True]*4,[False]*4,[True,True,False,True]]),
        normalized_advantage=torch.randn(3,4),returns=torch.randn(3,4))
    logits,_,_ = model(data['observations'],data['initial_hidden'],data['reset_before'])
    data['old_log_probability'] = logits.detach().log_softmax(-1).gather(-1,data['action_indices'][...,None]).squeeze(-1)
    return model,data,torch.arange(3)


def test_matched_frozen_settings_only_coefficient_differs():
    a,b = [run.authority(arm) for arm in run.ARMS]
    for key in ('parent','ppo','critic_lr','reward','reward_sha256','curriculum','opponents',
                'architecture_change','fresh_optimizer','worlds','physics_hz','policy_hz',
                'training_temperature','scenario_seed','evaluation_spec_sha256','initialization',
                'comparison_authority_sha256','decision_evidence'):
        assert a[key] == b[key],key
    assert a['parent']['accepted_updates'] == 750
    assert a['child_updates'] == b['child_updates'] == 30
    assert a['evaluation_boundaries'] == b['evaluation_boundaries'] == [10,30]
    assert a['exploration_objective']['coefficient'] == .01
    assert b['exploration_objective']['coefficient'] == 0
    b['exploration_objective']['coefficient'] = .01
    assert a['exploration_objective'] == b['exploration_objective']
    assert run.comparison_authority()['maximum_total_learner_decisions'] == 265420800


@pytest.mark.parametrize('arm',run.ARMS)
def test_context_is_scoped_and_restored(arm):
    before = (engine.VERSION,engine.EXTERNAL,engine.selection,engine.make_collector,
              engine.validate_resume,mixed.joint_sequence_loss,engine.base.joint_sequence_loss)
    with pytest.raises(RuntimeError,match='body'):
        with run.configured_engine(arm):
            assert engine.EXTERNAL == run.EXTERNAL/arm and engine.LIMIT == 30
            assert engine.EVALUATIONS == (10,30)
            assert engine.selection()['checkpoint'] == run.PARENT
            assert engine.make_collector is run.make_collector
            model,data,index = fixture()
            _,metrics = engine.base.joint_sequence_loss(model,data,index,ppo_config())
            assert float(metrics['exploration_coefficient']) == pytest.approx(run.ARMS[arm])
            assert mixed.joint_sequence_loss is engine.base.joint_sequence_loss
            raise RuntimeError('body')
    assert before == (engine.VERSION,engine.EXTERNAL,engine.selection,engine.make_collector,
                      engine.validate_resume,mixed.joint_sequence_loss,engine.base.joint_sequence_loss)


def test_withdraw_exact_original_loss_gradient_and_critic_parity():
    model,data,index = fixture()
    original,_ = joint_sequence_loss(model,data,index,ppo_config())
    withdrawn,metrics = run.loss_for('withdraw')(model,data,index,ppo_config())
    retained,_ = run.loss_for('retain')(model,data,index,ppo_config())
    assert torch.equal(original,withdrawn)
    params = tuple(model.parameters())
    ga = torch.autograd.grad(original,params,retain_graph=True)
    gb = torch.autograd.grad(withdrawn,params,retain_graph=True)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))
    critic = torch.autograd.grad(retained-withdrawn,tuple(model.critic.parameters()))
    assert all(not bool(g.any()) for g in critic)
    assert float(metrics['weighted_exploration_barrier']) == 0


@pytest.mark.parametrize('arm',run.ARMS)
def test_resume_rejects_cross_arm_closed_arm_and_stop(arm,tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    spec = run.authority(arm)
    directory = tmp_path/arm
    directory.mkdir()
    with run.configured_engine(arm):
        assert engine.validate_resume(None,None,spec) == (run.ROOT/run.PARENT['path'],run.PARENT['sha256'],0)
        path = directory/'rolling_0.pt'
        path.write_bytes(b'checkpoint fixture')
        digest = run.sha(path)
        latest = dict(branch_updates=10,path=str(path),sha256=digest)
        (directory/'latest.json').write_text(json.dumps(latest))
        state = dict(branch_updates=10,status='rollout',native_nexto_authority_sha256=content_hash(spec))
        (directory/'campaign_state.json').write_text(json.dumps(state))
        assert engine.validate_resume(path,digest,spec)[2] == 10
        other = run.authority('withdraw' if arm == 'retain' else 'retain')
        with pytest.raises(RuntimeError):
            engine.validate_resume(path,digest,other)
        state['status'] = 'complete_review'
        (directory/'campaign_state.json').write_text(json.dumps(state))
        with pytest.raises(RuntimeError):
            engine.validate_resume(path,digest,spec)
        (directory/'STOP').write_text('stop')
        with pytest.raises(RuntimeError,match='STOP'):
            engine.validate_resume(path,digest,spec)


def row_for(spec):
    beta = spec['exploration_objective']['coefficient']
    return dict(branch_updates=1,accepted_updates=751,native_nexto_authority_sha256=content_hash(spec),
        training=dict(trainable_agent_samples=4423680,nexto_training_sample_count=1474560),
        ppo=dict(kl_rejections=0,exploration_coefficient=beta,exploration_barrier=3.,
            weighted_exploration_barrier=beta*3,entropy=1.,completed_update_mean_kl=1e3,
            completed_update_sample_kl_max=1e9))


@pytest.mark.parametrize('arm',run.ARMS)
def test_report_accepts_exact_objective_and_large_finite_kl(arm):
    spec = run.authority(arm)
    row = row_for(spec)
    validate_rows([row],1,spec)
    payload = dict(native_nexto_package=dict(exploration_objective=spec['exploration_objective']))
    assert all(objective_checks(payload,[row],spec).values())


@pytest.mark.parametrize('fault',['coefficient','weighted','missing','nan','wrong_parent','gap','kl_rejection'])
def test_report_rejects_objective_or_lineage_errors(fault):
    spec = run.authority('withdraw')
    row = copy.deepcopy(row_for(spec))
    payload = dict(native_nexto_package=dict(exploration_objective=spec['exploration_objective']))
    if fault == 'coefficient': row['ppo']['exploration_coefficient'] = .01
    elif fault == 'weighted': row['ppo']['weighted_exploration_barrier'] = .001
    elif fault == 'missing': del row['ppo']['exploration_barrier']
    elif fault == 'nan': row['ppo']['completed_update_mean_kl'] = float('nan')
    elif fault == 'wrong_parent': row['accepted_updates'] = 681
    elif fault == 'gap': row['branch_updates'] = 2
    else: row['ppo']['kl_rejections'] = 1
    with pytest.raises(ValueError):
        validate_rows([row],1,spec)
        objective_checks(payload,[row],spec)


def test_unknown_arm_fails():
    with pytest.raises(ValueError):
        run.authority('third_sweep')


@pytest.mark.parametrize('status',['rollout','nonfinite_or_runtime_failure','stopped_at_accepted_boundary'])
def test_pair_never_restarts_unfinished_or_failed_worker(status,tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    monkeypatch.setattr(run,'verify',lambda arm: {})
    (tmp_path/'retain').mkdir()
    (tmp_path/'retain'/'campaign_state.json').write_text(json.dumps(dict(status=status,branch_updates=12)))
    launches = []
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k: launches.append(a))
    with pytest.raises(RuntimeError,match='Existing unfinished arm'):
        run.pair_run()
    assert not launches


def test_pair_stop_prevents_first_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    monkeypatch.setattr(run,'verify',lambda arm: {})
    (tmp_path/'STOP').write_text('User stop')
    launches = []
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k: launches.append(a))
    with pytest.raises(RuntimeError,match='Respect pair STOP'):
        run.pair_run()
    assert not launches


def test_worker_failure_never_launches_second_arm(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'EXTERNAL',tmp_path)
    monkeypatch.setattr(run,'verify',lambda arm: {})
    launches = []

    class Failed:
        pid,returncode = 123,1

        def poll(self):
            return self.returncode

    def spawn(command,**kwargs):
        launches.append(command)
        return Failed()
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError,match='Arm failed'):
        run.pair_run()
    assert len(launches) == 1 and launches[0][-1] == 'retain'
    assert json.loads((tmp_path/'pair_state.json').read_text())['status'] == 'worker_failed'
