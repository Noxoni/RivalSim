import copy
import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import run_direct_skills_exploration_continuation_v1 as run
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.report_direct_skills_exploration_continuation_v1 import validate_rows
from rivalsim import ssl_entity_mixed_training as mixed
from rivalsim.fresh_ground_30hz import content_hash


def test_only_continuation_not_learning_setting_change():
    spec = run.authority()
    old = run.prior.authority()
    for key in ('ppo', 'critic_lr', 'exploration_objective', 'reward', 'reward_sha256',
                'curriculum', 'opponents', 'architecture_change', 'fresh_optimizer', 'worlds',
                'physics_hz', 'policy_hz', 'training_temperature'):
        assert spec[key] == old[key], key
    assert spec['parent'] == run.PARENT and spec['parent']['accepted_updates'] == 680
    assert spec['child_updates'] == 70 and spec['evaluation_boundaries'] == [20, 45, 70]
    assert spec['original_control_parent'] == old['parent']


def test_context_binds_parent_paths_loss_and_factory_and_restores_on_exception():
    before = (engine.VERSION, engine.EXTERNAL, engine.selection, engine.make_collector,
              engine.validate_resume, mixed.joint_sequence_loss)
    with pytest.raises(RuntimeError, match='body'):
        with run.configured_engine():
            assert engine.VERSION == run.VERSION and engine.EXTERNAL == run.EXTERNAL
            assert engine.LIMIT == 70 and engine.EVALUATIONS == (20, 45, 70)
            assert engine.selection()['checkpoint'] == run.PARENT
            assert engine.make_collector is run.make_collector
            assert mixed.joint_sequence_loss is run.prior.frozen_loss
            assert engine.base.joint_sequence_loss is run.prior.frozen_loss
            raise RuntimeError('body')
    assert before == (engine.VERSION, engine.EXTERNAL, engine.selection, engine.make_collector,
                       engine.validate_resume, mixed.joint_sequence_loss)


def test_resume_is_explicit_latest_and_cannot_restart_closed_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'EXTERNAL', tmp_path)
    spec = run.authority()
    with run.configured_engine():
        source, digest, step = engine.validate_resume(None, None, spec)
        assert source == run.ROOT/run.PARENT['path'] and digest == run.PARENT['sha256'] and step == 0
        checkpoint = tmp_path/'rolling_0.pt'
        checkpoint.write_bytes(b'fixture')
        digest = run.sha(checkpoint)
        latest = dict(branch_updates=20, path=str(checkpoint), sha256=digest)
        (tmp_path/'latest.json').write_text(json.dumps(latest))
        state = dict(branch_updates=20, status='rollout', native_nexto_authority_sha256=content_hash(spec))
        (tmp_path/'campaign_state.json').write_text(json.dumps(state))
        assert engine.validate_resume(checkpoint, digest, spec)[2] == 20
        with pytest.raises(RuntimeError):
            engine.validate_resume(None, None, spec)
        state['status'] = 'complete_review'
        (tmp_path/'campaign_state.json').write_text(json.dumps(state))
        with pytest.raises(RuntimeError):
            engine.validate_resume(checkpoint, digest, spec)
        (tmp_path/'STOP').write_text('User stop')
        with pytest.raises(RuntimeError, match='STOP'):
            engine.validate_resume(checkpoint, digest, spec)


def valid_row(spec):
    return dict(branch_updates=1, accepted_updates=681, native_nexto_authority_sha256=content_hash(spec),
        training=dict(trainable_agent_samples=4423680, nexto_training_sample_count=1474560),
        ppo=dict(kl_rejections=0, completed_update_sample_kl_max=1e9))


def test_report_accepts_new_parent_count_and_large_kl():
    spec = run.authority()
    validate_rows([valid_row(spec)], 1, spec)


@pytest.mark.parametrize('fault', ['old_parent', 'gap', 'authority', 'samples', 'nexto', 'kl'])
def test_report_rejects_wrong_lineage_or_allocation(fault):
    spec = run.authority()
    row = copy.deepcopy(valid_row(spec))
    if fault == 'old_parent': row['accepted_updates'] = 651
    elif fault == 'gap': row['branch_updates'] = 2
    elif fault == 'authority': row['native_nexto_authority_sha256'] = 'old'
    elif fault == 'samples': row['training']['trainable_agent_samples'] = 1
    elif fault == 'nexto': row['training']['nexto_training_sample_count'] = 0
    elif fault == 'kl': row['ppo']['kl_rejections'] = 1
    with pytest.raises(ValueError):
        validate_rows([row], 1, spec)


def rng_fixture():
    identity = dict(version=engine.CONTROLLER, model_sha256='test-native-model', sampling_mode='native_v5')
    calls = []
    native = SimpleNamespace(generator=torch.Generator().manual_seed(1),
        checkpoint_state=lambda: identity, activate=lambda mask: calls.append(('activate', mask.numel())))
    collector = SimpleNamespace(env=SimpleNamespace(device='cpu'), nexto=native,
        opponent_generator=torch.Generator().manual_seed(2), nexto_sampling_mode='native_v5',
        nexto_seed=2026090691, is_nexto=torch.tensor([True, False]),
        assign=lambda mask: calls.append(('assign', mask.numel())))
    state = dict(version=engine.COLLECTOR_VERSION, controller_version=engine.CONTROLLER,
        nexto_sampling_mode='native_v5', nexto_seed=2026090691, nexto_probability=.5,
        generator_state=torch.Generator().manual_seed(3).get_state(),
        native_nexto=dict(identity, rng_state=torch.Generator().manual_seed(4).get_state(),
                          inference_calls=123, observation_builds=456,
                          controller={'old_shape': 32768}))
    return collector, state, calls


def test_rng_restore_preserves_streams_and_resets_only_fresh_worlds():
    collector, state, calls = rng_fixture()
    run.restore_new_episode_rng(collector, state)
    assert torch.equal(collector.opponent_generator.get_state(), state['generator_state'])
    assert torch.equal(collector.nexto.generator.get_state(), state['native_nexto']['rng_state'])
    assert collector.nexto.inference_calls == 123 and collector.nexto.observation_builds == 456
    assert calls == [('activate', 2), ('assign', 2)]


@pytest.mark.parametrize('fault', ['identity', 'counts', 'rng'])
def test_bad_rng_source_fails_before_live_streams_change(fault):
    collector, state, calls = rng_fixture()
    before = collector.opponent_generator.get_state().clone(), collector.nexto.generator.get_state().clone()
    if fault == 'identity': state['native_nexto']['model_sha256'] = 'wrong'
    elif fault == 'counts': state['native_nexto']['inference_calls'] = -1
    else: state['native_nexto']['rng_state'] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises((RuntimeError, ValueError)):
        run.restore_new_episode_rng(collector, state)
    assert torch.equal(before[0], collector.opponent_generator.get_state())
    assert torch.equal(before[1], collector.nexto.generator.get_state())
    assert not calls
