import ast
import json
from pathlib import Path

import pytest

from benchmarks import run_direct_skills_finishing_goal_v2 as recovery
from benchmarks import run_direct_skills_reset_recovery_v1 as prior


def test_frozen_branch_identity_and_learning_boundary():
    spec = recovery.authority()
    assert recovery.START == 625 and recovery.LIMIT == 25
    assert recovery.SOURCE.name == 'child_000025.pt'
    assert recovery.SOURCE_SHA == '8EBC09AF738CA89554854D9854EDAE46E01083A9B2841F9E0428D5A054E85AA3'
    assert recovery.CKPTS != prior.base.CHECKPOINTS and recovery.EXTERNAL != prior.base.EXTERNAL
    assert not spec['fresh_optimizer'] and spec['reward_change'] and not spec['architecture_change']
    assert recovery.ppo_config() == prior.ppo_config() and recovery.TEMPERATURE == 2
    assert recovery.load is prior.load
    assert recovery.base.mixed_joint_ppo_update is prior.base.mixed_joint_ppo_update


def test_same_bank_and_runtime_source_identity():
    assert recovery.scenario_hash(recovery.scenarios(1024)) == prior.scenario_hash(prior.scenarios(1024))
    fixed = recovery.runtime.verify(published=False)
    assert recovery.authority()['runtime_authority_sha256'] == recovery.runtime.digest(fixed)
    assert recovery.runtime.METHOD.endswith('HANDBRAKE_RESET_V3')


def test_actual_calls_not_mock_training_or_easier_eval():
    tree = ast.parse(Path(recovery.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    text = ast.unparse(run)
    assert 'bank = scenarios(32768)' in text
    assert "scenario_hash(bank) == package['scenario_sha256']" in text
    assert 'base.skill_evaluation(model)' in text and 'base.CandidateMatchRunner' in text
    assert 'base.mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)' in text
    assert 'while child < LIMIT' in text and 'if child == LIMIT:' in text
    assert 'runtime_package_sha256=runtime.digest(fixed)' in text
    assert 'match_reset_version=runtime.METHOD' in text
    assert 'exact_four_rng' in text and 'exact_optimizer_groups' in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'unlink' for n in ast.walk(run))


def test_fresh_branch_no_arbitrary_parent(tmp_path):
    assert recovery.validate_resume(None, None, tmp_path) == (recovery.SOURCE, recovery.SOURCE_SHA, 0)
    with pytest.raises(RuntimeError, match='frozen625'):
        recovery.validate_resume(tmp_path / 'other.pt', 'other', tmp_path)
    (tmp_path / 'campaign_state.json').write_text('{}')
    with pytest.raises(RuntimeError, match='operational audit'):
        recovery.validate_resume(None, None, tmp_path)


@pytest.fixture
def saved(tmp_path):
    checkpoint = tmp_path / 'rolling_0.pt'
    checkpoint.write_bytes(b'opaque metadata-only test checkpoint')
    digest = recovery.sha(checkpoint)
    latest = dict(branch_updates=4, path=str(checkpoint), sha256=digest)
    state = dict(branch_updates=4, status='rollout', finishing_authority_sha256=recovery.content_hash(recovery.authority()))
    def save():
        (tmp_path / 'latest.json').write_text(json.dumps(latest))
        (tmp_path / 'campaign_state.json').write_text(json.dumps(state))
    save()
    return checkpoint, digest, latest, state, save


def test_resume_exact_latest(saved, tmp_path):
    p, digest, latest, state, save = saved
    assert recovery.validate_resume(p, digest, tmp_path) == (p, digest, 4)
    state['branch_updates'] = 3
    save()
    assert recovery.validate_resume(p, digest, tmp_path)[2] == 4


@pytest.mark.parametrize('condition', ['stop', 'old_sha', 'other_path', 'corrupt', 'old_state', 'other_authority', 'complete', 'failure', 'out_of_range'])
def test_refuse_unsafe_resume(saved, tmp_path, condition):
    p, digest, latest, state, save = saved
    if condition == 'stop':
        (tmp_path / 'STOP').write_text('user stop')
    elif condition == 'old_sha':
        digest = 'wrong'
    elif condition == 'other_path':
        p = tmp_path / 'other.pt'
        p.write_bytes(b'opaque metadata-only test checkpoint')
    elif condition == 'corrupt':
        p.write_bytes(b'corrupt')
    elif condition == 'old_state':
        state['branch_updates'] = 1
    elif condition == 'other_authority':
        state['finishing_authority_sha256'] = 'wrong'
    elif condition == 'complete':
        state['status'] = 'complete_review'
    elif condition == 'failure':
        state['status'] = 'nonfinite_or_runtime_failure'
    else:
        latest['branch_updates'] = 26
    save()
    with pytest.raises(RuntimeError):
        recovery.validate_resume(p, digest, tmp_path)
    if condition == 'stop':
        assert (tmp_path / 'STOP').read_text() == 'user stop'


def test_only_reward_changes_and_native_paths_are_inherited():
    from rivalsim.direct_skills_v1 import DirectSkillsEnv
    from rivalsim.direct_skills_finishing_goal_v2 import FinishingGoalEnv
    spec = recovery.authority()
    assert spec['reward']['changed_role'] == 2
    assert spec['reward']['retired_finishing_rewards'] == dict(first_touch=0.,on_target_touch=0.,approach=0.)
    assert FinishingGoalEnv._step_impl is DirectSkillsEnv._step_impl
    assert recovery.base.DirectSkillCollector is prior.base.DirectSkillCollector
    assert recovery.CKPTS != prior.CKPTS and recovery.EXTERNAL != prior.EXTERNAL
    assert recovery.scenario_hash(recovery.scenarios(32768)) == json.loads((prior.OUT/'package.json').read_text())['scenario_sha256']


def test_real_parent_and_zero_step_metadata():
    import torch
    payload = torch.load(recovery.SOURCE,map_location='cpu',weights_only=False)
    assert recovery.sha(recovery.SOURCE) == recovery.SOURCE_SHA
    assert payload['accepted_updates'] == 625
    assert payload['cumulative_optimizer_steps'] == 141096
    assert payload['recovery_branch_updates'] == 25
    assert payload['recovery_authority_sha256'] == recovery.content_hash(prior.authority())
    assert payload['runtime_contract_hashes']['reward'] == recovery.content_hash(recovery.base.reward_authority())
    assert payload['ppo_config_sha256'] == recovery.ppo_config().content_hash
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {141096}
    tree = ast.parse(Path(recovery.__file__).read_text())
    preflight = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(preflight))
