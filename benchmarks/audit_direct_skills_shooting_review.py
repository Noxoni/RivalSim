"""CPU-only review of the frozen650->675 curriculum; no policy or optimizer execution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from benchmarks import run_direct_skills_shooting_progress_v1 as run  # noqa: E402
from benchmarks.audit_direct_skills_shooting_entry import exact, finite  # noqa: E402
from rivalsim.direct_skills_shooting_curriculum_v1 import curriculum_authority  # noqa: E402


def validate(parent, candidate, rows, authority_hash, package):
    offset = candidate['accepted_updates']
    checks = dict(
        offset=650 < offset <= 675,
        contiguous=[r['accepted_updates'] for r in rows] == list(range(651, offset + 1)),
        authority=candidate['exploration_amendment_sha256'] == authority_hash,
        source650=candidate['exploration_parent_sha256'] == run.SOURCE_SHA,
        ancestor=candidate['parent_sha256'] == parent['parent_sha256'] == run.base.PARENT_SHA,
        curriculum=candidate['training_curriculum_authority_sha256'] == run.content_hash(curriculum_authority()),
        effective_scenario=candidate['effective_training_scenario_sha256'] == package['scenario_sha256'],
        complete_package=candidate['exploration_package'] == package,
        same_root_package=candidate['package'] == parent['package'],
        contracts=all(exact(candidate[k], parent[k]) for k in (
            'runtime_contract_hashes', 'ppo_config_sha256', 'policy_config_sha256',
            'action_contract', 'observation_schema_sha256', 'reward_authority')),
        model_schema=candidate['model'].keys() == parent['model'].keys() and all(
            v.shape == parent['model'][k].shape and v.dtype == parent['model'][k].dtype
            for k, v in candidate['model'].items()),
        finite_model=finite(candidate['model']),
        finite_adam=finite(candidate['optimizer']),
        no_fresh_optimizer=not candidate['fresh_optimizer'],
        optimizer_groups=exact(candidate['optimizer']['param_groups'], parent['optimizer']['param_groups']),
        adam_count=candidate['cumulative_optimizer_steps'] == parent['cumulative_optimizer_steps']
            + sum(r['ppo']['optimizer_steps'] for r in rows),
        adam_uniform={int(s['step']) for s in candidate['optimizer']['state'].values()}
            == {candidate['cumulative_optimizer_steps']},
        samples=candidate['direct_skill_samples'] == parent['direct_skill_samples'] + (offset-650)*4423680,
        physics=candidate['direct_skill_physics_ticks'] == parent['direct_skill_physics_ticks'] + (offset-650)*11796480,
        curve_authority=all(r['exploration_amendment_sha256'] == authority_hash for r in rows),
        on_policy_temperature=candidate['training_distribution']['temperature'] == 2
            and candidate['training_distribution']['on_policy_sampling_and_likelihood']
            and all(r['training_temperature'] == 2 for r in rows),
        exact_nexto_third=all(r['training']['nexto_training_sample_count']*3
            == r['training']['trainable_agent_samples'] == 4423680 for r in rows),
        finite_ppo=all(math.isfinite(v) for r in rows for v in r['ppo'].values() if isinstance(v, (int, float))),
        no_kl_rejection=all(r['ppo']['kl_rejections'] == 0 for r in rows),
        four_rng=all(torch.is_tensor(candidate[k]) and candidate[k].dtype == torch.uint8
            and candidate[k].numel() > 0 for k in ('policy_generator_state', 'shuffle_generator_state',
            'torch_cpu_rng_state', 'torch_cuda_rng_state')),
    )
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    return checks


def aggregate(rows):
    assert rows and all(rows[i+1]['accepted_updates'] == rows[i]['accepted_updates']+1 for i in range(len(rows)-1))
    samples = sum(r['training']['trainable_agent_samples'] for r in rows)
    roles = {}
    for row in rows:
        for name, values in row['training']['by_reward_role_and_opponent'].items():
            totals = roles.setdefault(name, {k: 0. for k in values})
            assert totals.keys() == values.keys()
            for key, value in values.items():
                totals[key] += value
    assert sum(v['samples'] for v in roles.values()) == samples
    return dict(
        updates=[rows[0]['accepted_updates'], rows[-1]['accepted_updates']],
        samples=samples,
        contacts_per_learner_minute=sum(r['training']['touches'] for r in rows)/samples*30*60,
        movement_speed=sum(r['training']['speed'] for r in rows)/samples,
        entropy=sum(r['training']['entropy'] for r in rows)/samples,
        ended_episode_contact_fraction=sum(r['training']['episodes_with_touch'] for r in rows)
            /sum(r['training']['ended_player_episodes'] for r in rows),
        no_touch_truncations=sum(r['training']['no_touch'] for r in rows),
        roles={name: dict(raw=v, per_million_role_decisions={k: n/v['samples']*1e6
               for k, n in v.items() if k != 'samples'}) for name, v in roles.items()},
    )


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Never replace different evidence: {path}'
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def build(offset=675):
    _, package = run.verify()
    path = run.base.CHECKPOINTS / f'plus_{offset:06d}.pt'
    digest = run.sha(path)
    parent = torch.load(run.SOURCE, map_location='cpu', weights_only=False)
    candidate = torch.load(path, map_location='cpu', weights_only=False)
    raw = (run.base.EXTERNAL/'training_curve.jsonl').read_bytes().splitlines(keepends=True)
    all_rows = [json.loads(s) for s in raw if s.endswith(b'\n')]
    all_rows = [r for r in all_rows if r['accepted_updates'] <= offset]
    assert [r['accepted_updates'] for r in all_rows] == list(range(1, offset+1))
    rows = [r for r in all_rows if r['accepted_updates'] > 650]
    checks = validate(parent, candidate, rows, run.content_hash(run.amendment()), package)
    checks['source_file'] = run.sha(run.SOURCE) == run.SOURCE_SHA
    checks['stable_checkpoint_read'] = run.sha(path) == digest
    assert all(checks.values())
    curve_path = run.OUT/f'training_curve_000651_to_{offset:06d}.jsonl'
    encoded = ''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows)
    if curve_path.exists():
        assert curve_path.read_text() == encoded
    else:
        curve_path.write_text(encoded)
    result = dict(
        checks=checks, accepted_updates=offset, checkpoint=path.relative_to(ROOT).as_posix(),
        sha256=digest, authority_sha256=run.content_hash(run.amendment()),
        direct_skill_samples=candidate['direct_skill_samples'],
        direct_skill_physics_ticks=candidate['direct_skill_physics_ticks'],
        cumulative_optimizer_steps=candidate['cumulative_optimizer_steps'],
        new_optimizer_steps=sum(r['ppo']['optimizer_steps'] for r in rows),
        curve=curve_path.relative_to(ROOT).as_posix(), curve_text_sha256=run.base.text_sha(curve_path),
        max_completed_update_mean_kl=max(r['ppo']['completed_update_mean_kl'] for r in rows),
        max_sample_kl=max(r['ppo']['completed_update_sample_kl_max'] for r in rows),
        peak_cuda_bytes=max(r['cuda_peak_allocated_bytes'] for r in rows),
        optimizer_steps_in_audit=0, meaning='Integrity only; no capability conclusion or KL acceptance gate',
    )
    freeze(run.OUT/f'review_{offset:06d}.json', result)
    if offset == 675:
        reduction = dict(
            previous=aggregate([r for r in all_rows if 602 <= r['accepted_updates'] <= 650]),
            current=aggregate([r for r in all_rows if 652 <= r['accepted_updates'] <= 675]),
            first_half=aggregate([r for r in all_rows if 652 <= r['accepted_updates'] <= 663]),
            second_half=aggregate([r for r in all_rows if 664 <= r['accepted_updates'] <= 675]),
            meaning='Fresh-resume buffers601 and651 excluded only from these summaries, preserved in full curves. '
            'Rates use actual role decisions. Current curriculum is easier on half finishing starts; '
            'cross-curriculum rate changes are NOT proof of learning. Fixed original evaluation judges transfer.',
            optimizer_steps_in_reduction=0,
        )
        freeze(run.base.RESULTS/f'review_reduction_{offset:06d}.json', reduction)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, default=675)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(build(args.update), indent=2))
