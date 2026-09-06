"""Bounded, no-update reward/advantage/action-response diagnostic.

Never call an optimizer or change a campaign. Compare preserved parent and child
on identical parent-generated trajectories, not two diverging rollouts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.run_rival2_direct_skills_v1 import COLLISION, SEED, gpu_lease
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash
from rivalsim.direct_skills_exploration_v1 import TrainingExplorationPolicy
from rivalsim.direct_skills_kickoff_race_v1 import KickoffRaceEnv, scenarios
from rivalsim.direct_skills_native_nexto import DirectSkillNativeNextoCollector
from rivalsim.direct_skills_v1 import NAMES, player_roles
from rivalsim.fresh_ground_30hz import ppo_config, scenario_hash
from rivalsim.rival2_recurrent_ppo import _sequence_major
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data

OUT = ROOT / 'results/rival2/direct_skills_learning_diagnostic_v1'
PARENT = ROOT / 'checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt'
CHILD = ROOT / 'checkpoints/rival2/direct_skills_kickoff_race_v1/child_000015.pt'
IDENTITIES = {
    str(PARENT.relative_to(ROOT)): '939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D',
    str(CHILD.relative_to(ROOT)): '183C59CA2F7C04681C5EC92272F63A8494223BEC422B88F74AA48F23603B7957',
}
WORLDS = 2048


def save_json(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'Refuse different evidence: {path}'
    else:
        path.write_bytes(data)


def aggregate(rows, lower, upper):
    selected = [r for r in rows if lower <= r['branch_updates'] <= upper]
    assert len(selected) == upper - lower + 1
    result = {}
    for key in selected[0]['training']['by_reward_role_and_opponent']:
        entries = [r['training']['by_reward_role_and_opponent'][key] for r in selected]
        totals = {k: sum(x[k] for x in entries) for k in entries[0]}
        totals['score_fraction_of_ended_episodes'] = totals['goals'] / max(totals['episode_endings'], 1)
        totals['task_success_fraction_of_ended_episodes'] = totals['success_endings'] / max(totals['episode_endings'], 1)
        totals['contacts_per_1800_learner_decisions'] = 1800 * totals['touches'] / max(totals['samples'], 1)
        result[key] = totals
    return result


def describe(values):
    values = values.detach().double().flatten()
    if values.numel() == 0:
        return dict(count=0, mean=None, min=None, max=None, positive_fraction=None)
    return dict(count=values.numel(), mean=float(values.mean()), min=float(values.min()),
                max=float(values.max()), positive_fraction=float((values > 0).double().mean()))


@torch.no_grad()
def replay(model, data):
    result = {k: [] for k in ('logp', 'entropy', 'max_probability', 'jump_probability', 'argmax', 'value')}
    for start in range(0, data['observations'].shape[0], 32):
        end = start + 32
        logits, value, _ = model(data['observations'][start:end], data['initial_hidden'][:, start:end],
                                 reset_before=data['reset_before'][start:end])
        logp = logits.log_softmax(-1)
        probability = logp.exp()
        result['logp'].append(logp.gather(-1, data['action_indices'][start:end, :, None]).squeeze(-1))
        result['entropy'].append(-(probability * logp).sum(-1))
        result['max_probability'].append(probability.max(-1).values)
        result['jump_probability'].append((probability * model.action_table[:, 5]).sum(-1))
        result['argmax'].append(logits.argmax(-1))
        result['value'].append(value)
    return {k: torch.cat(v) for k, v in result.items()}


def summarize(data, before, after, roles, events, goals, mask):
    selected = data['train_mask'] & mask
    adv = data['normalized_advantage']
    delta = after['logp'] - before['logp']
    rows = {}
    for family in (0, 1):
        for role, name in enumerate(NAMES):
            subset = selected & (data['opponent_family'] == family) & (roles == role)
            if not bool(subset.any()):
                continue
            groups = {'all': subset, 'positive_advantage': subset & (adv > 0),
                      'negative_advantage': subset & (adv < 0),
                      'paid_first_touch': subset & events,
                      'native_score': subset & (goals == 1), 'native_concede': subset & (goals == -1)}
            rows[f'{name}_{("selfplay", "nexto")[family]}'] = {
                group: dict(advantage=describe(adv[choose]), raw_advantage=describe(data['advantages'][choose]),
                            sampled_action_logp_change=describe(delta[choose]),
                            advantage_times_logp_change=describe((adv * delta)[choose]),
                            parent_entropy=describe(before['entropy'][choose]),
                            child_entropy=describe(after['entropy'][choose]),
                            parent_jump_probability=describe(before['jump_probability'][choose]),
                            child_jump_probability=describe(after['jump_probability'][choose]),
                            parent_max_probability=describe(before['max_probability'][choose]),
                            argmax_changed_fraction=float((before['argmax'][choose] != after['argmax'][choose]).float().mean()) if bool(choose.any()) else None)
                for group, choose in groups.items()
            }
    return rows


def main():
    assert not (OUT / 'report.json').exists(), 'Immutable completed diagnostic'
    for path, digest in IDENTITIES.items():
        assert sha(ROOT / path) == digest
    rows = [json.loads(line) for line in (ROOT / 'results/rival2/direct_skills_kickoff_race_v1/through_000015.jsonl').read_text().splitlines()]
    parent_payload = torch.load(PARENT, map_location='cpu', weights_only=False)
    child_payload = torch.load(CHILD, map_location='cpu', weights_only=False)
    model = TrainingExplorationPolicy().cuda().eval()
    model.load_state_dict(parent_payload['model'], strict=True)
    parent_hash = tensor_hash(model.state_dict())
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    bank = scenarios(WORLDS)
    env = KickoffRaceEnv(WORLDS, COLLISION, device='cuda:0', seed=SEED, ssl_foundation_scenarios=bank)
    collector = DirectSkillNativeNextoCollector(env, model, seed=SEED,
        nexto_sampling_mode='native_v5', nexto_seed=2026090691)
    record = {key: [] for key in ('roles', 'first_touch_event', 'goals')}
    original_step = env._step_impl

    def observed_step(*args, **kwargs):
        roles = player_roles(env.family, env.focal).clone()
        transition = original_step(*args, **kwargs)
        record['roles'].append(roles)
        record['first_touch_event'].append(env.last_skill['weighted'][..., 0] > 0)
        winner = torch.arange(2, device=env.device)[None] == env.last_native['scoring_team'][:, None]
        record['goals'].append(transition.terminated[:, None].to(torch.int8) * (winner.to(torch.int8) * 2 - 1))
        return transition

    env._step_impl = observed_step
    rollout = collector.collect()
    assert tensor_hash(model.state_dict()) == parent_hash
    data = mixed_sequence_data(rollout, ppo_config())
    record = {key: _sequence_major(torch.stack(value)) for key, value in record.items()}
    before = replay(model, data)
    parity = float((before['logp'] - data['old_log_probability'])[data['train_mask']].abs().max())
    assert parity < 2e-4, f'Collection/loss likelihood mismatch: {parity}'
    model.load_state_dict(child_payload['model'], strict=True)
    child_hash = tensor_hash(model.state_dict())
    after = replay(model, data)
    assert tensor_hash(model.state_dict()) == child_hash
    assert all(bool(torch.isfinite(x).all()) for x in (*before.values(), *after.values()))
    all_samples = torch.ones_like(data['train_mask'])
    # Episode starts and pre-contact ground states are observed state subsets,
    # not new reward predicates, mechanic detectors or prescribed action targets.
    opening = data['reset_before']
    report = dict(version='RIVAL2_DIRECT_SKILLS_LEARNING_DIAGNOSTIC_V1',
        optimizer_steps=0, checkpoints=IDENTITIES, worlds=WORLDS, horizon=90,
        seed=SEED, nexto_seed=2026090691, scenario_sha256=scenario_hash(bank),
        collection_likelihood_max_abs_error=parity,
        parent_model_unchanged=True, child_model_unchanged=True,
        previous_block_updates_3_to_5=aggregate(rows, 3, 5),
        previous_block_updates_13_to_15=aggregate(rows, 13, 15),
        trajectory=summarize(data, before, after, record['roles'], record['first_touch_event'], record['goals'], all_samples),
        episode_starts=summarize(data, before, after, record['roles'], record['first_touch_event'], record['goals'], opening),
        limitations='New parent-generated 2048-world/90-decision diagnostic, not replay of the historical 32768-world training minibatches. Parent/child see identical observations and recurrent sequence starts. Conditional action-likelihood changes are associations, not proof of causal reward credit. No assisted/unassisted result split is inferred from aggregate logs. No student rollout or optimizer step.',
        training_metrics=collector.last_metrics)
    compact = dict(roles=record['roles'], first_touch_event=record['first_touch_event'], goals=record['goals'],
                   train_mask=data['train_mask'], opponent_family=data['opponent_family'], reset_before=data['reset_before'],
                   action_indices=data['action_indices'], advantages=data['advantages'], normalized_advantage=data['normalized_advantage'],
                   returns=data['returns'], **{f'parent_{k}': v for k, v in before.items()},
                   **{f'child_{k}': v for k, v in after.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'sample_evidence.npz').exists()
    np.savez_compressed(OUT / 'sample_evidence.npz', **{k: v.detach().cpu().numpy() for k,v in compact.items()})
    report['sample_evidence_sha256'] = sha(OUT / 'sample_evidence.npz')
    for path, digest in IDENTITIES.items():
        assert sha(ROOT / path) == digest
    save_json(OUT / 'report.json', report)
    print(json.dumps(dict(parity=parity, parent_unchanged=True, optimizer_steps=0,
                         report=str(OUT / 'report.json'))))


if __name__ == '__main__':
    torch.set_num_threads(8)
    with gpu_lease():
        main()
