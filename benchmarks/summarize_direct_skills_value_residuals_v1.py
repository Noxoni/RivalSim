"""CPU-only secondary analysis of an already committed matched-trajectory NPZ.

These are residuals against bootstrapped GAE targets, not Monte Carlo truth.
The child is the PREVIOUS kickoff arm, not the running log-barrier experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/rival2/direct_skills_learning_diagnostic_v1'
NAMES = ('natural', 'challenge', 'finishing', 'defense', 'kickoff')


def residual_stats(target, value):
    target = np.asarray(target, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if not target.size or target.shape != value.shape:
        raise ValueError('Nonempty matching arrays required')
    if not (np.isfinite(target).all() and np.isfinite(value).all()):
        raise ValueError('Nonfinite saved evidence')
    residual = target - value
    variance = target.var()
    return dict(count=target.size, target_mean=float(target.mean()), value_mean=float(value.mean()),
                residual_mean=float(residual.mean()), mse=float(np.mean(residual**2)),
                rmse=float(np.sqrt(np.mean(residual**2))),
                explained_variance=float(1-residual.var()/variance) if variance > 1e-12 else None)


def summarize(data):
    selected = data['train_mask']
    if selected.dtype != np.bool_ or not selected.any():
        raise ValueError('Current-policy Boolean mask required')
    error = float(np.abs(data['returns'][selected]-data['parent_value'][selected]
                        - data['advantages'][selected]).max())
    if error > 2e-4:
        raise ValueError('Recorded targets/parent values do not reconstruct raw advantage')
    rows = {}
    for role, name in enumerate(NAMES):
        for family, opponent in enumerate(('selfplay', 'nexto')):
            mask = selected & (data['roles'] == role) & (data['opponent_family'] == family)
            if not mask.any():
                continue
            rows[f'{name}_{opponent}'] = dict(
                parent=residual_stats(data['returns'][mask], data['parent_value'][mask]),
                previous_kickoff_child15_same_parent_targets=residual_stats(
                    data['returns'][mask], data['child_value'][mask]),
                raw_advantage_mean=float(data['advantages'][mask].astype(np.float64).mean()),
                normalized_advantage_mean=float(data['normalized_advantage'][mask].astype(np.float64).mean()),
                normalized_positive_fraction=float((data['normalized_advantage'][mask] > 0).mean()))
    return dict(max_advantage_identity_error=error, groups=rows)


def main():
    source = OUT / 'sample_evidence.npz'
    original = json.loads((OUT / 'report.json').read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    assert digest == original['sample_evidence_sha256']
    with np.load(source, allow_pickle=False) as data:
        result = summarize(data)
    result.update(version='RIVAL2_DIRECT_SKILLS_SAVED_VALUE_RESIDUALS_V1',
                  source_sha256=digest, checkpoints=original['checkpoints'],
                  optimizer_steps=0, new_simulator_rollouts=0, device='cpu',
                  limitations='Targets use parent bootstrap and parent-generated short trajectories, '
                  'not independent realized returns. Previous kickoff child15 is evaluated on identical '
                  'saved states against those fixed targets, NOT its own new-policy return distribution. '
                  'Neither comparison proves causal actor-credit failure. No current log-barrier child '
                  'is represented. Opponent-family normalization spans all task roles.',
                  inference='Large parent kickoff/Nexto residual existed, but the previous child already '
                  'greatly reduced its fixed-target error while failing full matches. This does not '
                  'support treating a critic-only upgrade as the established solution.')
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    destination = OUT / 'value_residuals.json'
    if destination.exists():
        assert destination.read_bytes() == encoded, 'Refuse changed secondary evidence'
    else:
        destination.write_bytes(encoded)
    print(json.dumps(result['groups']['kickoff_nexto']))


if __name__ == '__main__':
    main()
