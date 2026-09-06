"""One bounded, no-optimizer post-run gradient probe using the existing method.

Reuse the frozen calibration implementation on the preserved +30 checkpoint.
Its candidate table is diagnostic only; it does not authorize a new coefficient
or launch any training. Historical calibration evidence is never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from benchmarks import calibrate_direct_skills_log_barrier_v1 as calibration
from benchmarks.analyze_direct_skills_learning_v1 import save_json
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha
from benchmarks.finalize_direct_skills_log_barrier_v1 import active_campaign_pids

OUT = ROOT / 'results/rival2/direct_skills_post_exploration_probe_v1'
CHECKPOINT = ROOT / 'checkpoints/rival2/direct_skills_log_barrier_v1/child_000030.pt'
DIGEST = '16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD'


def main():
    assert not active_campaign_pids(), 'Completed campaign must be idle'
    assert sha(CHECKPOINT) == DIGEST
    assert json.loads((ROOT / 'results/rival2/direct_skills_log_barrier_v1/completion.json').read_text())['checkpoint']['sha256'] == DIGEST
    assert not (OUT / 'barrier_calibration.json').exists(), 'Completed probe is immutable'
    original = calibration.OUT, calibration.PARENT, calibration.IDENTITIES
    try:
        calibration.OUT = OUT
        calibration.PARENT = CHECKPOINT
        calibration.IDENTITIES = {str(CHECKPOINT.relative_to(ROOT)): DIGEST}
        calibration.main()
    finally:
        calibration.OUT, calibration.PARENT, calibration.IDENTITIES = original
    measured = json.loads((OUT / 'barrier_calibration.json').read_text())
    ratios = np.array([.01*r['norms']['unit_barrier']/r['norms']['policy'] for r in measured['records']])
    cosines = np.array([r['policy_barrier_cosine'] for r in measured['records']])
    result = dict(version='RIVAL2_DIRECT_SKILLS_POST_EXPLORATION_PROBE_V1',
                  checkpoint_sha256=DIGEST, raw_evidence_sha256=sha(OUT / 'barrier_calibration.json'),
                  current_coefficient=.01, median_gradient_ratio=float(np.median(ratios)),
                  maximum_gradient_ratio=float(ratios.max()),
                  median_gradient_cosine=float(np.median(cosines)),
                  mean_relative_component_along_PPO_gradient=float(np.mean(ratios*cosines)),
                  exceeds_original_initialization_criterion=bool(np.median(ratios)>.5 or ratios.max()>1),
                  optimizer_steps=0, optimizer_constructed=False, checkpoint_unchanged=sha(CHECKPOINT)==DIGEST,
                  new_training_authorized_by_probe=False,
                  limitation='Eight deterministic microbatches from a fresh 1024-world, 90-decision '
                  'training-distribution rollout. Not historical update-30 gradient replay and not '
                  'proof of gameplay causality. Raw calibration candidate selections are NOT adopted.')
    assert result['checkpoint_unchanged']
    save_json(OUT / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(8)
    with calibration.gpu_lease():
        main()
