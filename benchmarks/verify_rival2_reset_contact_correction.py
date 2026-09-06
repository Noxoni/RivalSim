"""Zero-learning native reset fixture before/after a contact-cache correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import warp as wp

from benchmarks.direct_skills_eval_stream import owned_match_stream
from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
from rivalsim.direct_skills_v1 import DirectSkillsEnv, scenarios


def capture_training_reset():
    env = DirectSkillsEnv(32, Path("G:/dev/RLBot-Rival/bot/collision_meshes"),
                         device="cuda:0", seed=192, ssl_foundation_scenarios=scenarios(32, seed=192))
    wheels = env.bridge.views['wheel_contact'].reshape(32, 2, 4)
    wheels.fill_(1)
    mask = wp.to_torch(env.world.rival2.reset_mask)
    mask.copy_((torch.arange(32, device=env.device) % 2 == 0).int())
    env.world.apply_interval_resets()
    assert not wheels[::2].any()
    assert wheels[1::2].all()
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in env.bridge.views.items()}
    arrays['final_observation'] = env.bridge.observation().cpu().numpy().copy()
    for group in ('vehicle', 'ball_world', 'car_car', 'lifecycle'):
        component = getattr(env.world, group)
        for name in dir(component):
            value = getattr(component, name)
            if isinstance(value, wp.array):
                arrays[f'{group}.{name}'] = value.numpy().copy()
    assert all(np.isfinite(v).all() for v in arrays.values())
    return arrays


def run(path, compare):
    with gpu_lease(), owned_match_stream():
        arrays = capture_training_reset()
    if compare:
        with np.load(path, allow_pickle=False) as original:
            assert set(original.files) == set(arrays)
            different = [key for key, value in arrays.items() if not np.array_equal(original[key], value)]
        assert not different, different
    else:
        assert not path.exists(), 'never replace a baseline'
        np.savez_compressed(path, **arrays)
    evidence = dict(
        schema='RIVAL2_RESET_CONTACT_TRAINING_PARITY_V1', comparing_to_original=compare,
        fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        arrays=len(arrays), exact_parity=True if compare else None,
        selected_worlds=16, unaffected_worlds=16, optimizer_steps=0,
        kernel_sha256=hashlib.sha256((ROOT/'rivalsim/kernels/rival2.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper(),
    )
    path.with_suffix('.after.json' if compare else '.before.json').write_text(json.dumps(evidence, sort_keys=True, indent=2)+'\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path', type=Path)
    p.add_argument('--compare', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    run(args.path, args.compare)
