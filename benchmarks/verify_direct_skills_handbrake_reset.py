"""Native before/after reset parity; no model, optimizer or reward tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'results/rival2/direct_skills_v1/handbrake_reset_v3'
OLD_KERNEL = '83CA8E172A08588C3CDA2E58DCD57F17EA3F3B93540D184D51DF6CEB9CA17C14'


def text_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper()


def capture(reference):
    import torch
    import warp as wp
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from rivalsim.direct_skills_v1 import DirectSkillsEnv, scenarios
    from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios as shooting_scenarios
    from rivalsim.open_play import world_array_paths
    from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim

    arrays, checks = {}, []
    collision = Path('G:/dev/RLBot-Rival/bot/collision_meshes')
    with gpu_lease(), owned_match_stream():
        for mode in ('standard', 'original_curriculum', 'shooting_curriculum'):
            n = 10 if mode == 'standard' else 32
            if mode == 'standard':
                world = Rival2WorldSim(n, collision, device='cuda:0', seed=192,
                                      kickoff_selector=np.arange(n, dtype=np.int32) % 5, car_lifecycle_seed=192)
                bridge = Rival2TensorBridge(world)
            else:
                bank = scenarios(n, seed=192) if mode == 'original_curriculum' else shooting_scenarios(n, seed=192)
                env = DirectSkillsEnv(n, collision, device='cuda:0', seed=192, ssl_foundation_scenarios=bank)
                world, bridge = env.world, env.bridge
            controls = torch.zeros((n, 2, 8), device='cuda')
            controls[:, :, 0] = .7
            controls[:, :, 1] = .3
            bridge.set_actions(controls)
            world.begin_decision()
            world.step(8)
            paths = world_array_paths(world)
            for component_name, component in (('controls', world.controls), ('template', world.ssl_foundation_reset)):
                if component is not None:
                    for name, value in vars(component).items():
                        if isinstance(value, wp.array):
                            paths[f'{component_name}.{name}'] = value
            views = {k: wp.to_torch(v).reshape(n, -1) for k, v in paths.items()}
            saved = {k: v.clone() for k, v in views.items()}
            clock = wp.to_torch(world.tick_counter).clone()
            handbrake = wp.to_torch(world.vehicle.handbrake_value).reshape(n, 2)
            for pattern in ('even', 'odd', 'all', 'none'):
                for k, v in views.items():
                    v.copy_(saved[k])
                wp.to_torch(world.tick_counter).copy_(clock)
                world.tick_count = 8
                selected = torch.arange(n, device='cuda') % 2 == (0 if pattern == 'even' else 1)
                if pattern in ('all', 'none'):
                    selected.fill_(pattern == 'all')
                handbrake.copy_(torch.linspace(0, 1, n * 2, device='cuda').reshape(n, 2))
                wp.to_torch(world.rival2.reset_mask).copy_(selected.int())
                if mode == 'standard':
                    wp.to_torch(world.lifecycle.kickoff_selector).copy_(torch.arange(n, device='cuda').int() % 5)
                bridge.observation()
                before = {k: v.clone() for k, v in views.items()}
                world.apply_interval_resets()
                if reference and mode == 'standard':
                    # Independent reference: old native reset followed by only
                    # the experimentally isolated selected-car cache clearing.
                    handbrake[selected] = 0
                assert not handbrake[selected].any()
                assert torch.equal(handbrake[~selected], before['vehicle.handbrake_value'][~selected])
                obs = bridge.observation()
                assert obs.shape == (n, 2, 182) and torch.isfinite(obs).all()
                for k, v in views.items():
                    assert torch.equal(v[~selected], before[k][~selected]), (mode, pattern, 'unselected', k)
                    arrays[f'{mode}/{pattern}/reset/{k}'] = v.cpu().numpy().copy()
                arrays[f'{mode}/{pattern}/reset/observation'] = obs.cpu().numpy().copy()
                for age in range(4):
                    world.begin_decision() if age == 0 else None
                    world.step(1)
                    for field in ('car_pos', 'car_vel', 'car_quat', 'car_ang_vel', 'wheel_contact', 'boost'):
                        arrays[f'{mode}/{pattern}/tick{age + 1}/{field}'] = bridge.views[field].cpu().numpy().copy()
                checks.append(dict(mode=mode, mask=pattern, selected=int(selected.sum()),
                                   unselected=n-int(selected.sum()), arrays_per_reset=len(views),
                                   unselected_exact=True, selected_handbrake_zero=True))
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays, checks


def run(reference):
    kernel = text_sha(ROOT / 'rivalsim/kernels/rival2.py')
    assert (kernel == OLD_KERNEL) == reference, 'Run reference before editing; verification after correction'
    OUTPUT.mkdir(exist_ok=True)
    archive = OUTPUT / 'native_reference.npz'
    report = OUTPUT / ('before.json' if reference else 'after.json')
    assert not report.exists()
    if reference:
        assert not archive.exists()
    arrays, checks = capture(reference)
    if reference:
        np.savez_compressed(archive, **arrays)
    else:
        with np.load(archive, allow_pickle=False) as old:
            assert set(old.files) == set(arrays)
            mismatched = [k for k, v in arrays.items() if not np.array_equal(old[k], v)]
        assert not mismatched, mismatched
    evidence = dict(reference=reference, exact_before_after=None if reference else True,
                    arrays=len(arrays), checks=checks, kernel_sha256=kernel,
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest().upper(),
                    optimizer_steps=0, policy_construction=False,
                    standard_reference='Old native reset plus selected handbrake-only clearing',
                    curriculum_reference='Unmodified native scenario reset, no intervention')
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', action='store_true')
    run(parser.parse_args().reference)
