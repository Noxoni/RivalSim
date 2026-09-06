"""Bounded private native continuation replay; no optimizer or production edits."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'results/rival2/direct_skills_v1'
OUTPUT = BASE / 'kickoff_native_isolation_000675'
CHECKPOINT = ROOT / 'checkpoints/rival2/direct_skills_v1/plus_000675.pt'
CHECKPOINT_SHA = '77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8'
ORIGIN_TICKS = [1756, 856, 644, 628, 668, 4716, 856, 644, 628, 668]
FIELDS = ('car_pos', 'car_vel', 'car_quat', 'car_ang_vel', 'boost',
          'on_ground', 'ball_pos', 'ball_vel', 'wheel_contact')
QUATERNIONS = ('state.car_quat', 'vehicle.solver_orientation')
HANDBRAKE = ('vehicle.handbrake_value',)
CHASSIS = ('vehicle.contact_count', 'vehicle.world_contact_normal')
WHEELS = ('vehicle.wheel_contact', 'vehicle.wheel_world_contact', 'vehicle.wheels_with_contact')
ARMS = {
    'initial_cold': ('initial', 0, ()),
    'initial_warm': ('initial', 1, ()),
    'postgoal_warm': ('postgoal', 1, ()),
    'postgoal_quaternion': ('postgoal', 1, QUATERNIONS),
    'postgoal_handbrake': ('postgoal', 1, HANDBRAKE),
    'postgoal_chassis': ('postgoal', 1, CHASSIS),
    'postgoal_wheels': ('postgoal', 1, WHEELS),
    'postgoal_all_caches': ('postgoal', 1, HANDBRAKE + CHASSIS + WHEELS),
    'postgoal_caches_quaternion': ('postgoal', 1, HANDBRAKE + CHASSIS + WHEELS + QUATERNIONS),
    'postgoal_caches_quaternion_cold': ('postgoal', 0, HANDBRAKE + CHASSIS + WHEELS + QUATERNIONS),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_map(layout, side):
    layout, side = np.asarray(layout), np.asarray(side)
    assert layout.shape == side.shape == (10,)
    assert np.isin(layout, range(5)).all() and np.isin(side, (0, 1)).all()
    return layout.astype(np.int64) + 5 * side.astype(np.int64)


def reduce_archive(arrays):
    mapping = arrays['source_map']
    assert mapping.shape == (10,) and np.isin(mapping, range(10)).all()
    controls_equal = (arrays['initial_controls'][:, mapping] == arrays['postgoal_controls']).all(axis=(0, 2, 3))
    reports = {}
    for arm in ARMS:
        reports[arm] = {}
        for reference in ('initial_cold', 'initial_warm', 'postgoal_warm'):
            reports[arm][reference] = {
                field: float(np.abs(arrays[f'{arm}/{field}'] - arrays[f'{reference}/{field}']).max())
                for field in FIELDS
            }
    return dict(
        comparisons=reports, actual_control_tape_equal_worlds=int(controls_equal.sum()),
        actual_control_tape_equal=controls_equal.tolist(),
        first_postgoal_ticks=arrays['origin_ticks'].tolist(), source_map=mapping.tolist(),
        private_interventions=True, production_interventions=0, optimizer_steps=0,
        baseline_exact=True, interpretation='Fixed controls isolate physical transition only; not match performance or a learning result.',
    )


def run():
    import torch
    import warp as wp

    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner
    from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from benchmarks.trace_rival2_direct_skills import temporary_attribute
    from rivalsim.open_play import world_array_paths

    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    package = json.loads((BASE / 'package.json').read_text())
    for name, digest in package['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper() == digest, name
    authority = OUTPUT / 'AUTHORITY.md'
    for path in (authority, Path(__file__)):
        remote = subprocess.check_output(['git', 'show', f'origin/main:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT)
        assert remote.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')
    target = OUTPUT / 'capture'
    assert not target.exists(), 'Never overwrite a completed or failed capture'
    with gpu_lease(), owned_match_stream():
        target.mkdir()
        arrays = {}
        try:
            runner = CandidateMatchRunner(CHECKPOINT, CHECKPOINT_SHA, entity=True)
            n = runner.num_worlds
            paths = world_array_paths(runner.world)
            for component_name, component in (('controls', runner.world.controls), ('match', runner.match), ('telemetry', runner.telemetry)):
                for name, value in vars(component).items():
                    if isinstance(value, wp.array):
                        paths[f'{component_name}.{name}'] = value
            views = {k: wp.to_torch(v).reshape(n, -1) for k, v in paths.items()}
            initial = {k: v.detach().clone() for k, v in views.items()}
            post = {k: torch.empty_like(v) for k, v in initial.items()}
            post_layout = np.full(n, -1, np.int32)
            captured = np.zeros((2, 32, n), bool)
            tapes = np.empty((2, 32, n, 2, 8), np.float32)
            physical = {field: np.empty((2, 32, n, *runner.bridge.views[field].reshape(n, -1).shape[1:]),
                                        dtype=runner.bridge.views[field].cpu().numpy().dtype) for field in FIELDS}
            common = np.load(BASE / 'kickoff_continuity_000675/trace/kickoff_reset.npz', allow_pickle=False)
            common_rows = {int(t): i for i, t in enumerate(common['host_tick'])}
            original_step = runner.world.step_graph

            def capture_step(*args, **kwargs):
                tick = runner.host_tick
                phase = 0 if tick < 32 else 1
                ages = np.full(n, tick, np.int64) if phase == 0 else tick - np.asarray(ORIGIN_TICKS)
                selected = np.where((ages >= 0) & (ages < 32))[0]
                if len(selected):
                    goal = runner.match_views['goal_count'].cpu().numpy()
                    episode = runner.bridge.views['rival2.episode_ticks'].cpu().numpy()
                    for w in selected:
                        age = int(ages[w])
                        assert goal[w] == phase and episode[w] == age
                        assert not captured[phase, age, w]
                        captured[phase, age, w] = True
                        tapes[phase, age, w] = runner.actions[w].cpu().numpy()
                        for field in FIELDS:
                            physical[field][phase, age, w] = runner.bridge.views[field].reshape(n, -1)[w].cpu().numpy()
                        if phase == 1 and age == 0:
                            for k in post:
                                post[k][w].copy_(views[k][w])
                            post_layout[w] = int(wp.to_torch(runner.world.lifecycle.kickoff_layout)[w])
                        if age % 4 == 0:
                            row = common_rows[tick]
                            assert common['eligible'][row, w]
                            assert np.array_equal(common['rival_action'][row, w], runner.rival_action[w].cpu().numpy())
                            for field in FIELDS:
                                if field in common.files:
                                    assert np.array_equal(physical[field][phase, age, w], common[field][row, w].reshape(-1)), (tick, w, field)
                return original_step(*args, **kwargs)

            with temporary_attribute(runner.world, 'step_graph', capture_step):
                runner.run_ticks(max(ORIGIN_TICKS) + 32)
            assert captured.all()
            common.close()
            side = runner.rival_side.cpu().numpy()
            mapping = source_map(post_layout, side)
            mapped_initial = {k: v[mapping].clone() for k, v in initial.items()}
            arrays.update(source_map=mapping, origin_ticks=np.asarray(ORIGIN_TICKS),
                          initial_controls=tapes[0], postgoal_controls=tapes[1], postgoal_layout=post_layout)
            for field, value in physical.items():
                arrays[f'captured_initial/{field}'] = value[0]
                arrays[f'captured_postgoal/{field}'] = value[1]
            for prefix, snapshot in (('origin_initial', initial), ('origin_postgoal', post)):
                for k, v in snapshot.items():
                    arrays[f'{prefix}/{k}'] = v.cpu().numpy().copy()

            def replay(snapshot, clock, controls, substitutions=()):
                for k, v in views.items():
                    source = mapped_initial[k] if k in substitutions else snapshot[k]
                    v.copy_(source)
                    assert torch.equal(v, source), k
                wp.to_torch(runner.world.tick_counter).fill_(clock)
                runner.world.tick_count = clock
                result = {f: [] for f in FIELDS}
                for age in range(32):
                    if age % 4 == 0:
                        runner.world.begin_decision()
                    for field in FIELDS:
                        result[field].append(runner.bridge.views[field].reshape(n, -1).cpu().numpy().copy())
                    runner.bridge.set_actions(torch.as_tensor(controls[age], device=runner.device))
                    runner.world.step_graph(1)
                return {f: np.stack(v) for f, v in result.items()}

            # Actual tapes, not assumed control parity, prove snapshot replay first.
            for phase, snapshot, clock in ((0, initial, 0), (1, post, 1)):
                observed = replay(snapshot, clock, tapes[phase])
                for field, value in observed.items():
                    arrays[f'baseline_{phase}/{field}'] = value
                    assert np.array_equal(value, physical[field][phase]), ('baseline', phase, field, float(abs(value - physical[field][phase]).max()))
            for arm, (origin, clock, substitutions) in ARMS.items():
                result = replay(mapped_initial if origin == 'initial' else post, clock, tapes[1], substitutions)
                for field, value in result.items():
                    arrays[f'{arm}/{field}'] = value
            assert all(np.isfinite(v).all() for v in arrays.values())
            archive = target / 'native_replay.npz'
            np.savez_compressed(archive, **arrays)
            with np.load(archive, allow_pickle=False) as recovered:
                assert set(arrays) == set(recovered.files)
                assert all(np.array_equal(v, recovered[k]) for k, v in arrays.items())
            assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
            assert sha(CHECKPOINT) == CHECKPOINT_SHA
            for name, digest in package['sources'].items():
                assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper() == digest
            report = reduce_archive(arrays)
            write_json(target / 'report.json', report)
            write_json(target / 'manifest.json', dict(
                checkpoint_sha256=CHECKPOINT_SHA, authority_sha256=sha(authority),
                script_sha256=sha(Path(__file__)), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                source_trace_sha256=sha(BASE / 'kickoff_continuity_000675/trace/kickoff_reset.npz'),
                frozen_sources=package['sources'], world_array_count=len(paths),
                optimizer_steps=0, production_interventions=0, model_checkpoint_unchanged=True,
                common_recorded_rows_exact=True, baseline_fixed_action_replays_exact=True,
                warm_clock_absolute_tick_equivalence_checked_by_postgoal_replay=True,
                scalar_clock='All positive clock values normalized to1 only in private replay; actual initial clock0.',
            ))
            print(json.dumps(report))
        except BaseException as exc:
            if arrays:
                np.savez_compressed(target / 'incomplete_raw.npz', **arrays)
            write_json(target / 'failure.json', dict(type=type(exc).__name__, reason=str(exc),
                       optimizer_steps=0, production_interventions=0, interpretable=False))
            raise


if __name__ == '__main__':
    import torch
    torch.set_num_threads(8)
    run()
