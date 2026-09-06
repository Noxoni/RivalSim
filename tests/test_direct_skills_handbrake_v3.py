import ast
import copy
import json

import numpy as np
import pytest

from benchmarks.evaluate_direct_skills_handbrake_v3 import BASE, OUTPUT, ROOT, METHOD
from benchmarks.report_rival2_direct_skills_matches import validate_comparability


def test_native_parity_evidence_complete():
    before = json.loads((OUTPUT / 'before.json').read_text())
    after = json.loads((OUTPUT / 'after.json').read_text())
    assert after['exact_before_after'] and after['arrays'] == 4368
    assert after['archive_sha256'] == before['archive_sha256']
    assert len(after['checks']) == 12
    assert all(x['unselected_exact'] and x['selected_handbrake_zero'] for x in after['checks'])
    assert after['optimizer_steps'] == 0 and not after['policy_construction']


def test_reference_contains_both_curricula_and_every_mask():
    with np.load(OUTPUT / 'native_reference.npz', allow_pickle=False) as a:
        for mode in ('standard', 'original_curriculum', 'shooting_curriculum'):
            for mask in ('even', 'odd', 'all', 'none'):
                hb = a[f'{mode}/{mask}/reset/vehicle.handbrake_value']
                selected = np.arange(len(hb)) % 2 == (0 if mask == 'even' else 1)
                if mask in ('all', 'none'):
                    selected[:] = mask == 'all'
                assert not hb[selected].any()
                if (~selected).any():
                    assert hb[~selected].any()


def test_kernel_change_is_only_cache_argument_and_clear():
    import subprocess
    prior = subprocess.check_output(['git', 'show', '732c008:rivalsim/kernels/rival2.py'], cwd=ROOT).decode()
    current = (ROOT / 'rivalsim/kernels/rival2.py').read_text()
    old_tree, new_tree = ast.parse(prior), ast.parse(current)
    old_fn = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rival2_interval_reset')
    new_fn = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rival2_interval_reset')
    assert new_fn.args.args.pop().arg == 'handbrake_value'
    loop = next(n for n in new_fn.body if isinstance(n, ast.For))
    writes = [n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'handbrake_value[car]']
    assert len(writes) == 1 and ast.unparse(writes[0].value) == '0.0'
    loop.body.remove(writes[0])
    assert ast.dump(old_tree) == ast.dump(new_tree)
    assert ast.dump(old_fn) == ast.dump(new_fn)


def test_matching_v3_allowed_mixed_versions_rejected():
    a = json.loads((BASE / 'full_match_000600.json').read_text())
    b = json.loads((BASE / 'full_match_000675.json').read_text())
    a['match_reset_version'] = b['match_reset_version'] = METHOD
    validate_comparability(a, b)
    bad = copy.deepcopy(b)
    bad['match_reset_version'] = 'RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2'
    with pytest.raises(ValueError, match='match_reset_version'):
        validate_comparability(a, bad)
