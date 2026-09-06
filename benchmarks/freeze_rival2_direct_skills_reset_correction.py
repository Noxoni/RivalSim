"""Bind the narrow native contact-reset correction without changing PPO authority."""

import json
import subprocess
import xml.etree.ElementTree as ET

from benchmarks.run_rival2_direct_skills_v1 import RESULTS, ROOT, EXTERNAL, verify, text_sha, write_json


def freeze():
    old = json.loads((RESULTS/'package.json').read_text())
    assert 'contact_reset_correction' not in old
    assert old == json.loads(subprocess.check_output(['git','show','a150965:results/rival2/direct_skills_v1/package.json'], cwd=ROOT))
    state = json.loads((EXTERNAL/'campaign_state.json').read_text())
    assert state['status'] == 'stopped_at_accepted_boundary' and state['accepted_updates'] == 451
    allowed = {'rivalsim/kernels/rival2.py', 'rivalsim/rival2_env.py', 'benchmarks/run_rival2_direct_skills_v1.py'}
    changed = {p for p,h in old['sources'].items() if text_sha(ROOT/p) != h}
    assert changed == allowed
    parity = json.loads((RESULTS/'training_reset_contact_fixture.after.json').read_text())
    assert parity['exact_parity'] and parity['arrays'] == 246 and parity['optimizer_steps'] == 0
    suites = ET.parse(RESULTS/'reset_contact_tests.xml').getroot().findall('testsuite')
    assert sum(int(s.get('tests')) for s in suites) == 17
    assert all(int(s.get('failures',0)) == int(s.get('errors',0)) == int(s.get('skipped',0)) == 0 for s in suites)
    write_json(RESULTS/'package_before_contact_reset_correction.json', old)
    new = dict(old)
    new['sources'] = {p:text_sha(ROOT/p) for p in old['sources']}
    new['evidence'] = dict(old['evidence'])
    for name in ('RESET_CONTACT_CORRECTION_PLAN.md', 'RESET_CONTACT_CORRECTION.md',
                 'package_before_contact_reset_correction.json', 'training_reset_contact_fixture.before.json',
                 'training_reset_contact_fixture.after.json', 'reset_contact_tests.xml'):
        new['evidence'][name] = text_sha(RESULTS/name)
    new['contact_reset_correction'] = dict(
        accepted_updates_before_correction=451,
        prospective_plan_commit='a1509658ce3b2e46542f4f621e3b8481fd4a990f',
        changed_sources=sorted(changed), authority_unchanged=True,
        training_reset_fixture_exact=True,
        match_reset_version='RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2',
        corrected_comparison='corrected_reset/full_match_000000.json versus full_match_000451.json',
        resume_checkpoint=state['latest_checkpoint'],
        semantics='Clear stale wheel contacts in standard kickoff reset; training scenario resets already cleared them. '
                  'Future matches explicitly tag corrected semantics; old evaluation evidence is immutable.',
    )
    write_json(RESULTS/'package.json', new)
    verify(published=False)


if __name__ == '__main__':
    freeze()
