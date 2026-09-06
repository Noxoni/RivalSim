import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.report_direct_skills_kickoff_continuity import build,pair_by_age,ROOT

DIRECTORY=ROOT/'results/rival2/direct_skills_v1/kickoff_continuity_000675'


@pytest.fixture(scope='module')
def arrays():
    with np.load(DIRECTORY/'trace/kickoff_reset.npz',allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def test_actual_report_rebuild_exactly():
    report=build(DIRECTORY)
    assert report==json.loads((DIRECTORY/'report.json').read_text())
    assert report['optimizer_steps']==0 and report['policy_or_physics_interventions']==0
    assert report['postgoal_kickoff_origins']==205
    assert all(r['rival_action_changed_pairs']==0 for r in report['ages'])


def test_layout_event_clears_after_first_tick_without_losing_episode_identity(arrays):
    assert (arrays['lifecycle_layout'][arrays['eligible'] & (arrays['episode_ticks']>0)]==-1).all()
    pairs=pair_by_age(arrays)
    assert len(pairs)==205*8
    assert len({key[0] for key,_,_ in pairs})==5


def test_missing_origin_rejected(arrays):
    a=dict(arrays);a['eligible']=arrays['eligible'].copy()
    later=arrays['eligible'] & (arrays['episode_ticks']==0) & (arrays['match_goal_count']>0)
    d,w=next(zip(*np.where(later)));a['eligible'][d,w]=False
    with pytest.raises(AssertionError):pair_by_age(a)


def test_invalid_origin_layout_rejected(arrays):
    a=dict(arrays);a['lifecycle_layout']=arrays['lifecycle_layout'].copy()
    later=arrays['eligible'] & (arrays['episode_ticks']==0) & (arrays['match_goal_count']>0)
    d,w=next(zip(*np.where(later)));a['lifecycle_layout'][d,w]=-1
    with pytest.raises(AssertionError):pair_by_age(a)
