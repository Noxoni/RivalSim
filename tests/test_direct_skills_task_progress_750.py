import copy

import pytest

from benchmarks.summarize_direct_skills_task_progress_750 import final_sources, markdown


def complete():
    return dict(status='complete_review_not_SSL', accepted_updates=70,
                total_exploration_updates=100, cumulative_updates=750,
                optimizer_steps_in_finalization=0, new_evaluations_in_finalization=0,
                checks={'no_active_worker': True, 'frozen_boundary': True},
                training=dict(updates=70, learner_decisions=309657600,
                              physics_ticks=825753600, kl_rejections=0),
                parent_comparisons={'20': {}, '45': {}, '70': {}},
                training_curve_sha256='ab' * 32)


def test_final_sources_use_original30_and_closed70_without_overlap():
    sources = final_sources(complete())
    assert [(row[2], row[3]) for row in sources] == [(650, 30), (680, 70)]
    assert sources[1][0].endswith('/through_000070.jsonl')
    assert sources[1][1] == 'AB' * 32


@pytest.mark.parametrize('fault', ['live', 'offset', 'samples', 'ticks', 'kl', 'missing_eval',
                                  'checks', 'empty_checks', 'hash', 'optimizer', 'new_eval'])
def test_no_partial_run_or_wrong_exposure_accepted(fault):
    value = copy.deepcopy(complete())
    if fault == 'live': value['status'] = 'optimizing'
    elif fault == 'offset': value['cumulative_updates'] = 725
    elif fault == 'samples': value['training']['learner_decisions'] -= 1
    elif fault == 'ticks': value['training']['physics_ticks'] -= 1
    elif fault == 'kl': value['training']['kl_rejections'] = 1
    elif fault == 'missing_eval': del value['parent_comparisons']['70']
    elif fault == 'checks': value['checks']['no_active_worker'] = False
    elif fault == 'empty_checks': value['checks'] = {}
    elif fault == 'hash': value['training_curve_sha256'] = 'z' * 64
    elif fault == 'optimizer': value['optimizer_steps_in_finalization'] = 1
    else: value['new_evaluations_in_finalization'] = 1
    with pytest.raises(ValueError):
        final_sources(value)


def test_markdown_preserves_unknown_fraction_and_pooled_rates():
    record = dict(totals=dict(goals=0, episode_endings=0),
                  success_fraction_of_observed_endings=None,
                  per_learner_minute=dict(touches=12.5, goals=0, control_gain=.25,
                                         shot_clear=.5, first_touch=.75))
    roles = {role + '_' + opponent: record
             for role in ('natural', 'finishing', 'challenge', 'defense', 'kickoff')
             for opponent in ('nexto', 'selfplay')}
    text = markdown(dict(windows=[dict(first_update=741, last_update=750, roles=roles)],
                         semantics=dict(censoring='Not an attempt cohort.')))
    assert '| 741-750 | 0 / 0 | unavailable | 0 / 0 | unavailable |' in text
    assert '| 741-750 | 12.5000 | 0.0000 | 0.2500 | 0.5000 | 0.7500 | 0.2500 |' in text
    assert 'Not an attempt cohort.' in text
