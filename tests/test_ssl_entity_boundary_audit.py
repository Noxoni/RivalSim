import copy
import json

import pytest
import torch

from benchmarks.audit_rival2_ssl_entity_boundary import finite_tree, ledger, read_prefix


def rows():
    return [
        dict(
            accepted_updates=101,
            cumulative_optimizer_steps=18204,
            training=dict(
                trainable_agent_samples=90,
                nexto_training_sample_count=10,
                action_index_counts=[40, 50],
                physical_physics_ticks=200,
            ),
            ppo=dict(optimizer_steps=4, kl_rejections=0, completed_update_mean_kl=123.0),
        )
    ]


def test_actual_samples_variable_adam_and_large_kl_are_telemetry():
    assert ledger(rows(), 101, 18200, 100, 200) == dict(
        additional_samples=90,
        nexto_current_agent_samples=10,
        additional_physical_ticks=200,
        cumulative_optimizer_steps=18204,
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.append(copy.deepcopy(r[0])), "noncontiguous_or_duplicate"),
        (lambda r: r[0].update(accepted_updates=102), "noncontiguous_or_duplicate"),
        (lambda r: r[0]["training"].update(nexto_training_sample_count=9), "not_conserved"),
        (lambda r: r[0]["training"].update(action_index_counts=[50, 50]), "not_learner_only"),
        (lambda r: r[0].update(cumulative_optimizer_steps=4), "counter_discontinuity"),
        (lambda r: r[0]["ppo"].update(kl_rejections=1), "unexpected_kl_rejection"),
        (lambda r: r[0]["ppo"].update(completed_update_mean_kl=float("nan")), "nonfinite"),
    ],
)
def test_bad_evidence_rejected(mutation, reason):
    data = rows()
    mutation(data)
    with pytest.raises(ValueError, match=reason):
        ledger(data, 101, 18200, 100, 200)


def test_prefix_excludes_later_and_partial_append(tmp_path):
    first = json.dumps(rows()[0]).encode() + b"\r\n"
    later = json.dumps(dict(accepted_updates=102)).encode() + b"\r\n"
    path = tmp_path / "live.jsonl"
    path.write_bytes(first + later + b'{"accepted_updates":')
    data, prefix = read_prefix(path, 101)
    assert data == rows()
    assert prefix == first.replace(b"\r\n", b"\n")
    data, _ = read_prefix(path, 103)
    assert [r["accepted_updates"] for r in data] == [101, 102]


def test_nested_finite_checks():
    assert finite_tree(dict(tensors=[torch.tensor([0.0, 1.0])], step=18204))
    assert not finite_tree(dict(state=dict(moment=torch.tensor(float("inf")))))
