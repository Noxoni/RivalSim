import torch

from rivalsim.fresh_ground_30hz import fresh_model
from rivalsim.rival2_contracts import ORANGE_PAD_REMAP
from rivalsim.rival2_policy import deterministic_hybrid_action
from rivalsim.ssl_entity_evaluation import evaluation_output
from rivalsim.ssl_entity_policy import EntityEncoder, EntityJointControlActorCritic, entity_schema
from rivalsim.ssl_joint_control_policy import JointControlActorCritic


def test_entity_fields_and_canonical_pad_geometry():
    encoder = EntityEncoder()
    obs = torch.arange(182, dtype=torch.float32).expand(3, -1).clone()
    before = obs.clone()
    own, opponent, ball, pads, context = encoder.raw_groups(obs)
    assert torch.equal(own[:, :39], obs[:, 9:48])
    assert own[:, 39:].count_nonzero() == 0
    assert torch.equal(opponent[:, :39], obs[:, 48:87])
    assert torch.equal(opponent[:, 39:], obs[:, 93:99])
    assert torch.equal(ball[:, :9], obs[:, :9])
    assert torch.equal(ball[:, 9:], obs[:, 87:93])
    assert torch.equal(context, obs[:, 167:182])
    assert torch.equal(pads[:, :, 4:6], obs[:, 99:167].reshape(3, 34, 2))
    torch.testing.assert_close(pads[:, :, 6:], encoder.pad_positions - obs[:, None, 9:12])
    torch.testing.assert_close(
        encoder.pad_positions[list(ORANGE_PAD_REMAP)] * torch.tensor([-1.0, -1.0, 1.0]),
        encoder.pad_positions,
        rtol=0,
        atol=0,
    )
    assert torch.equal(before, obs)
    assert entity_schema()["count"] == 38


def test_evaluation_interface_has_exact_controls_for_every_joint_action():
    model = JointControlActorCritic()
    reconstructed = deterministic_hybrid_action(evaluation_output(model.action_table))
    assert torch.equal(reconstructed, model.action_table)


def test_attention_is_permutation_equivariant_and_uses_all_entities():
    torch.set_num_threads(4)
    torch.manual_seed(4)
    encoder = EntityEncoder()
    obs = torch.randn(7, 182, requires_grad=True)
    tokens = encoder.tokens(obs)
    permutation = torch.cat((torch.zeros(1, dtype=torch.long), torch.randperm(37) + 1))
    result, weights = encoder.attend(tokens, weights=True)
    permuted, _ = encoder.attend(tokens[:, permutation], weights=True)
    torch.testing.assert_close(permuted, result, rtol=1e-5, atol=1e-6)
    assert weights.shape == (7, 4, 1, 38)
    torch.testing.assert_close(weights.sum(-1), torch.ones(7, 4, 1))
    result.square().sum().backward()
    assert torch.isfinite(obs.grad).all()
    for first, last in [(0, 9), (9, 48), (48, 87), (87, 99), (99, 167), (167, 182)]:
        assert obs.grad[:, first:last].abs().sum() > 0


def test_exact_initial_entity_parity_reset_semantics_and_critic_isolation():
    torch.set_num_threads(4)
    old = fresh_model(11)
    joint = JointControlActorCritic()
    entity = EntityJointControlActorCritic()
    joint.initialize_from_hybrid(old.state_dict())
    entity.initialize_from_hybrid(old.state_dict())
    obs = torch.randn(3, 5, 182)
    resets = torch.zeros(3, 5, dtype=torch.bool)
    resets[:, 0] = True
    resets[1, 3] = True
    h0 = joint.initial_hidden(3)
    a, v, h = joint(obs, h0, reset_before=resets)
    a2, v2, h2 = entity(obs, h0, reset_before=resets)
    for x, y in [(a, a2), (v, v2), (h, h2)]:
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    # Exercise a nonzero entity path, without taking an optimizer step.
    with torch.no_grad():
        entity.entity_actor.weight.normal_(0, 0.01)
        entity.entity_context.weight.normal_(0, 0.01)
    sequence, _, last = entity(obs, h0, reset_before=resets)
    pieces = []
    hidden = h0
    for i in range(5):
        piece, hidden = entity.forward_actor(obs[:, i], hidden, reset_before=resets[:, i])
        pieces.append(piece)
    torch.testing.assert_close(sequence, torch.stack(pieces, 1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(last, hidden, rtol=1e-5, atol=1e-6)
    assert not torch.equal(sequence, a2)
    entity.isolated_value(obs).square().mean().backward()
    assert all(p.grad is None for n, p in entity.named_parameters() if not n.startswith("critic."))
    entity.zero_grad(set_to_none=True)
    entity.forward_actor(obs, h0, reset_before=resets)[0].square().mean().backward()
    assert entity.entities.car.weight.grad.abs().sum() > 0
    assert entity.entities.attention.in_proj_weight.grad.abs().sum() > 0
