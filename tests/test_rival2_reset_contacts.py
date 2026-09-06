"""Native post-teleport cache invalidation; no optimizer or reward change."""

from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic


def test_all_layouts_masked_reset_contacts_and_policy_parity():
    root = Path('G:/dev/RLBot-Rival/bot/collision_meshes')
    if not torch.cuda.is_available() or not root.exists():
        pytest.skip('requires native CUDA simulator and arena')
    world = Rival2WorldSim(10, root, device='cuda:0', seed=192,
                          kickoff_selector=np.arange(10, dtype=np.int32) % 5,
                          car_lifecycle_seed=192)
    bridge = Rival2TensorBridge(world)
    initial = bridge.observation().clone()
    checkpoint = torch.load('checkpoints/rival2/direct_skills_v1/plus_000300.pt',
                            map_location='cpu', weights_only=False)
    model = EntityJointControlActorCritic().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    with torch.inference_mode():
        before_actions = model.deterministic(model.forward_actor(initial[:5].reshape(10, 182))[0])
    wheels = bridge.views['wheel_contact'].reshape(10, 2, 4)
    for pattern in (1, 0, 1):
        wheels.fill_(pattern)
        # Only the first half is reset; all five layouts and both perspectives.
        wp.to_torch(world.lifecycle.kickoff_selector).copy_(torch.arange(10, device='cuda').int() % 5)
        wp.to_torch(world.rival2.reset_mask).copy_((torch.arange(10, device='cuda') < 5).int())
        unselected_before = bridge.observation()[5:].clone()
        world.apply_interval_resets()
        actual = bridge.observation()
        assert not wheels[:5].any()
        assert (wheels[5:] == pattern).all()
        torch.testing.assert_close(actual[5:], unselected_before, rtol=0, atol=0)
        torch.testing.assert_close(actual[:5], initial[:5], rtol=0, atol=1e-6)
        with torch.inference_mode():
            after_actions = model.deterministic(model.forward_actor(actual[:5].reshape(10, 182))[0])
        assert torch.equal(before_actions, after_actions)
    assert all(torch.equal(v.cpu(), checkpoint['model'][k]) for k, v in model.state_dict().items())
