"""Explicit new collector: ordinary categorical exploration, native Nexto, two memories."""
import torch

from rivalsim.ssl_entity_mixed_training import MixedEntityRolloutCollector
from rivalsim.sustained_gameplay_v1 import COMPONENTS, SEED, ppo_config
from third_party.nexto.adapter import NextoStateTensors
from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter


class SustainedCollector(MixedEntityRolloutCollector):
    def __init__(self, env, model, seed=SEED):
        super().__init__(env, model, seed=seed, config=ppo_config(),
                         reward_component_names=COMPONENTS)

    def assign(self, reset):
        self.nexto_probability = .5
        self.is_nexto.copy_(self.rows.remainder(2) == 0)
        self.side.copy_(torch.where(reset, self.env.focal.long(), self.side))
        if self.nexto is None:
            self.nexto = NextoNativeV5PolicyAdapter(self.env.num_envs, device=self.env.device,
                sampling_mode="native_v5", seed=SEED+3)
            self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        self.nexto.set_player_index(1-self.side)
        self.nexto.activate(reset & self.is_nexto)
        self.env.learner.copy_(self.training_mask())
        self.env.opponent_family.copy_(self.is_nexto.long())

    def opponent_checkpoint_state(self):
        return dict(native_nexto=self.nexto.checkpoint_state(),
                    is_nexto=self.is_nexto.clone(), learner_side=self.side.clone(),
                    generator_state=self.opponent_generator.get_state())

    def collect(self):
        self.env.statistics.zero_()
        result = super().collect()
        from rivalsim.direct_skills_v1 import NAMES
        rows = self.env.statistics.cpu().tolist()
        self.last_metrics["by_start_family_and_opponent"] = {
            f"{name}_{opponent}": dict(zip(self.env.stat_names, rows[i*2+j], strict=True))
            for i,name in enumerate(NAMES) for j,opponent in enumerate(("selfplay","nexto"))}
        return result
