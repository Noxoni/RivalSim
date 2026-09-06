"""Use the existing recurrent PPO engine with fixed, sample-visible opponent slots."""

import torch

from rivalsim.direct_skills_v1 import NAMES
from rivalsim.ssl_entity_mixed_training import MixedEntityRolloutCollector
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors


class DirectSkillCollector(MixedEntityRolloutCollector):
    def assign(self, reset):
        # Fixed 50% world allocation => exactly one third of learner samples face
        # Nexto regardless of episode duration. The other half is evolving self-play.
        self.nexto_probability = 0.5
        self.is_nexto.copy_(self.rows.remainder(2) == 0)
        self.side.copy_(torch.where(reset, self.env.focal.long(), self.side))
        if self.nexto is None:
            self.nexto = NextoPolicyAdapter(self.env.num_envs, device=self.env.device)
            for parameter in self.nexto.actor.parameters():
                parameter.requires_grad_(False)
            self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        self.nexto.set_player_index(1 - self.side)
        self.nexto.activate(reset & self.is_nexto)
        self.env.learner.copy_(self.training_mask())
        self.env.opponent_family.copy_(self.is_nexto.long())

    def accept_evaluation(self, result):
        return 0.5  # Evaluation never silently changes the prospective opponent mix.

    def collect(self):
        self.env.reset_statistics()
        result = super().collect()
        stats = self.env.statistics.cpu().tolist()
        self.last_metrics["by_reward_role_and_opponent"] = {
            f"{role}_{opponent}": dict(zip(self.env.stat_names, stats[i * 2 + j], strict=True))
            for i, role in enumerate(NAMES)
            for j, opponent in enumerate(("selfplay", "nexto"))
        }
        return result
