"""Opt-in direct-skills collector with the versioned native-v5 Nexto controller.

No historical runner imports this module automatically. A new campaign must bind
this implementation, sampling mode and seed explicitly. Rewards, scenarios,
learner masks, recurrent PPO and the 50/50 world allocation are inherited intact.
"""

import torch

from rivalsim.direct_skills_training import DirectSkillCollector
from third_party.nexto.adapter import NextoStateTensors
from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter, VERSION

COLLECTOR_VERSION = "RIVAL2_DIRECT_SKILLS_NATIVE_NEXTO_COLLECTOR_V1"


class DirectSkillNativeNextoCollector(DirectSkillCollector):
    def __init__(self, env, model, *, seed, nexto_sampling_mode, nexto_seed):
        self.nexto_sampling_mode = nexto_sampling_mode
        self.nexto_seed = int(nexto_seed)
        super().__init__(env, model, seed=seed)

    def assign(self, reset):
        self.nexto_probability = 0.5
        self.is_nexto.copy_(self.rows.remainder(2) == 0)
        self.side.copy_(torch.where(reset, self.env.focal.long(), self.side))
        if self.nexto is None:
            self.nexto = NextoNativeV5PolicyAdapter(
                self.env.num_envs, device=self.env.device,
                sampling_mode=self.nexto_sampling_mode, seed=self.nexto_seed,
            )
            self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        self.nexto.set_player_index(1 - self.side)
        # Scenario replacement is a new virtual bot episode, not a native match
        # kickoff notification. Continuing matches must use notify_kickoff instead.
        self.nexto.activate(reset & self.is_nexto)
        self.env.learner.copy_(self.training_mask())
        self.env.opponent_family.copy_(self.is_nexto.long())

    def opponent_checkpoint_state(self):
        return dict(
            version=COLLECTOR_VERSION, controller_version=VERSION,
            nexto_sampling_mode=self.nexto_sampling_mode,
            nexto_seed=self.nexto_seed,
            nexto_probability=self.nexto_probability,
            is_nexto=self.is_nexto.clone(), learner_side=self.side.clone(),
            generator_state=self.opponent_generator.get_state().clone(),
            native_nexto=self.nexto.checkpoint_state(),
            resume_semantics="Exact controller continuation requires matching physical "
            "world state and assignments. For a fresh-episode resume, use "
            "restore_opponent_for_fresh_episodes; this is not exact physical replay.",
        )

    def restore_opponent_for_fresh_episodes(self, state):
        """Restore RNG lineage but explicitly discard old episode controller caches.

        Call only on a collector whose environment has already created fresh
        physical episodes. Never restore a legacy four-field Nexto cache here.
        """
        if (state.get("version") != COLLECTOR_VERSION
                or state.get("controller_version") != VERSION
                or state.get("nexto_sampling_mode") != self.nexto_sampling_mode
                or state.get("nexto_seed") != self.nexto_seed
                or state.get("nexto_probability") != 0.5):
            raise ValueError("wrong native Nexto collector checkpoint identity")
        probe = torch.Generator(device=self.env.device)
        probe.set_state(state["generator_state"].cpu())
        self.nexto.load_checkpoint_state(state["native_nexto"])
        self.opponent_generator.set_state(probe.get_state())
        self.nexto.activate(torch.ones_like(self.is_nexto))
        self.assign(torch.ones_like(self.is_nexto))
