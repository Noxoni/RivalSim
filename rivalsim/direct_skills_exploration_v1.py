"""One explicitly versioned on-policy exploration setting; no actor weight edits."""

from __future__ import annotations

from rivalsim.ssl_entity_policy import EntityJointControlActorCritic

VERSION = "RIVAL2_DIRECT_SKILLS_EXPLORATION_T2_V1"
TEMPERATURE = 2.0


class TrainingExplorationPolicy(EntityJointControlActorCritic):
    """Same state_dict/actor architecture, softened logits for training only.

    Both collection and PPO call this same forward, so their likelihoods refer
    to the same distribution. Native deterministic deployment/evaluation uses
    the original class and argmax. No temperature tensor is a learned parameter.
    """

    def _forward(self, *args, **kwargs):
        logits, value, hidden = super()._forward(*args, **kwargs)
        return logits / TEMPERATURE, value, hidden
