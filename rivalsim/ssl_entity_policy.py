"""Entity-aware recurrent joint-control actor; one network, no expert/router.

The external native observation remains exactly 182 fields. Explicit ball/car/
pad tokens expose their structure to self-query cross attention. Static pad
coordinates come from the same canonical Soccar map as the native observation.
No extra sensors, inferred mechanics, target actions or scenario IDs are added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.rival2_contracts import OBS_FIELD_NAMES, POSITION_SCALE
from rivalsim.ssl_joint_control_policy import (
    POLICY_VERSION as JOINT_VERSION,
)
from rivalsim.ssl_joint_control_policy import (
    JointControlActorCritic,
    JointControlConfig,
)

ENTITY_VERSION = "RIVAL2_ENTITY_JOINT_CONTROL90_RECURRENT_V1"
ENTITY_WIDTH = 64
ENTITY_HEADS = 4
ENTITY_COUNT = 38  # self, opponent, ball, 34 pads, history/lifecycle context


@dataclass(frozen=True, slots=True)
class EntityPolicyConfig(JointControlConfig):
    version: str = ENTITY_VERSION
    entity_width: int = ENTITY_WIDTH
    entity_heads: int = ENTITY_HEADS
    entity_count: int = ENTITY_COUNT

    def __post_init__(self):
        values = asdict(self)
        if values.pop("version") != ENTITY_VERSION:
            raise ValueError("entity policy version mismatch")
        for name, expected in (
            ("entity_width", ENTITY_WIDTH),
            ("entity_heads", ENTITY_HEADS),
            ("entity_count", ENTITY_COUNT),
        ):
            if values.pop(name) != expected:
                raise ValueError(f"unsupported {name}")
        JointControlConfig(version=JOINT_VERSION, **values)


def entity_schema():
    """Serializable, field-name-level mapping; no opaque guessed indices."""
    return dict(
        version=ENTITY_VERSION,
        count=ENTITY_COUNT,
        ball=list(OBS_FIELD_NAMES[0:9]) + list(OBS_FIELD_NAMES[87:93]),
        self_car=list(OBS_FIELD_NAMES[9:48]),
        self_relative="six exact zero self-to-self position/velocity components",
        opponent=list(OBS_FIELD_NAMES[48:87]) + list(OBS_FIELD_NAMES[93:99]),
        pads=[list(OBS_FIELD_NAMES[99 + 2 * i : 101 + 2 * i]) for i in range(34)],
        pad_geometry="Canonical SOCCAR_PAD_POSITIONS / POSITION_SCALE; large flag for indices "
        "0..5; canonical pad minus self position. Orange pad remap already applied "
        "by native observation.",
        context=list(OBS_FIELD_NAMES[167:182]),
        token_types=["self", "opponent", "ball", "boost_pad", "history_lifecycle"],
        attention="One 64-wide four-head self-query cross-attention over 38 tokens, "
        "dropout=0. No full quadratic entity self-attention.",
        architecture="Learned entity residual into both actor logits and pre-GRU context. "
        "Independent MLP critic unchanged; value loss cannot train entity/actor features.",
        initialization="Zero entity output projections preserve the joint-control policy "
        "logits, value and hidden exactly. Joint-control head projection itself is NOT "
        "hybrid policy parity.",
        no_task_id=True,
        supported_game="Soccar 1v1; this 182-field contract does not expose additional players.",
    )


class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.car = nn.Linear(45, ENTITY_WIDTH)
        self.ball = nn.Linear(15, ENTITY_WIDTH)
        self.pad = nn.Linear(9, ENTITY_WIDTH)
        self.context = nn.Linear(15, ENTITY_WIDTH)
        self.types = nn.Embedding(5, ENTITY_WIDTH)
        self.norm = nn.LayerNorm(ENTITY_WIDTH)
        self.attention = nn.MultiheadAttention(
            ENTITY_WIDTH, ENTITY_HEADS, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(ENTITY_WIDTH), nn.Linear(ENTITY_WIDTH, ENTITY_WIDTH), nn.SiLU()
        )
        self.register_buffer(
            "pad_positions",
            torch.as_tensor(SOCCAR_PAD_POSITIONS.copy()) / torch.tensor(POSITION_SCALE),
        )
        self.register_buffer("pad_large", (torch.arange(34) < 6).float().reshape(34, 1))
        self.register_buffer("type_indices", torch.tensor([0, 1, 2, *([3] * 34), 4]))

    def raw_groups(self, observation):
        if observation.ndim != 2 or observation.shape[-1] != 182:
            raise ValueError("entity encoder expects [N,182]")
        batch = len(observation)
        ball = torch.cat((observation[:, :9], observation[:, 87:93]), -1)
        own = torch.cat((observation[:, 9:48], observation.new_zeros(batch, 6)), -1)
        opponent = torch.cat((observation[:, 48:87], observation[:, 93:99]), -1)
        positions = self.pad_positions.expand(batch, -1, -1)
        pads = torch.cat(
            (
                positions,
                self.pad_large.expand(batch, -1, -1),
                observation[:, 99:167].reshape(batch, 34, 2),
                positions - observation[:, None, 9:12],
            ),
            -1,
        )
        return own, opponent, ball, pads, observation[:, 167:182]

    def tokens(self, observation):
        own, opponent, ball, pads, context = self.raw_groups(observation)
        cars = self.car(torch.stack((own, opponent), 1))
        encoded = torch.cat(
            (cars, self.ball(ball)[:, None], self.pad(pads), self.context(context)[:, None]), 1
        )
        return self.norm(torch.nn.functional.silu(encoded + self.types(self.type_indices)))

    def attend(self, tokens, *, weights=False):
        result, attention = self.attention(
            tokens[:, :1], tokens, tokens, need_weights=weights, average_attn_weights=False
        )
        return self.output(result[:, 0] + tokens[:, 0]), attention

    def forward(self, observation):
        return self.attend(self.tokens(observation))[0]


class EntityJointControlActorCritic(JointControlActorCritic):
    def __init__(self):
        super().__init__()
        self.config = EntityPolicyConfig(
            log_std_min=self.config.log_std_min, log_std_max=self.config.log_std_max
        )
        self.entities = EntityEncoder()
        self.entity_actor = nn.Linear(ENTITY_WIDTH, 90, bias=False)
        self.entity_context = nn.Linear(ENTITY_WIDTH, self.config.context_encoder_dim, bias=False)
        nn.init.zeros_(self.entity_actor.weight)
        nn.init.zeros_(self.entity_context.weight)

    def _forward(
        self,
        observation,
        hidden=None,
        *,
        reset_before=None,
        reset_metadata=None,
        include_value=True,
    ):
        # Deliberately separate from the frozen production hybrid path.
        step = observation.ndim == 2
        if step:
            observation = observation.unsqueeze(1)
            if reset_before is not None and reset_before.ndim == 1:
                reset_before = reset_before.unsqueeze(1)
        if observation.ndim != 3 or observation.shape[-1] != 182:
            raise ValueError("entity policy expects [B,182] or [B,T,182]")
        batch, ticks, _ = observation.shape
        if hidden is None:
            hidden = self.initial_hidden(batch, device=observation.device, dtype=observation.dtype)
        if hidden.shape != (1, batch, self.config.context_hidden_dim):
            raise ValueError("entity policy hidden shape mismatch")
        flat = observation.reshape(-1, 182)
        features = self.trunk(flat)
        entities = self.entities(flat).reshape(batch, ticks, ENTITY_WIDTH)
        encoded = self.context_activation(
            self.context_encoder(observation) + self.entity_context(entities)
        )
        context, next_hidden = self._context_with_resets(
            encoded, hidden, reset_before, reset_metadata
        )
        actor = (
            self.actor(features).reshape(batch, ticks, 90)
            + self.context_actor(context)
            + self.entity_actor(entities)
        )
        value = (
            self._value_from_features(flat, features).reshape(batch, ticks)
            if include_value
            else None
        )
        if step:
            actor = actor[:, 0]
            if value is not None:
                value = value[:, 0]
        return actor, value, next_hidden
