"""Read-only entity/joint90 inference and 30 Hz recurrent packet scheduling.

This does not reconstruct RLBot observations or claim packet-domain parity.
The scripted actor contains only copied inference weights, no critic/optimizer.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Hashable

import torch
from torch import nn

BRIDGE_VERSION = "RIVAL2_ENTITY_JOINT90_NATIVE_BRIDGE_V1"


class ExportEntities(nn.Module):
    """Scriptable transcription of EntityEncoder, preserving operation order."""

    def __init__(self, source):
        super().__init__()
        for name in ("car", "ball", "pad", "context", "types", "norm", "attention", "output"):
            setattr(self, name, deepcopy(getattr(source, name)))
        for name in ("pad_positions", "pad_large", "type_indices"):
            self.register_buffer(name, getattr(source, name).detach().clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch = observation.size(0)
        ball = torch.cat((observation[:, :9], observation[:, 87:93]), -1)
        own = torch.cat((observation[:, 9:48], observation.new_zeros((batch, 6))), -1)
        opponent = torch.cat((observation[:, 48:87], observation[:, 93:99]), -1)
        positions = self.pad_positions.expand(batch, -1, -1)
        pads = torch.cat((positions, self.pad_large.expand(batch, -1, -1),
                          observation[:, 99:167].reshape(batch, 34, 2),
                          positions - observation[:, None, 9:12]), -1)
        cars = self.car(torch.stack((own, opponent), 1))
        encoded = torch.cat((cars, self.ball(ball)[:, None], self.pad(pads),
                             self.context(observation[:, 167:182])[:, None]), 1)
        tokens = self.norm(torch.nn.functional.silu(encoded + self.types(self.type_indices)))
        result, _ = self.attention(tokens[:, :1], tokens, tokens,
                                   need_weights=False, average_attn_weights=False)
        return self.output(result[:, 0] + tokens[:, 0])


class EntityNativeActor(nn.Module):
    """Single-decision export; reset hidden externally at real episode boundaries."""

    def __init__(self, source):
        super().__init__()
        self.hidden_dim = int(source.config.context_hidden_dim)
        self.entities = ExportEntities(source.entities)
        for name in ("trunk", "actor", "entity_actor", "entity_context", "context_encoder",
                     "context_activation", "context_gru", "context_actor"):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.register_buffer("action_table", source.action_table.detach().clone())
        self.requires_grad_(False)
        self.eval()

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor):
        if observation.dim() != 2 or observation.size(1) != 182:
            raise RuntimeError("Expected observation [B,182]")
        if hidden.dim() != 3 or hidden.size(0) != 1 or hidden.size(1) != observation.size(0) or hidden.size(2) != self.hidden_dim:
            raise RuntimeError("Expected recurrent state [1,B,H]")
        features = self.trunk(observation)
        entities = self.entities(observation)[:, None]
        encoded = self.context_activation(self.context_encoder(observation[:, None])
                                          + self.entity_context(entities))
        context, next_hidden = self.context_gru(encoded, hidden)
        logits = (self.actor(features)[:, None] + self.context_actor(context)
                  + self.entity_actor(entities))[:, 0]
        return self.action_table[logits.argmax(-1)], next_hidden, logits


class RecurrentPacketScheduler:
    """One local car, one observed decision per four delivered physics ticks.

    Caller supplies a stable match/car/reset identity and active state. Lifecycle
    receives every unique active packet, including held ticks, before observation
    construction. Duplicate/backward packets in the same identity are ignored.
    After a gap we infer once from the received observation and reanchor cadence;
    skipped decisions are counted, never invented. Identity change resets state.
    Inactive->active also resets: no stale controls/memory across a kickoff pause.
    This explicit discontinuity is reported, not represented as exact history.
    """

    def __init__(self, model, hidden_dim: int):
        self.model = model
        self.hidden_dim = int(hidden_dim)
        self.hidden = torch.zeros(1, 1, self.hidden_dim)
        self.action = torch.zeros(8)
        self.last_frame = None
        self.last_decision = None
        self.identity = None
        self.was_active = False
        self.stats = dict(packets=0, decisions=0, held=0, duplicates=0, out_of_order=0,
                          missed_ticks=0, skipped_decisions=0, resets=0)

    def step(self, *, frame: int, identity: Hashable, active: bool,
             observation: Callable[[], torch.Tensor],
             lifecycle: Callable[[int, bool], None]) -> torch.Tensor:
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("Need nonnegative engine physics frame identity")
        self.stats["packets"] += 1
        if not active:
            self.was_active = False
            return torch.zeros(8)
        reset = identity != self.identity or not self.was_active
        if not reset and self.last_frame is not None:
            if frame == self.last_frame:
                self.stats["duplicates"] += 1
                return self.action.clone()
            if frame < self.last_frame:
                self.stats["out_of_order"] += 1
                return self.action.clone()
        delta = 0 if reset else frame - self.last_frame
        if reset:
            self.hidden = torch.zeros(1, 1, self.hidden_dim)
            self.action = torch.zeros(8)
            self.last_decision = None
            self.stats["resets"] += 1
        else:
            self.stats["missed_ticks"] += max(0, delta - 1)
        lifecycle(delta, reset)
        self.last_frame, self.identity, self.was_active = frame, identity, True
        if self.last_decision is not None and frame - self.last_decision < 4:
            self.stats["held"] += 1
            return self.action.clone()
        obs = observation()
        if obs.shape != (1, 182) or obs.dtype != torch.float32 or not torch.isfinite(obs).all():
            raise RuntimeError("Invalid policy observation; no inference accepted")
        with torch.inference_mode():
            action, hidden, logits = self.model(obs, self.hidden)
        if (action.shape != (1, 8) or hidden.shape != (1, 1, self.hidden_dim)
                or logits.shape != (1, 90)
                or not all(bool(torch.isfinite(x).all()) for x in (action, hidden, logits))
                or not bool((action[:, :5].abs() <= 1).all())
                or not bool(((action[:, 5:] == 0) | (action[:, 5:] == 1)).all())):
            raise RuntimeError("Invalid actor output; no hidden/action accepted")
        if self.last_decision is not None:
            self.stats["skipped_decisions"] += max(0, (frame - self.last_decision) // 4 - 1)
        self.hidden, self.action = hidden.detach().clone(), action[0].detach().clone()
        self.last_decision = frame
        self.stats["decisions"] += 1
        return self.action.clone()
