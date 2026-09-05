"""Reset-aware GRU execution without changing recurrent model/state semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

RESET_EXECUTION_VERSION = "RIVAL2_RESET_BOUNDARY_GRU_V1"


@dataclass(frozen=True)
class ResetMetadata:
    """Minibatch-local reset times, reusable only with the same unmodified mask."""

    mask: torch.Tensor
    version: int
    boundaries: tuple[int, ...]

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> ResetMetadata:
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("cached reset mask must be bool [batch, sequence]")
        return cls(mask, mask._version, tuple(mask.any(dim=0).nonzero().flatten().cpu().tolist()))

    def validate(self, mask: torch.Tensor) -> None:
        if mask is not self.mask or mask._version != self.version:
            raise ValueError("reset metadata does not match this unmodified mask")


def gru_reset_spans(
    gru: nn.GRU,
    encoded: torch.Tensor,
    hidden: torch.Tensor,
    reset_before: torch.Tensor | None,
    metadata: ResetMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse reset-free spans; preserve gradients until each row's actual reset.

    A reset at t=0 must not make a whole rollout run as single-tick GRU calls.
    Find the union of reset times with one metadata transfer, zero only the
    affected rows at each boundary, and retain the full autograd graph. There
    are no detached states, packed/padded episode copies, or parameter changes.
    The one-tick inference path needs no host synchronization at all.
    """
    if reset_before is None:
        if metadata is not None:
            raise ValueError("reset metadata requires a mask")
        return gru(encoded, hidden)
    if reset_before.shape != encoded.shape[:2]:
        raise ValueError("reset_before must have shape [batch, sequence]")
    resets = reset_before.to(torch.bool)
    if metadata is not None:
        metadata.validate(reset_before)
    if encoded.shape[1] == 1:
        return gru(encoded, hidden.masked_fill(resets[:, 0].view(1, -1, 1), 0.0))
    if metadata is not None:
        boundaries = metadata.boundaries
    else:
        boundaries = resets.any(dim=0).nonzero().flatten().cpu().tolist()
    if not boundaries:
        return gru(encoded, hidden)
    outputs = []
    start = 0
    for boundary in boundaries:
        if boundary > start:
            output, hidden = gru(encoded[:, start:boundary], hidden)
            outputs.append(output)
        hidden = hidden.masked_fill(resets[:, boundary].view(1, -1, 1), 0.0)
        start = boundary
    output, hidden = gru(encoded[:, start:], hidden)
    outputs.append(output)
    return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], hidden
