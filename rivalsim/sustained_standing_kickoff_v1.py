"""Stationary kickoff correction only; historical scenario authorities stay intact."""
from __future__ import annotations

import numpy as np

from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.sustained_acquisition_v1 import training_starts

VERSION = "RIVAL2_SUSTAINED_STANDING_KICKOFF_V1"


def specification():
    return dict(version=VERSION,
        selection="Every kickoff_indicator row, including dedicated and natural-family kickoffs",
        initialization="Exact make_standard_kickoff_state for the existing layout: both cars stationary, zero linear/angular velocity, native positions/orientations, 100/3 boost, centered stationary ball, cleared prior inputs and jump/flip state",
        preserved="Every non-kickoff state and all family/focal/layout metadata remain byte-identical; acquisition probes and full-match evaluation unchanged",
        retirement="Apply the same stationary correction to the original five-family bank after acquisition retirement; never restore assisted kickoff momentum",
        controls="Policy acts from first actionable tick; no controller prefix, speedflip script or opening-specific reward",
        rewards="Unchanged sustained-gameplay reward and goal/inactivity episode semantics",
        continuation="Exact accepted update76 model, Adam, RNG and counters; documented fresh physical episodes and cleared actor/critic memory at process restart")


def standing_training_starts(worlds, retired=False):
    batch = training_starts(worlds, retired)
    rows = np.flatnonzero(batch.kickoff_indicator)
    if len(rows):
        native = make_standard_kickoff_state(len(rows), batch.kickoff_layout[rows])
        for name in batch.state.__dataclass_fields__:
            getattr(batch.state, name)[rows] = getattr(native, name)
    batch.state.validate()
    return batch


def validate_resume(source, source_sha256, package, package_sha256):
    """An old checkpoint may enter only at the prospectively frozen stop boundary."""
    identity = source.get("standing_kickoff_amendment")
    if identity is None:
        assert source["accepted_updates"] == package["resume"]["accepted_updates"]
        assert source_sha256 == package["resume"]["sha256"]
    else:
        assert identity == dict(version=VERSION, sha256=package_sha256,
                                from_update=package["resume"]["accepted_updates"])
        assert source["accepted_updates"] >= package["resume"]["accepted_updates"]
