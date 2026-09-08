"""Scenario-only ablation with new random weights, not a new reward contract."""
from dataclasses import asdict, replace

import numpy as np

from rivalsim.sustained_acquisition_v1 import acquisition_starts, SEED as BANK_SEED, specification
from rivalsim.sustained_gameplay_v1 import fresh_model, ppo_config, reward_authority, POLICY_VERSION

VERSION = "RIVAL2_FRESH_ACQUISITION_ONLY_V1"
SEED = 2026090810
MAX_UPDATES = 150
WORLDS = 32768


def training_bank(worlds=WORLDS):
    if worlds < 8 or worlds % 8:
        raise ValueError("World count must be divisible by eight")
    # Exactly the same 16,384 acquisition examples used in the previous full bank.
    # Each appears once in the fixed even Nexto lanes and once in odd self-play
    # lanes. Do not correlate easy/varied alternating source rows with opponents.
    batch = acquisition_starts(worlds // 2, BANK_SEED)
    rng = np.random.default_rng(SEED + 1)
    order = np.empty(worlds, dtype=np.int64)
    order[::2] = rng.permutation(worlds // 2)
    order[1::2] = rng.permutation(worlds // 2)
    state = type(batch.state)(**{name: getattr(batch.state, name)[order].copy()
                                for name in batch.state.__dataclass_fields__})
    batch = replace(batch, state=state, **{name: getattr(batch, name)[order].copy()
        for name in ("family", "focal_side", "kickoff_indicator", "kickoff_layout", "wall_aerial_variant")})
    batch.state.validate()
    return batch


def new_model():
    return fresh_model(SEED)


def authority():
    return dict(version=VERSION, parent=None, seed=SEED, worlds=WORLDS,
        policy_version=POLICY_VERSION, ppo=asdict(ppo_config()), reward=reward_authority(),
        initialization="New random actor/entities/actor GRU and independent critic MLP/GRU; empty fresh Adam. No trained checkpoint initializes this experiment.",
        scenarios="100% acquisition starts; same underlying acquisition source population as mixed campaign, duplicated across opponent families and shuffled within each. No other reset family, scripted control or automatic curriculum graduation.",
        source_acquisition_seed=BANK_SEED,
        source_unique_states=WORLDS//2, reset_bank_rows=WORLDS,
        opponent_worlds=dict(current_selfplay=.5, native_v5_nexto=.5),
        opponent_semantics="Existing collector unchanged: both current agents learn in self-play; only current side learns vs Nexto. Opponent sampler seed unchanged.",
        physics_hz=120, policy_hz=30, held_physics_ticks=4,
        episode_semantics="Unchanged: continue after touch until goal, or45s without either car contacting. No first-touch bonus or short first-touch terminal; no new task reward.",
        exploration="Unchanged joint90 categorical temperature1, entropy0.001, KL telemetry only, corruption rollback and stop.",
        probe=specification()["probe"], probe_seed=BANK_SEED+100,
        evaluation_offsets=list(range(0,MAX_UPDATES+1,10)), max_updates=MAX_UPDATES,
        budget="Bounded150-update diagnostic to observe early learning and persistence through the previous regression interval. STOP or numerical/corruption failure ends earlier. No automatic extension or promotion.",
        control_limit="Fresh initialization and scenario mixture both differ from the stopped trained model. Improvement is evidence of learnability under isolation, not causal proof of which old setting harms growth. No fresh mixed-scenario control or multi-seed replication is included.",
        resume="Exact latest same-lineage model/Adam/counters/RNG only. Process restart creates fresh physical episodes and clears both memories; no synthetic rewards or GAE splice. In-process evaluations preserve training worlds and memories.")
