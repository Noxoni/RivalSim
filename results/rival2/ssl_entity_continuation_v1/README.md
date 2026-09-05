# Entity-aware joint-control continuation v1

This is continuation of the completed entity-aware +100 pilot, **not a restart**
and not an SSL promotion. User authorization remains continued development and
training until stopped. The immutable pilot authority still stops at +100;
this separate prospectively published authority governs later updates.

## Why continue this checkpoint

The fixed +100 development evaluation reached 51/64 acquisition cases and 35
finishing goals, versus the original hybrid parent's 16/64 and 14. More
importantly, the predeclared ten-full-match Nexto comparison recorded 245
contacts versus 127 and 324 concessions versus 408. Every paired team/layout
had more contacts and fewer concessions. Neither policy scored or won any
match. These are basic-gameplay gains, not strong play or proof that attention
alone caused improvement. Kickoff competitiveness and natural scoring remain
unresolved. Detailed unchanged comparison: `../ssl_entity_joint_control_v1/`.

Parent: `checkpoints/rival2/ssl_entity_joint_control_v1/plus_100.pt`

SHA256: `B5F7D19471257758966EB0B407797CFCFBD0F6CE8E86BCF9E13D41FBA4EA7ABA`

The actor, recurrent state parameters, attention, critic and **existing Adam
moments/counters** are loaded. The parent has 18,200 optimizer steps and
589,824,000 pilot agent decisions. The optimizer is not restarted. These
weights originate from the fresh 30 Hz lineage, not V5 or Human BC. Physical
episodes and live recurrent hidden states start fresh; checkpoint provenance
explicitly does not claim exact simulator-state resumption.

## Unchanged learning contract

- 120 Hz physics, 30 Hz policy, four-tick action hold; 32,768 worlds.
- 90-decision/three-second rollouts; complete recurrent sequences and real
  episode reset masks; gamma .995 and lambda .9973145188572297.
- Same entity-aware recurrent model, 182 native observations, 90 joint actions
  producing eight native controls. No router, copied Nexto parameters or script
  controls Rival. Same independently trained multilayer critic.
- Policy/trunk LR 1e-4, critic LR 3e-4, two epochs, PPO clip .2,
  max gradient norm .5, value coefficient .5, entropy coefficient .001.
- Exactly the same +/-10 terminal outcome and seven discounted potential
  differences, weights, physically valid scenario bank and reset cadence.
  No action, touch, named-mechanic, speed, demo or occupancy rewards added.
- KL telemetry only. Full parameter/gradient/Adam nonfinite checks remain;
  corrupt updates restore model/Adam/shuffle RNG and stop. No KL rejection.

## Staged Nexto, without contaminating PPO samples

Continue the original fresh30Hz schedule: two consecutive scheduled acquisition
evaluations with touch coverage >= .60, conditional median first touch <=5s,
and at least one finishing goal. +100 qualifies; +50 did not. Initial streak
is one and Nexto probability remains zero. If +150 qualifies, future physical
episode resets select 20% Nexto and 80% current self-play. No mid-episode
opponent changes. Activation is persistent thereafter, as in the old schedule.

Both current agents supply PPO samples in current self-play. Against Nexto,
only the current player supplies actor/critic/entropy/advantage samples.
Nexto's frozen adapter emits actual controls at each physics tick, retaining
its native 15 Hz neural cadence and stock kickoff behavior. Its policy is
inference-only. Opponent family and learner side are redrawn only at physical
resets, and all advantage statistics exclude ignored samples. Full sequences
containing a reset can change learner eligibility; per-tick masks and recurrent
resets preserve this distinction. Entirely ineligible sequences are skipped.

The categorical buffer stores dummy current-policy proposals for ignored
opponent slots, **not** native Nexto action targets. Tests observe actual bridge
inputs every physics tick, prove unchanged learner controls, and prove the
masked opponent actions/rewards cannot influence the update. Exposure counts
record actual trainable decisions, separately from physical world ticks.
Adam step counts are stored explicitly because mixed updates need not have
the pilot's fixed 182 minibatches.

## Validation and prospective freeze

`focused_tests.xml`: seven focused tests, including exact pure-self-play
equivalence to the frozen pilot, family normalization, ignored-opponent
invariance, two-boundary scheduling, corruption rollback, native collector
parity, reset-only assignment and actual per-physics-tick Nexto controls.

`native_preflight.json`: full 32,768-world, 90-decision rollout with 20% Nexto
enabled **for preflight coverage only**, forward/backward through a complete
728-sequence minibatch, gradient/value isolation checks and exact probability
recomputation. Zero optimizer steps, unchanged parent and teacher. All checks
passed; peak Torch allocated memory 16,291,090,432 bytes and rollout 8.716s.
These are correctness/resource checks, not a learning or match result.

`authority.json` and `package.json` bind sources, parent, scenarios, evidence,
tests and preflight. New text identities normalize CRLF to LF explicitly;
checkpoint hashes are raw bytes. Remote contents must match before training.
The original pilot package and operational recovery remain independently
verified and unchanged. No observation adapter, reward, simulator kernel,
dataset, contract or old checkpoint is modified here.

## Operations

From `G:\dev\RivalSim`, using `.venv\Scripts\python.exe`:

```powershell
python benchmarks/run_rival2_ssl_entity_continuation.py preflight
python benchmarks/run_rival2_ssl_entity_continuation.py prepare
# Commit/push the sources, tests, preflight, authority and package now.
python benchmarks/run_rival2_ssl_entity_continuation.py verify
python benchmarks/run_rival2_ssl_entity_continuation.py run
```

The actual campaign launcher uses the repository venv and a hidden Windows
process with stdout/stderr redirected outside Git. External state and alternating
rolling checkpoints: `G:\dev\RivalSim-runs\ssl-entity-continuation-v1`.
All GPU operations share the original pilot/evaluation lease, preventing
simultaneous learners or competing full-match workers.

On operational interruption, preserve `latest.json` and its exact checkpoint,
inspect/fix only the operational fault, publish recovery evidence, then use:

```powershell
python benchmarks/run_rival2_ssl_entity_continuation.py run --resume <latest-path> --resume-sha256 <latest-sha256>
```

An external `STOP` file stops at the next accepted boundary. Numerical or
capability/reward faults require diagnosis, not weakening protection. No
automatic restart of old lineages or silent authority changes is permitted.
Missing scheduled boundary evaluation is performed before resumed training.

Permanent checkpoint and deterministic development evaluation every 50 global
entity updates, beginning +150; rolling checkpoint every update. No arbitrary
update/time ceiling replaces the user's continue-until-stop instruction.

After a boundary evaluation completes, reproduce its saved checkpoint and sample
ledger audit with `python benchmarks/audit_rival2_ssl_entity_boundary.py --update N`
using that exact scheduled update number. This CPU-only report does not interrupt
the learner or perform evaluation/training. See `BOUNDARY_AUDIT.md`; its integrity
PASS is not a gameplay verdict. It freezes only the completed curve prefix, never
the racing live append log.

Inspect newly completed evaluations for sustained actual progress; investigate
stagnation rather than silently changing rewards or spending billions without
measurement. Full-match follow-ups remain separate bounded comparisons and
must not compete with a live GPU learner.

The monitor reports only new evaluation results, material findings or actionable
failures. Training/serialization PASS, scenario improvement, natural match
strength, deployment and SSL capability remain separate verdicts.
