# Fresh sustained-gameplay restart

This is a new random-weight lineage, **not a continuation of update 780**, V5,
Human BC, or any prior PPO optimizer. Old checkpoints and STOP files are retained.
Implementation integrity is not a claim of learned gameplay capability.

## User-facing behavior

The existing 32,768-state start bank remains unchanged: natural play, challenge,
finishing, defense and kickoff. It includes both genuine standing kickoffs and
the previously used momentum-assisted kickoff exercises. The policy receives
no scenario ID, scripted actions, or task-specific reward. Native controller
outputs are its own from the first decision.

Every start becomes sustained play. A touch, won challenge, save, control change,
or the old 12/30-second episode age **does not end it**. It ends on a native goal,
or after **45 seconds with no actual car-ball contact by either player**. Continuous
physical contact counts as activity, not only new contact onsets. Moving without
contact does not reset this clock. Thus 45 seconds is an inactivity timeout, not
a hard cap on an actively played episode.

At a goal, pay +10 to the scorer and -10 to the other player, use zero terminal
potential/value, then reset. At inactivity, apply -1 to both players and reset;
this is a truncation, not a scored terminal. The return includes the recurrent
critic's value of the **pre-reset successor**, and the GAE chain does not cross
into the new scenario. A goal overrides an inactivity flag in the same decision.

## Reward and temporal credit

Same seven state potentials and weights as the preceding natural-game reward:
field 1.25, access 0.75, control 0.75, defense 0.50, alignment 0.30, boost 0.15,
goal-directed ball velocity 0.25. All scenarios receive the same combined
`gamma*Phi(next)-Phi(before)` plus goals and the two explicit additions below.
There are no direct touch, shot projection, save, race, movement, mechanic,
flip, jump, or boost-pickup bonuses.

Control is now also an **ongoing payment**, at most **0.02 reward/second**:
native exclusive last-contact ownership multiplied by positive own-minus-opponent
physical controllability. Controllability combines 150–500 uu proximity with
0–1200 uu/s relative-speed matching. Payment continues as long as that control
persists, and stops when it does not. Simultaneous contact clears exclusive
ownership; no payment occurs on or after a goal physics tick. These are physical
proxies, not a claim to identify tactical possession perfectly. There is no
eight-frame prerequisite or once-only control bonus.

This occupancy term is explicitly **not policy-invariant shaping**. Its maximum
discounted infinite lifetime sum is approximately 0.866 reward, versus 10 for a
goal. That keeps its numerical scale small; it cannot prove that every delayed
goal is preferred to every possible control strategy. Monitor real goal outcomes
and control-without-scoring behavior instead of claiming the proxy solves them.

Reward discount half-life is **30 seconds**. GAE trace half-life is **3 seconds**.
The exact constants are derived and frozen in `authority.json`, not rounded
independent settings. A goal's discounted value is 5 after 30 seconds. A TD signal
propagated through 90 decisions has half its weight after three seconds. A
three-second rollout still bootstraps at its right boundary; this is not an
exact causal attribution algorithm or a guarantee of correct credit.

## Actor, critic and PPO

- 120 Hz native physics; 30 Hz policy; one joint90 action held for four ticks.
- Fresh entity-aware recurrent actor, unchanged external 182 fields and action
  table. No human adapter or old policy weights.
- **Independent recurrent critic**: 182→512→512→512→256 MLP, GRU256, scalar value.
  Value loss cannot train actor/entity/actor-GRU parameters.
- Actor and critic have independent hidden banks. Both persist between rollouts
  and reset only on actual goal/inactivity boundaries. Training unrolls complete
  90-decision sequences and detaches only at rollout/BPTT boundaries.
- Successor value peeks use critic state after the current observation. The peek
  state is discarded: the successor is not consumed twice by the stored memory.
- 32,768 worlds; horizon 90; two PPO epochs; minibatch budget 65,536 (728 full
  sequences); clip 0.20; value coefficient 0.50; entropy coefficient 0.001;
  global gradient clip 0.50; actor LR 1e-4, critic LR 3e-4; fresh Adam.
- Categorical temperature 1. No inherited T2/uniform-log-barrier intervention
  intended to dislodge the old saturated policy. Initial categorical entropy is
  near log(90); this is genuinely exploratory random initialization.
- 50% worlds current self-play, 50% native-v5 Nexto. Both self-play agents train;
  Nexto is inference-only. Consequently one-third of learner samples face Nexto.
- Family-local advantage normalization. KL telemetry only, no preservation loss
  or KL rejection. Nonfinite/corruption protection and complete-update rollback
  remain; there is no automatic capability-changing recovery.

## Validation and operation

Focused tests include timing/GAE, absorbing goals on all four hold ticks,
truncation bootstrap, old-timeout suppression, new-clock behavior, two-memory
sequence/step parity, critic-gradient isolation, Nexto masks, finite PPO and
unchanged behavior of the old stateless-critic path. Disposable unit-test models
take optimizer steps; the full-scale campaign preflight takes **zero** steps.

`preflight_32768.json` records actual full-scale rollout/gradient/memory evidence.
The initialized checkpoint, complete authority, source identities, scenario
identity and passing evidence are committed/pushed before the campaign starts.
The runner compares those artifacts against `origin/main` before training.

```powershell
.\.venv\Scripts\python.exe benchmarks/run_sustained_gameplay_v1.py verify
.\.venv\Scripts\python.exe benchmarks/run_sustained_gameplay_v1.py run
```

External runtime: `G:\dev\RivalSim-runs\sustained-gameplay-v1`.
An existing `STOP` file requests a stop after the current accepted update.
Alternating atomic rolling checkpoints are saved each update. Permanent snapshots
are kept at initialization, +1, +10, +20 and every 50. Evaluation runs at 0, 10,
20 and every 50 using ten complete deterministic Nexto development matches,
five standing kickoff layouts on both sides. Evaluations do not change reward,
weights, selection, exploration or training scenario state.

There is no fixed total update cap: the latest user direction is continue until
stopped. Numerical/corruption or evaluation execution failures stop the process.
The monitor must distinguish training telemetry from completed match evaluation,
publish only newly completed evaluations, and never call a changing contact rate
proof of SSL ability.

Process-resume limitations are explicit. Saved policy, Adam and RNG are resumable,
but a process restart initializes fresh physical worlds and clears both memories;
it is **not exact physical-world replay**. Unfinished old episodes are not assigned
fake terminal rewards or joined to new GAE. In-process checkpointing and evaluation
do not reset worlds. Old lineages cannot pass the new loader identity checks.
