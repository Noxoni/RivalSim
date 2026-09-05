# Fresh random-weight 30 Hz ground PPO v1

This is a **new random initialization**, not Unified V5, BC, or an SSL descendant.
The user authorized continuous training **until they stop it**, superseding the
old 100-update cap and any proposed ten-hour budget. No deadline/update ceiling
is embedded in this runner. Previous checkpoints and concurrent work remain intact.

## Frozen experiment

- 32,768 worlds; 120 Hz physics; 30 Hz current-policy decisions; four-tick hold.
- 90-decision/three-second recurrent rollout; gamma 0.995;
  lambda 0.9973145188572297. The GAE discount-trace product has a three-second
  half-life. This is credit assignment, not a three-second predictive world model.
- Reused **architecture implementation only**: 3x512 SiLU actor trunk plus
  GRU256 residual context; separately initialized 3x512 critic. The inherited
  architecture-version string does not mean inherited learned capabilities.
- Fresh Adam; actor/recurrent parameters 1e-4, critic 3e-4; two epochs;
  clip 0.20, value coefficient 0.50, global gradient norm 0.50.
- Learned analog standard deviation, initialized at 0.65 and bounded to
  [0.20,0.90], is used identically in sampling, old/new likelihood, entropy,
  and diagnostics. Buttons are learned Bernoulli distributions at temperature 1.
  Initial jump/boost/handbrake probabilities are 0.12/0.35/0.08, not scripted
  controls. Entropy regularization is 0.001. These are prospective choices for
  exploring from a random policy, not claims of an optimal configuration.
- All actor/trunk/recurrent and critic parameters learn; critic loss has **no
  gradient path into actor features**. Complete sequences and episode reset masks
  are preserved; no frame-shuffling through the GRU. Effective minibatches hold
  728 complete 90-step sequences (65,520 samples), with the final remainder.
- KL is telemetry only. No preservation loss, KL stop, KL rejection, LR backoff,
  or KL rollback. Nonfinite loss/gradient/parameters and corruption stop training
  with transactional rollback. No automatic reward/capability changes to repair them.

## Reward

`r = terminal_outcome + gamma * Phi(next) - Phi(current)`.

| Bounded state potential | Weight | Definition |
| --- | ---: | --- |
| Field progress | 1.25 | Existing canonical ball Y / 5120 |
| Loose-ball access | 0.75 | `(1-max(Cself,Copp)) * (.75*exp(-dself/2000) + .25*(exp(-dself/2000)-exp(-dopp/2000)))` |
| Physical controllability advantage | 0.75 | Existing proximity and relative-velocity controllability difference |
| Defensive coverage | 0.50 | Existing threat-weighted goal-side coverage difference |
| Approach geometry | 0.30 | Existing car-ball-target geometry, not car facing direction |
| Boost resource | 0.15 | Existing bounded square-root boost fraction |
| Goal-directed ball velocity | 0.25 | Bounded dot product of ball velocity and unit ball-to-opponent-goal-center vector, divided by 3000 |

`Phi` is the weighted sum. No additional ordinary touch, named mechanic, speed,
flip, jump, save, demo, pad, or possession-occupancy reward exists. The new access
term avoids the old large-distance-gap clipping plateau; it still fades when
either car has physical controllability. The velocity potential is a direction
proxy, **not a goal/shot detector or a proven remedy**.

The physical discount is `0.995**(1/4) = 0.9987476493904754`. Four physical
potential differences telescope exactly to the decision-level difference.
Goals at sub-tick k (zero-based) pay `physical_gamma**k * (+10,-10)`; the
successor potential is absorbing zero immediately. Remaining sub-ticks cannot
leak shaping from a new episode: reset occurs once at the held-action boundary.
Native scoring latches are inspected every physics tick and cross-checked
against the decision termination flag. Both signs and all four scoring times
are covered by live native tests, repeated across consecutive resets.

30-second administrative limits and 15-second no-touch limits truncate rather
than score: use the **pre-reset final observation** for value bootstrapping and
cut the GAE trace before the new episode. Neither limit pays a penalty.

## Initial teaching distribution and opponents

50% loose-ground-ball acquisition, 25% achievable finishing, 15% all five real
standard kickoff layouts, 10% ground ongoing 1v1. Half the non-kickoff balls
start stationary; the others roll slowly. Heading and momentum agree, with
off-angle approaches; both cars are controlled from tick zero. All roles are
side-balanced. No scenario/task identifier or scripted controller prefix is
fed to the policy. Source-bank resets use the existing full-cycle coprime
stride, so worlds do not remain stuck on their initial draw.

Initially 100% current self-play, with both agents trainable. Nexto is initially
evaluation-only. After two consecutive evaluation boundaries with acquisition
focal touch fraction >=0.60, conditional median first touch <=5 seconds, and at
least one focal finishing goal, new episodes use 80% current / 20% Nexto.
Nexto is inference-only and retains its own adapter cadence. No frozen V5 or
other old Rival is instantiated. This transition is prospective and irreversible
within v1; poor Nexto results are reported rather than concealed by easier tests.

## Evaluation and interpretation

Fixed deterministic evaluations at 0,10,20,50, then every 50 updates:
64 new-seed worlds each for acquisition self-play, finishing self-play, and
standard kickoffs against Nexto, each observed for at most 30 seconds. Only
the original episode in each world contributes; reset episodes do not inflate
counts. Report touch fraction, conditional first-touch median, contact rate,
goalward-contact fraction, goals/concedes, and no-touch truncations.

These are **scenario outcomes, not full-match win rates**. Positive canonical
ball velocity at the end of a contact decision is only goalward movement, not
proof of possession, an on-target shot, aerials, or competent mechanics.
First-touch medians are conditional on obtaining a touch and must always be read
beside touch fraction. Moving reset states can generate contacts even before a
random policy learns: compare to the frozen update-zero baseline.

Stochastic training metrics use **30 policy decisions per second** for contact
rates, not the old 120 Hz denominator. All native contacts within each held
action are counted. Goals and concedes are equal in pure self-play by design;
that equality is not evidence of opponent strength. The evaluators are separate
from the live training worlds and preserve the learner RNG around evaluation.

## Validation and artifacts

- `focused_tests.xml`: 13 new focused tests; `adjacent_tests.xml`: 23 combined
  new/recurrent/legacy-SSL regression tests passed, including native forced goals at
  each sub-tick, repeated resets, no-touch/time-limit truncation, physical-time
  reward telescoping, fresh-init reproduction, critic isolation, action likelihood,
  recurrent PPO numerical smoke, and same-lineage resume. The small unit-test
  optimizer exercise is discarded and is not campaign training.
- `package.json`: exact local source hashes, scenario-bank hash, fresh random
  model tensor hash, initial checkpoint hash, empty optimizer state.
- `preflight.json`: full **32,768 x 90** native rollout and real recurrent
  minibatch backward, **no optimizer step**. PASS: all values/gradients finite,
  unchanged model and optimizer, 4.545 GiB rollout storage, 11.757 GiB measured
  Torch peak, native reward/cadence checks passed. This is allocation/gradient
  validation, not evidence that the policy has learned gameplay.
- `initial_deterministic_validation.json`: full update-zero evaluator smoke.
- `evaluations/u*.json`, `training_curve.jsonl`, `snapshots.jsonl`: real run evidence.

The package and preflight must be committed and present in `origin/main` before
the production runner accepts an optimizer step. The training runner checks
that requirement itself. Detailed results are subsequently committed by the
monitor; no GPU benchmark/evaluator runs alongside the learner.

## Operation

```powershell
.\.venv\Scripts\python.exe benchmarks/run_rival2_fresh_ground_30hz_v1.py prepare
.\.venv\Scripts\python.exe benchmarks/run_rival2_fresh_ground_30hz_v1.py preflight
# Commit/push the prospective package and preflight before the next command.
.\.venv\Scripts\python.exe -u benchmarks/run_rival2_fresh_ground_30hz_v1.py run
```

External run directory: `G:/dev/RivalSim-runs/fresh-ground-30hz-v1`.
`campaign_state.json` identifies the active process/stage, `latest.json` names
and hashes the latest accepted checkpoint. Two atomically published rolling
slots alternate every update; snapshots at 10,20,50 and every 50 thereafter are
permanent. Check `latest.json`, not a guessed slot filename. Never overwrite old
V5 checkpoints. A Windows file lease prevents two simultaneous learners.

The existing heartbeat was repointed to **Monitor fresh 30 Hz Rival PPO**, every
10 minutes, with the old lineage/cap/deadline explicitly removed. It reports
new evaluations and actionable failures, not repeated unchanged status. The
CPU-only `benchmarks/report_rival2_fresh_ground_30hz_v1.py` validates a stable
rolling checkpoint and writes a complete-record curve snapshot for Git without
interrupting or competing with the GPU learner. Checks distinguish finite-state
integrity from demonstrated gameplay ability.

To request a clean stop, create `STOP` in this run directory. The current update
and any boundary evaluation finish, the accepted checkpoint is preserved, and
no next update starts. On a user stop the monitor must remain stopped. After
an explicitly authorized resume, pass `run --resume <verified latest path>`.
The same model, Adam, counters, and RNG restore; simulator episodes/GRU states
restart cleanly rather than pretending an interrupted physics world was saved.

## RLGym guidance considered

The official [RLGym training example](https://rlgym.org/Rocket%20League/training_an_agent/)
illustrates simple learning signals, parallel simulation, and PPO; it is not a
mandatory recipe and its action repeat differs from this experiment.
The simulator author's [training guide](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md)
recommends initially teaching ball acquisition and later reducing direct touch
rewards. Those are **direct behavioral rewards**, unlike this user's stipulated
potential-based design. We retain fixed potentials here, avoid introducing a
touch/movement payment, and change initial problem difficulty/opposition instead.
Potential weights are not retuned or faded mid-run merely because a behavior
looks imperfect. Neither the guide nor this implementation proves that 120 Hz
caused the previous lack of improvement.

No game deployment or claim that a freshly initialized policy can beat Nexto
is implied by launch readiness. The early evaluations are the evidence to inspect.
