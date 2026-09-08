# Fresh full-scenario contact-reward restart

Active user instructions supersede the acquisition-only draft before its first
training step: start from fresh random weights; reward the first touch; add a
bonus for contacts while the ball is off the ground; use all existing scenarios.
Retain all original sustained-gameplay rewards and continue playing after touches.

## Reward actually sent to PPO

1. **+1 on each player's own first native contact per episode**, across all starts.
   Opponent-first contact does not remove eligibility. Both players can earn it.
2. **+0.25 on each distinct native contact with an off-ground ball**, up to10 paid
   contacts per player per episode (maximum2.5). This prevents arbitrarily many
   contact events from outweighing a goal. Continuous contact is not repaid.
3. **All prior terms remain**: goal/concede+10/-10, the seven unchanged potential
   differences, sustained control up to0.02/s, and inactivity penalty-1.

First-touch, off-ground-touch and scoring/control rewards may stack. Native
120Hz event timing is retained in the 30Hz decision by multiplying the event
payment by physics_gamma**substep. Events before or on a goal tick count; events
after the goal do not. Payout latches/budgets reset only at real episode resets,
never at PPO update, checkpoint or evaluation boundaries.

"Off ground" is a physical height test at contact, not a named-mechanic claim:
use each contacting car's captured **pre-contact** ball center, strictly above
the simulator sphere radius plus2uu clearance. Radius is91.2499964uu; threshold
is93.2499964uu. The tolerance excludes resting-floor contact noise. Low bounces
count; neither the car nor ball needs to be high. This is not proof of an aerial
shot. A ground touch that pops the ball cannot retroactively earn the air bonus.
The kernel only reads native state and adds reward accounting; physics is unmodified.

There is no generic jump/flip/action reward, mechanic classification, scripted
prefix, task ID, or preservation KL objective. Nexto's contact rewards are
excluded from PPO by the current-agent training mask.

## Scenarios and training

Use the full corrected six-family bank, including all native stationary kickoff
layouts: approximately50% acquisition,15% natural play,10% challenge,10% finishing,
7.5% defense and7.5% dedicated kickoff. Natural kickoff rows are also stationary.
This is the existing sustained-acquisition mix before retirement, not a selection
of every historical specialist training package. No family is automatically
removed during this run. Scenarios continue until a goal or45s without a contact
by either player, not until the first touch. No fixed episode-age timeout.

Fresh construction of actor, entity modules, actor GRU, independent critic MLP/GRU,
and empty Adam, seed2026090810. No trained model/optimizer is loaded. Reusing this
initialization seed reproduces the prior *untrained* tensors, not its learning.
The earlier completed acquisition-only experiment and all other trained models
remain stopped and unmodified. The draft first-touch-only acquisition package
in fresh_acquisition_touch_v1 is marked SUPERSEDED_PRELAUNCH; never launch it.

Preserve32,768 worlds,120Hz physics,30Hz decisions/four-tick actions,182 observations,
joint90 categorical actions,temperature1,entropy0.001,90-decision rollouts,two
passes,actor LR1e-4,critic LR3e-4,30s reward half-life and3s trace half-life.
Half the worlds are current self-play (both learn); half active native-v5 Nexto
(only Rival learns). Finite/corruption rollback-and-stop remains; KL is telemetry.

## Verified before campaign optimizer entry

25 focused tests passed in14.79s. Native fixtures confirmed ground-versus-air
contact timing, one-off first payout, no per-tick contact payment, both-player
eligibility, air-budget cap, post-goal exclusion, goal availability, no touch reset,
timeout/re-arm behavior, and exact unchanged original rewards and observations.
Kickoff tests verify exact stationary native states on initialization and reset.
A tiny disposable PPO update verified that both bonuses reach learner samples.

Actual32,768-world preflight passed all12 checks with zero campaign optimizer
steps and no model mutation. It observed5,384 learner first-touch awards
(5,382.414241313934 discounted reward) and1,121 learner off-ground contact awards
(280.1654955595732 reward). It separately excluded14,359 Nexto first awards and
3,784 Nexto off-ground awards. No air budget was reached during that preflight.
All six scenario families had measured exposure, both recurrent memory banks and
critic isolation were verified, and all outputs/gradients were finite.

Sources, tests, authority, package and initialization must be committed, pushed
and remotely verified before the first campaign optimizer step.

## Monitoring, checkpointing and interpretation

Retain a bounded150-update diagnostic window, baseline and unchanged deterministic
1,024-start acquisition probe every10 updates. Preserve every probe checkpoint
and alternating rolling checkpoints each accepted update. The evaluator is
read-only and ignores reward, so it uses the existing physics/observation/reset
path to compare contacts without training from evaluation data.

Record both new reward contributions separately, including learner award counts,
excluded-opponent events, repeated contacts without first bonus, raw eligible
off-ground events, and decisions at the air cap. Verify discounted reward/count
accounting on every rollout before optimizer entry. Also retain all existing
family/opponent gameplay telemetry and PPO diagnostics.

The monitor checks every2minutes and reports only newly completed evaluations
(or a genuine blocker), not partial or unchanged results. At150 finalize and
stop; no automatic reward retuning, promotion, or additional campaign.
This is not a single-factor comparison: the user requested reward and reset-mix
changes together. Contact success and off-ground contact counters are not proof
of possession, aerial offense, scoring competence or SSL.

Commands using project .venv Python:

- benchmarks/run_fresh_sustained_contact_v1.py preflight
- benchmarks/run_fresh_sustained_contact_v1.py freeze
- benchmarks/run_fresh_sustained_contact_v1.py verify
- benchmarks/run_fresh_sustained_contact_v1.py run
- benchmarks/run_fresh_sustained_contact_v1.py report --offset N

External run: G:/dev/RivalSim-runs/fresh-sustained-contact-v1.
Checkpoints: checkpoints/rival2/fresh_sustained_contact_v1.
Recovery only with explicit same-lineage --resume PATH --resume-sha256 HASH.
It restores model/Adam/counters/RNG but starts fresh physical episodes, clears
both memories and payout budgets, and does not splice old episodes or GAE.
Never clear an old STOP or resume a different lineage.
