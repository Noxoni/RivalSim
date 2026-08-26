# Rival 2.0 mechanics-corrective full-match curriculum

This handoff records the user-authorized corrective campaign following the
RocketSim reciprocal-validation diagnosis. It is a v0.5/Rival 2.0 correction;
it does not begin v0.6.

## Source and mechanical authority

- RocketSim source commit: `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`.
- RocketSim Python binding commit: `2da51b1dac7b8127127613a5ff30e490bdd70dd8`.
- Car: stock Octane; arena: stock Soccar; physics cadence: 120 Hz.
- Rival policy cadence remains 30 Hz with each action held for four physics ticks.
- The direct RocketSim car pre-tick order is authoritative for jump, held jump,
  double jump, directional dodge, flip torque/cancel, stall, auto-flip, air
  control, boost, steering, powerslide, braking, supersonic state, wheel forces,
  rigid-body solve, and integration.
- The open-loop movement gate must compare identical 120 Hz controls and pass
  every declared mechanic family before training starts. Policy feedback is not
  part of that parity gate.
- Existing source-backed static-world, car-ball, car-car, lifecycle, kickoff,
  and visitation-order semantics remain in force.

The prior 45B checkpoint was learned with a simulator path that changed public
jump/flip flags after the chassis solver had already consumed stale rigid-body
velocity. It is evidence about the old environment, not a valid initialization
for this corrected campaign. Training therefore starts from the same seeded
model initialization and PPO configuration used by the original Reward V2
campaign, not from the old learned weights.

## Full-match episode contract

Every training world is one ordinary standard 1v1 Soccar match:

- five minutes of regulation at 120 Hz;
- two cars, one Blue and one Orange;
- the five standard kickoff layouts, assigned evenly and advanced by lifecycle
  state;
- a scored goal increments the persistent score and performs a standard kickoff
  reset without ending the match;
- a non-tied score at the end of regulation ends the match;
- a regulation tie enters unbounded sudden-death overtime after a standard
  kickoff, and the overtime goal ends the match;
- demolition/respawn, boost pads, goals, kickoffs, and controller cadence retain
  normal simulator behavior;
- there is no no-touch truncation, hard time-limit truncation, synthetic
  mid-play reset, or short training episode.

The historical 15-second no-touch condition is retained only as a read-only
counterfactual acquisition metric for each kickoff segment. It must never
reset, truncate, pause, shorten, or otherwise alter a match.

PPO rollouts may cross internal update boundaries. Match state, score, clock,
overtime, policy history, and opponent assignment persist across those update
boundaries. An opponent assignment may change only after the complete match
ends.

## Phase A: contact acquisition

- Fresh seeded initialization: `20260826`.
- Worlds: `131072`.
- PPO/model/self-play configuration: unchanged Campaign 03 entropy-off
  configuration.
- Reward: base `RIVAL2_REWARD_V1` plus the exact
  `RIVAL2_REWARD_V2` car-to-ball approach-distance term.
- Evaluation: every 30 completed PPO updates, 4,096 held-out stochastic
  current-policy self-play worlds at seed `920260826`, each running one complete
  standard match.
- Acquisition gate: the aggregate counterfactual no-touch kickoff-segment
  fraction is at most `0.01` on two consecutive held-out evaluations.
- There is no sample cap. Do not select a checkpoint retrospectively.

## Reward transition

At the confirming Phase A update boundary, save the acquisition checkpoint and
change only the reward contract to `RIVAL2_REWARD_GOAL_ONLY_V1`. That reward is
`+10/-10` for a goal and contains no approach, progress, touch, demolition, or
other shaping term.

Preserve the model, optimizer, CPU/CUDA/policy/opponent RNG state, iteration and
sample counters, historical opponent pool, live opponent assignments, and all
live full-match world state. Publish an exact field-by-field transition record
and a reloadable post-transition checkpoint.

## Phase B: score-only continuation

Follow the original post-acquisition curriculum shape with the goal-only reward:

1. complete exactly 239 additional PPO updates
   (`2,004,877,312` agent decision samples), with checkpoints/evaluations at
   offsets `60`, `120`, `180`, and `239`;
2. from that exact boundary, start one fresh standard match in each of the
   `131072` resident worlds and continue goal-only PPO only for samples from
   those matches; as each world completes, mask its later resident transitions
   out of PPO while the remaining overtime matches finish;
3. stop when every world has completed that one match, then save and verify the
   final resumable checkpoint and publish the complete
   acquisition and scoring curves, and stop.

The final bound is match-based, not time- or sample-based. Every Phase B
training rollout and every evaluation uses the full-match episode contract.

## Required evaluation evidence

For every evaluation, retain at minimum:

- completed matches and simulated match time;
- Blue wins, Orange wins, and overtime matches;
- Blue/Orange goals and goals per simulated minute;
- touches and touches per simulated minute;
- kickoff segments, first-touch-within-15-second segments, and counterfactual
  no-touch segments;
- aggregate counterfactual no-touch kickoff-segment fraction;
- match duration and goal-count distributions;
- controller activation summaries;
- exact checkpoint, contract, source, seed, and configuration identities.

Do not call the counterfactual metric a real truncation in new evidence. Do not
reuse old short-episode evaluation results as full-match results.

## Stop boundary

Stop after the final one-match-set goal-only checkpoint, its single scheduled
full-match evaluation, compact evidence publication, and push to `origin/main`.
Do not begin v0.6, change the model/PPO/action/observation contracts, add reward
terms, train against Nexto, or build the viewer.
