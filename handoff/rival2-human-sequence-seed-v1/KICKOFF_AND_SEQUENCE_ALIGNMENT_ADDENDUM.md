# Human Sequence Seed v1 — Kickoff and Sequence Alignment Addendum

This addendum is authoritative for `handoff/rival2-human-sequence-seed-v1/CODEX_START_PROMPT.md`.
It corrects two recurrent-sequence details without changing the task's model, observation-view, data-source, or no-PPO scope.

## 1. Human/native recurrent start-state alignment

Do not allow native Rocket League countdown/frozen kickoff frames to create recurrent history unless RivalSim reproduces the same frozen/control-disabled sequence exactly.

The human recording contains kickoff/countdown states in which the physical car/ball state can remain essentially unchanged while the recorded controller target changes. RivalSim standard-kickoff evaluation begins from a playable simulator kickoff state and must not inherit hidden state generated from human-only countdown history.

For every kickoff/episode segment used for Human Sequence training:

- identify the first **controllable/post-countdown gameplay tick** corresponding to the playable kickoff state;
- reset the recurrent hidden state to zero at that aligned playable boundary;
- exclude earlier frozen/countdown frames from supervised loss;
- do not use those earlier frozen/countdown frames as GRU burn-in/context;
- the first recurrent input presented for the human segment must correspond semantically to the first recurrent input presented after a RivalSim kickoff reset.

If RivalSim is found to reproduce the native countdown/freeze sequence exactly, document and prove that equivalence before retaining those frames. Otherwise, use the playable-boundary rule above.

This alignment must also be used during deterministic Nexto evaluation: hidden state starts at zero on the playable kickoff boundary, not at an earlier countdown state that the simulator never executes.

## 2. Recurrent split boundaries

The original handoff says to preserve the existing chronological 80/10/10 frame boundaries. For a recurrent policy, do not cut a live gameplay episode merely to hit the exact frame percentage.

Instead:

- keep the split chronological;
- partition by whole playable kickoff/goal episode segments;
- choose whole-segment boundaries nearest to 80% train / 10% validation / 10% test by frame count;
- no episode segment may appear in more than one split;
- reset hidden state only at real aligned episode/kickoff boundaries, never because an arbitrary percentage boundary landed mid-play;
- record the exact resulting frame counts and segment identities in the split authority.

The untouched test remains unopened until checkpoint selection is complete.

## 3. Explicit Stage-1 supervised action loss

To avoid ambiguity in the recurrent implementation:

- analog channels `throttle, steer, pitch, yaw, roll`: supervise `tanh(actor_mean)` against the exact recorded analog action using MSE;
- button channels `jump, boost, handbrake`: supervise raw button logits against exact 0/1 targets using binary cross-entropy with logits;
- actor log-standard-deviation rows are not human targets and remain fixed/documented during Stage 1;
- critic receives no Stage-1 gradient.

Checkpoint ranking remains only the held-out validation complete-action RMSE. Use the same deterministic/probability action representation consistently for every candidate so no secondary score can override a lower RMSE.

## Everything else remains unchanged

All other requirements in `CODEX_START_PROMPT.md` remain authoritative, including:

- fresh random recurrent model;
- only the reviewed 58,306-frame gameplay recording;
- shared direct human/native observation projection;
- no Observation Adapter V2 in the final policy input path;
- previous-action fields always zero;
- no previous Rival/BC/PPO weights;
- no retention or old-policy KL objective;
- no reward optimization and no PPO;
- actual recurrent training followed by deterministic closed-loop Nexto evaluation.
