# Natural ground-to-air V7 review

## Verdict

**NOT PROMOTED.** V7 preserved the proven fixed pop pitch and learned only
bounded steer/yaw/roll corrections during the inherited jump sequence. The
controls were causal and training was finite, but the best deterministic
boundary still had a zero worst-row gate, no high follow touches, and no
productive continuations. The untouched test was not opened and neither
external checkpoint is an accepted parent.

## What changed physically

- Initial setup contact remained common: about 74% to 100% across the twelve
  low-bounce, incoming-chip, and matched-dribble rows.
- A small number of deterministic elevated reconnections appeared and moved
  between families across boundaries. At block 16, four rows had at least one
  elevated event and the best row was 3/256 (1.17%).
- No row produced a high follow touch or a productive continuation at any
  validation boundary.
- The best block-16 aggregate tiebreaker was `0.032958984375`, but the primary
  worst-row ratio remained exactly zero.
- Approximate KL stayed between roughly `0.0006` and `0.0016` near the end;
  optimization stability was not the limiting factor.

## Why training stopped at block 16

V7 established that small orientation corrections can change a few outcomes,
but it did not consolidate a common launch-and-follow motion. Continuing toward
the 64-block ceiling would optimize a control variable that had already failed
to produce vertical reconnection across most deterministic rows. The runner was
interrupted immediately after its durable block-16 validation/checkpoint write.
This was a bounded capability decision, not a hard safety failure.

## Prospective correction

The physical evidence and external mechanics research point to takeoff timing.
The inherited first jump is held for only 8 ticks (about 67 ms at 120 Hz), while
Rocket League applies additional first-jump force for up to about 200 ms. The
next experiment should sweep bounded first-hold/release/second-jump schedules on
the unchanged protected scorer and unchanged natural setup corpus before any
optimizer step. It must preserve the light bounce/chip contact, fixed nose-up
pitch, six-contact ceiling, and lack of raw-airtime or named-mechanic reward.
