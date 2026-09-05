# Completed reset-only pilot: negative/inconclusive

Exactly 30 accepted updates completed from immutable update 597, with the
prospectively frozen easier reset bank. No reward, architecture, PPO, exploration
or opponent changes were made. No hard numerical failure was reported.

| Offset | Acquisition touched /64 | Matched control /64 | Finishing goals | Control goals | Nexto kickoff goals for/against |
|---|---:|---:|---:|---:|---:|
| 0 | 16 | 16 | 14 | 14 | 0/64 |
| 10 | 15 | 14 | 12 | 14 | 0/64 |
| 20 | 13 | 16 | 14 | 17 | 0/64 |
| 30 | 16 | 15 | 12 | 16 | 0/64 |

The pilot failed its frozen continuation criteria. The final acquisition result
did not improve over the parent; final finishing scored four fewer goals than
the matched control. Intermediate Nexto contact fluctuations did not translate
into scoring. These are bounded scenario evaluations, not full-match win rates.
No-touch timeout counts can overlap cases that made an earlier touch.

All boundaries and the complete 30-row curve are retained, together with the
final model/optimizer/hash audit. Training integrity PASS is not a capability
PASS. No checkpoint is promoted and this pilot is not extended. The old update
597 campaign remains paused. The broader SSL-development goal is unfinished.

The next implementation candidate changes how the policy represents entities
and selects joint controls; it requires its own prospective configuration and
native validation. This pilot does not prove that easier resets are useless in
general, nor that any proposed new architecture will work.
