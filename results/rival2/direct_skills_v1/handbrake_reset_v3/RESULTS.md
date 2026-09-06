# Corrected reset: +600 is the stronger development candidate

Implementation/authority commit: `93746d4bd9baf5f8db3ed50f239a20e034f39c81`.
All 14 implementation/authority artifacts were read back from origin/main
before either evaluation. Both fixed ten-match evaluations completed, exit 0,
zero optimizer steps, unchanged checkpoint/model hashes, all 11 match integrity
checks per checkpoint passed. See raw JSON and deterministic comparison JSON.

| Same V3 reset method | +600 | +675 |
|---|---:|---:|
| Wins / losses | 8 / 2 | 1 / 9 |
| Goals / concedes | 169 / 122 | 71 / 136 |
| Rival contacts | 486 | 508 |
| Touches / regulation minute | 9.72 | 10.16 |
| Matches without a Rival contact | 0 | 0 |
| Rival kickoff first contacts | 0 | 0 |
| Same-player / resolved follow-ups | 50 / 311 | 158 / 431 |
| Same-player follow-up fraction | 16.08% | 36.66% |

Nine of ten matched goal-difference cases worsen from +600 to +675; one
improves. Additional contacts/follow-ups did not translate into better results.
Follow-up fraction is not possession duration or evidence of advanced mechanics.
No-touch resets are disabled in this match protocol, so their zero count is not
a behavioral accomplishment. Neither model wins the first kickoff contact.
Ten fixed development cases are not independent ranked games, SSL evidence,
or native Rocket League deployment validation.

Old-method +600 was 65/144 goals, zero wins; old-method +675 was 37/168, zero
wins. Those old-to-new differences are effects of the handbrake reset correction
at identical weights, NOT learning. Only the new same-method comparison above
supports the checkpoint ordering. Training curriculum resets already cleared
handbrake; all 4,368 pre/post native arrays/traces passed exact parity, including
both training banks. Do not attribute the training regression to this evaluation
fix or assume its narrow causal proof explains every remaining gameplay issue.

Preserve +675 and its whole history. Retain +600 as the stronger measured
development candidate, not an automatic RLBot deployment. Its exact checkpoint:

`checkpoints/rival2/direct_skills_v1/plus_000600.pt`

SHA256 `8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2`.

## Shooting and the next finite learning decision

Shooting is already a first-class family: 20% of source starts, with native
goal +/-10 and the existing once-per-episode on-target contact outcome. This
is not a claim that the keeper-aware finishing problem is solved (+600 scored
6/64 original set-keeper cases; +675 scored 5/64). The 650 diagnostic separately
found 49/64 initially-open and 50/64 recovering-keeper successes, versus 5/64
set-keeper successes, motivating intermediate pressure instead of only easy nets.

Use a SEPARATE 25-update development branch from the exact +600 model, Adam,
counters and four RNG states. Apply the already-implemented intermediate-pressure
shooting bank (half finishing starts, only opponent pose/velocity changed),
retaining original hard finishing cases and every other reward/PPO/opponent
setting. This is a deliberate rollback to a measured stronger parent, NOT a
continuation of +675 and NOT a fresh random run. Preserve all old artifacts.
Fresh physical episodes/zero recurrent hidden at branch entry are explicit.

Freeze/publish the new runtime/training authority, verify exact entry model and
optimizer/RNG parity, then train 25 child updates. Review original skill cases
and the SAME V3-method Nexto matches before any extension. Use actual goals and
match outcomes, not a rise in task-proxy rewards, as transfer evidence. Keep
+600 intact if the child regresses. No preservation KL gate, reward redesign,
new named-mechanic detector, physics change or automatic deployment is authorized
by this decision. Finite/corruption protection remains; KL is telemetry only.
