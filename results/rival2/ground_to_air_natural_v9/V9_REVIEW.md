# Natural ground-to-air V9 review

## Verdict

**DIAGNOSTIC NON-PROMOTION.** V9 was stopped after the durable block-12
boundary. Its block-8 strict-event score was not accompanied by any high
follow contact, second airborne contact, productive continuation, or controlled
goal, and the strict-event score regressed at block 12. The untouched test was
not opened and neither V9 checkpoint is eligible for V23 integration.

## Training result

V9 accepted 12 blocks and 96 balanced optimizer steps from the unchanged
controlled scorer. It remained finite and preserved the frozen critic. The
aggregate physical score rose from `0.0045166` at preflight to `0.0089111` at
block 8, then fell to `0.0067139` at block 12. The strict elevated-follow event
appeared in six of twelve rows at block 8 and five rows at block 12, but every
row remained zero for high follow contacts, second airborne contacts,
productive continuations, and controlled goals.

The block-8 diagnostic checkpoint is
`G:/dev/RivalSim-runs/ground-to-air-natural-v9/best_validation.pt`, SHA-256
`2EBF0A052E6C75C5EA4B4AB8F9133B785EBAA8DFCC381042A435A357BD13756F`.
The block-12 rolling checkpoint is
`G:/dev/RivalSim-runs/ground-to-air-natural-v9/rolling.pt`, SHA-256
`B2B44EC3B8576A33B3E8C788F2CC8C586D90E9674DDA04328587BDB6A3AD3C31`.
Both are diagnostic only.

## Native touch-geometry finding

The read-only probe replayed the same 12 setup/defender/side rows with 256
worlds per row and seed `2026110912`. It separates any distinct follow touch,
an airborne follow, a prompt airborne follow no more than 60 ticks after the
setup touch, and the old strict event requiring car height at least 150 uu and
ball height at least 250 uu.

| Checkpoint | Prompt airborne follow | Any airborne follow | Old strict event |
| --- | ---: | ---: | ---: |
| Protected controlled scorer | 39.91% | 42.61% | 0.33% |
| V9 best block 8 | 40.07% | 42.58% | 0.26% |

The protected scorer therefore already produces many genuine prompt low
airborne recontacts. Low-bounce recontacts occur around 21–23 ticks after the
setup touch at median car height 71–76 uu and ball height 180–183 uu.
Matched-dribble recontacts occur around 24 ticks after setup at median car
height about 81 uu and ball height 195–197 uu. The old strict event excludes
nearly all of these causal first follow contacts.

Incoming-chip recontacts were much rarer and usually happened hundreds of ticks
later, so an unrestricted “any airborne touch” is also too permissive. The
60-tick prompt category preserves the causal launch connection without using a
named-mechanic classifier or rewarding airtime.

## Decision

V9 was optimizing a sparse later-stage event while lacking explicit credit for
the already-attainable first low airborne recontact. Its tiny strict-score
changes did not improve real prompt recontact and should not be promoted.

The successor must restart from the protected controlled scorer, not V9. It may
add a bounded physical event for the first separated airborne follow within 60
ticks of the setup touch. It must retain the old strict elevated/high/second
events as subsequent stages, preserve the six-contact maximum, and still
require productive continuation and scoring before opening the untouched test
or integrating with V23.
