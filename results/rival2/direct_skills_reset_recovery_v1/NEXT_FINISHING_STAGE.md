# Next experiment: graduate solved shooting subgoals, retain actual goal objective

This is a prospective implementation decision under the continuing user goal,
not an already executed training result. No reward was changed during the
completed child25block. The new reward must receive a separate version/hash.

Evidence: original set-keeper evaluation still6/64goals and64/64contacts, with
53on-target projection events. Late training reports10,111proxy success endings
from12,641finishing-vs-Nexto episodes despite only1,125goals and8,266concedes.
The native goal reward is working; old shot projection intentionally ignores
the goalkeeper. Existing650pressure diagnosis already showed that nearby-ball
access/boosting is not the main failure. Do not repeat that broad diagnosis.

## Scope to implement before the next optimizer step

1. Retain the complete accepted child25 model, Adam, counters and four RNG as a
   separate research parent (SHA8EBC09AF738CA89554854D9854EDAE46E01083A9B2841F9E0428D5A054E85AA3).
   Preserve600as the better current full-match reference. Choosing child25 here
   tests improving finishing from the policy with more repeat contacts; it is
   NOT a claim of promotion, a new random run, or a return to old675.
2. New finishing reward only: retire its first-touch bonus, its projected
   on-target-touch bonus, and its pre-first-touch approach payment. These are
   the already-solved intermediate signals. Native score/concede remains+/-10
   with identical within-hold discount and immediate terminal handling.
3. Finishing success means an actual native goal, not a projected shot. A
   finishing time-limit ending without a goal receives the existing-1failed
   attempt amount even when an earlier shot projected into the net. A true
   goal or concede never also receives the time-limit penalty. Do not add a
   penalty for being saved, jumping, flipping, boosting or any named behavior.
4. Preserve old contact/projection telemetry as raw detected events if useful,
   but make their zero reward explicit. Do not claim detected counts are paid
   bonuses. No new detector, keeper-interception heuristic or mechanic label.
5. All non-finishing role outputs, reward components and state remain exact.
   Preserve natural PBRS, challenge/defense/kickoff objectives, curricula,
   opponent shares, actor/critic/GRU/entity structure, PPO,T2exploration and
   KL-telemetry-only semantics. Critic/optimizer are resumed, not reinitialized.
   This is one scoped objective-graduation experiment, not several tunings.
6. Verify CPU matched-input non-finishing parity, reward decomposition, actual
   goal versus truncation cases, no terminal double-payment, detected/paid
   distinction, and focused native rollout/finite/gradient checks. Require the
   normal 30Hz/fourtick goal timing/reset/bootstrap invariants. No broad
   classifier-calibration project. Failures must not be papered over.
7. Freeze/publish/read back the exact next authority, source/checkpoint hashes,
   tests and no-step preflight BEFORE training. Use another separate bounded
   25-update branch with entry/first/rolling/final snapshots. Original skill
   cases and same V3fullmatch evaluation at its end; no easier evaluation.

No guarantee this improves shots: removing auxiliary signals can also slow
learning. The hypothesis is that the current agent no longer needs payment for
nearby-ball acquisition or keeper-ignorant shot projection in finishing drills.
Judge by native scoring, ongoing play and retained useful contact behavior,
not only losses/proxies. Do not tune again mid-block. Keep every candidate and
report tradeoffs rather than declaring all learned behavior invalid.
