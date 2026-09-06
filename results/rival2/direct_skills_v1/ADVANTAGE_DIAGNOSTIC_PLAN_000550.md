# Bounded no-learning diagnostic after the +550 plateau

Prospective scope: inspect whether the current direct-skill and natural-gameplay
training signals are reaching the policy, and whether task gradients conflict.
This is not a new campaign, training authority, reward change, acceptance gate,
model selection, or permission to weaken finite checks. Commit/push this plan
before collecting the diagnostic native rollout.

## Immutable identities

- Analyze the exact evaluated +550 model, not a selected best-of-many candidate:
  `checkpoints/rival2/direct_skills_v1/plus_000550.pt`, SHA256
  `C8564101BEAE09219026A11ED5F5433E6CEC460EFAE808326F7B8D7FB5DB33FE`.
- Latest accepted +554 stays preserved for same-lineage continuation:
  `checkpoints/rival2/direct_skills_v1/paused_000554.pt`, SHA256
  `CF5C9D022FFB4EF18DBD728206D78A3601AB8C9128BFA0F0E9ADE31BCF961BAD`.
- Existing Direct Skills reward, 30Hz/120Hz timing, 90-decision horizon, gamma,
  lambda, inference-only Nexto/current self-play split, action table, and native
  182-field observation stay unchanged. Verify all 20 frozen runtime hashes.
- Diagnostic bank: existing `scenarios(1024, seed=2026090655)`, SHA256
  `D46A5DACAA3E742BF64E4EA06EB3642F3F370D4AEA2F73FF2791369018A9A73B`.
  Collector seed 2026090655. This new bounded bank is diagnostic training-style
  experience, not a replacement evaluation corpus or changed live curriculum.

## Collection and checks

1. Confirm worker30272 has stopped at the accepted boundary. Hold the same
   exclusive GPU lease and owned stream used by the existing native evaluator.
   Do not overlap another training/evaluation process.
2. Construct only a private +550 policy and the existing collector/environment.
   Do not construct or step an optimizer. Collect **four consecutive 90-decision
   rollouts** with no model change: 12 simulated seconds per world, 1,440 physics
   ticks/world total. Keep hidden state/episodes continuous across the four
   rollout boundaries exactly as in collection; reset only as the runtime does.
3. Read-only step taps retain role/opponent identity, real goal/concede/reset,
   native touches, existing skill events and reward components. Never override
   actions, contacts, observations, values, returns, or opponent behavior.
4. On each native buffer, compute the existing GAE and opponent-family
   normalization. Independently reconstruct the backward recurrence on CPU;
   confirm goal-terminal bootstrap suppression and truncation final-state
   bootstrap/reset cuts. Check collection/recurrent-replay log probabilities.
5. Report counts, raw/normalized advantage distributions, critic values/returns,
   prediction error and actor entropy/probabilities by role/opponent and around
   genuine goal/concede/contact/control/on-target events. Separate the initial
   fresh-state rollout from later experience. Do not relabel a sampled action
   as causally good/bad just because a later goal occurs.
6. Inspect actor/shared-feature and independent-critic gradients without taking
   a step. Use the first eligible production-sized complete-sequence minibatch
   of each rollout. Measure role-gradient norms and pairwise cosine, and critic
   isolation. Negative cosine is evidence of local conflicting objectives, not
   by itself proof of the plateau's cause. Clear all gradients afterward.
7. Store bounded native numeric evidence and summaries, exact identities and
   finite/model/checkpoint immutability checks. Abort on a real runtime/nonfinite
   problem; do not turn any KL magnitude into a rejection condition.

The diagnostic must lead to a concrete continuation decision. Do not repeatedly
replay the already-completed +250 finishing or +300 reset investigations. If
the active reward/gradient path is correct, report that honestly and use the
measured task tradeoff/exploration evidence to define any next experiment
prospectively. Do not silently alter the frozen running contract. The pause is
temporary diagnostic work, not cancellation of the user's continue-until-stop
training objective, and not an irrecoverable blocker.
