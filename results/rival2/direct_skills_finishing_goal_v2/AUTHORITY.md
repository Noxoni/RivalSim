# Finishing-only objective graduation: prospective 25-update experiment

Continue the preserved completed shooting-pressure child25, cumulative direct625,
SHA `8EBC09AF738CA89554854D9854EDAE46E01083A9B2841F9E0428D5A054E85AA3`.
This research parent increased repeat-contact fraction but lost aggregate match
strength versus600. Keep600 as the better measured full-match reference; neither
is SSL or native Rocket League proof. No new random model, old BC/V5 parent,
specialist routing, optimizer reset, or promotion is implied.

Implement the decision committed in
`results/rival2/direct_skills_reset_recovery_v1/NEXT_FINISHING_STAGE.md`.
Only finishing role2 changes. Native goals remain +/-10, with the same first
goal tick discount, absorbing terminal/reset and pre-reset truncation bootstrap.
Retire finishing first-contact +0.5, projected-shot +1 and approach payments.
Finishing success now means a real native goal. A time-limit ending without a
goal pays the existing -1 failure amount even after a projected on-target shot;
a true goal/concede never also pays timeout failure. No penalty for being saved,
flipping, jumping or boosting. No new detector or reward term.

Raw first_touch and on_target_touch events and their once-event tracker remain
for continuity; they do NOT indicate paid bonuses in finishing. Weighted positive
event amounts and approach there are exactly zero. Success_endings in finishing
now counts real goals. All non-finishing reward/tracker outputs remain exact.
The original skill evaluation is unchanged; its old success/projection counters
are diagnostic proxies, not this new training reward. Judge native goals first.

Same 32,768-state intermediate-pressure bank, role proportions, opponent shares,
entity/GRU/independent-critic architecture, 30Hz four-tick policy, 120Hz physics,
90-decision rollout, PPO gamma/lambda/LRs/two passes and temperature2 sampling AND
likelihood. Half Nexto worlds / half selfplay; one-third learner samples Nexto.
Natural PBRS, challenge, defense and kickoff rewards unchanged. KL telemetry only;
finite/Adam/corruption protection unchanged. Preserve model/Adam/counters/four RNG.
Fresh physical episodes and zero hidden at declared branch entry, not exact
continuation of interrupted physical world state.

New branch offsets0..25 correspond to cumulative ancestry625..650. Separate
entry/first/final and alternating rolling snapshots; never overwrite another
lineage. End at25 with original64cases/family and same ten V3 Nexto matches,
then review. No automatic extension or deployment and no mid-block tuning.
Unknown/user STOP and failures require audit, not blind retries.

Before any optimizer step: focused native tests, matched-input non-finishing
parity, actual goals at all four hold ticks, timeout bootstrap, no double payment,
full runner/resume guards and 1024-world no-step rollout/backward/gradient checks.
Commit/push/read back code, authority, bank/source hashes and this evidence.
Verify actual saved entry model/Adam/groups/four RNG/counters against the parent.

The initial native parity test used a nonexistent Rival2Step.action attribute;
it stopped with AttributeError after12passes, not a measured behavior mismatch.
The test was corrected to emitted_action; all13native tests passed. The first
failed XML is retained as fixture-error evidence. No training occurred during
either test run. This correction does not change runtime behavior.
