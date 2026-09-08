# Acquisition-only diagnostic: update 130

Both groups slipped from update 120. No settings changed.

| Deterministic acquisition measure | Update 120 | Update 130 |
| --- | ---: | ---: |
| Easy own contact within 5 seconds | 170/512 (33.203125%) | 166/512 (32.421875%) |
| Varied own contact within 8 seconds | 123/512 (24.0234375%) | 112/512 (21.875%) |
| Easy conditional median seconds | 0.925 | 0.95 |
| Varied conditional median seconds | 2.2 | 2.175 |

Baseline was 31/512 easy and 11/512 varied. Total successes fell 293 to 278/1,024;
the observed combined peak remains 333 at update 70. Medians exclude failures and
compare different subsets. No possession, first-touch-win or scoring claim follows.

Stochastic rollout contacts increased 1.1877441406 to 1.3208007813/minute and speed
301.444527 to 309.941494 uu/s. These changing-state endpoints differ from the fixed
deterministic probe and must not be substituted for it.

All 16 CPU audit checks passed. Counters: 575,078,400 learner decisions and 18,076
fresh Adam steps. Model/Adam finite; no KL rejection or non-acquisition exposure.
All 1,024 raw ticks were checked (-1 or 1..960), with counts, fractions, conditional
lower medians and no-contact fractions independently recomputed. Result and started
receipt hashes agree with actual checkpoint bytes. Scenario/specification hashes
match baseline. Evaluation took zero optimizer steps, changed neither model nor
Nexto nor checkpoint, and was not rerun. Frozen sources/package were verified in
this monitoring turn. No interruption, extra GPU work, recovery or retuning occurred.

Continue the frozen diagnostic to 150 unless stopped. The observed gains coexist
with setbacks; the cause cannot be assigned to a particular old setting from this
single fresh isolated arm. The previous mixed campaign stays stopped at 156.

- Checkpoint: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000130.pt`
- Checkpoint SHA-256: `9DCB4AA5F02A7BD523F9B3E34799886E731C74BADFA991E193BAF10009FFE0EB`
- Result SHA-256: `25C85024FDD68EFD47B2BE95A0BD64D29DDA05E1CEA574C39C1BE16B4ACDD093`
- Scenario SHA-256: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`
- Specification SHA-256: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`

Full evidence: acquisition_000130.json, its started receipt, audit_000130.json,
and the closed update-1..130 prefix through_000130.json.
