# Handbrake reset correction and bounded same-method evaluation

Implements the prospective plan published in732c008. Only standard-kickoff
reset receives the handbrake smoothing array and clears selected cars to0.
The scenario template already clears it. No new state/reward/action contract,
opponent, first-tick timing, policy, optimizer or training change.

Before editing, capture native reference with the original kernel:
`benchmarks/verify_direct_skills_handbrake_reset.py --reference`.
Standard reference is old reset plus isolated selected handbrake clearing.
Original and shooting curricula are unmodified references. Repeat after edit
without `--reference`; require all4,368 arrays/traces exactly equal across
three runtime modes and four masks. All five layouts/both cars,0..1 residual
handbrake, selected/unselected worlds and four subsequent physics ticks.
This is a bounded native parity proof, not a whole-state-space theorem.

Freeze authority.json after focused tests; it binds changed runtime sources,
prior immutable package, all unchanged runtime hashes and exact600/675 models.
Commit/push/read back before the no-learning match evaluation. Do not overwrite
old package.json, old results or model bytes; expired runners remain unavailable
against stale runtime identities. New method is
`RIVAL2_STANDARD_KICKOFF_HANDBRAKE_RESET_V3`.

Run ten original deterministic regulation matches percheckpoint, all five
layouts/both sides, only existing bounded overtime, stock Nexto15Hz, Rival30Hz,
120Hz native physics and every native goal reset. No no-touch/admin skill resets.
Corrected675 versus corrected600 is a same-method checkpoint comparison.
Old675 versus corrected675 and old600 versus corrected600 are runtime effects,
never training gains. Report every match/side and actual follow-ups/contacts.
No skill re-evaluation needed: its native transitions must be bit-identical.

Entry remains agent675 STOP. No optimization during implementation/evaluation.
After actual outcomes, publish a prospective next training decision; neither
an evaluation improvement from this fix nor narrow shooting proxies prove SSL.
