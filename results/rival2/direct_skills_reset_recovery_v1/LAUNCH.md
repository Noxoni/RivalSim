# Actual branch entry and first accepted update

The prospective authority/code/tests/corrected evaluations were committed at
`4f95625255092cc281c5d56969bc20bdd68b11e1`, pushed, and all13files remotely
read back BEFORE launch at2026-09-06T11:31:09Z. Separate hidden process:
launcher39460, Python18848. Use actual process creation/command/log progress,
not those possibly stale IDs alone, for monitoring.

`entry_integrity.json` is from the actual full32,768-world learner before its
first PPO step. All6checks pass, including exact model, complete Adam/groups,
counters and four RNG. The CPU-only audit independently compares parent,
actual entry and first accepted checkpoint: all20checks pass. No audit optimizer
step. Parent600 and old675 are untouched.

Entry checkpoint SHA256:
`FA8A7D11F0880632F6BB27AB51E93D57E6670E34AADE365AA346D0FBA49D3AFD`.

First accepted child1 (cumulative direct ancestry601) SHA256:
`4FB0D6C864E3940568AE2A6D1A50BA2AC76390C46C0A768C63A11EE5908DA5AB`.

The first update adds136actual Adam steps,137694cumulative;4,423,680learner
decisions and11,796,480physical world ticks. Exactly one-third learner decisions
face Nexto. Model and Adam remain finite. Raw PPO/reward/skill telemetry is in
`first_training_update.json`; this is training health, not new capability proof.

At2026-09-06T11:32:35Z child3was accepted and the next rollout was active.
The external stderr was empty. Full curves/rolling checkpoint remain under
`G:/dev/RivalSim-runs/direct-skills-reset-recovery-v1`; do not conflate them with
the old675run or its retained intentional review STOP.

The existing ten-minute goal monitor now follows this branch, stays quiet on
unchanged state, reports newly completed evaluations once, and requires review
after child25. The branch runner does not extend itself beyond25. Original
skill cases and V3-method ten-match Nexto evaluation execute at that boundary.
Current corrected600 remains the stronger measured candidate until the child
demonstrates otherwise; neither this launch nor fixed development wins prove SSL.
