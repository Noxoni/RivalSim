# Update 10: first learned checkpoint, not gameplay acceptance

The campaign is **RUNNING**, with no fixed end time or accepted-update limit.
The user will decide when to stop. This report does not finalize the campaign.

Implementation/authority remote commit before learning:
`15abcdfa1e2a9808f726ee333f225b8bff28538d`.

Preserved accepted update-10 checkpoint:
`checkpoints/rival2/fresh_ground_30hz_v1/u000010.pt`.
SHA-256: `995EAAEAD431E43AC3A9161560E54845EE394F859C50AA96560498789B079379`.
Fresh random origin, 58,982,400 trainable samples, 1,820 accepted Adam minibatch
steps across ten PPO updates (two passes per update). This is not any previous
checkpoint continued under a new name.

| Deterministic scenario metric | Random initial policy | Update 10 |
| --- | ---: | ---: |
| Acquisition focal touch fraction | 12/64 (18.75%) | 12/64 (18.75%) |
| Acquisition touches | 17 | 18 |
| Acquisition first-touch median, touched episodes only | 3.500 s | 4.942 s |
| Acquisition no-touch truncations | 63 | 64 |
| Finishing focal touch fraction | 24/64 (37.50%) | 30/64 (46.875%) |
| Finishing touches | 67 | 84 |
| Finishing first-touch median, touched episodes only | 1.192 s | 1.075 s |
| Finishing focal goals | 3 | 7 |
| Finishing no-touch truncations | 58 | 57 |
| Standard kickoff against Nexto: goals for / conceded | 0 / 64 | 0 / 64 |
| Standard kickoff against Nexto: Rival touches | 0 | 0 |

All cases use the same frozen 64-world seeds and up to 30 seconds per original
episode. A no-touch truncation means 15 seconds **since the last contact**, not
necessarily that no contact occurred anywhere in that episode. Initial car
momentum and rolling balls can create touches even under random weights, which
is why the update-zero comparison matters. These are not full-match win rates.

There is an early improvement on the easier finishing cases. Acquisition did
not improve in coverage and its conditional first-touch median worsened. Rival
has **not** demonstrated functional general gameplay or competitiveness against
Nexto. There is no basis to claim possession, aerials, dashes, or demos from
these metrics. Ten updates are not proof that 30 Hz fixed the learning problem.

Numerical/lineage audit: PASS. Model tensors changed from the fresh initialization;
Adam has real positive step counters; model/optimizer tensors are finite; fresh
lineage, 30Hz cadence, independent critic, optimizer LRs, authority hashes and
KL-telemetry-only policy match. The complete accepted curve is snapshotted in
`monitoring/curve_through_u000010.json`; live append-only logs continue separately.

The routine-acquisition criterion is not met. Training remains pure self-play;
Nexto remains evaluation-only. No automatic reward or exploration retuning was
performed after evaluation. The next scheduled checkpoint/evaluation is update
20, followed by 50 and every 50 thereafter. Monitoring is active and reports
new results or actionable failures without interrupting a healthy learner.
