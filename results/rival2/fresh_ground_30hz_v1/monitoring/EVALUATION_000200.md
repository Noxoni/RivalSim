# Update 200: finishing contact coverage improves slightly; acquisition remains poor

Evaluation completed 2026-09-05 15:57:51 UTC at 1,179,648,000 accepted trainable
samples. The live learner was not interrupted. Each deterministic case consists
of 64 original episodes from the same frozen scenario seeds, evaluated for up
to 30 seconds. These are scenario outcomes, not full-match win rates.

| Metric | Initial | Update 150 | Update 200 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 11/64 (17.1875%) | 9/64 (14.0625%) |
| Acquisition contacts/min | 0.97275 | 0.68807 | 0.55856 |
| First-touch median, touched cases only | 3.500 s | 0.683 s | 0.683 s |
| Acquisition goals scored / conceded | 0 / 0 | 1 / 0 | 0 / 0 |
| Acquisition no-touch truncations | 63/64 | 63/64 | 64/64 |
| Finishing touch coverage | 24/64 (37.5%) | 46/64 (71.875%) | 47/64 (73.4375%) |
| Finishing contacts | 67 | 52 | 54 |
| Finishing goals scored / conceded | 3 / 0 | 15 / 1 | 15 / 0 |
| Finishing no-touch truncations | 58/64 | 48/64 | 49/64 |
| Nexto kickoff goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

## Interpretation

One additional finishing setup obtains a touch, without an increase in goals.
Acquisition contact coverage and rate are below both initial and update-150
values. The unchanged fast first-touch median applies to only nine contacting
episodes; it does not establish fast pursuit across the corpus. A no-touch
truncation means 15 seconds since the last contact, not necessarily zero
contacts during the entire episode. All 64 acquisition episodes ultimately
hit that inactivity limit. There is still no demonstrated Nexto kickoff
competitiveness or reliable general acquisition. No possession or named
mechanic competence is inferred from these metrics.

The previous read-only initial-action diagnostic remains available in
`initial_action_diagnostic_through_u000150.json`. It ruled out simply refusing
boost or initially jumping/handbraking as an explanation for that checkpoint;
it did not establish the cause of later trajectory failures. It was not
rerun or extrapolated as a measurement of update 200.

## Integrity and continuation

Frozen source, authority, and preflight hashes were verified. CPU-only audit
of accepted update 212 passed at 1,250,426,880 accepted trainable samples:
fresh lineage, finite model and Adam, real optimizer steps, changed weights,
30 Hz decisions / 120 Hz physics, unchanged actor/critic learning rates,
KL telemetry only, and preserved nonfinite rollback. These checks establish
execution integrity, not learning quality. Worker PID 35748 remains active;
stderr is empty, with no failure record or STOP marker.

The routine-acquisition transition criterion remains unmet. Opponents are
still current self-play only; Nexto is evaluation-only. No reward, curriculum,
exploration, optimizer, model, or training-source changes were made. The run
continues under the user's explicit until-stopped instruction, with another
evaluation at update 250. The lack of broad gameplay improvement is reported,
not treated as success or hidden by stochastic training-contact increases.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000200.pt`.
SHA-256: `89B4379AB35B82FDFE5BB9F7440A9D5E8E014972E52F281D8539218F6D5714E7`.
The permanent artifact hash matches the evaluated rolling-checkpoint hash.
Full evaluation: `../evaluations/u000200.json`.
Stable accepted-curve and checkpoint audit: `curve_through_u000212.json`,
`u000212.json`. Live append-only training data is not staged mid-write.
