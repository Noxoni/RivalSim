# First monitor follow-up: accepted update 20 evaluation

The GPU learner was running normally and was not interrupted. During this
monitor, the latest accepted rolling checkpoint audited was update 47:
277,217,280 trainable samples. CPU-only checkpoint audit: PASS. Model/Adam
finite, fresh lineage, cadence, learning rates, numerical protection, and
KL-telemetry-only policy remain unchanged. Frozen package/preflight hashes
reverified; no implementation changes or competing GPU evaluation were made.

Newly reportable evaluation: **update 20**, completed 2026-09-05 14:50:18 UTC.
The previous notification was update 10. These are fixed-seed, deterministic,
64-world-per-case scenario evaluations, not full-match win rates.

| Metric | Random initial | Update 10 | Update 20 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 12/64 (18.75%) | 13/64 (20.3125%) |
| Acquisition native contacts | 17 | 18 | 25 |
| Acquisition contact rate/min | 0.97275 | 1.03507 | 1.46972 |
| Acquisition median first touch, touched cases only | 3.500 s | 4.942 s | 2.208 s |
| Acquisition no-touch truncations | 63/64 | 64/64 | 64/64 |
| Finishing touch coverage | 24/64 (37.50%) | 30/64 (46.875%) | 35/64 (54.6875%) |
| Finishing native contacts | 67 | 84 | 70 |
| Finishing focal goals | 3 | 7 | 8 |
| Finishing median first touch, touched cases only | 1.192 s | 1.075 s | 0.925 s |
| Finishing no-touch truncations | 58/64 | 57/64 | 56/64 |
| Nexto kickoff goals for / against | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

Interpretation: modest further improvement in finishing coverage/scoring.
Acquisition has one more successful-contact case and shorter conditional
first-touch time, but coverage remains poor and all acquisition cases still
end in a no-touch timeout (15 seconds since the last contact, not necessarily
zero lifetime contacts). Fewer repeat finishing contacts are not automatically
a regression or improvement; do not infer possession quality from that count.
There is still no demonstrated competitive kickoff/Nexto gameplay. These
results do not establish mechanics or a broad capability gain.

The prospective acquisition criterion is not met. Training remains pure
current self-play; Nexto remains evaluation-only. No reward/sampling/optimizer
retuning occurred, no guard fired, and training continues until the user stops it.

Permanent update-20 checkpoint:
`checkpoints/rival2/fresh_ground_30hz_v1/u000020.pt`
SHA-256: `28A5DD9CAB4775C0F8C2D8E2931228D829952893470A1532C5FFCAE6F5CEED6E`.

Full evaluation: `../evaluations/u000020.json`.
Frozen accepted curve through the audited update: `curve_through_u000047.json`.
Audit: `u000047.json`. The monitor's rolling-slot hash is a point-in-time audit;
permanent snapshots, not overwritten rolling-slot names, are historical artifacts.
