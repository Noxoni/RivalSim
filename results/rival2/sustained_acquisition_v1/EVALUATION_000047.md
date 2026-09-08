# Completed acquisition evaluation: total update 47

This is an acquisition development probe, not a full-match result or promotion.
Training and its active evaluator were not interrupted for this CPU-only audit.

| Native focal-contact metric | Parent 27 | Update 37 | Update 47 |
| --- | ---: | ---: | ---: |
| Easy contact within 5 s | 114/512 (22.265625%) | 205/512 (40.0390625%) | 277/512 (54.1015625%) |
| Varied contact within 8 s | 49/512 (9.5703125%) | 81/512 (15.8203125%) | 113/512 (22.0703125%) |
| Median easy first-contact time, successful attempts only | 0.566667 s | 0.575 s | 0.600 s |
| Median varied first-contact time, successful attempts only | 1.100 s | 1.141667 s | 1.141667 s |

Update 47 adds 72 easy successes and 32 varied successes relative to update 37.
Both success fractions improved, but neither meets its frozen 95%/90% threshold.
The passing streak remains zero and the acquisition source-bank share stays active.
The slightly higher conditional easy median is not by itself a regression: the
set of successful attempts has changed. No assertion of faster acquisition,
possession quality, kickoff wins, scoring improvement or SSL ability is made.

The same frozen 1,024-state probe bank and evaluation specification were used.
Native first-contact ticks retain every miss. No opponent contact or contact in a
replacement episode is credited to Rival's original attempt. Policy outputs are
deterministic; Nexto uses the frozen seeded native-v5 mode. There are no evaluation
optimizer steps. Recorded model, Nexto and checkpoint integrity flags passed.

Update 37 was already reported interactively before this publication. Its
publication here is not a new result notification.

- Update 37 checkpoint SHA: `1338EC2EDD4742BF3BD596BCC097CE3C667D70643ABF45223B969101AFCA7056`
- Update 37 result SHA: `3869762FE79932EFF65539D4024B338B4F91C0167995DA11A63865E96533C288`
- Update 47 checkpoint SHA: `1C6C1DFCD45F65393710E9BE1E539A7EA054816ED53B3E0D12222A434C83CE57`
- Update 47 result SHA: `B1A698F82A626755F91361DD839E7CC7910128D92106808F61F71E62A3DF145D`

`audit_000037.json` and `audit_000047.json` verify model/Adam finiteness, exact
lineage/authority/contracts, all optimizer counters, continuous accepted-update
logs, sample/physics counts, native Nexto identity and KL telemetry-only behavior.
`through_000047.json` is the closed training prefix through this accepted boundary,
not a staged live training log. The checkpoint parent and runtime were not changed.
