# Rival 2.0 human-demo review V2

Generated: `2026-08-28T22:08:49.808139Z`

This is an inventory, paired-stream integrity review, reset-aware segmentation, and source-bound mechanic adjudication pass. It identifies high-confidence demonstrations for later behavior-cloning curation, but does not behavior-clone, train, define a production mechanic detector, or assign rewards. Successful, failed, and ambiguous evidence is preserved together.

## Inventory and validation

| Session | Class | Declared label | Frames | Attempts | Review-usable | Strict recorder diagnostic |
|---|---|---:|---:|---:|---:|---|
| `1ED91132-BEF2-4B18-8872-364E7893B89A` | freeplay_mechanic_practice | wavedash | 2252 | 4 | true | none |
| `151E5828-251B-420D-8798-EE14B3A204C5` | freeplay_mechanic_practice | zapdash | 3824 | 7 | true | none |
| `529360F1-9944-42E9-BDB4-9C39C29CD2A3` | freeplay_mechanic_practice | walldash | 1570 | 5 | true | none |
| `8104A81F-9570-4E8D-9EF5-D310FFE78528` | freeplay_mechanic_practice | musty | 8787 | 11 | true | none |
| `E9068B84-AABB-4EFE-B8F6-F028419DE642` | freeplay_mechanic_practice | breezi | 4781 | 9 | true | active simulation continues beyond the final demonstration frame: 123 physics ticks, 1.0249939 engine seconds |
| `24A888CE-9F8A-4FBB-8FC9-F26730314ACF` | freeplay_mechanic_practice | groundtoairdribble | 30003 | 46 | true | missing physics frames: 545 |
| `CEEC32EE-65B6-45BE-AD55-8ACAA11F89A3` | freeplay_mechanic_practice | powerflick | 5207 | 9 | true | active simulation continues beyond the final demonstration frame: 425 physics ticks, 3.54168701 engine seconds |
| `B6B93AB6-A20C-4AF6-8BAD-6E17624BDFB9` | freeplay_mechanic_practice | forward45fastflick | 1093 | 1 | true | active simulation continues beyond the final demonstration frame: 272 physics ticks, 2.2666626 engine seconds |
| `ED4D364B-DA86-483C-A668-4FAC04887F3A` | freeplay_mechanic_practice | forward45highflick | 2383 | 3 | true | none |
| `F4B86029-AB4E-4725-AFF8-590C71F722D0` | freeplay_mechanic_practice | forward45delayed | 1794 | 1 | true | none |
| `FF4ABCAB-659F-462C-A6B4-3DDA3E4260BE` | freeplay_mechanic_practice | powerflickdelayed | 2485 | 3 | true | missing physics frames: 184 |
| `A3B739EF-71DD-4825-AB72-17662B81D173` | freeplay_mechanic_practice | 180flick | 975 | 1 | true | none |
| `B919DC95-DDD0-4C92-A908-F4C9C9D93C93` | freeplay_mechanic_practice | flipreset | 7539 | 7 | true | active simulation continues beyond the final demonstration frame: 360 physics ticks, 3 engine seconds |
| `3B97377E-EFFD-43C5-B8D3-E59BE304D9CC` | freeplay_mechanic_practice | aerialdribble | 22467 | 23 | true | active simulation continues beyond the final demonstration frame: 451 physics ticks, 3.75842285 engine seconds |
| `5D36C080-1C98-4191-AED8-DD5B7C37F5BB` | freeplay_mechanic_practice | ceilingpinch | 26799 | 35 | true | none |
| `513F33EB-27E7-430D-A3DD-035DA8912B2F` | freeplay_mechanic_practice | wallpinch | 5268 | 6 | true | active simulation continues beyond the final demonstration frame: 163 physics ticks, 1.35827637 engine seconds |
| `3B0EDCE9-E093-44EC-8D08-A378873600C0` | freeplay_mechanic_practice | groundpinch | 17180 | 24 | true | missing physics frames: 1420 |
| `CD6E7DB1-2761-4B8B-BD37-F21C7F135722` | gameplay | nexto_1v1 | 58306 | 0 | true | session termination is incomplete; missing physics frames: 2547; identity failures: 1 |

## Reset-aware segmentation authority

Version: `RESET_AWARE_PAIRED_STATE_ACTION_V2`. Local-human native jump-onset events are activity anchors. Global physics-frame discontinuities and native reset, respawn, and local-car rebind events are hard boundaries. A sustained 24-tick period with at least 2 world-contact wheels, grounded state, and inactive jump/dodge/flip components separates attempts. The segmentation rule never branches on the declared mechanic label. Events after the last paired action frame are excluded rather than clamped onto it.

## Mechanic-specific source-bound adjudication

Each exact source recording is reviewed against a declared-mechanic physical criterion using native 120 Hz state, effective action, and event evidence. Only high-confidence successes are listed as later behavior-cloning candidates. This adjudication is not a reusable production detector.

| Label | Attempts | BC candidates | Failed | Ambiguous |
|---|---:|---:|---:|---:|
| wavedash | 4 | 4 | 0 | 0 |
| zapdash | 7 | 0 | 0 | 7 |
| walldash | 5 | 3 | 2 | 0 |
| musty | 11 | 3 | 6 | 2 |
| breezi | 9 | 3 | 4 | 2 |
| groundtoairdribble | 46 | 35 | 11 | 0 |
| powerflick | 9 | 5 | 4 | 0 |
| forward45fastflick | 1 | 1 | 0 | 0 |
| forward45highflick | 3 | 2 | 1 | 0 |
| forward45delayed | 1 | 1 | 0 | 0 |
| powerflickdelayed | 3 | 3 | 0 | 0 |
| 180flick | 1 | 1 | 0 | 0 |
| flipreset | 7 | 5 | 2 | 0 |
| aerialdribble | 23 | 18 | 5 | 0 |
| ceilingpinch | 35 | 13 | 21 | 1 |
| wallpinch | 6 | 6 | 0 | 0 |
| groundpinch | 24 | 7 | 15 | 2 |

## Evidence files

- `source_inventory.json` binds every native manifest, chunk, event stream, and marker stream by byte count and SHA-256.
- `validation.json` preserves every validator verdict and diagnostic.
- `sessions/*.json` contains session metadata, classification, validation, paired-stream verdict, event inventory, segmentation audit, and criteria.
- `attempts/*.jsonl` preserves every attempt with raw timing references, physical outcome telemetry, and its source-bound adjudication.
- `mechanic_assessments.json` summarizes criteria and all verdict counts.
- `behavior_cloning_candidates.json` contains only high-confidence source spans.
- `groupings.json` is a compact descriptive grouping index.
- `artifact_manifest.json` hashes the generated review package.
- Re-run `python benchmarks/review_rival2_human_demos.py --verify-only` to verify source and review hashes plus attempt/index invariants.

## Known evidence boundary

The strict recorder completeness verdict is retained unchanged as a diagnostic. The review verdict instead asks whether every retained frame has a valid paired state/action and whether global frame-ID gaps are explained by native lifecycle events. No tick is synthesized. The Nexto match is complete at the captured match-ended state even though a later post-match car-spawn callback caused a local-human identity shutdown.
