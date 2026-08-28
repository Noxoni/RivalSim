# Rival 2.0 human-demo review V1

Generated: `2026-08-28T18:59:08.214651Z`

This is an inventory, integrity review, generic activity segmentation, and descriptive physical-outcome pass. It does not train, behavior-clone, detect named mechanics, judge mechanic correctness, or assign rewards. Relative outcome buckets are review candidates only; successful and failed evidence is preserved together.

## Inventory and validation

| Session | Class | Declared label | Frames | Attempts | Ingestion-safe | Principal validation issue |
|---|---|---:|---:|---:|---:|---|
| `D373ED52-3F55-4082-A111-4CD64FC48ACD` | recorder_smoke | recorder_smoke | 5052 | 0 | false | active simulation continues beyond the final demonstration frame: 10359 physics ticks, 86.3246641 engine seconds; duplicate physics frames: 1 |
| `080AD805-D5BF-47E6-B55E-89D1CE4B84E8` | recorder_smoke | replacement_smoke | 12214 | 0 | true | none |
| `4AC8B3D3-B14B-4246-8FFF-4F6E5DB87559` | freeplay_mechanic_practice | wavedash | 5008 | 18 | true | none |
| `3C33CD5C-A488-4D32-8996-3009094C7BBA` | freeplay_mechanic_practice | zapdash | 5188 | 13 | true | none |
| `5263CB76-F043-47B7-892C-0650DA92028C` | freeplay_mechanic_practice | walldash | 12004 | 61 | true | none |
| `AEDCE532-EFC1-4F5D-8B84-F9065F37247A` | freeplay_mechanic_practice | musty | 5961 | 8 | false | missing physics frames: 244 |
| `555A703B-E76A-48C7-8C27-51B642C335F8` | freeplay_mechanic_practice | flipreset | 2009 | 1 | true | none |
| `C6856FFC-DB9A-4E7D-BC5A-F35AEF1BBC66` | freeplay_mechanic_practice | aerialdribble | 1515 | 1 | false | active simulation continues beyond the final demonstration frame: 279 physics ticks, 2.32501221 engine seconds |
| `3C6E58BE-0858-4FD4-B30E-0A74F78299A1` | freeplay_mechanic_practice | ceilingpinch | 1447 | 1 | true | none |
| `D4416964-6CA8-4E2C-A82D-C9F21761D3BA` | freeplay_mechanic_practice | wallpinch | 1257 | 1 | true | none |
| `0260EE5A-8B8B-4A98-8750-01A4D909A22A` | freeplay_mechanic_practice | groundpinch | 9496 | 15 | true | none |
| `023B9FC4-4B94-40FC-AE9A-BC47128C2645` | freeplay_mechanic_practice | breezi | 897 | 1 | false | missing physics frames: 91 |
| `0F9A6033-F1D5-4A29-AEA6-0CD78C1D9988` | freeplay_mechanic_practice | groundtoairdribble | 4994 | 6 | false | missing physics frames: 76 |
| `C528AB34-9A4F-4448-866A-8192F188E002` | freeplay_mechanic_practice | stallflipreset | 8355 | 8 | false | invalid Rival action at sequence 4188; invalid Rival action at sequence 4189; invalid Rival action at sequence 4190; invalid Rival action at sequence 4191; invalid Rival action at sequence 4192; invalid Rival action at sequence 4193; invalid Rival action at sequence 4194; invalid Rival action at sequence 4195; invalid Rival action at sequence 4196; invalid Rival action at sequence 4197; invalid Rival action at sequence 4198; invalid Rival action at sequence 4199; invalid Rival action at sequence 4200; invalid Rival action at sequence 4201; invalid Rival action at sequence 4202; invalid Rival action at sequence 4203 |
| `DF89DF92-6FB6-4249-9DFF-134052D54D7D` | freeplay_mechanic_practice | forward45flick | 2110 | 4 | false | active simulation continues beyond the final demonstration frame: 228 physics ticks, 1.90002441 engine seconds |
| `0F09D892-CCA4-4228-BE29-338957392E05` | freeplay_mechanic_practice | powerflick | 1901 | 3 | false | missing physics frames: 361 |
| `D1CB476C-0190-4817-B562-A6322433DA4A` | freeplay_mechanic_practice | 180 flick | 3982 | 7 | false | active simulation continues beyond the final demonstration frame: 184 physics ticks, 1.5333252 engine seconds |
| `CF6347E4-44EF-4C98-89F9-B232921D962A` | freeplay_mechanic_practice | delayedflick | 5309 | 10 | true | none |
| `4FAFD72D-F7D3-4B97-A484-883BB5D1113B` | gameplay | nexto_1v1 | 51816 | 0 | false | session termination is incomplete; missing physics frames: 1723; identity failures: 1 |

## Generic segmentation authority

Version: `GENERIC_ACTIVITY_RECOVERY_V1`. Local-human native jump-onset events are activity anchors. Recorder sequence/physics discontinuities and native reset, respawn, and local-car rebind events are hard boundaries. A sustained 24-tick period with at least 2 world-contact wheels, grounded state, and inactive jump/dodge/flip components separates attempts. The rule never branches on the declared mechanic label.

## Descriptive outcome groupings

Contact-oriented sessions are selected only when at least 25% of generic segments contain a native local-human ball-touch episode. Their review extremes use exact 12-tick ball velocity change, goal timing, and sustained recovery. Other sessions use within-session planar speed gain and sustained recovery. Quartiles are computed independently inside each recording. These are not training labels or named-mechanic detectors.

| Label | Attempts | Stronger candidate | Limited/failed candidate | No measured human contact | Ambiguous |
|---|---:|---:|---:|---:|---:|
| wavedash | 18 | 4 | 7 | 0 | 7 |
| zapdash | 13 | 4 | 4 | 0 | 5 |
| walldash | 61 | 8 | 37 | 0 | 16 |
| musty | 8 | 1 | 3 | 3 | 1 |
| flipreset | 1 | 0 | 0 | 0 | 1 |
| aerialdribble | 1 | 0 | 1 | 0 | 0 |
| ceilingpinch | 1 | 0 | 0 | 0 | 1 |
| wallpinch | 1 | 0 | 0 | 0 | 1 |
| groundpinch | 15 | 4 | 6 | 1 | 4 |
| breezi | 1 | 0 | 0 | 0 | 1 |
| groundtoairdribble | 6 | 1 | 5 | 0 | 0 |
| stallflipreset | 8 | 2 | 4 | 0 | 2 |
| forward45flick | 4 | 1 | 1 | 2 | 0 |
| powerflick | 3 | 0 | 1 | 1 | 1 |
| 180 flick | 7 | 2 | 4 | 0 | 1 |
| delayedflick | 10 | 3 | 3 | 2 | 2 |

## Evidence files

- `source_inventory.json` binds every native manifest, chunk, event stream, and marker stream by byte count and SHA-256.
- `validation.json` preserves every validator verdict and diagnostic.
- `sessions/*.json` contains session metadata, classification, validation, event inventory, segmentation audit, and grouping criteria.
- `attempts/*.jsonl` preserves each generic attempt with raw timing references and physical outcome telemetry.
- `groupings.json` is a compact descriptive grouping index.
- `artifact_manifest.json` hashes the generated review package.
- Re-run `python benchmarks/review_rival2_human_demos.py --verify-only` to verify source and review hashes plus attempt/index invariants.

## Known evidence boundary

A readable source prefix is still reviewed when the current validator rejects the complete demonstration. Those sessions remain `ingestion_eligible_under_current_validator=false`. Missing physics ticks are never synthesized, invalid native action values are not clamped, and incomplete termination is not upgraded to success.
