# RivalSim v0.3 Native Oracle Cache

## Purpose

v0.3 parity runs consume immutable cached RocketSim authority. Cache construction is native
oracle-data generation, not a RivalSim acceptance run. Once a phase cache is complete, its GPU
runner has no live-RocketSim fallback: missing, corrupt, incomplete, or identity-mismatched data
is an error.

Large cache chunks remain local and ignored under `.tools/v0.3/`. Compact custody is published in
`results/v0.3/oracle_data.json`.

## Frozen identities

| Phase | Authority identity SHA-256 | Cases | Native frames | Chunks |
| --- | --- | ---: | ---: | ---: |
| A ball/world | `11E307ABD14C8D79C2BFBDEC7F20C6A45716AD361609A901C517DCA98C804ED2` | 31,216 | 374,592 | 122 |
| B car/ball | `AEEF0E721D995393ACFA0632EE934DCC1DFBE8EC223E81F4A10D48EC63F75A9B` | 8,192 | 98,304 | 64 |
| C car/car | `7A1D369F58CF3FAA0BA06D6D105CCF896A8FD9DE2C994471D2649F852AEC9EC9` | 8,192 | 196,608 | 64 |
| D integrated | `998010324B30E8196429ACF61ACC524780139E5D958543237D121D4886066B08` | 512 | 12,288 | 16 |

Phase C and D have two complete labeled native branches, so their frame totals include both
`a_then_b` and `b_then_a`. Across all phases the cache contains 681,792 tick frames and 56,816
immediate post-`SetState` readbacks.

## Identity inputs

Each authority identity hashes every input capable of changing native truth:

- RocketSim primary commit `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- RocketSim Python-binding commit `2da51b1dac7b8127127613a5ff30e490bdd70dd8`;
- package version 2.2.1 and installed `RocketSim.pyd` SHA-256
  `E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`;
- exact 16-file Soccar CMF set, combined SHA-256
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`;
- phase corpus-generator source and generator schema/config;
- phase seed and exact frozen corpus bytes;
- authority oracle and builder source;
- captured fields/ticks, game mode, body configuration, collision switches, and other relevant
  authority settings.

Phase C/D additionally bind the logical-order diagnostic patch SHA-256
`68A180A6E620153CF34BC2DC748B879656C0A809DE341B6F2194E5ED23C3CBD3` and diagnostic extension
SHA-256 `A92F6680284A7149843AFC1041C70DB574ED4CAEA5176F3EF0F1E0E216763807`.

The identity does not hash RivalSim GPU kernels, Warp compilation output, pilot selection,
comparison code, or reporting changes because those cannot change native truth. Tolerance changes
after freeze are nevertheless prohibited without new authority.

## State custody

Every case is created with isolated native-world semantics. The cache stores both the exact frozen
source state supplied to RocketSim and the immediate native readback after `SetState`. Every native
tick 1–12 is captured; ticks 1/4/8/12 are the blocking acceptance horizons.

Chunk manifests record range, metadata hash, array hash, field schema, captured ticks, extension
hash, and creation process. Finalization reopens and verifies every chunk before writing
`COMPLETE_NATIVE_AUTHORITY`.

## Multi-outcome relation

The source-proven lifecycle for two-car visitation is:

- construction or membership insertion/removal establishes or may change order;
- ordinary ticks and physical state resets preserve order;
- demolition/respawn preserves order when membership does not change;
- arena reconstruction establishes a new order.

The diagnostic extension returns stable logical car IDs only. It does not publish native pointer
values or allocator addresses and does not change physical behavior. Phase C/D authority creation
prospectively constructs fresh arenas until each logical source-valid order is observed, then
applies the frozen physical state and records a complete deterministic trajectory for that label.

A RivalSim trajectory passes when it matches one complete labeled native-valid branch. Comparison
never mixes metrics across branches. RivalSim carries the selected order as internal lifecycle
state; it does not choose an order by consulting case IDs or expected outputs.

## Invalidation

Regenerate a phase cache only when its identity changes. This includes a changed RocketSim or
binding revision, extension binary, CMF bytes, corpus generator/config/seed/corpus, oracle or
builder source, or relevant authority settings. Do not silently regenerate from an acceptance
runner.

A current-source identity mismatch is intentionally fatal. During release validation the final
Phase D cache was prospectively rebuilt from identity `95B216…` to `998010…` because its hashed
corpus/oracle/builder sources had changed. The old data was not reused or blessed.

## Local directory layout

```text
.tools/v0.3/
  phase-a/oracle-cache/<identity>/
  phase-b/oracle-cache/<identity>/
  phase-c/oracle-cache-relational/<identity>/
  phase-d/oracle-cache-relational/<identity>/
```

Each complete directory contains `identity.json`, the frozen corpus artifact, per-range metadata
and arrays, and `manifest.json`. These artifacts are intentionally excluded from Git.
