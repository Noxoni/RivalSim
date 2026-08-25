# RivalSim v0.4 Native Lifecycle Authority

## Purpose

v0.4 uses a small, content-addressed RocketSim authority cache for standard-Soccar lifecycle
behavior. Cache construction is native oracle-data generation, not an acceptance run. The GPU
lifecycle gate only consumes a complete matching cache; missing, corrupt, or mismatched data is a
hard error and there is no live fallback.

Large data remains local and ignored under `.tools/v0.4/oracle/`. Compact custody is published in
`results/v0.4/oracle_data.json` and the source/contract map in
`results/v0.4/rules_source.json`.

## Frozen identity

Final authority identity SHA-256:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

Frozen authority artifact SHA-256:

`6A24DF7C483B4F43F94E7FD2B302516A317E7D0C051A0B42C6A52C6D85C1188D`

The identity hashes every input capable of changing the native lifecycle result:

- RocketSim primary commit `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- RocketSim binding commit `2da51b1dac7b8127127613a5ff30e490bdd70dd8`;
- `rocketsim==2.2.1` and installed `RocketSim.pyd` SHA-256
  `E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`;
- all 16 external Soccar CMFs, each file hash, and combined content SHA-256
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`;
- `rivalsim/v04_authority.py` source bytes and generator seed `20260825`;
- exact corpus configuration: 34 pads × two cars, five kickoff layouts/seeds, four respawn
  locations/seeds, and six strict goal-boundary inputs;
- authority settings: Soccar, 120 Hz, Octane, blue/orange teams, and score-callback-to-explicit-
  kickoff composition;
- bounded RivalSim selector, raw-event, and terminal/truncation contract.

The cache was regenerated when the collector's collision search initially failed to descend into
the `soccar/` directory. The final identity above includes all 16 CMFs; the earlier incomplete
identity was rejected and is not release authority.

## Source map

The bounded translation follows these pinned functions:

- `Arena::Step` for pre-tick, Bullet step, post-tick, finish-tick ordering;
- `Arena::IsBallScored` for the strict goal threshold and team attribution;
- `Arena::ResetToRandomKickoff` for standard ball/car reset state and layout families;
- `BoostPad::_PreTickUpdate`, `_CheckCollide`, and `_PostTickUpdate` for recharge, lock,
  contention, grant, and cooldown behavior;
- `Car::Demolish`, `Car::Respawn`, and `Car::_PreTickUpdate` for disabled state, timer, selection,
  respawn state, and re-entry;
- `Arena::_BtCallback_OnCarCarCollision` for the already-accepted v0.3 physical demo event;
- the Python binding's goal callback behavior for first-tick entry events.

Source-file byte hashes and handoff/spec hashes are recorded in
`results/v0.4/rules_source.json`.

## Explicit composition decisions

RocketSim reports a goal but does not itself define a training environment's replay/reset policy.
The frozen v0.4 standard headless contract invokes a deterministic kickoff reset immediately after
the source-valid goal callback. The selected layout is explicit per-world state and advances
modulo five.

RocketSim's default respawn uses ambient random selection when passed seed `-1`. RivalSim exposes
the selected source-valid respawn location as explicit internal state, initialized per world and
advanced modulo four for each car. This makes the lifecycle reproducible without emulating the
host RNG or fitting outputs.

RocketSim defines no regulation-time, overtime, score-limit, or training episode termination
contract for this bounded authority. v0.4 therefore publishes raw score/goal/reset/clock state and
keeps `terminated=truncated=0`. Selecting an episode policy belongs to v0.5.

## Car visitation lifecycle

The v0.3 source experiment established that car-container membership determines a persistent
logical visitation state. Ordinary ticks and physical resets preserve it. Demolition and respawn
do not remove the car object from `_cars`, so v0.4 preserves the selected per-world order across
demo, respawn, goals, and kickoffs. Pad contention uses that same order.

The runtime does not expose native pointers, emulate allocator/heap layout, use case-specific
tables, or choose a branch from expected outputs.

## Cache layout and validation

```text
.tools/v0.4/oracle/
  <authority-identity>/
    identity.json
    authority.json
    frozen.json
```

`identity.json` contains every hashed input. `authority.json` contains the native pad, goal,
kickoff, demolition timer, and respawn records. `frozen.json` binds the identity to the authority
artifact hash, marks the cache complete, and records `live_fallback: false`.

## Invalidation

Regenerate only when an identity input changes, including any RocketSim/binding revision,
extension binary, CMF byte, collector source, corpus/config/seed, authority setting, selector
contract, or raw-event policy. Never bless an identity mismatch or regenerate data implicitly
inside the acceptance runner.
