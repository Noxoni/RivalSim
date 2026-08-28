# Rival 2.0 native human demonstration recorder

## Scope and safety boundary

The Rival recorder is recording infrastructure only. It does not behavior-clone or train Rival,
change rewards, alter RivalSim physics, choose controls, inject input, or automate gameplay. The
plugin records only Rocket League state and the effective `ControllerInput` passed to the local
human car. It does not access microphone/audio, video, unrelated keyboard or controller activity,
network traffic, or process memory.

The plugin refuses to start in an online game. Intended contexts are:

- Rocket League Freeplay;
- Rocket League local/offline matches;
- RLBot-created local games.

The implementation contains no call to `SetInput`, `SetVehicleInput`, `OverrideParams`,
`ExecuteUnrealCommand`, car/ball transform setters, spawn APIs, or other gameplay mutation APIs.
The input hook parameter is copied through a `const ControllerInput*` and is never written.

## Capture architecture and SDK pin

The plugin uses the native pre-call hook:

```text
Function TAGame.Car_TA.SetVehicleInput
```

through `GameWrapper::HookEventWithCaller<CarWrapper>`. This is the action Rocket League consumes,
after the game's controller/keyboard binding logic, at the native physics application cadence. The
plugin does not poll XInput and needs no controller calibration.

The official BakkesMod SDK is pinned to:

```text
repository: https://github.com/bakkesmodorg/BakkesModSDK
commit:     479e8f571cf554b25f4eeb64d611dec4133edcaf
subject:    BM218 SDK
API build:  95
```

The lock is machine-readable in
`tools/rival_demo_recorder/sdk.lock.json`. CMake rejects a checkout at any other revision.

## Windows build

Requirements are Visual Studio 2022 with the Desktop development with C++ workload, CMake 3.24 or
newer, Git, and the pinned SDK checkout.

```powershell
git clone https://github.com/bakkesmodorg/BakkesModSDK G:\dev\BakkesModSDK
git -C G:\dev\BakkesModSDK checkout --detach 479e8f571cf554b25f4eeb64d611dec4133edcaf

cmake -S tools\rival_demo_recorder `
  -B build\rival_demo_recorder `
  -G "Visual Studio 17 2022" -A x64 `
  -DBAKKESMOD_SDK_ROOT=G:\dev\BakkesModSDK
cmake --build build\rival_demo_recorder --config Release --parallel
```

Outputs:

```text
build/rival_demo_recorder/Release/rival_demo_recorder.dll
build/rival_demo_recorder/Release/rivalrec_format_fixture.exe
```

The compiled plugin embeds the RivalSim Git SHA, plugin build identity, and pinned SDK revision.

## Install and load

1. Copy `rival_demo_recorder.dll` to the BakkesMod plugins directory, normally:

   ```text
   %APPDATA%\bakkesmod\bakkesmod\plugins\rival_demo_recorder.dll
   ```

2. Start BakkesMod and Rocket League.
3. Open the BakkesMod console with F6 and run:

   ```text
   plugin load rival_demo_recorder
   ```

4. The console must print the recorder build and exact SDK revision. Any load error is a failed
   install; do not record until it is resolved.

The default output root is the folder returned by BakkesMod `GetDataFolder()`, followed by
`rival2/human_demos`. A typical path is:

```text
%APPDATA%\bakkesmod\bakkesmod\data\rival2\human_demos\<session-uuid>
```

To use a Rival data drive instead, set the persistent CVar before starting:

```text
rivalrec_data_dir G:\RivalData\human_demos
```

## Recorder commands

Normal local or RLBot 1v1:

```text
rivalrec_start match nexto_1v1
rivalrec_status
rivalrec_mark clean_recovery
rivalrec_note opponent=Nexto map=DFH
rivalrec_stop
```

Freeplay mechanic practice:

```text
rivalrec_start freeplay musty
rivalrec_status
rivalrec_mark especially_clean_attempt
rivalrec_note practicing 45-degree and 90-degree variants
rivalrec_stop
```

The optional label is stored as `opponent_label` for a match and `mechanic_label` for Freeplay. It
may contain spaces. Recording is continuous from start through stop; markers and notes are optional
metadata and are never needed for later segmentation or mechanic detection.

Optional BakkesMod keybind examples, kept separate from controller input:

```text
bind F8 "rivalrec_start freeplay musty"
bind F9 "rivalrec_mark clean_attempt"
bind F10 "rivalrec_stop"
```

Do not bind a controller button if the bind would interfere with gameplay. Console commands are the
authoritative workflow.

## Human-car identification

At start and at every relevant `SetVehicleInput` callback, the plugin requires all of the
following to agree:

1. `GameWrapper::GetLocalCar()`;
2. `GameWrapper::GetPlayerController().GetPRI()`;
3. exactly one PRI in `ServerWrapper::GetPRIs()` whose PRI and owned car match both values.

The session identity is strictly `PriWrapper::GetUniqueIdWrapper().GetIdString()`. A missing stable
unique ID is an identity failure; the recorder does not substitute a car address, index, team,
player name, or other guess.

The car UObject address is a replaceable binding, not player identity. If the same stable player
uniquely owns a different current car after a goal, round reset, respawn, Freeplay reset, or other
supported local replacement, the recorder atomically rebinds to that car and continues. Every
transition emits `local_car_rebind` with the previous and new addresses, stable player ID, physics
frame/time, sequence boundary, and most recent lifecycle-event context. If the logical player
relationship becomes null, ambiguous, or changes stable ID, recording stops as incomplete and an
`identity_failure` event preserves the reason. Every frame contains all nonspectator PRIs/cars and
exactly one `is_local_human` flag.

## Native inputs captured

Every native field in the pinned SDK `ControllerInput` is stored:

| Native field | Stored type |
| --- | --- |
| `Throttle` | IEEE-754 float32 |
| `Steer` | IEEE-754 float32 |
| `Pitch` | IEEE-754 float32 |
| `Yaw` | IEEE-754 float32 |
| `Roll` | IEEE-754 float32 |
| `DodgeForward` | IEEE-754 float32 |
| `DodgeStrafe` | IEEE-754 float32 |
| `Handbrake` | boolean |
| `Jump` | boolean |
| `ActivateBoost` | boolean |
| `HoldingBoost` | boolean |
| `Jumped` | boolean |

The direct `RIVAL2_ACTION_V1` projection is stored beside the raw input:

```text
throttle, steer, pitch, yaw, roll, jump,
boost = ActivateBoost OR HoldingBoost,
handbrake
```

No analog channel is quantized or downsampled. Native float32 values are serialized by exact bit
representation. The five bitfields are stored as their exact logical true/false values.

The top-level frame `native_input` copied from the hook parameter is the authoritative action
label. `cars[*].native_input`, obtained separately from `CarWrapper::GetInput()`, is state/debug
telemetry observed at a different engine phase and is not interchangeable with the applied action.
The first live smoke differed on 4,104 of 5,052 frames, so converters must never substitute the
per-car snapshot for the top-level hook input.

## Native state captured per input application

Each frame represents state read synchronously at the native input-application hook plus the exact
action passed to that call.

Frame/timing identity:

- monotonically increasing recorder sequence;
- `EngineTAWrapper::GetPhysicsFrame()`;
- `EngineTAWrapper::GetReplicatedPhysicsFrame()`;
- `EngineTAWrapper::GetPhysicsTime()`;
- configured native engine physics framerate in the session manifest;
- server seconds elapsed;
- high-resolution local monotonic nanoseconds;
- UTC nanoseconds;
- deltas from the preceding published physics tick;
- out-of-order and missing physics-frame flags and counts;
- duplicate-hook, suppressed-duplicate, and retained-duplicate counters.

`SetVehicleInput` can be invoked more than once for the local car in one physics frame. The
recorder holds one frame pending until the physics-frame identity advances. A later same-tick
callback replaces the pending frame because it is the final input application observed before the
engine advances. Each replaced callback is preserved as a `duplicate_hook_callback_suppressed`
event containing both raw native inputs. Only the final callback is published, so the binary frame
stream has one unambiguous frame per physics tick. The counters are:

- `duplicate_hook_callbacks`: extra same-tick hook invocations observed;
- `duplicate_frames_suppressed`: earlier pending frames replaced by the deterministic rule;
- `duplicate_frames_retained`: duplicate physics frames published; production value must remain
  zero.

No missing physics tick is synthesized. If the bounded writer queue is exhausted, that frame is
dropped, the recorder sequence advances, and the final manifest reports the drop; a reader then
observes the explicit sequence gap.

Ball:

- position;
- native integer Unreal rotation;
- linear and angular velocity;
- effective gravity Z and replicated ball-gravity scale;
- latest native touch time;
- latest native world-hit time;
- latest hit team;
- current affector car/player identity when present.

Every nonspectator participant/car:

- stable unique ID, player name, numeric player ID, team, and bot flag;
- local-human flag;
- car present and demolition/respawn lifecycle state;
- position and native integer Unreal rotation;
- linear and angular velocity;
- raw native current boost amount;
- on-ground, supersonic, jumped, double-jumped, can-jump, and has-flip flags;
- time off ground and time on ground;
- last ball-touch and ball-impact physics frames;
- native PRI respawn-time-remaining value;
- wheel-contact and wheel-world-contact counts;
- boost, jump, double-jump, dodge, and flip component active/activity-time state;
- native dodge direction;
- native flip time and direction;
- current native `ControllerInput` when the car wrapper exists;
- every exposed wheel: index, contact and world-contact flags, contact-change time, contact location,
  contact normal, lateral and longitudinal directions, reference location, suspension distance, and
  spin speed.

Match/field:

- native match type/game mode, current map, and match GUID;
- seconds elapsed and remaining;
- total game time and overtime time played;
- paused, overtime, round-active, match-ended, ball-hit, and kickoff/countdown flags;
- team 0 and team 1 scores;
- round, replicated countdown, and remaining-countdown numbers;
- event-discovered boost pad identity, position, native boost amount and boost type, type-derived full
  boost class, pickup/spawn active state, native respawn delay, and explicitly quality-marked derived
  cooldown remaining.

Every ball, car, and match block has an availability mask. Default numeric storage for an absent
car or unavailable block is never evidence of availability; converters must consult the mask.

## Native event journal

`events.jsonl` stores native event name, recorder-sequence boundary, physics frame, engine/game
time, monotonic/UTC time, caller identity, and relevant actor identity for:

- ball touch/car hit, including SDK-exposed hit location, hit normal, ball address, and contacting
  car identity when the hook parameter layout is valid;
- wheel-ball contact, including wheel address;
- goal;
- demolition;
- respawn;
- kickoff/round reset and game reset;
- jump onset;
- dodge/flip impulse onset;
- Freeplay/player reset;
- boost-pad pickup/spawn;
- ball out-of-world.

Recorder lifecycle evidence is also journaled:

- `local_car_rebind`: `previous_car_address`, `new_car_address`, `stable_player_id`,
  `reason=resolved_local_car_changed`, and `event_context` in addition to the common timing fields;
- `identity_failure`: the unresolved/changed identity reason before an incomplete stop;
- `duplicate_hook_callback_suppressed`: same-tick identity, deterministic rule, sequence, and the
  suppressed and retained raw `ControllerInput` objects.

`markers.jsonl` stores optional `rivalrec_mark` and `rivalrec_note` entries with the same timing
identity. Events are segmentation evidence only; the recorder makes no reward or mechanic-quality
judgment.

## `RIVALRL_NATIVE_DEMO_V1` format

Session layout:

```text
<session-uuid>/
    manifest.json
    events.jsonl
    markers.jsonl
    chunks/
        000000.rvr
        000001.rvr
        000002.rvr.partial     # possible after crash
```

All binary values are explicitly little-endian. Game scalars/vectors remain float32, Unreal
rotations remain signed int32, counters/timestamps use fixed-width integers, and strings are
length-prefixed UTF-8. No lossy compression is used.

Each chunk has:

1. 40-byte header: magic `RIVRDMO1`, schema version, chunk index, session UUID bytes, first sequence;
2. append-only records with magic `RVRC`, type, byte length, and CRC32;
3. frame records;
4. clean footer containing frame count and last sequence.

The active file uses `.rvr.partial`. Every 120 frames by default it is flushed. At 3,600 frames by
default, the chunk receives a footer, is closed, atomically renamed to `.rvr`, SHA-256 hashed, and
published in an atomically replaced manifest. The queue defaults to 4,096 records and is bounded.
These values are configurable through `rivalrec_flush_frames`, `rivalrec_chunk_frames`, and
`rivalrec_max_queue_frames`.

A crash leaves previously completed, hash-published chunks intact. The deterministic reader can
recover only complete CRC-valid records from the final `.partial` chunk and does not synthesize its
truncated tail. A `.rvr` without a valid footer is corruption, not a clean chunk.

On clean stop the manifest includes:

- session UUID, UTC/local start and UTC end;
- match/freeplay mode and user labels;
- Rocket League, BakkesMod, plugin, SDK, schema, and RivalSim Git identities;
- map and local player/team identity;
- session wall duration, final frame count, and session-wide capture rate;
- active-capture first/last physics frame and engine time, active duration, and active capture rate;
- stop physics frame, engine time, and local monotonic time;
- attempted, enqueued, written, queue-dropped, duplicate-hook, suppressed/retained duplicate,
  out-of-order, missing, identity-failure, and local-car-rebind counts;
- the same-tick last-callback resolution rule and authoritative action-source declaration;
- per-chunk frame/sequence ranges, bytes, clean-complete flag, and SHA-256;
- event/marker bytes and SHA-256;
- clean/incomplete flag and termination reason.

## Python reader, validation, and inspection

Run from the RivalSim repository with its environment:

```powershell
$session = "$env:APPDATA\bakkesmod\bakkesmod\data\rival2\human_demos\<session-uuid>"

.\.venv\Scripts\python.exe -m rivalsim.human_demo validate $session
.\.venv\Scripts\python.exe -m rivalsim.human_demo inspect $session
.\.venv\Scripts\python.exe -m rivalsim.human_demo action-variation $session
.\.venv\Scripts\python.exe -m rivalsim.human_demo action-variation $session1 $session2 $session3
.\.venv\Scripts\python.exe -m rivalsim.human_demo mapping-report
.\.venv\Scripts\python.exe -m rivalsim.human_demo action-alignment
```

`validate` reports three distinct verdicts:

```text
container_valid
capture_complete
overall_demonstration_valid = container_valid AND clean stop AND capture_complete
```

`container_valid` covers binary framing, CRCs, complete footers, chunk/session/index metadata,
per-chunk frame counts and sequence ranges, chunk and journal SHA-256/size, deterministic sequence
order, action ranges/types, exactly one human car per frame, and clean final frame count.

`capture_complete` separately compares the final frame with the final event, marker, and recorded
stop physics identity. It reports all final identities, trailing uncovered ticks/engine seconds,
active-capture bounds/rate, session wall duration/rate, and rejects duplicate published ticks,
missing/out-of-order ticks, queue drops, retained duplicates, identity failures, or material active
simulation after the final frame. The legacy `valid` field and CLI exit status now use the safe
`overall_demonstration_valid` verdict. Use `--strict-complete` to reject rather than recover a final
partial chunk. Add `--output report.json` to write machine-readable JSON.

The API is deterministic and streaming:

```python
from rivalsim.human_demo import SessionReader

reader = SessionReader(session_path)
for frame in reader.iter_frames():
    consume(frame)
```

## Current 182-field Rival observation mapping

The raw format does not store or freeze the current 182-vector. The command
`mapping-report` enumerates every field by current schema name and classifies it as direct,
derivable, approximate, or unavailable. At this SDK/schema revision:

| Classification | Fields | Meaning |
| --- | ---: | --- |
| exact/direct | 16 | Native boolean/contact value is already in the field's logical form. |
| exactly derivable | 58 | Deterministic normalization, team transform, orientation, difference, or event derivation. |
| approximately derivable | 102 | Boost-pad coverage, the first-frame missing pre-session action, and timer/state semantics that do not exactly match RivalSim. |
| unavailable | 6 | No exact source in the pinned SDK. |

The six unavailable fields are:

```text
self.time_since_boosted
self.supersonic_time
self.sticky_ticks
opponent.time_since_boosted
opponent.supersonic_time
opponent.sticky_ticks
```

The full field-level report, including source and reason for all 182 fields, is produced by the
command. An adapter must not fill unavailable values with plausible defaults.

## Direct 120 Hz action alignment

`action-alignment` emits the versioned action relation for the active V2 line:

```text
native Rocket League physics frame N
-> frame.rival_action[N]
-> RIVAL2_ACTION_V2_120HZ decision N
```

There is no averaging, subsampling, four-frame combination, or other temporal reduction. The
action relation is exact and independent of the observation mapping report: approximate or
unavailable observation fields remain explicitly classified and are never synthesized. The first
recorded frame cannot provide the action from before the recording session; later contiguous
frames provide the immediately preceding action required by `RIVAL2_OBS_V2_120HZ`.

## Historical four-tick variation diagnostic

`action-variation` retains the native frames, then examines nonoverlapping groups of four only when
both recorder sequence and physics frame are contiguous. It reports:

- valid and gap-skipped four-tick windows;
- fraction of windows in which all eight channels were constant;
- per-channel changed-window count/fraction;
- mean and maximum within-window range;
- session class assembled from the match/freeplay/mechanic/opponent labels.

Analog equality uses a configurable diagnostic epsilon (default `1e-4`); buttons require exact
equality. The report deliberately leaves `decision` null. It is retained as historical evidence
about what the former 30 Hz reduction problem would have lost; it does not alter the direct V2
120 Hz target relation.

When two or more session paths are supplied, the collection report aggregates independent windows
by session/mechanic class and orders `classes_by_intra_window_variation` by the fraction containing
any changed channel. It also retains every per-session report, so repeated mechanic classes can be
audited without joining frames across session boundaries.

## Manual replacement/lifecycle smoke test

Use this procedure on a machine with Rocket League and BakkesMod. It is a failed gate if any step
cannot be verified; do not replace it with synthetic data.

1. Build and install the DLL, launch BakkesMod and Rocket League, and run
   `plugin load rival_demo_recorder`.
2. Enter ordinary Freeplay. Do not load an online match.
3. Run `rivalrec_start freeplay replacement_smoke`.
4. Drive, steer, jump, boost, powerslide, rotate in the air, and touch the ball.
5. Trigger at least two explicit Freeplay resets. Continue driving/jumping/boosting/touching after
   every reset; do not stop recording between replacements.
6. Score or otherwise trigger a normal goal/round reset if the active Freeplay mode supports it.
7. Near the end run `rivalrec_mark after_multiple_replacements`, continue driving for several more
   seconds, then run `rivalrec_status` and `rivalrec_stop`.
8. Confirm a clean stop, zero queue drops/out-of-order/missing frames, one or more
   `local_car_rebind` events when addresses changed, nonzero `duplicate_hook_callbacks` only if each
   has a matching suppressed event, and `duplicate_frames_retained=0`.
9. Run `validate`, `inspect`, `action-alignment`, and `action-variation` on the emitted UUID. Require
   `container_valid=true`, `capture_complete=true`, `overall_demonstration_valid=true`, no material
   trailing uncovered time, active capture rate near the declared engine cadence, stable player ID,
   continuous sequence/physics identity after every rebind, exact hashes/CRCs, and exactly one human
   car per frame.
10. Repeat in an RLBot-created local 1v1 with `rivalrec_start match nexto_replacement_smoke`. Score at
    least one goal and continue playing after the kickoff/respawn. Confirm the opponent identity and
    label remain present and the sequence crosses every replacement.
11. Attempt start from an online game only as a refusal check; it must reject recording. Do not play
    or automate an online match for this test.

## Failed first smoke retained as regression evidence

Session `D373ED52-3F55-4082-A111-4CD64FC48ACD` is intentionally not rewritten or ingested. With the
corrected validator it reports a structurally valid container and native-cadence prefix, but fails
capture completeness because event/marker activity continues 10,359 physics ticks and 86.324664
engine seconds beyond the final frame. Its active rate is approximately 120.0475 Hz while its
session-wide rate is approximately 32.5540 Hz. This distinction is the regression contract for the
completeness gate.

## Exact native-state limitations

The pinned SDK does not expose every desirable value as a complete authoritative array or timer:

- there is no complete boost-pad enumeration/cooldown array. Pads are discovered by native
  pickup/spawn events after recording starts; active state and respawn delay are native, while
  cooldown remaining is event-time-derived and quality-marked. Unseen pads remain absent;
- Rocket League's full mutator configuration is not exposed by one supported read-only SDK wrapper;
  effective ball gravity and native game/match identifiers are recorded instead;
- demolished lifecycle is derived from a participant with no current car and positive native respawn
  time; the SDK does not expose one standalone per-PRI demolished bit;
- exact RivalSim timer semantics for boost history, supersonic duration, sticky ticks, jump timing,
  and demolition timing are not native Rocket League telemetry contracts;
- an authoritative general ball-world contact callback with contact point/normal is not exposed by
  the pinned supported wrappers. Car-ball and wheel-ball contact geometry is recorded where exposed;
- a session begun after a kickoff, pad pickup, or earlier lifecycle event cannot reconstruct the
  pre-session event history.

These limitations are represented as unavailable/approximate data. The recorder never invents
missing ticks, pads, timers, events, or state.
