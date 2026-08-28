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

At start and before every captured frame, the plugin requires all of the following to agree:

1. `GameWrapper::GetLocalCar()`;
2. `GameWrapper::GetPlayerController().GetPRI()`;
3. exactly one PRI in `ServerWrapper::GetPRIs()` whose PRI and owned car match both values.

The stable player identity is `PriWrapper::GetUniqueIdWrapper().GetIdString()`, with the native
player ID as a fallback only if the unique-ID string is unavailable. No car index, team, or player
name assumption is made. If the relationship is null or not unique, start is rejected. If it stops
being unique while recording, capture stops as incomplete and the identity failure is reported.
Every frame contains all nonspectator PRIs/cars and exactly one `is_local_human` flag.

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
- deltas from the preceding captured hook;
- duplicate, out-of-order, and missing physics-frame flags and counts.

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
- duration, final frame count, and observed capture rate;
- attempted, enqueued, written, queue-dropped, duplicate, out-of-order, missing, and identity-failure
  counts;
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
```

`validate` verifies binary framing, CRCs, complete footers, chunk/session/index metadata, per-chunk
frame counts and sequence ranges, chunk and journal SHA-256/size, deterministic sequence order,
physics duplicates/order, action ranges/types, exactly one human car per frame, and clean final
frame count. Use `--strict-complete` to reject rather than recover a final partial chunk. Add
`--output report.json` to any command to write machine-readable JSON.

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
| approximately derivable | 102 | Boost-pad coverage, 30 Hz previous-action choice, and timer/state semantics that do not exactly match RivalSim. |
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

## 120 Hz to candidate 30 Hz diagnostic

`action-variation` retains the native frames, then examines nonoverlapping groups of four only when
both recorder sequence and physics frame are contiguous. It reports:

- valid and gap-skipped four-tick windows;
- fraction of windows in which all eight channels were constant;
- per-channel changed-window count/fraction;
- mean and maximum within-window range;
- session class assembled from the match/freeplay/mechanic/opponent labels.

Analog equality uses a configurable diagnostic epsilon (default `1e-4`); buttons require exact
equality. The report deliberately leaves `decision` null. It does not select first-action,
averaging, action chunks, rejection, or a higher policy rate.

When two or more session paths are supplied, the collection report aggregates independent windows
by session/mechanic class and orders `classes_by_intra_window_variation` by the fraction containing
any changed channel. It also retains every per-session report, so repeated mechanic classes can be
audited without joining frames across session boundaries.

## Manual read-only smoke test

Use this procedure on a machine with Rocket League and BakkesMod. It is a failed gate if any step
cannot be verified; do not replace it with synthetic data.

1. Build and install the DLL, launch BakkesMod and Rocket League, and run
   `plugin load rival_demo_recorder`.
2. Enter ordinary Freeplay. Do not load an online match.
3. Run `rivalrec_start freeplay recorder_smoke`.
4. For 10 seconds, drive, steer both ways, jump, boost, powerslide, pitch/yaw/roll in the air, and
   hit the ball at least once.
5. Run `rivalrec_mark smoke_midpoint`, then `rivalrec_status`, then `rivalrec_stop`.
6. Confirm the console reports a clean stop, no queue drops, no out-of-order frames, and a capture
   rate near the engine cadence. Occasional duplicate/gap counts must be investigated, not hidden.
7. Run `validate`, `inspect`, and `action-variation` on the emitted UUID directory.
8. Confirm validation is true, exactly one human car exists in every frame, native analog values are
   in their actual game ranges, physics frames progress coherently, state changes are continuous,
   the ball-touch and marker journals are present, and hashes verify.
9. Repeat in an RLBot-created local 1v1 using `rivalrec_start match nexto_smoke`; confirm the opponent
   name/bot identity and user label are retained.
10. Attempt start from an online game only as a refusal check; it must reject recording. Do not play
    or automate an online match for this test.

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
