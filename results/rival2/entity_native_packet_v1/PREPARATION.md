# Native packet integration and bounded comparison

This is the next no-learning step after09f6618. Two unchanged entity checkpoints,
exact90-way greedy action decoding,120Hz packets/30Hz recurrent decisions,
four-tick held action. No model, optimizer, reward, physics or curriculum changes.
The prior installed Rival and native Nexto files remain unmodified.

## Completed preparation

- Separate runtime and four bot definitions under deployment/entity_native_v1.
- Existing packet observation builder vendored without numerical formula changes;
  source identity is pinned by the preceding input audit. Its unavailable wheel
  broadcasts and sticky counter heuristic are explicit proxies, never measurements.
- New manifests bind the actual OBS_V1 history/event contract and joint90 action
  contract, not the obsolete120Hz hybrid manifest. Complete field quality mask is
  recorded alongside observations; it is NOT fed as a new model input or used to
  zero/change any observation field.
- Warm-up uses20throwaway zero-hidden calls. Game hidden starts fresh at zero.
- Unique player_id/team relationship, not array index, selects the local car.
- Goals/countdown/rebind reset hidden. Pauses suspend decision clock without
  resetting memory or accumulating active episode timers. Held/duplicate/backward
  packets do not advance the GRU. Missing active packets are counted and history
  quality degrades explicitly; missing observations/GRU updates are never invented.
- Touch/demo events accumulate through the four-tick interval and clear only after
  a decision. Previous action retains the actual emitted table action until reset.
-19focused bridge/packet tests passed, including a real exported actor.
-12000captured simulator states converted into packet-shaped fixtures.135shared
  directly representable/derived fields matched to5.364418e-7normalized maximum.
  This does NOT mean actual Rocket League packets equal simulator observations.
  Internal timer/wheel fields were not secretly seeded from simulator internals.
  A first fixture conversion used an ill-conditioned Euler extraction near vertical
  cars (maximum3.17e-4); corrected only the fixture inverse to basis dot products.
  The actual packet builder/model were not modified to hide this fixture error.
- Actual RLBot Python imports the bot/runtime without importing RivalSim/Warp.

## Prospective native evaluation

native_authority.json pins all source/config identities and installed native
Nexto executable/config hashes. It must be committed and remotely read back
before any game launch. Four five-minute standard native local Soccar matches:
reference600Blue, finishing650Orange, finishing650Blue, reference600Orange.
Game-native kickoff choices; no state setting, scripted prefixes, new opponents,
training or candidate selection during evaluation. Default physics/mutators.
At most120seconds overtime per case and600seconds wall time including startup.
Initial packet wait bounded150seconds; stalled stream30seconds stops the task.
This small development comparison is not a ranked/SSL assessment.

No existing RocketLeague process may be interrupted to launch this comparison.
Run definitions stay in RivalSim rather than replacing the old installed bot.
The manager may start RLBotServer and RocketLeague for this authorized offline
comparison, and stops only its own match at completion/failure/user STOP.

External run directory: G:/dev/RivalSim-runs/entity-native-comparison-v1.
User STOP file in that directory ends the comparison; never clear it to recover.
Each case preserves exact source readiness, current summary, native match result,
all delivered tick/action records, and per-decision observations/actions/quality
plus the corresponding packed native packet. Gzip streams are bounded-memory;
flush every120decisions and close on match end/retirement. Raw packets are at
decision times, NOT a claim of full120Hz raw packet capture. Tick/action log
records every packet delivered to the bot, with gaps explicitly counted.

Afterward verify export/checkpoint hashes, replay recorded policy observations
through the exact exported actor with captured reset boundaries, and summarize
actual goals, touches, motion, packet cadence and input qualifications. Do not
reinterpret unavailable input as exact if native results are poor. No new PPO
run is authorized by this document; use the native evidence to choose the next
specific correction toward the continuing SSL goal.
