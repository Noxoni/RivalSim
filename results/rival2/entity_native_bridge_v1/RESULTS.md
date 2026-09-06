# Native bridge preparation: inference verified, packet integration still required

No training, source checkpoint mutation, reward/physics change, RLBot source
change, installation or native Rocket League match occurred. This moves toward
native comparison; it is not a new gameplay result or SSL claim.

## Verified inference

Prospective implementation/protocol commit:
`671000a98dee3643e6321c3bc49ceab33677484b`, pushed before capture.
The existing published handbrake-corrected match runtime was verified before
either capture. Two2400-tick/ten-world captures cover20seconds of every standard
layout on each side against Nexto. Original source policies controlled them.

| Candidate | Native decision samples | Actual goal resets | CPU source/export logit and hidden error | GPU-source/CPU-export action differences |
|---|---:|---:|---:|---:|
| Reference600 | 6000 | 17 | 0 | 0 |
| Finishing650 | 6000 | 18 | 0 | 0 |

Both CPU recurrent trajectories were independently evolved from zero with only
captured episode resets. All eight action channels matched. Scheduler replay
also checked36000held physics ticks without inference/hidden advancement.
Source GPU vs CPU hidden state was NOT byte-identical: maximum absolute
differences0.0059038103 and0.0056702793 respectively, despite identical selected
actions. This finite-duration result does not guarantee arbitrary future
GPU/CPU trajectory identity. Source weights remain exactly unchanged.

Both serialized artifacts loaded in the actual external RLBot Python3.12.14,
PyTorch2.13.0+cpu environment without importing RivalSim. Another12000sequential
actions matched captured source actions exactly. Median CPU inference was
approximately0.199ms;99th percentiles0.323ms/0.314ms. Maxima60.89ms/39.67ms include
cold-start execution. Warm-up with throwaway hidden state is needed before a
live bot reports ready. These are forward-only measurements, not packet/network
latency guarantees. TorchScript deprecation warnings from the source runtime
are preserved in tests; target interpreter compatibility was tested, not assumed.

Artifacts (whole resumable originals remain separately preserved):

- `checkpoints/rival2/entity_native_bridge_v1/reference600.ts`
  SHA256 `7C034675349144A4BBEF588F6954CF0D786217164052428777742C57F109EB72`
- `checkpoints/rival2/entity_native_bridge_v1/finishing650.ts`
  SHA256 `37262522D808CD65E374271DC6004E2C2E2C84422191D7F627B222082EC59DE0`

Each is4809176bytes, one entity/shared/GRU actor, exact90-choice argmax decoder,
no critic, optimizer, router, task input or training-temperature division.
Manifests bind exact source checkpoint/model/config, action table, observation
contract, captured corpus and serialized artifact hashes. `.ts` is inference-only,
not PPO-resumable. Full originals still are.

## Packet observation audit, not domain parity

`input_availability.json` enumerates all182fields and hashes every inspected
source, including installed RLBot stubs and old packet builder. Classification:
39direct,104derived,29approximate,10unavailable. Direct/derived denotes accessible
data/formula, NOT proof that a reconstructed native observation equals a
simulator observation across all contact/timer/phase transitions.

The10unavailable fields are eight individual wheel flags and two internal
sticky-force counters. Existing aggregate-ground broadcasts and sticky heuristics
are documented proxies, not measured values. Approximations include force-state
enums, accumulated jump/air/boost/supersonic timers, dodge availability, flip-time
semantics and lifecycle event/reset accounting. Do not silently mask inputs,
retrain adapters, or relabel these exact to pass an audit.

All182field ordering/scales and canonical pad metadata match the installed
builder's layout. **Temporal contract does not match its old manifest.** Current
policy uses OBS_V1 decision-interval history/events at30Hz; old live manifest
uses OBS_V2_120HZ. A new manifest must bind the actual
`10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF`
observation hash, not reuse the old120Hz hash because dimensions happen to agree.

Primary documentation cross-checks:
[RLBot game data](https://wiki.rlbot.org/v5/botmaking/game-data/) and
[official game-data schema](https://raw.githubusercontent.com/RLBot/flatbuffers-schema/main/schema/gamedata.fbs).
Installed stubs/source hashes, rather than mutable web HEAD, bind this audit.

## Checks, operational issues and next action

Initial9bridge tests passed;15with adjacent entity/joint tests passed before
capture. Final suite adds full182-field classification/unknown-field refusal.
Evidence is in focused_tests.xml and final_tests.xml.
An audit-script import initially referenced a nonexistent pad-duration constant;
corrected to existing six-big/28-small canonical duration definition, verified
against the actual observation builder. No simulation/training semantic change.
First target-interpreter probe repeatedly decompressed its NPZ inside the loop;
the owned read-only CPU diagnostic was interrupted before completion, then
corrected to load the bounded arrays once. It produced no accepted report.
`target_runtime.json` is the completed corrected probe, not a training rerun.

**Next concrete step:** integrate the verified scheduler/actor with a separately
versioned packet adapter and manifest inside RivalSim, leaving external installed
bot untouched. Use actual packet-shaped fixtures from recorded/native data,
preserve30Hz interval event/history semantics, qualify gaps and unavailable
fields in per-decision telemetry, and test real score/countdown/rebind boundaries
versus ordinary pauses. Do not reset memory merely because a nonterminal pause
occurred. Prove cold warm-up does not alter initial policy hidden state.

Then define a bounded native local-game comparison of the two exact candidates
before installation/launch. No additional PPO block or new imitation experiment
is started by this result. The development goal remains open.
