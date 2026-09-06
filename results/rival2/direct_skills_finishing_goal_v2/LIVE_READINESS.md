# Native-play readiness: existing exporter/runtime are not compatible

Read-only CPU probe during the unchanged finishing25training block. This is a
concrete deployment prerequisite, not a reason to interrupt healthy training or
declare the simulator policy broken. No exporter artifact, RLBot installation,
simulator step, optimizer step or source/checkpoint mutation occurred.

The exact625research parent loads strictly into EntityJointControlActorCritic.
Its actor produces90joint-action logits, whose argmax indexes the frozen90x8
action table. The GRU hidden shape is recorded in live_loader_readiness.json.
A zero-observation call only establishes the interface and finite output; it
does not establish sensible driving or observation parity.

The installed G:/dev/RLBot-Rival/scripts/export_rival2_unified_v5.py rejects that
same checkpoint with its correct expected SHA, before any export:

`TypeError: Rival2UnifiedPolicyConfig.__init__() got an unexpected keyword argument 'critic_architecture'`

This is the expected old/new architecture mismatch, not corrupt weights.
The old exporter also decodes5tanh analog means and3sigmoid buttons from hybrid
outputs; those slices are not a joint90 action decoder. The installed recurrent
unified_runtime.py requires120Hz policy/one-tick hold and advances the GRU every
packet. The older feedforward runtime supports action holding, but invokes
model(observation) without hidden state. Neither is an existing30Hz recurrent
entity/joint90 deployment path. Do not merely relax a manifest check.

Required native-play work, separate from current training:

1. Export the selected entity model with the exact joint90 argmax action-table
   decoder, observation182input and recurrent hidden input/output. Verify model
   and sequential hidden/action parity on captured inputs and reset boundaries.
2. Use120Hz packet/lifecycle tracking but30Hz policy/GRU advancement with four
   physics ticks of held action. Handle duplicate/missing packets explicitly;
   no hidden advancement during held ticks and no fabricated missing packets.
3. Audit the actual packet-to-observation mapping and field availability against
   the simulator semantics. The earlier individual-wheel versus aggregate-ground
   diagnostic remains a separate unresolved input gap, not permission to replace
   features silently. Do not retrain/mask the running policy based on this probe.
4. Only after an honest bridge/contract audit, perform actual native local-game
   testing. Fixed RivalSim Nexto wins are not native Rocket League or SSL proof.

All three inspected external source hashes and the exact probed checkpoint SHA
are in the JSON; all remain unchanged afterward. There is no new human-data,
mechanic-detector, reward or PPO change in this work. Existing old artifacts and
bot configuration remain untouched. The finishing branch still follows its
original25-update and fixed evaluation boundary.
