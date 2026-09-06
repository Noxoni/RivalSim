# Entity/joint90 native bridge preparation V1 (no learning)

Next step from results commit368bf8fe2ce2699d657c4002dd07f9024c19da5a.
Preserve exact reference600 and finishing650 sources bound in
benchmarks/prepare_entity_native_bridge.py. No checkpoint selection or training.
No writes to external RLBot source, installed configuration or Rocket League.

Export only copied actor/shared/entity/GRU inference tensors, retaining exact
argmax over the recorded90x8table. No temperature or hybrid decoder. Critic and
optimizer omitted from the inference artifact; whole resumable originals remain.
Use a scripted single-decision wrapper, not a trace with frozen batch sizes.

Capture2400physics ticks per candidate in the existing native RivalSim ten-world
Nexto match runner, five layouts and both sides, with owned CUDA stream and GPU
lease. Verify the published handbrake-corrected runtime first. Source actor runs
normally and controls all ticks, including goal-driven resets. Capture every
fourth tick's actual pre-trunk observation, source actions, hidden in/out and
reset identity. Require real goal resets; never synthesize them as coverage.
This is simulator input evidence, NOT Rocket League packet equivalence.

Replay all ten sequences separately on CPU through source and exported actor,
with independently evolving hidden states, zeroing only at captured resets.
Require identical controller outputs for all samples, including captured GPU
source actions. Numerical maxima reported; CPU export/source logit tolerance
2e-4absolute+2e-6relative, hidden2e-5absolute+2e-6relative. Do not weaken on failure.
Feed the same observations through the packet scheduler with all intervening
held ticks: no hidden advancement, no unseen observation reconstruction.
Separate focused fixtures cover duplicates, backwards packets, gaps, rebinds,
inactive boundaries and nonfinite output. Gaps count missed ticks/decisions and
reanchor to the received frame; do not pretend missing native history is known.

The existing packet observation builder needs a complete field availability
audit and live-interpreter checks before deployment. Inference parity here is
necessary but does not prove domain parity or capability. Individual wheel and
reconstructed timer gaps cannot be silently renamed exact or masked. No new
adapter learning, rewards, physics, model architecture or curriculum change.
Prepare that mapping and bounded native-match protocol before installation.

Retain all raw archives, manifests, exact hashes, tests and negative evidence.
Do not extend the completed25-update block. The wider SSL goal remains open.
