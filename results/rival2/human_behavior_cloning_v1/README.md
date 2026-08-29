# Rival human behavior cloning V1

This directory is the prospectively frozen authority and evidence package for
Rival's first authorized human behavior-cloning stage. The only trainable object
is a byte-identical copy of the iteration-479 120 Hz bootstrap. The bootstrap
teacher and external observation adapter V2 are frozen, and the optimizer is a
fresh supervised AdamW instance that has no PPO resumability claim.

The frozen campaign uses every paired native 120 Hz action without averaging or
subsampling. Gameplay and mechanic data are separate sampling families. Mechanic
sampling is a 90% natural-frame / 10% label-balanced mixture and retains the
label -> whole attempt -> frame hierarchy; its runtime probability audit must
remain under the frozen 4x per-frame oversampling cap.

Every supervised step also includes whole-world observations from the committed
authoritative RivalSim retention corpus. Candidate checkpoints are transactional
32-step boundaries, selected only on human validation and simulator validation.
The human and simulator test splits are opened once after selection.

Run the no-update authority gate first:

```powershell
.venv\Scripts\python.exe benchmarks\run_rival2_human_behavior_cloning_v1.py --preflight-only
```

After the config/code commit and preflight evidence are pushed and remotely
verified, run the bounded campaign:

```powershell
.venv\Scripts\python.exe benchmarks\run_rival2_human_behavior_cloning_v1.py
```

The run performs no PPO update, reward change, mechanic-detector work, or native
recording mutation. Mechanic closed-loop evaluation fails closed when a native
start cannot be represented exactly in RivalSim; open-loop imitation is never
reported as learned closed-loop capability.
