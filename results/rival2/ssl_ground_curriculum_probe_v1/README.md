# Reset-only ground acquisition probe

This is a bounded experiment under the user's continuing SSL-development goal,
not a new random-weight restart or an SSL promotion. The preceding matched
analog-noise pilot was negative/inconclusive; native diagnosis found repeated
pass-by and poor return-to-ball control despite sufficient observation/control
information. See the committed preceding pilot report.

Start from the preserved fresh 30Hz update-597 checkpoint, SHA256
`B0B35CDAF3B3551EC667776EB99C3822F863AAA1F17A0BA2F013B5F216BD87A5`.
Preserve every learned tensor, Adam moment, optimizer counter and RNG state.
The only training change is the reset-state bank: 80% short achievable ground
approaches/finishes, with two off-angle difficulty bands; 20% unchanged original
curriculum states. Both players are freely controlled by the same current
policy. There are no action masks, scripted prefixes, teacher actions, task IDs,
new rewards, Nexto training samples, architecture changes or sigma multipliers.
The original potential-only goal contract and all numerical guards are retained.

The experiment takes exactly 30 accepted updates, preserving a rolling checkpoint
every update and permanent +10/+20/+30 snapshots. Evaluate the original fixed
acquisition/finishing/Nexto cases at +0/+10/+20/+30. Compare +20 and +30 against
the already measured unchanged-noise control from the same parent, not against
an easier test that merely mirrors the new resets. The prospective success rule
is an experiment continuation criterion, not a permanent deployment gate; all
results, including negative results, must be reported. No silent extension.

Full-scale no-optimizer-step forward/backward preflight must pass and authority,
sources and evidence must be committed and remotely verified before training.

Commands from the repository root:

```
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_ground_curriculum_probe.py prepare
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_ground_curriculum_probe.py preflight
# Commit, push, verify prospective authority before invoking run.
.venv\Scripts\python.exe -u benchmarks/run_rival2_ssl_ground_curriculum_probe.py run
```

External state: `G:\dev\RivalSim-runs\ssl-ground-curriculum-probe-v1`.
Create `STOP` there to request an accepted-boundary stop. Recovery requires
`run --resume <verified same-probe checkpoint> --resume-sha256 <exact SHA256>`.
Resume preserves model/optimizer/RNG/counters and starts fresh scenario episodes,
as explicitly declared for the existing simulator checkpoint format. Never run
two GPU learners or resume the original paused campaign automatically.
