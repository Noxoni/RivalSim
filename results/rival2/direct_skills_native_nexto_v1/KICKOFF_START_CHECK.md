# Read-only kickoff-start check during the frozen block

Question from the completed child10 result: could zero Rival kickoff-first
contacts be explained by missing kickoff starts or disabled Nexto kickoff
admission in the training bank? This focused check does not rerun the completed
controller oracle, native matches, observation audits or GPU preflight.

## Evidence

Rebuilt the exact32,768-row source scenario bank on CPU with the frozen default
seed2026090601. Its digest matches the committed training package:

`A095350CA07C88AC4AAB7C9DEB1A668374B49D64A4023C8D1DB01CC68FE52E66`

| Measurement | Count |
| --- | ---: |
| Dedicated kickoff-role source rows |4915|
| Dedicated kickoff rows with noncentered ball xy |0|
| Additional natural-role centered kickoff rows |4900|
| Centered rows in Nexto-controlled even-numbered worlds |4908|
| Those Nexto rows denied the nearest-car admission criterion |0|
| Centered starts in challenge/finishing/defense roles |0|

All five standard layout IDs0,1,2,3,4 are represented. Dedicated kickoff is15%
of starts; another roughly15% of total starts are the kickoff half of the30%
natural family. Do not confuse the source distribution with the duration-weighted
PPO sample distribution. Shooting/finishing remains20% of source starts.

Read the actual call path:

- `DirectSkillNativeNextoCollector.assign` explicitly selects the native-v5
  controller, the opponent side and active mask, and activates new scenario
  episodes. It does not use the legacy adapter or leave kickoff admission denied.
- The inherited per-physics-tick provider identifies centered ball xy and passes
  that flag to `nexto.tick_action` before applying the emitted controller action.
- The native controller admits the closest car, allowing the existing10uu
  distance tolerance. Every active centered source row meets that criterion.
- The full-match evaluator starts with its kickoff latch active and uses the
  same controller. Its latch clears at first contact or ball-y departure.

**Limited conclusion:** the source starts and initial admission do not support
the hypothesis that kickoff training is absent or Nexto's kickoff routine is
disabled. This does not prove whole-trajectory phase equivalence: training uses
centered xy whereas the evaluator has a contact latch. A vertical contact which
leaves xy exactly centered can differ after that contact. This check does not
quantify that case, change it, or claim it explains who made the first contact.
No reward, control, scenario, model, optimizer or live process was modified.
Zero optimizer steps; no simulated rollout or GPU kernel campaign was run.

## Reproduction

Run this code in `.venv/Scripts/python.exe` from the repository root. It reads
source state only; the imported simulator package may enumerate CUDA devices
but this computation uses NumPy arrays and does not construct an environment.

```python
import numpy as np
from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios
from rivalsim.fresh_ground_30hz import scenario_hash

bank = scenarios(32768)
xy = bank.state.ball_pos[:, :2]
centered = np.all(xy == 0, axis=1)
active = np.arange(32768) % 2 == 0
dist = np.linalg.norm(
    bank.state.car_pos[:, :, :2].astype(np.float64)
    - xy[:, None, :].astype(np.float64), axis=2)
own = dist[np.arange(32768), 1 - bank.focal_side]
eligible = np.abs(own - dist.min(axis=1)) <= 10
print(scenario_hash(bank))
print(int((bank.family == 4).sum()))
print(int(((bank.family == 4) & ~centered).sum()))
print(int(((bank.family == 0) & centered).sum()))
print(int((active & centered).sum()))
print(int((active & centered & ~eligible).sum()))
print(int((centered & (bank.family != 0) & (bank.family != 4)).sum()))
print(np.unique(bank.kickoff_layout[centered]).tolist())
```

Continue the healthy, frozen block to child25. Do not redesign rewards or infer
a learned fake/win from this source-state check. The actual child10 result still
has zero first contacts and ten losses; preserve it without a competence claim.
