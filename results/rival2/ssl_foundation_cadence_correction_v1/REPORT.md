# SSL Foundation 120 Hz cadence correction

## Verdict

The stopped SSL Foundation PPO V1 update-1 through update-4276 lineage is
invalid as evidence for the intended terminal-goal-plus-potential reward
contract. It is retained for inspection only and must not be resumed or used
as the parent of a corrected campaign.

## Root cause

`rival2_begin_decision` clears the policy-interval state at the beginning of
every decision. The shared reward kernel previously inferred a four-tick
decision interval from the `GOAL_ONLY` reward mode, even when the environment
was configured for one physics tick per 120 Hz decision. The interval therefore
advanced only from zero to one before being cleared again. Terminal, truncation,
reward finalization, and reset-mask logic were unreachable for this lane.

Because the SSL potential shaping is composed outside the native goal-only
kernel, the policy still received geometric potential differences. It did not
receive the intended terminal goal objective, terminal absorbing potential, or
episode reset stream.

## Correction

- The native kernel now receives the environment's actual
  `physics_ticks_per_decision`; reward type no longer determines cadence.
- SSL scenario resets use a deterministic coprime full-cycle schedule through
  the complete 32,768-state source bank. A world no longer repeats one immutable
  source template at every reset.
- Standard kickoff is an explicit 10% reset family, taking 10 percentage points
  from the former 25% natural-ongoing family. All five authoritative Soccar
  kickoff layouts are represented.
- Runtime preflight authority now binds the one-tick cadence, full-cycle reset
  bank, and kickoff coverage. The superseded V1 authority fails closed against
  these requirements.

The reward function, six potential definitions and weights, terminal score
values, observation/action contracts, policy architecture, and PPO algorithm
were not changed by this correction.

## Validation

CUDA integration validation forces, in the same 120 Hz step:

- a Blue goal and verifies the terminal-only component is exactly `[+10, -10]`;
- an Orange goal and verifies the terminal-only component is exactly `[-10, +10]`;
- a no-touch timeout and verifies truncation/reset;
- physical scenario reset and lifecycle `reset_required` consumption;
- a second goal and a second, distinct deterministic curriculum-bank source.

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rival2_ssl_foundation_v1.py tests/test_rival2_120hz.py tests/test_rival2_unified_ground_selfplay_ppo_v1.py tests/test_rival2_unified_ground_curriculum_ppo_v2.py -q
```

Result: `23 passed`.

Adjacent trainer/optimizer tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v05_rival2.py tests/test_rival2_recurrent_ppo.py tests/test_rival2_opponent_curriculum.py tests/test_rival2_mixed_ppo.py tests/test_rival2_human_bc_ppo_v1.py tests/test_rival2_unified_ground_selfplay_ppo_v1.py tests/test_rival2_unified_ground_curriculum_ppo_v2.py -q
```

Result: `23 passed, 6 skipped`; the skips are existing environment-gated tests.

The exact-scale deterministic scenario build contains 32,768 states:

- natural ongoing: 4,915
- standard kickoff: 3,277
- loose-ball access: 4,915
- catch/control/possession: 4,915
- shooting/finishing: 3,277
- defensive/shadow/save: 3,277
- contested 50: 3,277
- wall/aerial: 3,277
- recovery/scramble/low boost: 1,638

Kickoff layout counts are `655, 655, 655, 656, 656`.

No PPO optimizer step was run as part of this correction.
