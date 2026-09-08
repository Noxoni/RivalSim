# Temporary ball-acquisition starts for sustained gameplay

User amendment: add an initial learning-to-touch scenario, retained until no-touch
acquisition is no longer an issue, then drop that extra scenario. This is a
curriculum amendment, **not another random restart or a new reward contract**.

## Preserved policy and training

Continue the stopped sustained-gameplay model at total accepted update **27**:
`checkpoints/rival2/sustained_gameplay_v1/plus_000027.pt`, SHA-256
`C2532D3D347A13C23919D52CA5A1E30409BBA0E5102577CB5DE2FDC9888D8DAF`.
Parent has 119,439,360 learner decisions and 3,990 Adam steps. All model weights,
Adam moments/counters and learner RNG states are restored, not reset. The original
random initialization remains the lineage root. No old BC/V5 weights are loaded.

Preserve the previous reward, actor and independent memory critic, observation
and action contracts, 120 Hz physics / 30 Hz action decisions, 90-decision rollout,
two PPO passes, LR, exploration, discount/trace and 50% self-play / 50% Nexto world
mix. No direct skill/mechanic counters or rewards are reinstated. Nexto is active,
inference-only; both current self-play players contribute training samples.

The predecessor is deliberately left stopped. Process entry starts fresh physical
episodes and clears both recurrent histories, exactly as the parent resume protocol
already specifies; it is not an exact physics continuation. Unfinished old episodes
are not converted into failures or spliced into new GAE. Thereafter both memories
and physical episodes persist across rollout and evaluation boundaries.

## Reset distribution, not a controller or shortcut

Half of the 32,768 source-bank rows become **ball_acquisition**. Half remain the
existing natural/challenge/finishing/defense/kickoff starts, selected proportionally
within each original family. Retained rows remain byte-identical. Source-bank
percentage is not necessarily percentage of training frames: episodes have different
lengths. Actual per-family/per-opponent world-seconds and contacts are logged.

New starts are grounded, non-overlapping, centrally placed stationary/slow rolling
balls. Half are 350–850 uu away, half 850–1800 uu away. Heading and momentum are
coherent but not perfect; offsets span easy and wider approaches. Half the focal
cars stand still initially; others move slowly. Both player sides are represented.
The other car is active, initially 3000–3400 uu upfield, leaving a learnable contact
opportunity. There is no scripted controller prefix, frozen defender or injected
touch. Ball rolling angular motion is initialized consistently with linear motion.

**First contact never ends a training episode and earns no new touch bonus.**
The unchanged access/other potentials, small ongoing control payment and ultimate
goal outcome remain the learning signal. Continue until a goal, or 45 seconds
without any physical contact by either car. The latter retains the same -1 penalty
and correct pre-reset recurrent value bootstrap. There is no new fixed-age limit.

## Prospective retirement rule

At parent entry and every ten additional updates, a separate read-only process
runs the current actor deterministically in 1,024 held-out-generation development
starts against active native-v5 Nexto (512 easy, 512 varied; both sides). The
generation seed is different from the training bank; definitions/hashes are frozen
in the package. This is a repeatedly used development check, not an untouched test.

Required for **two consecutive probes**:

- At least 95% of easy starts produce Rival's own native contact within 5 seconds.
- At least 90% of varied starts produce Rival's own native contact within 8 seconds.

An opponent contact does not count. Goals/reset boundaries before Rival's contact
are failures for that original attempt; replacement episodes cannot rescue it.
First-contact physics-tick records retain failures as -1. Conditional contact time
is reported with success rate, not used to hide misses. The eight-second diagnostic
window does not generate PPO data or shorten any training episode.

Before retirement, the **same exact checkpoint** must also complete the existing
ten full deterministic Nexto matches with at least **5 Rival contacts/minute** and
**zero entirely touchless matches**. This is a modest gameplay-transfer floor, not
proof of good possession, winning kickoffs, or SSL gameplay. If the probes pass but
ordinary matches remain nearly touchless, the temporary scenario remains.

Retirement replaces only future reset sources with the exact original five-family
bank. Existing physical games, focal sides, Nexto states, optimizer and both memory
banks are untouched. Existing acquisition-started games play out. Retirement and
streak/checkpoint identities persist in each checkpoint; there is no automatic
reinsertion or reward retuning. Later probes continue every 50 total updates so
regression can be reported without silently changing the curriculum.

Full ten-match Nexto evaluations still run every 50 total updates, and additionally
when a passing two-probe streak makes retirement possible. Probes happen at 27,
37,47,57,... while active. Every accepted update has an atomic rolling checkpoint;
entry, first new update, all diagnostic boundaries and every 50 have permanent files.

## Evidence and commands

`tests.xml`: native contact-does-not-reset, physical start validity, deterministic
bank, preservation of original starts, retirement rule/deduplication, future-only
reset replacement, six-family telemetry, a disposable finite PPO update, and the
existing sustained reward/GAE/goal-timing/recurrent-critic tests.
`preflight_32768.json`: actual full-scale 90-decision rollout and backward/memory
check on the +27 parent, with **zero optimizer steps** and exact parent/Adam parity.

The first preflight caught a diagnostic-only missing `model.train()` before cuDNN
GRU backward after rollout's eval mode. It stopped before any optimizer step; the
check was corrected to enter train mode, matching the production PPO update.
Neither the saved parent nor production policy/reward implementation was changed.

```powershell
.\.venv\Scripts\python.exe benchmarks/run_sustained_acquisition_v1.py verify
.\.venv\Scripts\python.exe benchmarks/run_sustained_acquisition_v1.py run
```

The runner requires prospective source/evidence/parent bytes on `origin/main`.
External state is `G:\dev\RivalSim-runs\sustained-acquisition-v1`. STOP is honored
at the next accepted boundary; do not clear old runs' STOP files. Resume only with
the exact latest accepted path and `--resume-sha256`. KL remains telemetry only;
finite/corruption failures stop without automatic capability-changing recovery.

This amendment provides more acquisition opportunities. It does not establish
that the bot has learned them yet. Compare the native-contact probes **and** actual
full-match gameplay results before claiming improvement.
