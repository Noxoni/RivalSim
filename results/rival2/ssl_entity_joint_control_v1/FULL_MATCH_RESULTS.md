# Fixed parent versus +100 full-match Nexto comparison

Completed under the unchanged prospective `full_match_protocol.json`: ten
deterministic matches per policy, five standard layouts on each team, 300 seconds
of regulation, no training/no-touch/curriculum reset. Every match finished in
regulation; no overtime or unresolved match occurred.

| Metric | Original hybrid u597 | Entity/joint-control +100 |
|---|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 |
| Goals for | 0 | 0 |
| Goals against | 408 | 324 |
| Rival contacts | 127 | 245 |
| Contacts / player-minute | 2.54 | 4.90 |
| Matches without a Rival contact | 0 | 0 |
| Rival first touches on kickoff | 0 | 0 |
| Measured execution time, seconds | 109.452 | 112.159 |

The candidate conceded 20.59% fewer goals and made 92.91% more contacts.
Every paired starting-layout/team match has fewer concessions and more
contacts, but neither policy scored or won any match. This supports real
basic-gameplay improvement beyond the short development scenarios, not
competitive strength, good possession, aerial competence, SSL rank, or an
isolated causal claim about attention. Both the architecture/action
parameterization and subsequent learning differ from the original parent.

| Layout | Rival team | Parent conceded | +100 conceded | Parent contacts | +100 contacts |
|---:|---:|---:|---:|---:|---:|
| 0 | Blue | 42 | 35 | 12 | 20 |
| 1 | Blue | 39 | 31 | 12 | 27 |
| 2 | Blue | 42 | 37 | 12 | 29 |
| 3 | Blue | 40 | 38 | 12 | 22 |
| 4 | Blue | 41 | 36 | 13 | 32 |
| 0 | Orange | 40 | 31 | 16 | 21 |
| 1 | Orange | 42 | 30 | 9 | 20 |
| 2 | Orange | 43 | 29 | 9 | 24 |
| 3 | Orange | 39 | 28 | 16 | 21 |
| 4 | Orange | 40 | 29 | 16 | 29 |

## Integrity and limitations

All twenty matches record exactly 36,000 physics ticks. Every recorded goal
has a valid goal-plane entry (408/408 and 324/324), no goal-buffer overflow
occurred, and recurrent reset counts equal native goal reset counts per match.
Both models and checkpoint files remained unchanged; zero optimizer steps
occurred. `full_match_parent597.json`, `full_match_entity100.json`, and
`full_match_comparison.json` retain raw per-world and per-goal telemetry and
bound checkpoint/protocol hashes. Full-match reward buffers are unused.

The interface check passed before these matches. Its earlier stream and goal
fixture failures are preserved separately and were not gameplay episodes.
No-touch truncations are zero by the evaluation protocol, not an acquisition
success claim. The parent and candidate are fixed in advance; no checkpoint
was selected from their match results. This is a small deterministic
development comparison, not an untouched broad benchmark or online/ranked test.

## Next development action

Retain +100 and its Adam/RNG state as the continuation parent, not a fresh
restart or deployed/promoted SSL model. Acquisition and finishing improved
enough to continue this lineage; Nexto scoring and kickoff competitiveness
remain unresolved. Prospectively specify the next continuation before updates,
preserve rewards/physics/control contracts, and implement the existing staged
Nexto schedule with correct inference-only opponent masks and family-local
advantage normalization. The +100 acquisition result is one qualifying
boundary; +50 was below the existing 60% two-boundary criterion. Do not silently
extend the completed 100-update pilot authority.
