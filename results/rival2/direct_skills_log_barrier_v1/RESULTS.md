# Thirty-update exploration experiment: no competitive improvement

Exactly 30 accepted PPO updates completed; actual worker exited normally.
132,710,400 trainable decisions; 4,294 optimizer steps. Final cumulative update
680 / optimizer step 148,930. All three fixed evaluations are preserved.

| Metric | Parent 650 | +5 | +15 | +30 |
| --- | ---: | ---: | ---: | ---: |
| Wins against Nexto | 0/10 | 0/10 | 0/10 | 0/10 |
| Goals scored | 7 | 1 | 1 | 2 |
| Goals conceded | 196 | 264 | 291 | 228 |
| Touches/minute | 9.12 | 5.74 | 6.64 | 8.88 |
| Kickoff first contacts | 0 | 0 | 0 | 0 |

Update 30 recovered relative to update 15: goal difference improved in all ten
paired worlds. But versus the actual starting model, one world improved, two
were unchanged and seven worsened. Overall scoring/conceding remained worse.
No competitive promotion or SSL capability is established.

Training entropy increased from 0.84835 to 3.38218; the uniform log barrier fell
from 8.52074 to 0.98925. The exploration intervention changed the distribution
substantially, but that fact is not a gameplay success. Final next-contact
identity remained with Rival in 88/435 resolved followups (20.23%); this is not
a measurement of continuous possession.

The CPU closeout verified all three checkpoint/objective audits, all three
full-match integrity records, frozen source/evidence hashes, complete training
curve, exact final rolling/permanent hash agreement, and no live worker. Model
and Adam are finite; no KL rejection occurred. Checkpoints retain optimizer and
RNG lineage. No safety boundary or reward changed during the run.

Final checkpoint:
`checkpoints/rival2/direct_skills_log_barrier_v1/child_000030.pt`

SHA-256: `16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD`.

Decision: close this arm without automatic extension or deployment. Preserve
the parent and all evidence. Next is one bounded no-optimizer check of the
post-run exploration-to-PPO gradient ratio, not a blind coefficient change.
The broader SSL-development goal remains active and unachieved.
