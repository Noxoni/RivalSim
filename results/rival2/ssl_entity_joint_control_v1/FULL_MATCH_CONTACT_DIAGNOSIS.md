# What the +100 full-match improvement does and does not show

Posthoc CPU-only reduction of the already committed ten-match-per-policy
comparison. No new match, checkpoint selection, optimizer step, reward change,
or detector. The active continuation remains unchanged.

| Existing native counter | Original hybrid u597 | Entity/joint-control +100 |
|---|---:|---:|
| Rival contacts | 127 | 245 |
| Contact changed longitudinal ball velocity toward opponent end by >=100 uu/s | 127 | 233 |
| Contact changed it away by >=100 uu/s | 0 | 11 |
| Completed post-contact intervals with net forward ball displacement >=100 uu | 1 | 35 |
| Completed intervals with net backward displacement >=100 uu | 41 | 41 |
| Resolved next contact was Rival again | 0 / 126 | 8 / 245 |
| Resolved next contact was Nexto | 126 / 126 | 237 / 245 |
| Conceded goals with at most one registered contact since kickoff reset | 200 / 408 | 69 / 324 |
| Native Rival demolition events | 0 | 27 |

The majority of Rival's new contacts still do not retain the next touch:
96.73% of resolved next-contact identities belong to Nexto. There is a small
appearance of repeated contacts and more useful post-contact field movement,
but **reliable possession is not established**. This is the central missing
piece beyond simple acquisition; no extra direct possession reward is added.

The candidate concedes markedly fewer one-contact kickoff goals, despite
never winning the first kickoff touch. This supports improved interruption
of immediate scoring sequences rather than a claim of good kickoff execution.
Goals with later contact exchanges increase from 208 to255: that is partly
expected when more kickoff sequences survive beyond a single contact. It
does not alone prove open-play defending became worse, since exposure to
open play also changed. Both policies still scored zero and lost every match.

## Read the counter semantics literally

- `direction_count` bins the **change** in canonical longitudinal ball velocity
  at contact, not resulting ball velocity. A positive change can merely slow
  an incoming ball that still travels toward Rival's own goal. The original
  parent's 127/127 positive changes alongside zero scoring illustrates why
  this must not be described as 100% good shots or goalward ball movement.
- `displacement_count` uses net longitudinal displacement until the next
  contact or goal, with a 100-uu threshold. The helper function shares the
  direction category but this quantity is distance, not uu/s. An unfinished
  end-of-match interval can be censored; totals need not equal contact count.
- `possession_same/opponent/total` describes **next contact identity**. It
  does not measure control duration, catch quality, shielding, or intentional
  possession. Goal/end censoring explains differences in denominators.
- `kickoff_goal_count` means at most one registered distinct contact since
  reset. It is not an arbitrary elapsed-time cutoff or every early goal.
- The 27 native demolition events do not establish intentional tactical demos.
  Wall/backboard interval counts concern the **ball** contacting a surface;
  they do not prove wall mechanics or aerial ability by the car.

The next scheduled +150 development evaluation should be read alongside this
natural-gameplay deficit. Continue the current model/Adam and frozen staged
Nexto schedule; do not reset weights, add mechanics payments or claim SSL
progress from training loss alone. Natural match follow-ups must eventually
show actual scoring and possession gains, not only more contacts.

## Reproduce

```powershell
.venv\Scripts\python.exe benchmarks/report_rival2_ssl_entity_match_contacts.py
```

`full_match_contact_breakdown.json` binds both raw source files and the
existing counter implementation by hash. The script asserts source model and
checkpoint immutability, zero optimizer steps, exact match duration, direction
count totals, next-contact conservation and kickoff-event/count parity.
Rebuilding does not read live training state or use the GPU.
