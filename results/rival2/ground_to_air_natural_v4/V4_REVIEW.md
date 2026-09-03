# Natural ground-to-air V4 review

## Verdict

**NOT PROMOTED.** The bounded 96-block natural-entry/live-defender curriculum
finished normally, but no validation-selected checkpoint passed every frozen
row.  The untouched test seed remained unopened and the competitive V23
policies remained byte-identical.

This was not a broad capability failure.  The option substantially improved on
incoming chips and modestly improved live-defender matched-dribble offense.  It
did not reliably raise low-bounce and matched-dribble follow contacts above 300
uu on every physical side, and the easiest incoming-chip family began producing
slightly too many chains beyond the six-contact maximum.

## Selection result

- Parent: controlled ground-to-air scorer V3,
  `F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154`
- Best validation block: 88
- Baseline score: `2.455078125`
- Best score: `4.347005208333333`
- Stop reason: `maximum_blocks`
- Untouched test opened: `false`
- Promoted checkpoint: none
- Final rolling diagnostic (block 96, external only):
  `G:/dev/RivalSim-runs/ground-to-air-natural-v4/rolling.pt`, SHA-256
  `9213D60CEB65B1A6B9414531C18C81FABB34732D5EE8802EED571FF45B496822`

The block-96 rolling file is diagnostic only.  It is not an accepted parent.

## Baseline to best validation movement

The following values average the two physical sides within each frozen setup
and defender mode:

| setup / defender | elevated follow | high follow | productive continuation | goal <= 6 contacts | over-contact |
|---|---:|---:|---:|---:|---:|
| incoming chip / live, baseline | 35.9% | 15.6% | 10.0% | 4.5% | 0.0% |
| incoming chip / live, best | 59.8% | 34.6% | 18.6% | 8.8% | 0.2% |
| incoming chip / parked, baseline | 21.3% | 20.5% | 5.1% | 4.5% | 0.0% |
| incoming chip / parked, best | 60.4% | 48.8% | 23.6% | 8.4% | 1.2% |
| low bounce / live, baseline | 9.6% | 0.4% | 1.4% | 0.4% | 0.0% |
| low bounce / live, best | 12.9% | 0.4% | 2.0% | 1.2% | 0.0% |
| low bounce / parked, baseline | 19.5% | 4.5% | 6.4% | 2.5% | 0.2% |
| low bounce / parked, best | 21.1% | 5.1% | 7.0% | 3.1% | 0.2% |
| matched dribble / live, baseline | 16.6% | 0.2% | 2.7% | 1.4% | 0.0% |
| matched dribble / live, best | 22.1% | 0.4% | 5.7% | 2.1% | 0.0% |
| matched dribble / parked, baseline | 69.7% | 13.7% | 28.1% | 3.3% | 0.0% |
| matched dribble / parked, best | 65.6% | 11.5% | 33.2% | 6.2% | 0.2% |

Exact per-side values are in `result.json` and the complete curve is in
`training_curve.jsonl`.

## Failed frozen checks at the best block

- Low bounce / live defender / side 1: zero high follow contacts in 256
  attempts and productive continuation `0.78% < 1.5%`.
- Matched dribble / live defender / side 0: one high follow contact in 256
  attempts (`0.39% < 0.5%`).
- Matched dribble / live defender / side 1: zero high follow contacts in 256
  attempts.
- Incoming chip / parked / both sides: contact-budget overrun exceeded the
  frozen 1% maximum (three or more seven-contact failures per 256 attempts).

## Diagnosis and prospective correction

Each V4 training rollout mixed every setup/defender stratum and normalized its
advantages globally.  Incoming-chip trajectories generated much more reward
and improved rapidly.  Their signal consequently dominated the rarer,
lower-return low-bounce and matched-dribble-under-pressure states.  This also
increased repeated incoming-chip contacts until the six-contact cap became a
failure.

A successor must restart from the accepted V3 scorer, not the blocked V4
rolling file.  Before any new optimizer step it should prospectively freeze:

1. equalized setup and defender strata at the PPO-update boundary so each
   family's advantages are normalized locally;
2. a stronger physical over-contact failure for a seventh touch;
3. explicit validation preservation of the already-passing uncontested scorer;
4. the same 20-boost floor, six-contact maximum, no raw-airtime reward, and
   no named-mechanic classifier.

The result supports continuing capability training.  It does not support
full-match promotion yet.
