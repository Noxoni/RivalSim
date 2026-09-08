# Update60: acquisition continues to decline

Scheduled deterministic evaluation against active Nexto completed on the frozen
1,024 original-episode starts. No extra GPU evaluation or training interruption.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |
| 50 | 39/512 (7.6171875%) | 27/512 (5.2734375%) |
| 60 | 25/512 (4.8828125%) | 20/512 (3.90625%) |

Both measures decline again. Easy acquisition is now numerically below the
untrained baseline; varied remains above it but far below its update40 peak.
This is ongoing behavioral regression, not progress or a promotion candidate.
The fixed probe does not establish a population significance claim.

Conditional lower-median successful-contact times are0.825s easy and
2.6083333333333334s varied. No-own-contact fractions over the full8s are
0.9453125 and0.9609375. Easy success uses5s and is not the complement of its
full8s no-contact fraction. The probe measures own contact, not first-touch
victory against Nexto, possession, scoring or aerial mechanics.

## Training51-60

All six families remain exposed. Learner first-contact awards16,537 and
discounted reward16532.25196123123; off-ground awards4,488 and reward
1121.672954082489. Nexto's31,301 first and65,577 off-ground awards excluded.
Repeated contacts without another first payout5,295; capped agent-decisions0.
Off-ground includes low bounces and is not proof of an aerial mechanic.

Compared with41-50, mean learner speed declines from611.4676901584202 to
473.1836117440683uu/s. Nexto-world Rival touches decline23,766 to14,565 over
equal approximately491,520 world-seconds. Rival goals decline44 to11; opponent
goals decline57,436 to34,087; inactivity resets increase0 to186. These are
stochastic training scenario outcomes, not full-match scores. Fewer concessions
alone must not be reported as improved gameplay while acquisition, own scoring
and movement deteriorate. No causal mechanism has been established by this audit.

## Integrity

All17 CPU report checks passed: finite model/Adam, fresh lineage/contracts,
contiguous samples/updates/physics/Adam counters, no KL rejection, every-family
exposure, exact bonus accounting, checkpoint identity and read-only evaluation.
Independently recomputed raw first-contact counts, fractions, lower medians and
no-contact fractions. Started receipt and actual checkpoint hash match evaluation;
scenario/spec hashes match baseline. All frozen source hashes match under the
specified LF text convention; authority artifacts remain unchanged.

Update60 contains265,420,800 learner decisions and8,702 Adam steps. Completed
update mean KL0.004549074230817496, max sample KL0.15760314464569092 and
entropy4.194553852081299; KL is telemetry only. These finite/integrity checks do
not imply behavioral acceptance or explain the regression.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000060.pt`

SHA-256: `A4E18DFFCD0D042FD5C749F856BA750D9A940E1537BC60C50CFDD4564DECE4FE`

Raw probe/start receipt, audit and closed training prefix are preserved. The
healthy process remains under the frozen bounded150-update authority. No
automatic reward retuning, new guard, extra campaign, restart or promotion.
