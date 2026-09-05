# Update 100: stronger exploration completed, gameplay result mixed

The bounded campaign stopped at total local update 100. Only updates 85-100
used the stronger exploration override: sigma 0.30 / button temperature 1.0.
Those 16 updates produced 275,548,987 trainable samples in 1,032.37 seconds of
update execution, with 11,552 Adam steps. Total lineage training samples are
1,604,423,795. No further updates or new campaign were launched.

## Deterministic evaluation

The established protocol is unchanged: 1,024 scenario worlds, 512 against each
opponent, 3,600 physics ticks (30 seconds) per world. These are aggregate scenario
results, not full-match scores or win rates. Each side/opponent uses the same
fixed assignment as previous evaluations. The update-100 evaluation ran once.

| Opponent | Update | Goals for / against | Difference | Touches/min | Speed uu/s | No-touch resets | Goalward-touch fraction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Nexto | 10 | 185 / 923 | -738 | 13.609375 | 1192.313647 | 0 | 0.877727 |
| Nexto | 50 | 178 / 930 | -752 | 14.140625 | 1176.966676 | 0 | 0.864641 |
| Nexto | 84, before stronger exploration | 172 / 992 | -820 | 13.828125 | 1169.243372 | 0 | 0.859887 |
| Nexto | 100 | 149 / 991 | -842 | 15.152344 | 1160.587036 | 2 | 0.879866 |
| Frozen V5 | 10 | 362 / 375 | -13 | 16.007813 | 1184.560331 | 1 | 0.822596 |
| Frozen V5 | 50 | 359 / 355 | +4 | 15.945313 | 1163.937856 | 3 | 0.824596 |
| Frozen V5 | 84, before stronger exploration | 388 / 333 | +55 | 15.722656 | 1169.342244 | 1 | 0.859627 |
| Frozen V5 | 100 | 357 / 362 | -5 | 16.078125 | 1162.583867 | 0 | 0.838678 |

Compared with 84, Nexto touches/min increased 9.58%, with a 2.00 percentage-point
increase in goalward touches. But Rival scored 23 fewer goals and conceded only
one fewer: goal differential worsened by 22. Against frozen V5, touches/min
increased 2.26%, but goals declined by 31 and concedes increased by 29, worsening
goal differential by 60. Mean movement speed declined slightly against both.

Versus the update-10 baseline, Nexto goal differential is worse by 104 despite
1.543 more touches/min; frozen V5 differential is better by 8 with almost unchanged
touch frequency (+0.070/min). Complete precise deltas for all metrics versus 10,
50 and 84 are in `completion_audit.json`.

This is evidence of more deterministic ball contact, **not** improved scoring
or a solution to the user's gameplay concerns. Goalward touch means positive
canonical ball-Y velocity following contact, not an independently validated
shot at the goal mouth. No possession duration, backwards driving, aerials,
dashes or demo skill is inferred from these fields. Sixteen stronger-exploration
updates and one fixed evaluation corpus do not establish broad generalization
or isolate exploration from resumed episode initialization.

Recommendation: keep this run stopped. Preserve both 84 and 100 for comparison;
do not automatically extend it or launch the proposed fresh-weight/30 Hz restart.

## Learning and integrity

All 16 stronger updates were accepted with 722 Adam steps each. Completed-update
mean KL ranged 0.005336-0.006967; maximum post-step minibatch mean KL was 0.008735.
KL remained telemetry only. No nonfinite/corruption/reward guard fired. Final
checkpoint tensors and optimizer are finite; all Adam counters advanced exactly.
The 120 Hz contracts, original V5 source, independent critic, 360-tick trace,
PPO parameters, opponent mixture and six-potential reward authority are unchanged.
Peak allocation over the stronger updates was 21.073 GiB.

The 39 pre-launch tests and earlier publication regression pass. Two additional
CPU publication-recovery tests pass. Finalization only read checkpoint tensors
and existing evaluation data; no optimizer step or additional evaluation ran.

## Recovered final-file publication failure

After training **and** evaluation completed, the worker failed while overwriting
the old `final.pt`: Windows error 1224, ERROR_USER_MAPPED_FILE. The mapping owner
was not identified. The old file remains the intact update-84 artifact, SHA
`B30BCFE1D4860C6F13F8A8E52A1991D481037909151EA78378A911979A05E2A0`;
it must not be mistaken for update 100.

The accepted rolling100 and periodic snapshot100 were saved before evaluation.
Their complete decoded checkpoint payloads match exactly; different ZIP archive
names account for different file hashes. The completed evaluation was already
persisted in the manifest. Recovery authority/code/tests were committed and pushed
at `32db4a1` before final-only publication. No training/runtime code changed.

The new final pointer names exact rolling100 bytes, with all Adam/RNG state
preserved. Resume semantics remain fresh episodes and zero recurrent hidden as
specified by the existing checkpoint contract. The checkpoint contains the RNG
state from the accepted boundary, before evaluation. It is not a reconstructed
post-evaluation checkpoint.

- Final: `checkpoints/rival2/ssl_foundation_v5_long_trace_v1/strong_exploration_verified_u0100.pt`
- SHA-256: `76AD396FDB541865CACEEE07A9AD564B13649D853C5BE6188F82C4ECE40EFFCF`
- Periodic snapshot copy: `checkpoints/rival2/ssl_foundation_v5_long_trace_v1/evaluation_u0100.pt`
- Snapshot SHA-256: `5E861D27C12F1137955AB41F237A112F6B8703948F06081FBC75D0E1A5D69A46`

`publication_failure.stderr.log` preserves the actual failure; stdout and the old
summary are retained alongside it. `production_verification.json` was written
before the final-file error and its empty-stderr check refers to that instant.
The later `completion_audit.json` and summary explicitly record this recovered
operational failure. An integrity/completion PASS is not a gameplay-improvement PASS.
