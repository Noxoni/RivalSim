# Fixed +200 natural-match follow-up

Freeze this plan and `full_match_000200_protocol.json` while the current
learner is still below200, before inspecting that checkpoint's evaluation.
The target is exactly `plus_000200.pt`, not the most favorable checkpoint.
This is a bounded read-only follow-up within the ongoing SSL development goal,
not another training campaign, acceptance framework, reward change or test search.

Use the exact existing `CandidateMatchRunner` and full-match lifecycle:
ten deterministic matches against pinned Nexto, five standard layouts on each
team, 300-second regulation, up to120 seconds overtime if tied. Rival acts at
30Hz and Nexto's neural policy at15Hz, on120Hz physics. Hidden state resets
only at native goal/kickoff boundaries. No training/no-touch/curriculum resets
are used in these matches. Rival has no scripted controller prefix.

Reuse the completed, hashed +100 raw match result. Do **not** rerun +100 or
choose another baseline. Bind the future +200 candidate to its scheduled
evaluation's checkpoint hash and verify continuation lineage before loading.

## Controlled pause and resume

1. Let normal training and the scheduled +200 development evaluation finish.
2. Record a dedicated evaluation-pause reason outside Git and create the run's
   `STOP` marker. Allow the current update to finish, preserve `latest.json`,
   and verify the actual worker exits. The pause checkpoint can be slightly
   later than200; that does not change the fixed evaluation target.
3. Verify the latest accepted checkpoint hash. Do not delete or alter it.
4. Run the new read-only helper only after `stopped_at_accepted_boundary` and
   acquisition of the same exclusive training/evaluation GPU lease:

   ```powershell
   .venv\Scripts\python.exe benchmarks/evaluate_rival2_ssl_entity_continuation_match.py run
   ```

5. Preserve result, checkpoint/hash immutability, match/goal/reset integrity,
   and raw next-contact/direction counters. Score, concede, contact and
   next-contact outcomes are more useful than claiming a mechanic from a
   coarse counter. Do not claim SSL or native Rocket League rank from this
   small simulator comparison.
6. Remove **only the agent-owned evaluation-pause marker**, provided the user
   has not asked to stop and no corruption/capability fault occurred. Resume
   the exact last accepted rolling checkpoint with `--resume` and its SHA.
   Keep its Adam moments/counters, RNG and staged Nexto state. Fresh physical
   episodes/hidden on process resume remain the already declared semantics.

This is a brief evaluation pause, not a training time/update cap. A user stop
overrides the planned resume immediately. Never run another GPU learner or
evaluation alongside the healthy current worker. Do not remove someone else's
STOP file, reset optimizer moments, or reuse an older checkpoint to recover.

The helper rejects wrong lineage/offset, an unpaused worker, a changed protocol,
unpublished sources, altered baseline and already completed output. Existing
native interface validation is reused because the controller/runtime code is
unchanged. The two focused tests cover exact method reuse and checkpoint
identity rejection; they are not evidence of a completed live match.

Report separately whether +200 improves actual Nexto scoring, concessions,
contacts, kickoff first touches and repeated-contact outcomes. If it does
not, retain that negative evidence and use it to guide the next bounded
investigation. Do not compensate by silently adding rewards or changing the
live frozen training contract.
