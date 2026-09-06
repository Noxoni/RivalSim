# Latest-checkpoint recovery guard

The active temperature-2 learner was not interrupted or changed for this work.
Its frozen runner/source hashes remain exactly those published before launch.

An operational review identified a sharp edge: the launch runner defaults to
the original +554 source if the caller omits its resume arguments. The monitor
already required explicit latest-checkpoint arguments, but a mistaken manual
command could restart the segment from stale weights/Adam and append duplicate
update offsets. No such restart was observed.

Use the separate guard for any **future operational recovery**:

```powershell
.venv\Scripts\python.exe -u benchmarks/resume_direct_skills_exploration_v1.py --resume <verified-latest.pt> --resume-sha256 <exact-latest-SHA256>
```

The guard requires both arguments; acquires the same exclusive GPU lease;
rejects every STOP, a completed review boundary, the wrong amendment, stale
checkpoint SHA, file corruption or an inconsistent state/latest pair; and then
calls the original frozen runner under the held lease. It permits a durable
latest checkpoint one boundary ahead of status, accounting for process exit
between atomic checkpoint/latest and status publication. It does not select an
older model, delete markers, modify rewards, change optimizer semantics or
alter temperature. The frozen runner still performs its complete payload,
source, contract, Adam and parent validation.

`--check-only` performs no PPO call. During a healthy active learner it should
fail to acquire the exclusive lease, not disturb training. This was tested
against actual learner PID35488: the check-only command raised Windows
`PermissionError: [Errno 13] Permission denied` at the nonblocking lease,
before resume validation or PPO. PID35488 remained live and advanced to +570.
That expected refusal is not a learner failure.

Unit tests cover stale identity, corruption, STOP preservation, intentional
review stop, wrong amendment, atomic publication skew, mandatory CLI arguments,
and validation-before-dispatch/check-only behavior. Adjacent full-match
comparability tests ensure the upcoming +600 review still rejects mismatched
reset methods/cases. No new evaluation or learning result is claimed here.

At +600, a deliberate `stopped_for_exploration_review` is not an operational
crash. Complete and publish the scheduled evaluation review and make the next
prospective decision under the ongoing user goal. Do not use this guard to
bypass that review.
