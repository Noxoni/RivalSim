# Resumed and learning; capability evaluation pending

Authority/implementation were pushed at
`447a67a2562aae9797fd45e67a1d293d5720ce5f` before optimization. The initial
device-check operational failure accepted zero updates and rolled back.
Correction `b178629adb80a722d7002ae5501b951203380939` and its hashes were
remotely verified before resuming the preserved zero-update checkpoint.

The resumed process is launched hidden, with `resume1.stdout.log` and
`resume1.stderr.log` under the external run directory. WorkerPID38704 was
verified active. PID is a historical launch detail; always inspect live state.

Two accepted updates were inspected without running competing GPU work:

- All expected parameter groups updated, including entity attention and both
  entity residual outputs; all model/Adam tensors finite.
- Every Adam counter is364, matching2 updates times182 minibatches.
- Static entity-map buffers and the action table are unchanged.
- Immutable parent is preserved. No reward, dataset or physics changes.

The audited +2 checkpoint and its SHA are in `accepted_002_integrity.json`.
This is implementation/training integrity evidence, not a gameplay verdict.
The initial categorical policy baseline acquired11/64 cases and scored11
finishing goals; original hybrid parent acquired16/64 and scored14. Head
projection is not policy parity. Initial Nexto kickoff scoring remained0/64.
Judge later learning against these explicit baselines; do not claim attention
already improved play. Scheduled deterministic evaluations remain0/10/20/50/100.

The existing ten-minute development heartbeat was updated to follow this run,
respect the frozen experiment, preserve evidence, and notify only on newly
completed evaluations/material findings/failures. The broader SSL goal is
unfinished and this checkpoint is not automatically deployed to RLBot.
