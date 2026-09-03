# Unified Capability Distillation V1 Diagnosis

V1 stopped at its prospectively frozen ten-boundary plateau condition with no
validation-eligible checkpoint.  No V1 checkpoint was emitted or promoted.

The recurrent student reduced held-out aerial-teacher expected-action RMSE from
`0.2548483610` to `0.038142` at step 900 while its natural V23 error remained
approximately `0.0018`.  The blocker was the requirement that every Capability
V2 family improve over the original V23-to-Capability teacher discrepancy.
Those discrepancies were only `0.00040525` for demo, `0.00035382` for floor
landing, and `0.00016205` for wall landing.  Learning the much larger aerial
residual perturbed those families by roughly `0.002`, so no boundary could
satisfy the frozen rule.

Inspection of the immutable checkpoints explains the scale mismatch.  The
Capability V2 trunk and critic are byte-identical to V23; only its actor changed,
and the actor-weight relative L2 displacement is approximately `0.000114`.
Ground-to-Air Goal V3 changed its actor by approximately `0.0228` and also made
small trunk changes.  Treating both teachers as equally independent recurrent
residual targets therefore turned preservation of a minute already-aligned
actor delta into the dominant eligibility bottleneck.

The prospective correction is a new authority, not a reinterpretation of V1:
initialize the one policy's frozen base actor from the elementwise mean of the
side-specific Capability V2 actor heads (their trunks and critics are already
identical), preserve V23 natural actions by rehearsal, and train the recurrent
residual against the aerial teacher.  Demo/floor/wall teacher rows remain in
rehearsal and validation, but are judged relative to that capability-baked
initialization.  Physical direct-control evaluation remains decisive.
