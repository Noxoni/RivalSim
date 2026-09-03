# Rival 2 official capability bundle V1 authority

The deployment artifact is one inference-only checkpoint containing five
immutable trained model components:

- V23 Blue and Orange competitive base policies;
- the Ground-to-Air Goal V3 aerial specialist;
- Capability Curriculum V2 Blue and Orange specialist policies.

The models are not weight-averaged. A deterministic observation-only router
selects a component. The final playable configuration is fail-closed:
automatic aerial, recovery, and offensive-demolition takeovers are disabled.
All specialist components remain embedded and hash-bound for subsequent
targeted integration work.

This is required by the measured physical evidence. Candidate 1 enabled all
three routes and failed the ten-match Nexto matrix at 2-8 (105-123), with 577
recovery and 33 aerial activations. Candidate 2 disabled recovery, retained
aerial/demo routing, and failed at 3-7 (111-132), with 20 aerial and five demo
activations. Candidate 3 disabled automatic specialist takeovers and exactly
reproduced the accepted V23 result: 8-2, 159-111, and 687 touches.

The official play build must therefore default to Candidate 3 semantics. It
must not claim that the embedded specialist behaviors are naturally integrated
or automatically active. Enabling a specialist is experimental until a new
physical authority passes the same whole-match gate.

No optimizer step, reward change, or policy-parameter mutation occurs during
bundle construction or routing.
