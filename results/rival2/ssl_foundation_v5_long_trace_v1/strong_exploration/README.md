# Stronger exploration on the current 120 Hz policy

User requested a substantial exploration increase **before** any fresh-weight or
30 Hz transition. This is a continuation of the same bounded long-trace campaign,
not a new model lineage or a fresh optimizer.

The prospective settings are analog pre-tanh Gaussian sigma **0.30**, up from
0.04 (7.5x standard deviation, 56.25x variance), and button temperature **1.0**,
up from 0.25. Learned means and button logits are not overwritten. These settings
are constant from the first new rollout after the accepted-boundary pause.
Rollout sampling, old log probabilities, PPO recomputation and entropy/KL
telemetry all use the same override. No noisy old rollout is recycled across
the distribution change. The raw learned log-std remains bypassed as before.

This explicitly changes the behavior policy; it is not an execution-parity
optimization. Increasing temperature makes finite nonzero button logits less
committed, not uniformly random, and leaves the deterministic threshold unchanged.
Sigma describes the Gaussian **before tanh**, not a guaranteed controller-space
standard deviation. Deterministic evaluation is unchanged and is the check for
learned behavior, distinct from stochastic training telemetry.

Unchanged: 32768 worlds, 120 Hz/one tick per decision, 360-tick recurrent rollout,
three-second trace half-life, two PPO epochs, actor LR1e-4, critic LR3e-4,
independent critic, minibatches, resets, six potentials, +/-10 goals, 40/40/20
opponents, entropy objective coefficient0, KL telemetry-only, nonfinite rollback,
original wall-clock deadline and **total update100** limit. No new action bias,
scripted controls, blanket action penalty, correlated-noise scheme, or 30 Hz
transition is introduced. Existing resumed-episode/zero-hidden semantics remain.

The only new production code is
`benchmarks/run_rival2_ssl_foundation_strong_exploration.py`. Original runtime
files and older authority archives remain byte-identical. The original runner
correctly refuses the new authority. Use the new runner and latest accepted
checkpoint for operational recovery, never a pre-amendment checkpoint.

`anchor.json` binds the pre-change checkpoint and activation boundary.
`transition.json` proves every learning tensor, optimizer field, counter, RNG,
and other non-provenance checkpoint field was preserved. `original/` retains the
preceding authorities and boundary evaluation. The rebound checkpoint truthfully
retains the last *old* rollout's exploration metadata; the next accepted checkpoint
must name the stronger contract, checked by the new resume validator.

Focused tests check exact probability recomputation, distribution breadth,
button behavior, finite gradients, preserved settings, and invalid-resume
rejection. `no_step_validation.json` checks one actual 32768-world x360-tick
rollout and one full182-sequence recurrent backward with no Adam step. The
production runner requires this evidence bound to the new authority before
learning. This validates execution, not gameplay improvement. The accepted
update100 deterministic evaluation remains the next planned outcome comparison.

The initial focused test invocation exposed a pre-existing feedforward test
fixture using `gae_ready=True` on uninitialized `torch.empty` advantage/return
buffers. The fixture now computes GAE first; no production PPO code changed.
The first invocation also encountered a Windows default temporary-directory
permission error. Subsequent tests use a new task-owned temporary directory.
The original failing XML is retained for provenance.
