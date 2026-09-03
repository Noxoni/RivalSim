# Rival 2 Unified Capability Distillation V1 Authority

This experiment creates one policy whose weights contain ordinary V23 play,
the Ground-to-Air Goal V3 behavior, and the Capability Curriculum V2 demo and
surface-recovery behaviors.  There is no runtime policy router, expert index,
task identifier, or action splice.

The V23 Blue policy is copied into a parity-preserving base path.  A single
recurrent residual actor is zero-initialized and is the only trainable module
during consolidation.  Consequently step zero is exactly V23 and the frozen
base trunk, actor, and critic cannot be forgotten.  The recurrent residual is
part of the one deployed policy and carries temporal continuation intent.

Teacher identity is training metadata only.  Teacher/scenario labels are not
policy inputs.  The training mixture contains ordinary V23-vs-Nexto sequences,
Ground-to-Air Goal V3 sequences, and Capability Curriculum V2 offensive-demo,
floor-landing, and wall-landing sequences.  The V2 high-ball row is excluded
because its accepted checkpoint recorded zero high contacts; aerial supervision
comes from the passing V3 aerial teacher instead.

Checkpoint selection uses only the frozen validation corpora.  The bound test
seeds are not generated or evaluated until exactly one validation checkpoint is
selected.  The final physical evaluation directly controls the car with the one
unified policy on every tick.

This V1 authority permits supervised teacher consolidation only.  It does not
authorize PPO, a reward change, a mechanics classifier, simulator mutation, or
promotion over the existing official fail-closed build.
