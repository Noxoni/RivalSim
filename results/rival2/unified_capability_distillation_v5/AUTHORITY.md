# Unified Capability Distillation V5 Authority

V5 starts from the selected V4 single-network checkpoint.  The V4 physical
evaluation proved that ordinary action matching on student observations is not
sufficient for a recurrent policy: aerial supervision started with a zero GRU
state even though deployed control inherited the scripted launch prefix.

V5 freezes prefix-aligned DAgger rehearsal.  The immutable V4 student records
the complete recurrent prefix from scenario tick zero.  Aerial loss begins only
when learned control is active; demo and landing loss uses the active scenario
mask.  All prefix sequences are replayed from a zero episode-boundary hidden
state, exactly matching deployment.  Natural V4-vs-Nexto DAgger and ordinary
teacher trajectories remain in the mixture.

Only the existing recurrent context modules are trainable.  Deployment remains
one network with one actor output and no router, expert selection, task id, or
scenario id.  No PPO, reward change, detector change, simulator change, or
official promotion is authorized by this authority.
