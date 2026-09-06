# Reduced-world preflight setup correction

The first preflight stopped before collecting a rollout or taking an optimizer
step: the existing exact-size controller restore rejected a 32,768-world cache
when loading the parent into the 1,024-world preflight environment.

The continuation explicitly begins fresh physical episodes; it must preserve
the two opponent RNG streams and counters, not carry obsolete world caches.
The new entry helper validates native identity and both RNG states, restores
those streams/counters, and activates only the fresh environment's worlds.
The same helper runs at full production scale. Same-arm resumptions continue
to use the preserved exact-size production validator.

Tests cover exact streams, fresh-world reset size, telemetry counters and
failure before mutation on invalid identity/counters/RNG. No model, optimizer,
reward, controller action logic, or physical simulation semantics changed.
