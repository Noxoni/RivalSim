# Native match-interface check: initial-only result and CPU limitation

The optional CPU dynamic check failed before the first physical tick completed.
Warp's generated CPU vehicle source could not compile these existing native
snippets:

```
error: use of undeclared identifier 'fminf'
error: use of undeclared identifier 'fmaxf'
Module rivalsim.kernels.vehicle 2f213aa load on device 'cpu' ... (error)
Exception: CPU kernel build failed with error code -1
```

The failure occurred in `train.step(action)`, through `wheel_pre_tick`, before
the forced goal fixture. The compiler retried without precompiled headers and
failed again. This is not a policy training failure: the separate CUDA worker
continued accepting updates and its stderr stayed empty. No physics source,
training source, checkpoint, CUDA configuration or safety guard was modified.
The failed CPU dynamic test is **not complete** and provides no reset-parity
result. Dynamic/CUDA full-match behavior must still be checked after training.

A separately invoked `--initial-only` mode successfully instantiated the native
training and match worlds at all five standard kickoff layouts on both teams.
All182 observation fields, actor logits and deterministic eight-channel
controls matched exactly at initialization. This is real native initialization,
not observations reconstructed from a hand-written state approximation, but it
does not advance physics or establish CPU/CUDA trajectory parity.

For the immutable +20 candidate, all10 initial kickoff decisions requested:

`throttle=1, steer=0, pitch=0, yaw=0, roll=0, jump=0, boost=1, handbrake=0`.

Thus the policy is not initially idle or immediately jumping in these layouts.
This says nothing about subsequent steering, contact, recovery or winning:
the completed +20 Nexto evaluation still reports0 goals for and64 against.

The initial-only report verifies zero CUDA allocation, no optimizer steps and
unchanged model/checkpoint hashes. Its limited PASS must not be reported as a
dynamic full-match test PASS.

```
.venv\Scripts\python.exe benchmarks/validate_rival2_ssl_entity_match_interface.py --initial-only
```
