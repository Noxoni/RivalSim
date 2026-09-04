# Pre-learning operational corrections

Two issues were found in the new preflight entry point before any amended PPO
optimizer step. Neither involved reward semantics or a failed learning guard.

1. The checkpoint hash utility accepts `Path`, not a CLI string. The first
   invocation stopped before constructing the trainer. Corrected in `bb4c2fbe`.
2. `collect_rollout` deliberately puts the model into evaluation mode. The
   standalone backward preflight must call `model.train()` afterward, just as
   the unchanged production `trainer.update` already does. The second invocation
   collected the full 360-tick rollout but stopped at cuDNN's training-mode check
   before any optimizer step. Corrected in `fac06da0`.

The non-active, pre-learning authority/transition drafts are preserved in
`preflight_path_fix/` and `preflight_training_mode_fix/`. They are diagnostic
history only. The top-level authority and transition bind the corrected code.
The original update-10 checkpoint remains unchanged. No failed rollout or
preflight gradient is included in accepted training counters/checkpoints.

3. The unsplit full-minibatch backward completed but reached 46.31 GiB of tensor
   allocation / 49.01 GiB reserved, beyond physical GPU memory. Its functional
   PASS was not accepted as evidence that training would fit physical VRAM.
   That draft report is archived in `preflight_memory_fix/`. Before learning,
   added complete-sequence gradient accumulation (32 sequences at a time,
   trainable-frame weighted, unchanged effective minibatch and Adam-step count)
   and an explicit physical-memory-headroom preflight check. Finite-rollout
   validation is chunked to avoid multi-gigabyte temporary Boolean masks.
