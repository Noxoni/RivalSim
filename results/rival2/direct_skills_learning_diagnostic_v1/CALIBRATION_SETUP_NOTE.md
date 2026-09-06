# Calibration setup correction

The first gradient-calibration invocation collected its diagnostic rollout but
failed before saving calibration evidence or taking any optimizer step:

`RuntimeError: cudnn RNN backward can only be called in training mode`

The collector correctly switches the model to evaluation mode. The new probe
omitted switching it back for its backward-only gradient measurement. The probe
now calls model.train() after collection, matching the production PPO path.
The policy contains no dropout. This is a diagnostic harness setup correction,
not evidence of a production PPO failure or a change to a trained checkpoint.
The next invocation repeats the same frozen parent/seed/corpus and coefficient
selection rule. No calibration result existed to overwrite.
