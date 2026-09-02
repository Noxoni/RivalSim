# Rival 2 Codex autonomous campaign v7

V7 gives the promoted V4 model one protected 30-microstep learning window. Each
microstep still contains only one optimizer minibatch, but 30 fresh 128-tick rollouts
cover 32 seconds per world—long enough for normal goal and defensive outcomes to
reach PPO. Deterministic Nexto and read-only human validation run every five steps;
the source checkpoint remains the default unless a boundary genuinely improves it.
