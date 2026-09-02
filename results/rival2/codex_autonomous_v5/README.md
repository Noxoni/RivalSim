# Rival 2 Codex autonomous campaign v5

V5 branches from the promoted V4 step-3 checkpoint (103-153 against Nexto), never
from V4's regressed rolling state. It keeps analog-only one-minibatch PPO and shifts
world assignment to 80% frozen Nexto / 20% current self-play. Because current-current
worlds train both cars, this yields approximately two-thirds Nexto and one-third
self-play policy samples. Every microstep remains closed-loop gated.
