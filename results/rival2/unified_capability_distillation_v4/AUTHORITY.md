# Unified Capability Distillation V4 Authority

V4 starts from the selected V3 single-network checkpoint. V3 improved bounded
Nexto scoring from 0-256 to 51-154 and preserved demo/dash behavior, but its
aerial closed-loop state distribution diverged even though teacher-trajectory
validation remained strong.

V4 freezes symmetric student-state rehearsal. It records deterministic V3
trajectories in both pinned-Nexto natural play and the controlled aerial task,
labels the visited states with the appropriate frozen teachers, and jointly
rehearses those corpora with fresh teacher-induced natural, aerial, demo,
floor-landing, and wall-landing sequences.

Only recurrent context modules are trainable. The deployed policy remains one
network with one actor output and receives no teacher, route, task, or scenario
identifier. No PPO, reward change, mechanic detector, simulator change, or
official promotion is authorized.
