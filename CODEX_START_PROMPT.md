# Active Codex Handoff — Rival 2.0 Behavioral Trajectory Evaluation

The Rival 2.0 overnight curriculum is complete at commit `fe057b12253d4e416a30c187853b03f2ec8f4d26` with final checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Start from current `origin/main` and read `handoff/rival2-behavioral-eval/README.md` in full. Treat that file as the controlling requirement.

Mission: run one evaluation-only, large held-out stochastic current-policy self-play analysis of the final 45B checkpoint that records what the ball actually does after every accepted touch and where every scored ball enters the goal. Publish touch-trajectory, possession/next-touch, wall/backboard continuation, touch-to-goal, and goal-mouth X/Z placement evidence.

Do not train. Do not change or remove the current ball-progress reward or any other reward based on the result. Do not alter model/PPO/observation/action/episode/self-play behavior. Do not build the viewer and do not begin v0.6. No preflight/regression/parity/pytest/Ruff/compileall ceremony is authorized.

Implement only the telemetry/evaluator needed for this analysis, run it once, publish the evidence, and stop for review.