# Active Codex Handoff — Behavioral Evaluation -> Nexto Port -> Full-Match Benchmark

The Rival 2.0 overnight curriculum is complete with final checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Start from the current `origin/main` and read both controlling handoffs in order:

1. `handoff/rival2-behavioral-eval/README.md`
2. `handoff/rival2-nexto-port/README.md`

The user has now explicitly authorized continuation into the Nexto work without returning for another approval.

## Phase 1 — finish the existing final-45B behavioral evaluation

If it has not already been completed in the current lineage, implement and run the single authorized behavioral trajectory / goal-placement evaluation exactly as defined by `handoff/rival2-behavioral-eval/README.md`. Publish its evidence.

Do not change Rival's reward or training based on that result.

## Phase 2 — port public Nexto into RivalSim

Follow `handoff/rival2-nexto-port/README.md` exactly. Pin `Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`, implement a faithful batched GPU-native Nexto observation/model/action adapter, preserve its 15 Hz neural cadence and exact stock 120 Hz hard-coded kickoff controller, and pass the targeted observation/action/kickoff parity gates.

## Phase 3 — build and run the full-match benchmark

Build the separate 120 Hz full-match RivalSim runtime defined by the handoff. The frozen Rival training episode semantics must remain unchanged. Run the canonical deterministic 10-match side/layout matrix and the secondary batched stochastic-Rival robustness suite against deterministic stock Nexto, publish the complete evidence, and stop.

Do not train Rival against Nexto yet. Do not change rewards, PPO, model architecture, Rival observation/action contracts, simulator physics, or begin v0.6. Do not build the viewer. Avoid unrelated release/lint/regression ceremony; only the targeted adapter fidelity checks required by the handoff are authorized.

When finished, commit and push all implementation/evidence to `origin/main` and return the final commit SHA plus the behavioral-evaluation summary and Rival-vs-Nexto match results.