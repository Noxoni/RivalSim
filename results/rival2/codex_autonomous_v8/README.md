# Codex autonomous V8

V8 is a bounded supervised bridge between the reviewed human demonstrations and
the pinned Nexto controller. It collects native RivalSim observations paired with
the exact eight-channel actions emitted by Nexto, trains those samples jointly with
the unchanged reviewed human training split, freezes the critic, and promotes only
checkpoints that improve deterministic closed-loop Nexto play without crossing the
human validation floors. It performs no PPO and does not change rewards.
