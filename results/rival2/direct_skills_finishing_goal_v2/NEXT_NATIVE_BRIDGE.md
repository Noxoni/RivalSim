# Next concrete task: native entity/joint90 recurrent30Hz preparation

The completed finishing25block is preserved, not automatically extended. It wins
8/10fixed RivalSim Nexto matches but scores7/64hardfinishing and2/29ongoing tests.
An older600reference also wins8/10 with higher goal difference. These are research
candidates, not demonstrated SSL or a native RLBot result.

Implement a separate native-play preparation path, without any learning or
changes to model weights, rewards, physics, curricula or observation contract.
Do not copy a new checkpoint over the installed old V5 bundle/configuration.

1. Use exact600and newfinishing650 as separately identified candidate inputs:
   600SHA8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2;
   new650SHA939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D.
   Do not confuse new650with old direct650/675. Preserve both whole checkpoints.
2. Build a self-contained entity policy inference/export wrapper using the exact
  90-logit argmax action-table mapping. Preserve182input fields and GRU state;
   no hybrid tanh/BCE decoding, temperature sampling, task/router/action scripts.
3. Validate action and recurrent-state parity on captured native simulator
   observations, sequential hidden evolution and real reset boundaries. Random
   or zero inputs can supplement shape checks, never stand in for gameplay.
4. Prepare a30Hz recurrent packet scheduler:120Hz lifecycle/packet updates,
   four-tick held action, no GRU advancement on held/duplicate packets. Handle
   missed packets explicitly without fabricating unseen observations/history.
5. Audit the complete installed packet observation builder against the actual
   model fields; explicitly distinguish direct, derived, approximate, unavailable.
   Individual-wheel availability remains a known gap, not permission to silently
   rewrite training fields or label aggregate state exact. Do not train an adapter
   or turn features off as an unreviewed workaround. Use existing recorded/native
   evidence where relevant; no new human recording required.
6. Store exports, manifests, code, parity/cadence/input-availability evidence in
   this repo first. Keep external RLBot source read-only during preparation and
   leave game/installed bot untouched. Then assess the prepared path and define
   the bounded native local-game comparison before any installation/launch.

The existing loader diagnostic is complete; do not repeat it as new research.
Actual error: old Rival2UnifiedPolicyConfig rejects critic_architecture; the old
recurrent runtime independently requires120Hz/hold1 and its exporter uses hybrid
output slices. Build the correct interface rather than loosening its guard.
No new PPO campaign is authorized by this note. It is the next no-learning step
toward the ongoing goal, not cancellation of that goal or a claim it is blocked.

## Preparation progress (2026-09-06)

The inference export, recurrent packet scheduler, native-sequence parity,
target RLBot interpreter loading and complete source-level input availability
audit are now implemented. Read `results/rival2/entity_native_bridge_v1/RESULTS.md`
and its manifests before continuing; do not repeat the completed12000-decision
capture/export probe or old-loader incompatibility investigation.

Remaining next action is the separately versioned packet integration inside
RivalSim, with real phase/reset/ordinary-pause handling, interval metadata,
unavailable-field qualifications and cold warm-up validation. Then freeze the
bounded native local-game comparison before installation. Neither export has
been installed; no native match or new learning occurred during preparation.

## Superseding next action after native comparison (2026-09-06)

Packet integration, bounded protocol publication and the first native launch
are now complete. Read `results/rival2/entity_native_packet_v1/RESULTS.md` and
`reference600_blue/audit.json` before acting. Do not repeat preparation above.

Reference600 trailed native Nexto 0-37 when the frozen600-second wall-clock
limit stopped its still-incomplete match, with about80seconds regulation left.
No finishing650 or opposite-side game ran. This is a severe performance gap,
not a completed four-case comparison and not a native promotion.

All9171recorded policy decisions reproduce exactly. Actual native consumed
controls match the previous command on all9132eligible uninterrupted decision
endpoints; the source/export hashes are unchanged. Rival was moving, requesting
boost and touching the ball, but not scoring. No learning occurred.

Next: a bounded read-only discrepancy audit using these preserved actual native
packets, especially initial kickoff/approach segments. Check physical inputs,
phase semantics, short motion/contact fidelity and simulator-versus-native Nexto
integration/identity. Do not invent unknown physics state as exact, assume a
root cause, mask fields, rewrite rewards, or restart PPO as a workaround. Do not
extend the expired native comparison deadline or present its partial game as a
full result. Both candidate checkpoints and all prior evidence stay preserved.

## Superseding measured diagnosis (2026-09-06)

The initial native packet wheel proxy versus simulator reset-cache mismatch is
now demonstrated to matter: changing only those first-decision wheel fields in
the same20second simulator test changes14/3goals to4/14, and34touches to8.
All38native kickoffs were checked;30first commands change under that ablation.
The short grounded motion replay is close, not a gross acceleration failure.
See `results/rival2/entity_native_gap_v1/RESULTS.md`; no production fix or learning
has occurred. Follow its `NEXT_RESET_ALIGNMENT.md` for a separate, explicit,
qualified research-runtime compatibility test. Do not repeat completed audits
or silently zero unavailable fields as if they were measured native zeros.

## Superseding native reset V2 launch (2026-09-06)

The separate research runtime and 27 focused tests are published at
`9c9dffe8b59c6b2bdd909bf672fdc630ea7fd4f8`. The bounded two-team native comparison
is now launched, with first actual kickoff projection verified in the first120
flushed decisions. Follow `results/rival2/entity_native_reset_v2/NEXT_ACTION.md`
and its external campaign state. Do not repeat implementation, restart the run,
or treat the launch receipt as a completed result. No learning is active.
