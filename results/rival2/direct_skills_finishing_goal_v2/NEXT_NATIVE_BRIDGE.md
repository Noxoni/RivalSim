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
