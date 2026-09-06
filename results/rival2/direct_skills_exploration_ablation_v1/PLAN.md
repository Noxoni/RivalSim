# Matched exploration-pressure comparison from update 750

This is a finite learning experiment under the continuing SSL-development goal,
not a new claim of gameplay competence. The previous turn restated measured
regression; this experiment now tests one possible cause by changing one loss
coefficient. Do not resume any completed earlier campaign.

## Frozen design

- Common parent: `checkpoints/rival2/direct_skills_exploration_continuation_v1/child_000070.pt`.
- SHA-256: `517A2217CBDF4124612B518E10F30BC4D5F00F5C1DC77A5119D10D2B1DD89613`.
- Retain arm: uniform-to-policy exploration coefficient 0.01.
- Withdraw arm: coefficient 0.0. The unweighted barrier remains telemetry;
  the weighted loss is exactly zero. This is not removal of ordinary entropy.
- Both restore the same model, Adam/moments/groups/counters, four learner RNGs
  and native-opponent RNGs. Fresh physical episodes and zero recurrent hidden
  are explicit. Neither arm resumes the other's learned state.
- Same 32,768 worlds, 30 Hz decisions/120 Hz physics, horizon 90, two PPO passes,
  actor LR 1e-4, critic LR 3e-4, temperature 2 for collection AND likelihood,
  ordinary entropy 0.001. Same entity-aware recurrent actor/independent critic.
- Same goal-based finishing, challenge, defense and kickoff rewards, original
  scenario bank and half native-v5 Nexto/half current-self-play worlds. No new
  task identifiers, scripted Rival controls, mechanic detectors or rewards.
- 30 accepted updates per arm, 132,710,400 learner decisions each; total
  265,420,800. Sequential independent workers: retain, then withdraw.
- Preserve entry 0, first 1, permanent 10/30 and alternating durable rolling
  checkpoints after every accepted update. Equal offsets mean cumulative
  760/780 in separate lineages, not a single sequential update-810 policy.
- Same ten full 300-second native-v5 Nexto development matches at +10 and +30,
  deterministic raw argmax, standard unassisted standing kickoffs. No reruns,
  cherry-picked worlds or substitute easier benchmarks.
- Compare final goal difference primarily, with goals/concedes reported
  separately, wins, contacts, next-contact identity and kickoff first contacts.
  Intermediate +10 does not tune, stop or select either arm. Numerical failure
  and user STOP still stop work; KL magnitude never does.
- No automatic deployment, promotion or extension. A single training seed and
  ten reused development matches cannot establish statistical significance,
  native Rocket League parity, or SSL skill. More contacts are not possession.

The completed 750 baseline scored 2/conceded 194, 0/10 wins, 14.34 contacts/min,
and zero standing-kickoff first contacts. Original 650 scored 7/conceded 196.
The later contact-sequencing change is retained in the common parent, rather
than silently discarding it by returning to 650 or restarting random weights.

## Validation and launch

Focused tests prove coefficient-zero loss AND gradient parity with original
PPO, exact critic-gradient independence of the added term, arm-scoped bindings,
frozen common settings, cross-arm/closed-run resume rejection, nonfinite
telemetry detection, large finite KL acceptance, STOP and worker-failure behavior.
Both real-simulator preflights use 1,024 worlds and 90 decisions with zero
optimizer steps. They verify unchanged parent/model/Adam, finite gradients,
critic isolation, native Nexto, exact action targets, masks and reward identity.
Their parent/model/Adam identities and complete rollout training telemetry
match exactly; gradient norms differ as expected from the two objectives.
This is not a claim that the later learned trajectories will be identical.

Before the first optimizer step: freeze both authorities/packages, commit and
push all sources/tests/preflights, remotely read back their Git objects, then
run the committed verifier. No parent checkpoint may be overwritten.

Commands from `G:/dev/RivalSim` using `.venv/Scripts/python.exe`:

```text
benchmarks/run_direct_skills_exploration_ablation_v1.py freeze
benchmarks/run_direct_skills_exploration_ablation_v1.py verify
benchmarks/run_direct_skills_exploration_ablation_v1.py pair
benchmarks/report_direct_skills_exploration_ablation_v1.py 10 --arm retain
benchmarks/report_direct_skills_exploration_ablation_v1.py 10 --arm withdraw
benchmarks/report_direct_skills_exploration_ablation_v1.py 30
```

The pair supervisor launches separate workers under the existing shared GPU
lease. State/logs live under
`G:/dev/RivalSim-runs/direct-skills-exploration-ablation-v1/{retain,withdraw}`.
Root `STOP` is propagated to the active arm; each worker preserves its latest
accepted checkpoint. It never automatically restarts a failed/interrupted arm.
After a narrowly audited operational recovery, explicit same-arm latest
path/hash resume is available through `run --arm NAME --resume PATH
--resume-sha256 SHA`. Do not clear user/unknown STOP or change learned semantics.
The second arm launches only after first-arm completion and CPU audits.

The existing monitor will report newly completed evaluations only, not repeat
old results on every heartbeat. Following OpenAI Docs, update the existing
chat schedule rather than create a duplicate:
[official scheduling documentation](https://learn.chatgpt.com/docs/automations?surface=app).

All detailed checkpoint audits, curve prefixes, per-world comparisons, logs
and final checkpoint artifacts belong in Git. Runtime health alone never
constitutes success on the user's gameplay goal.
