# Rival 2.0 mixed-opponent transition strategy diagnostic

Verdict: `PASS_GREEN`.

One deterministic update-360 rollout, fixed sample order, fixed model state, and fixed Adam state were used for every variant. No checkpoint was written and no campaign training was resumed.

## Variant comparison

| variant | status | optimizer steps | first minibatch guard step @ KL | maximum minibatch KL | completed mean KL | completed guard | maximum steering KL | value-to-trunk gradient |
|---|---|---:|---|---:|---:|---|---:|---:|
| Baseline | WOULD_REJECT_MINIBATCH_KL | 154 | 6 @ 0.106567591 | 0.235667706 | 0.029711639 | pass | 0.198114648 | 29.287244797 |
| A - family-normalized advantages | WOULD_REJECT_MINIBATCH_KL | 154 | 17 @ 0.110854432 | 0.184228927 | 0.024760458 | pass | 0.132982433 | 29.050451279 |
| B - critic isolated from shared trunk | WOULD_REJECT_BOTH_KL_GUARDS | 154 | 3 @ 0.109817237 | 0.243869528 | 0.119447585 | REJECT | 0.118781656 | 0.000000000 |
| A+B | WOULD_REJECT_MINIBATCH_KL | 154 | 3 @ 0.348160684 | 1.131129503 | 0.010661046 | pass | 0.201196924 | 0.000000000 |
| A+B with actor/trunk learning rate 1e-4 | PASS_COMPLETE | 154 | none | 0.017498828 | 0.010052922 | pass | 0.009296965 | 0.000000000 |

## Effective PPO sample share

| family | nominal world probability | realized initial worlds | realized world share | trainable samples | effective PPO sample share |
|---|---:|---:|---:|---:|---:|
| current | 0.200000 | 26075 | 0.198936 | 1668800 | 0.331855 |
| historical | 0.100000 | 13027 | 0.099388 | 416864 | 0.082897 |
| nexto | 0.350000 | 45859 | 0.349876 | 1467488 | 0.291822 |
| wisp | 0.350000 | 46111 | 0.351799 | 1475552 | 0.293426 |

Current-vs-current worlds contribute two trainable Rival cars; frozen-opponent worlds contribute only Rival's car. The sample mixture therefore differs from the nominal world-assignment mixture by design.

The complete per-step sequence—including pre/post empirical KL, clip fraction, losses, gradient norms, parameter-step norms, every action-channel analytic KL, and all four family-specific KL values—is retained in the machine-readable diagnostic JSON.

## Family-normalized advantage verification

| variant | family | normalized mean | normalized std |
|---|---|---:|---:|
| A - family-normalized advantages | current | -0.000000005 | 1.000000000 |
| A - family-normalized advantages | historical | -0.000000004 | 0.999999940 |
| A - family-normalized advantages | nexto | -0.000000019 | 1.000000000 |
| A - family-normalized advantages | wisp | -0.000000018 | 0.999999940 |
| A+B | current | -0.000000005 | 1.000000000 |
| A+B | historical | -0.000000004 | 0.999999940 |
| A+B | nexto | -0.000000019 | 1.000000000 |
| A+B | wisp | -0.000000018 | 0.999999940 |
| A+B with actor/trunk learning rate 1e-4 | current | -0.000000005 | 1.000000000 |
| A+B with actor/trunk learning rate 1e-4 | historical | -0.000000004 | 0.999999940 |
| A+B with actor/trunk learning rate 1e-4 | nexto | -0.000000019 | 1.000000000 |
| A+B with actor/trunk learning rate 1e-4 | wisp | -0.000000018 | 0.999999940 |

## Architecture implications

The exact replay results and ranked transition recommendation are recorded in the machine-readable `analysis` section and summarized below.

- **Recommended immediate transition:** option 4: A+B with a temporary `1e-4` actor/shared-trunk learning rate while retaining `3e-4` for the critic head. It is the only tested variant whose full 154-step sequence stayed inside both unchanged guards; maximum minibatch KL was `0.017498828` and completed-update mean KL was `0.010052922`.
- **Family normalization works as a statistical operation, but not as a standalone safety fix.** All family means are numerical zero and standard deviations are one; A still rejected at step 17 with KL `0.110854432`.
- **Critic isolation is exact, but not sufficient.** Applied value-loss gradient to the shared trunk was exactly zero in B, A+B, and the reduced variant. B rejected at step 3 with KL `0.109817237`.
- **Do not use A+B at `3e-4`.** It rejected at step 3 with KL `0.348160684`; pitch contributed `0.326307207` analytic KL. Its counterfactual final mean KL fell to `0.010661046`, but the full-sequence maximum reached `1.131129503`; this is direct evidence that a low end-of-update mean cannot replace the minibatch guard.
- **The passing result still learns.** Every reduced-variant optimizer step had nonzero within-family PPO gradients and nonzero actor-head/trunk parameter movement; its smaller displacement is not a frozen-policy artifact.
- **Architecture:** separate actor/critic trunks have qualified long-term support, and critic-only opponent-family conditioning is plausible, but neither is established by this replay or required before the recommended bounded transition. No architecture was changed here.

## Safety

Source checkpoint remained `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`.

Rollback checkpoint remained `430F0113AEF437CA778552A40654AE09263CAC4487A455CC013467FAFD271481`.

Nexto, Wisp, and historical Rival opponents remained frozen and non-trainable.
