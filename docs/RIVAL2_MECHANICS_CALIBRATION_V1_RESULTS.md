# Rival 2.0 Mechanics Calibration V1 Results

Source head: `0124cd7f29278702158c9cbba9c741c11a29f111`
Handoff source: `1da8557f32a94e6a8e96d1acbb0103656e203e27`
Arena geometry SHA-256: `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`
Gameplay V1 +239 checkpoint SHA-256: `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`
Mode: calibration plus read-only shadow telemetry; mechanics reward remained exactly disabled.

No Rival training or opponent training ran. No policy, PPO, observation, action, physics, reward, or episode-lifecycle contract was changed.

Targeted correction baseline: `f49768368377dcb5aa0cc67f3a08f79bd68538a3`; corrected families: Musty, Breezi, Redirect. The six other calibration records/corpora are preserved exactly. See `docs/RIVAL2_MECHANICS_CALIBRATION_TARGETED_CORRECTION_V1.md` for the reviewer package and legacy-event replay.

## Corpus and contracts

- 648 real 120 Hz RivalSim traces: 72 per continuous detector.
- Per detector: 24 positives, 24 near misses, and 24 ordinary controls; 16 derivation plus 8 held-out cases from each class.
- Calibration seed: `2026082701`; shadow seed: `2026082702`.
- Policy cadence: 30 Hz; physics cadence: 120 Hz.
- Policy iteration: `359`; policy config hash: `58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136`.
- Frozen contract hashes: `{"RIVAL2_ACTION_V1": "145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B", "RIVAL2_EPISODE_V1": "E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E", "RIVAL2_OBS_V1": "10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF", "RIVAL2_REWARD_GAMEPLAY_V1": "48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072"}`.

## Source-exact regression

Focused result: `PASS_GREEN` (8 tests passed). The suite covers ball/car reset resource/body identity, chain/pre-flip re-arm, frozen dash timing and surface classes, same-family de-duplication, compound-family observability, the 72-case split, midpoint derivation, and a GPU-resident zero-reward observer smoke.

## Continuous detector results

| Detector | Status | Boundaries | Held-out FP | Held-out FN |
|---|---:|---|---:|---:|
| speedflip | CALIBRATED | actual_dodge min 0.5 (margin 1); cancel_ticks max 8 (margin 10); pitch_rotation max 0.487357 (margin 0.418773); alignment min 0.898344 (margin 0.0742735) | 0 | 0 |
| half_flip | CALIBRATED | actual_dodge min 0.5 (margin 1); cancel_ticks_min_feature min 28.5 (margin 11); cancel_ticks_max_feature max 518.5 (margin 961); pitch_rotation max 2.94978 (margin 0.461565); heading_dot max -0.374697 (margin 0.64389); new_forward_speed min 18.3763 (margin 121.202); supported_completion min 0.5 (margin 1) | 0 | 0 |
| possession | NOT_READY_FOR_REWARD | touch_onsets min 5 (margin 2); control_distance max 5085.91 (margin 9826.17); control_relative_speed max 5164.95 (margin 9668.1); contact_gap_ticks max 613 (margin 772) | 1 | 0 |
| ground_carry | NOT_READY_FOR_REWARD | support_ticks min 1 (margin 2); control_distance max 154.739 (margin 2.37064); control_relative_speed max 438.029 (margin 20.1516) | 8 | 0 |
| musty | CALIBRATED | actual_backward_dodge min 0.5 (margin 1); contact_age_ticks min 4.5 (margin 7); rotational_closing_speed min 154.479 (margin 11.6513); rotational_fraction min 0.264016 (margin 0.0179901); sweep_closure min 13.4876 (margin 0.0750389); sweep_path_length min 23.0173 (margin 0.430721); ball_delta_v min 354.312 (margin 103.652) | 0 | 0 |
| breezi | CALIBRATED | terminal_musty min 0.5 (margin 1); ordered_orientation min 0.5 (margin 1); nose_up_peak min 0.265685 (margin 0.53137); inverted_depth min 0.770279 (margin 0.0489773); nose_down_depth min 0.0662821 (margin 0.132564); roll_path min 3.81543 (margin 0.486946); yaw_path min 1.27793 (margin 0.291349); combined_roll_yaw_ticks min 130 (margin 2); roll_yaw_overlap_fraction min 0.981506 (margin 0.021836); control_max_distance max 394.808 (margin 342.654) | 0 | 0 |
| redirect | CALIBRATED | legitimate_contact min 0.5 (margin 1); incoming_speed min 514.813 (margin 125.104); outgoing_speed min 249.117 (margin 12.1976); direction_change min 0.939067 (margin 1.20002); approach_cross_fraction min 0.37283 (margin 0.11374); contact_normal_cross_fraction min 0.186995 (margin 0.268654); speed_retention min 0.339024 (margin 0.0512823) | 0 | 0 |
| pinch | CALIBRATED | overlap_ticks max 6.5 (margin 11); opposition_sign min 0.5 (margin 1); opposition min -0.482971 (margin 1.03406); closing_speed min 178.269 (margin 356.537); ball_delta_v min 506.547 (margin 85.6791) | 0 | 0 |
| pogo | CALIBRATED | chassis_contact min 0.5 (margin 1); corner_region min 0.570819 (margin 0.113546); incoming_normal_speed min 223.147 (margin 446.295); outgoing_normal_speed min 3.6041 (margin 3.89449); wheel_support max 1 (margin 2); separation_ticks max 10 (margin 4) | 0 | 0 |

Thresholds are binary physical event-identity boundaries. They are not quality scores and no threshold changes reward magnitude.

## Explicit overlap evidence

### possession

This family remains `NOT_READY_FOR_REWARD`; no runtime event is emitted. The following physically different cases remain inside the full trace-derived conjunction:

- `possession-near_miss-D03` (derivation near_miss): `{"contact_gap_ticks":52.0,"control_distance":134.14712524414062,"control_relative_speed":182.4742431640625,"support_ticks":4.0,"touch_onsets":22.0,"velocity_change":0.11914200335741043}`
- `possession-near_miss-D10` (derivation near_miss): `{"contact_gap_ticks":49.0,"control_distance":134.303466796875,"control_relative_speed":170.1923065185547,"support_ticks":3.0,"touch_onsets":15.0,"velocity_change":0.39488035440444946}`
- `possession-near_miss-H17` (heldout near_miss): `{"contact_gap_ticks":54.0,"control_distance":134.08737182617188,"control_relative_speed":189.18972778320312,"support_ticks":6.0,"touch_onsets":26.0,"velocity_change":0.7934455275535583}`

The complete parameters, extrema, labels, and measured features are preserved in `case_results.jsonl` and `thresholds.json`.

### ground_carry

This family remains `NOT_READY_FOR_REWARD`; no runtime event is emitted. The following physically different cases remain inside the full trace-derived conjunction:

- `ground_carry-near_miss-D00` (derivation near_miss): `{"contact_gap_ticks":51.0,"control_distance":153.51309204101562,"control_relative_speed":179.06866455078125,"support_ticks":3.0,"touch_onsets":7.0,"velocity_change":574.05419921875}`
- `ground_carry-near_miss-D01` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":151.28411865234375,"control_relative_speed":310.2514343261719,"support_ticks":2.0,"touch_onsets":1.0,"velocity_change":580.5184936523438}`
- `ground_carry-near_miss-D02` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":151.0008087158203,"control_relative_speed":258.2755432128906,"support_ticks":3.0,"touch_onsets":1.0,"velocity_change":584.3273315429688}`
- `ground_carry-near_miss-D03` (derivation near_miss): `{"contact_gap_ticks":151.0,"control_distance":152.18617248535156,"control_relative_speed":206.89576721191406,"support_ticks":3.0,"touch_onsets":3.0,"velocity_change":555.7503662109375}`
- `ground_carry-near_miss-D04` (derivation near_miss): `{"contact_gap_ticks":89.0,"control_distance":153.00823974609375,"control_relative_speed":192.4700164794922,"support_ticks":3.0,"touch_onsets":4.0,"velocity_change":570.620361328125}`
- `ground_carry-near_miss-D05` (derivation near_miss): `{"contact_gap_ticks":44.0,"control_distance":143.42684936523438,"control_relative_speed":305.7281799316406,"support_ticks":2.0,"touch_onsets":4.0,"velocity_change":594.8442993164062}`
- `ground_carry-near_miss-D06` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":131.57504272460938,"control_relative_speed":228.325439453125,"support_ticks":3.0,"touch_onsets":1.0,"velocity_change":598.4424438476562}`
- `ground_carry-near_miss-D07` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":152.13331604003906,"control_relative_speed":342.541259765625,"support_ticks":2.0,"touch_onsets":1.0,"velocity_change":575.46484375}`
- `ground_carry-near_miss-D08` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":153.02499389648438,"control_relative_speed":318.22467041015625,"support_ticks":2.0,"touch_onsets":1.0,"velocity_change":579.9117431640625}`
- `ground_carry-near_miss-D09` (derivation near_miss): `{"contact_gap_ticks":152.0,"control_distance":151.14598083496094,"control_relative_speed":268.1210632324219,"support_ticks":2.0,"touch_onsets":2.0,"velocity_change":562.472412109375}`
- `ground_carry-near_miss-D10` (derivation near_miss): `{"contact_gap_ticks":162.0,"control_distance":151.0486297607422,"control_relative_speed":205.14190673828125,"support_ticks":3.0,"touch_onsets":2.0,"velocity_change":564.9484252929688}`
- `ground_carry-near_miss-D11` (derivation near_miss): `{"contact_gap_ticks":82.0,"control_distance":153.3294677734375,"control_relative_speed":175.9705047607422,"support_ticks":2.0,"touch_onsets":8.0,"velocity_change":559.930908203125}`
- `ground_carry-near_miss-D14` (derivation near_miss): `{"contact_gap_ticks":999.0,"control_distance":150.98936462402344,"control_relative_speed":345.2641906738281,"support_ticks":2.0,"touch_onsets":1.0,"velocity_change":576.0650634765625}`
- `ground_carry-near_miss-D15` (derivation near_miss): `{"contact_gap_ticks":175.0,"control_distance":151.79124450683594,"control_relative_speed":293.1713562011719,"support_ticks":2.0,"touch_onsets":2.0,"velocity_change":554.111572265625}`
- `ground_carry-near_miss-H16` (heldout near_miss): `{"contact_gap_ticks":999.0,"control_distance":151.14535522460938,"control_relative_speed":268.8185729980469,"support_ticks":3.0,"touch_onsets":1.0,"velocity_change":583.99365234375}`
- `ground_carry-near_miss-H17` (heldout near_miss): `{"contact_gap_ticks":167.0,"control_distance":151.50352478027344,"control_relative_speed":216.61178588867188,"support_ticks":3.0,"touch_onsets":2.0,"velocity_change":563.7492065429688}`
- `ground_carry-near_miss-H18` (heldout near_miss): `{"contact_gap_ticks":76.0,"control_distance":152.44357299804688,"control_relative_speed":169.69674682617188,"support_ticks":3.0,"touch_onsets":4.0,"velocity_change":555.0311279296875}`
- `ground_carry-near_miss-H19` (heldout near_miss): `{"contact_gap_ticks":43.0,"control_distance":137.28195190429688,"control_relative_speed":234.68588256835938,"support_ticks":2.0,"touch_onsets":4.0,"velocity_change":596.318359375}`
- `ground_carry-near_miss-H20` (heldout near_miss): `{"contact_gap_ticks":83.0,"control_distance":141.15655517578125,"control_relative_speed":318.7574157714844,"support_ticks":4.0,"touch_onsets":2.0,"velocity_change":596.5260620117188}`
- `ground_carry-near_miss-H21` (heldout near_miss): `{"contact_gap_ticks":999.0,"control_distance":152.3328857421875,"control_relative_speed":340.78515625,"support_ticks":2.0,"touch_onsets":1.0,"velocity_change":577.3102416992188}`
- `ground_carry-near_miss-H22` (heldout near_miss): `{"contact_gap_ticks":183.0,"control_distance":152.10704040527344,"control_relative_speed":293.5357971191406,"support_ticks":2.0,"touch_onsets":2.0,"velocity_change":553.2384643554688}`
- `ground_carry-near_miss-H23` (heldout near_miss): `{"contact_gap_ticks":999.0,"control_distance":149.6942138671875,"control_relative_speed":243.5624237060547,"support_ticks":3.0,"touch_onsets":1.0,"velocity_change":584.862060546875}`

The complete parameters, extrema, labels, and measured features are preserved in `case_results.jsonl` and `thresholds.json`.

## Shadow gate

Episodes: 256 (Nexto 128, Wisp 128); stochastic Gameplay V1 +239 Rival, side-balanced.

| Family | Count | Events/min |
|---|---:|---:|
| speedflip | 13 | 0.189786 |
| half_flip | 0 | 0.000000 |
| possession | 0 | 0.000000 |
| ground_carry | 0 | 0.000000 |
| musty | 0 | 0.000000 |
| breezi | 0 | 0.000000 |
| redirect | 84 | 1.226307 |
| pinch | 4 | 0.058396 |
| pogo | 105 | 1.532884 |

| Opponent | Episodes | Sim min | Goals | No-touch | Hard-time |
|---|---:|---:|---:|---:|---:|
| Nexto | 128 | 28.914444 | 128 | 0 | 0 |
| Wisp | 128 | 39.583889 | 111 | 1 | 16 |

| Opponent | Rival side | Family | Count | Events/min |
|---|---|---|---:|---:|
| Nexto | Blue | speedflip | 1 | 0.073299 |
| Nexto | Blue | half_flip | 0 | 0.000000 |
| Nexto | Blue | possession | 0 | 0.000000 |
| Nexto | Blue | ground_carry | 0 | 0.000000 |
| Nexto | Blue | musty | 0 | 0.000000 |
| Nexto | Blue | breezi | 0 | 0.000000 |
| Nexto | Blue | redirect | 15 | 1.099483 |
| Nexto | Blue | pinch | 0 | 0.000000 |
| Nexto | Blue | pogo | 19 | 1.392678 |
| Nexto | Orange | speedflip | 4 | 0.261923 |
| Nexto | Orange | half_flip | 0 | 0.000000 |
| Nexto | Orange | possession | 0 | 0.000000 |
| Nexto | Orange | ground_carry | 0 | 0.000000 |
| Nexto | Orange | musty | 0 | 0.000000 |
| Nexto | Orange | breezi | 0 | 0.000000 |
| Nexto | Orange | redirect | 29 | 1.898941 |
| Nexto | Orange | pinch | 2 | 0.130961 |
| Nexto | Orange | pogo | 26 | 1.702499 |
| Wisp | Blue | speedflip | 5 | 0.235590 |
| Wisp | Blue | half_flip | 0 | 0.000000 |
| Wisp | Blue | possession | 0 | 0.000000 |
| Wisp | Blue | ground_carry | 0 | 0.000000 |
| Wisp | Blue | musty | 0 | 0.000000 |
| Wisp | Blue | breezi | 0 | 0.000000 |
| Wisp | Blue | redirect | 24 | 1.130831 |
| Wisp | Blue | pinch | 2 | 0.094236 |
| Wisp | Blue | pogo | 28 | 1.319303 |
| Wisp | Orange | speedflip | 3 | 0.163394 |
| Wisp | Orange | half_flip | 0 | 0.000000 |
| Wisp | Orange | possession | 0 | 0.000000 |
| Wisp | Orange | ground_carry | 0 | 0.000000 |
| Wisp | Orange | musty | 0 | 0.000000 |
| Wisp | Orange | breezi | 0 | 0.000000 |
| Wisp | Orange | redirect | 16 | 0.871433 |
| Wisp | Orange | pinch | 0 | 0.000000 |
| Wisp | Orange | pogo | 32 | 1.742867 |

Impossible/pathological classifications: 0.
Mechanics reward contribution: `0.0` (required exact zero).

Bounded per-event raw features are retained in `shadow_event_evidence.json`; all calibration case parameters and measured features are retained in `case_results.jsonl`.

No detector fired on an impossible-state assertion. Calibrated-family frequencies were bounded by physical family lockout/re-arm state; the two telemetry-only families emitted zero events by construction. No suspicious case required threshold retuning after held-out evaluation.

## Reproduction

```powershell
$env:RIVALSIM_COLLISION_DIR='G:\dev\RLBot-Rival\bot\collision_meshes'
.venv\Scripts\python.exe -m pytest -q tests/test_rival2_mechanics_calibration.py --basetemp .pytest_cache\mechanics-final
.venv\Scripts\python.exe benchmarks/run_rival2_mechanics_calibration.py --collision-root $env:RIVALSIM_COLLISION_DIR
```

Machine-readable artifacts and their SHA-256 hashes are indexed by `results/rival2/mechanics_calibration_v1/calibration_manifest.json`.
