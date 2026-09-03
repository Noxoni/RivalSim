# Ground-to-air takeoff timing V8 review

## Verdict

**CALIBRATION PASS; NO POLICY PROMOTION.** The no-learning matched-state sweep
proved that the inherited 8-tick first-jump hold leaves substantial vertical
lift unused. The frozen lexicographic rule selected `hold_24_release_4` as the
physical timing basis for the next prospectively authorized learning run. No
model parameter changed, no test corpus was opened, and V23 remains protected.

## Matched physical result

All five candidates replayed the same 3,072 scenario worlds: three setup
families, parked and live defenders, both field directions, and 256 worlds per
row.

| First hold / release | First hold | Second jump | Worst-row median max car Z | Worst-row median max vertical speed | Nonzero elevated rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 8 | 66.7 ms | 133.3 ms | 92.72 uu | 288.69 uu/s | 7/12 |
| 12 / 4 | 100.0 ms | 133.3 ms | 95.78 uu | 312.97 uu/s | 5/12 |
| 16 / 4 | 133.3 ms | 166.7 ms | 110.32 uu | 338.65 uu/s | 6/12 |
| 20 / 4 | 166.7 ms | 200.0 ms | 126.92 uu | 363.30 uu/s | 6/12 |
| 24 / 4 | 200.0 ms | 233.3 ms | 141.76 uu | 387.39 uu/s | 7/12 |

From 8/8 to 24/4, the worst-row median maximum car height improved by 52.9%
and worst-row median maximum upward speed improved by 34.2%. The worst-row
takeoff fraction remained 78.9%, the worst-row within-160-uu fraction remained
5.86%, and the maximum row median closest ball distance remained 175.20 uu.
Thus the added lift did not worsen the frozen broad proximity measures.

## Interpretation

The physical probe confirms that Rival generally initiates a jump after the
light setup touch, but the inherited timing does not use the full first-jump
force window. Timing alone still does not produce a complete deterministic
aerial: every candidate has at least one zero-elevated row, and 24/4 has no
high follow rows in this fixed parent evaluation. The selected schedule is
therefore a better launch manifold for training, not a learned capability.

The next authority must restart from the protected controlled scorer, set the
first hold/release schedule to 24/4, preserve fixed pitch and the narrow
steer/yaw/roll residual, and retain equal setup/defender/side gradients. It must
not load V7 weights, change the production reward, or promote on aggregate
height alone.
