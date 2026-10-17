# CETACEAN-004b receipt (2026-10-16, America/Regina)
Robustness check of the CETACEAN-004 species and ecotype results.
- Same 55-file DCLDE 2027 DFO subset.
- Feature extraction is copied verbatim from run004.py.
- Run: `/workspace/cetacean-001/.venv/bin/python run004b.py .` (946 s).
- Local only: no uploads, sends or commits.

## Method
- **Seeds:** 5 seeds (0–4). Each seed changes the shuffled day folds (GroupKFold shuffle=True) and the random-forest seed.
  - Cross-deployment splits are fixed. LR is deterministic there (sd 0).
- **Confidence intervals:** 95% intervals from a cluster bootstrap over recording days (2,000 resamples) on the seed-0 out-of-fold predictions.
- **Permutation test:**
  - 200 permutations, applied to the call features with the automatic band only.
  - Labels are shuffled at event level within each deployment.
  - The random forest uses 100 trees during permutations.
  - The minimum possible p is 1/201 = 0.005.

## Key results (balanced accuracy: mean of 5 seeds; 95% CI from seed 0)

Species:

| Model | Day holdout | North→West | West→North | p-value |
|---|---|---|---|---|
| Call features, auto band, LR | .793 [.749, .837] | .780 [.674, .841] | .725 [.640, .778] | .005 (all splits) |
| Call features, auto band, RF | .763 [.668, .784] | .695 [.585, .777] | .764 [.676, .831] | .005 (all splits) |
| Trivial, RF | .590 [.491, .630] | .639 [.568, .711] | .651 [.552, .694] | – |
| Trivial, LR | .592 [.481, .700] | .537 [.487, .609] | .566 [.473, .708] | – |
| Run 003 baseline, RF | .654 [.589, .702] | .603 [.503, .647] | .608 [.505, .725] | – |
| Run 003 baseline, LR | .633 [.526, .703] | .500 | .500 | – |

Ecotype (floor 0.333):

| Model | Day holdout | North→West | West→North | p-value |
|---|---|---|---|---|
| Call features, auto band, RF | .491 [.444, .544] | .469 [.420, .546] | .506 [.450, .578] | .005 (all splits) |
| Trivial, RF | .419 [.267, .409] | .307 | .319 | – |
| Run 003 baseline, LR | .457 [.306, .482] | .338 | .441 [.290, .536] | – |

The permutation null averaged 0.50 for species and 0.33 for ecotype.

## Verdict
**Species:** the result is above chance in every split. All CI lower bounds are above 0.5, and p = 0.005, the smallest value 200 permutations can give.
- **Day holdout:** the result beats the trivial floor clearly. LR's CI lower bound (.749) is above the trivial upper bound for RF (.630), though not for trivial LR (.700).
- **Across deployments:** the result is better than trivial on average, but the CIs overlap.
  - LR, West→North: lower bound .640 vs trivial RF mean .651.
  - RF, North→West: lower bound .585 vs trivial RF mean .639.

**Ecotype:** weak but above chance (about 0.47–0.51 vs 0.333, p = 0.005). The CIs overlap the comparators in places.

## Caveats
- Event-level shuffling breaks the correlation between events from the same day. That makes the null optimistic, so the p-values are likely too small. A day-level permutation would be stricter.
- CIs reflect day sampling only; they do not cover the choice of model.
- The day-fold seed moves the trivial floor by up to about ±0.04.
- 55 files.
