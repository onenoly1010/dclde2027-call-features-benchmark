# CETACEAN-005 receipt (2026-10-16, America/Regina)
Local only: no uploads, sends or commits.

## Data
- Source: DCLDE 2027 DFO_CRP portion. Public NOAA bucket `noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/dfo_crp/audio/`, DOI 10.25921/15ey-mh50. Cite Palmer et al. 2025, Sci Data.
- 300 files, 1.03 GB:
  - NorthBc: 150 files, 75 days.
  - WVanIsl: 150 files, 128 days.
  - 55 of them are the CETACEAN-003 files.
- Every file was checked against the GCS object: md5 300/300 and size 300/300 match. FILE_MANIFEST.csv lists URL, GCS generation, md5, SHA-256, sample rate and duration for each file.
- NorthBc includes 8 short files of about 91 s. These are native durations in the archive, not altered.
- Selection is deterministic (code/select_005.py): a round-robin over HW/NRKW/TKW/OKW that prefers unused days, then files containing both HW and KW. 66 files have both HW and KW.
- Annotations: 11,637 events in the selected files. 7 that ran past the end of their file were dropped, leaving 11,630. 0 events above Nyquist.
- Species set: HW 4,992, KW 6,548.
- Ecotype set: NRKW 2,769, TKW 2,539, OKW 1,133. OKW comes from only 8 days per deployment.

## Run
- Command: `/workspace/cetacean-005/.venv/bin/python run005.py .`
- Environment: sklearn 1.9.1, librosa 1.0.0, numpy 2.5.4.
- The box restarted during the first attempt and the cetacean-001 venv disappeared. The run was resumed with a new venv and checkpointing. The species model numbers came out identical in both attempts.
- Feature extraction is copied verbatim from run004/004b. Protocol is the same as 004b:
  - 5 seeds. Seeds change the shuffled day folds and the RF seed.
  - 95% CIs from a day-cluster bootstrap with 2,000 resamples.
- Paired difference: call minus comparator. Same day-bootstrap resamples for both models, seed-0 out-of-fold predictions.
- Stricter permutation test: block permutation within each deployment. The chronological label sequence is circularly shifted by an offset at least as long as the largest day, which keeps within-day label runs together. 200 permutations; minimum possible p is 0.005. RF uses 100 trees during permutations.
- Full numbers are in table.txt and results.json.

## Headline: species balanced accuracy, mean of 5 seeds [95% CI]
| | Day holdout | North->West | West->North |
|---|---|---|---|
| Call features, auto band, LR | .834 [.809,.860] | .844 [.815,.868] | .775 [.722,.820] |
| Call features, auto band, RF | .845 [.821,.869] | .848 [.823,.871] | .787 [.735,.830] |
| Trivial, RF (stronger floor) | .669 [.638,.700] | .540 [.507,.567] | .635 [.589,.680] |
| Trivial, LR | .543 | .511 | .500 |
| Run 003 baseline, RF | .795 | .626 | .641 |
| Run 003 baseline, LR | .739 | .500 | .543 |

Paired call minus trivial RF [95% CI]:
| | Day holdout | North->West | West->North |
|---|---|---|---|
| LR | +.166 [.135,.198] | +.306 [.269,.341] | +.138 [.090,.186] |
| RF | +.177 [.149,.206] | +.311 [.279,.343] | +.149 [.105,.197] |

- Call minus run 003 baseline RF, across deployments: +.129 to +.225, every lower bound above 0.
- Block permutation: p = .005 for all 6 call-feature tests. The largest null value was 0.666.

## Verdict
**Species:** the result now beats the trivial floor across deployments.
- Every paired-difference CI excludes 0, in both directions, for both models, against both trivial models and both baselines.
- Within a single deployment (day holdout), the run 003 baseline gets closer (.795). Across deployments it falls to about .63, while the call features hold at .78–.85.

**Ecotype:** weak.
- Call features score .46–.49 against a floor of .333, and the block permutation gives p = .005.
- Against the run 003 baseline, the paired CIs include 0 in all splits except one RF-vs-RF comparison.
- Against trivial RF, call RF beats it in every split. Against trivial LR, the North->West CI includes 0.
- **OKW recall** (seed 0): call RF .09 (day holdout), .27 (North->West), .02 (West->North). Call LR .43, .22, .40. Offshore killer whales remain poorly recognised.

## Caveats
- All variants use the annotated event start and end times.
- The 800 Hz high-pass detector under-samples low humpback calls.
- Only two deployments and two recorders, so this does not show transfer to other sites or instruments.
- Permutation RF uses 100 trees while the observed score uses 300. This makes the test slightly conservative, if anything.
- Patterns, not meaning.
