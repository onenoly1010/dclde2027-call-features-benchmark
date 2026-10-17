# CETACEAN-004 receipt (2026-10-16, America/Regina)
Data: same 55-file DCLDE 2027 DFO subset as CETACEAN-003 (3,061 events). Run: `/workspace/cetacean-001/.venv/bin/python run004.py .` (seed 0, 262 s). Local only.
Holdouts: 5-fold GroupKFold by deployment+day; NorthBc->WVanIsl; WVanIsl->NorthBc. Classifiers: LR (scaled, class_weight=balanced), RF (300 trees, balanced_subsample). Floor = majority class (balanced acc 0.5 species / 0.333 ecotype) + duration+peak-freq features.
Call features: background = median spectrum of up to 3 s each side, non-event frames only (local bg found for 100% of events); band-limited, bg-subtracted spectrum; peak-frequency track (start/end/min/max/median, slope, curvature, range, jump/up fractions, 8-point contour shape, SNR). Band AUTO = bins within 10 dB of bg-subtracted peak above 200 Hz (median 289-3180 Hz). LABELLED band = annotator box -> LEAK, reported separately.
Deployment check: RF predicting deployment from the same features, day-grouped 5-fold.

## Key numbers: balanced accuracy (day-5fold | N->W | W->N); deployment check
Species (floor 0.50): trivial RF .62|.64|.65, dep .86 ; baseline003 LR .64|.50|.50 RF .62|.58|.62, dep 1.00 ;
call_AUTO LR .80|.78|.73 RF .78|.70|.76, dep .55 ; baseline+call_AUTO LR .75|.50|.51 RF .72|.79|.74, dep 1.00 ; call_LABELLED(LEAK) LR .71|.73|.67 RF .68|.69|.66, dep .55
Ecotype (floor 0.333): trivial RF .36|.31|.32 ; baseline003 LR .42|.34|.44 ; call_AUTO LR .44|.42|.45 RF .48|.46|.50, dep .58 ; LABELLED(LEAK) RF .42|.47|.41
Unsupervised on call features: ARI vs species <=0.05, mapped balanced acc ~0.5; nothing clusters by itself.
Per-class detail: table.txt; everything: results.json.

## Verdict
Supervised call-focused AUTO-band features separate humpback from killer whale beyond recording conditions: ~0.73-0.80 balanced accuracy on unseen days and on the unseen deployment, while the same features predict deployment only at 0.55 (near chance). The 003 baseline features predict deployment perfectly (1.00) but species barely (0.50-0.64), confirming 003. Ecotype: weak signal (0.44-0.50 vs 0.333), NRKW/TKW partly separable, OKW essentially not recognised (recall 0.04-0.17). Unsupervised clustering still finds no species structure. The leaky labelled band did NOT help (it scored below AUTO).
## Caveats
Event timing comes from human/detector annotation in all variants (only freq bounds treated as leak). Detector was 800 Hz high-passed. 55 files, ~24 days per deployment; no CIs computed. Mixing baseline features back in reintroduces deployment signal (LR collapses cross-deployment). Single seed, untuned. Patterns, not meaning.
