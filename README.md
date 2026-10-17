# dclde2027-call-features-benchmark

Author: Kris Olofson (OINIO). Code: MIT. Derived data/results: CC BY-SA 4.0. See [ATTRIBUTION.md](ATTRIBUTION.md).

## What this is, in plain language

We asked a narrow question: **can simple acoustic features tell humpback whale sounds from killer whale sounds, and one killer whale ecotype from another, in real ocean recordings, without the classifier just learning which recorder made the recording?**

- Data: the public DCLDE 2027 dataset, DFO Cetacean Research Program portion, from NOAA's bucket.
  - Two long hydrophone deployments in British Columbia: "NorthBc" (SM2M recorder, 16 kHz) and "WVanIsl" (AURAL recorder, 16,384 Hz, resampled to 16 kHz).
  - Experts labelled each sound as humpback (HW) or killer whale (KW). Killer whales also carry an ecotype: Northern Resident (NRKW), Transient (TKW) or Offshore (OKW).
- Main scoring rule: test on **recording days the model never saw**, and on **the other deployment entirely**. Train on one recorder, test on the other, both directions.

**This is species/ecotype sound classification. It says nothing about what the sounds mean.** It is not communication, translation, or "decoding" of whale language.

## Headline result (CETACEAN-005: 300 files, 203 recording days, 11,540 species-labelled events)

Balanced accuracy, where 0.50 is chance for two classes. Values are the mean of 5 seeds; brackets are a 95% CI from resampling whole recording days.

| Features / model | Unseen days | Train NorthBc → test WVanIsl | Train WVanIsl → test NorthBc |
|---|---|---|---|
| **Call-focused features (auto band), logistic regression** | **.834** [.809, .860] | **.844** [.815, .868] | **.775** [.722, .820] |
| **Call-focused features (auto band), random forest** | **.845** [.821, .869] | **.848** [.823, .871] | **.787** [.735, .830] |
| Trivial floor: duration + peak frequency (random forest) | .669 [.638, .700] | .540 [.507, .567] | .635 [.589, .680] |
| Generic MFCC features from run 003 (random forest) | .795 [.753, .827] | .626 [.575, .674] | .641 [.564, .725] |

**Gain over the trivial floor.** Paired, same day resamples: call features minus trivial random forest.

| Model | Unseen days | NorthBc → WVanIsl | WVanIsl → NorthBc |
|---|---|---|---|
| Logistic regression | +.166 [.135, .198] | +.306 [.269, .341] | +.138 [.090, .186] |
| Random forest | +.177 [.149, .206] | +.311 [.279, .343] | +.149 [.105, .197] |

- Every paired CI excludes zero.
- A stricter permutation test shuffles labels in whole-day blocks within each deployment. It gives p = 0.005 for all 6 call-feature tests (200 permutations; 0.005 is the smallest possible p).

**Bottom line:** on this data, call-focused features separate humpback from killer whale beyond recording conditions, and this holds across the two recorders.

## What did not work (reported on purpose)

1. **Run 003: unsupervised clustering found the recorder, not the species.**
   - k-means on generic MFCC features recovered the deployment almost perfectly: ARI 0.925 (species set), 0.979 (ecotype set).
   - ARI vs species was 0.002.
   - Normalising each file removed the recorder signal, but no species signal appeared.
2. **Unsupervised methods failed throughout.** With call-focused features, clustering still gave ARI ≤ 0.06 vs species and vs ecotype. Structure appears only with labelled training data.
3. **The run 003 generic features mostly encode the recorder.** A classifier predicts deployment from them at 0.998–1.00 balanced accuracy. The call-focused features predict deployment only at about 0.55–0.58 (pilot, run 004).
4. **Ecotype is weak.**
   - Call features reach .46–.49 balanced accuracy against a .333 floor. That is above chance (block permutation p = .005).
   - But it is not reliably better than the generic run 003 features: most paired CIs include zero.
5. **Offshore killer whales are poorly recognised.** OKW recall (seed 0) is .09 / .27 / .02 for the random forest across the three splits, and .43 / .22 / .40 for logistic regression.
6. **The annotators' own frequency box did not help.** Using it leaks annotator information. That variant scored *below* the automatically estimated band (pilot, run 004).

## Caveats (read before citing)

- **Annotated start/end times are used in every variant.** The model is told *when* a call happens and only judges *what* it is. This is not a detector.
- **Detector gap below 800 Hz.** DFO labels come from a PAMGuard whistle-and-moan detector with an 800 Hz high-pass filter, so low humpback sounds are under-represented. An unlabelled stretch is not proof of silence.
- **Harmonics may be separate events.** One call can appear as several annotations (the source paper notes this for DFO data). Grouping by day stops leakage between train and test, but event counts overstate the number of independent calls.
- **Only 2 deployments and 2 recorders, both in BC.** Transfer to other sites or instruments is untested.
- **Pilot scale.** 300 of 9,374 DFO_CRP files. OKW comes from only 8 days per deployment.
- **Other limits.**
  - Single feature design, untuned models.
  - The bootstrap CIs reflect day sampling, not the choice of model.
  - Runs 003–004 had no CIs; 004b used a weaker event-level permutation test, superseded by 005.

## Runs in this repo

| Run | Data | What it does | Folder |
|---|---|---|---|
| 003 | 55 files | Unsupervised baseline (k-means/HDBSCAN), confound checks | `runs/cetacean-003` |
| 004 | 55 files | Supervised LR/RF; call-focused features (auto vs labelled band); deployment check | `runs/cetacean-004` |
| 004b | 55 files | 5 seeds, day bootstrap CIs, event-level permutation | `runs/cetacean-004b` |
| 005 | 300 files | 004b protocol at larger scale, paired differences, day-block permutation | `runs/cetacean-005` |

- Each folder holds:
  - the exact script as run (`run*.py`)
  - `results.json`, `table.txt`, `RECEIPT.md` and `run.log`
  - `SHA256SUMS.original`, the hash list written on the analysis machine. It also lists caches that are not shipped.
- `runs/cetacean-004b/table.txt` was generated at release time from its results.json.
- `code/` holds the deterministic file-selection, download/verification and annotation-subsetting scripts.
  - They contain the original absolute paths and need `Annotations.csv`; they are kept for provenance.

## Reproduce

```bash
./reproduce.sh                       # downloads ~1.03 GB FLAC + 50 MB Annotations.csv, verifies hashes, reruns all runs
./reproduce.sh --verify-only         # download + verify only
./reproduce.sh --local-audio /path/to/flacs --local-annotations /path/Annotations.csv --verify-only
./reproduce.sh --runs "005"          # rerun one run
```

- `FILE_MANIFEST.csv` lists every audio file with its public URL, GCS generation, GCS md5 and SHA-256. Audio is not stored in this repo.
- Python dependencies are pinned in `requirements.lock.txt`. Tested with Python 3.13, scikit-learn 1.9.1, librosa 1.0.0, numpy 2.5.4.
- Runs 003–004b were originally executed in an earlier environment that was lost. It had the same scikit-learn 1.9.1, but other package versions are unrecorded. Run 005 in the locked environment reproduced its own interrupted first attempt exactly. Small numeric drift in 003–004b is possible.
- A full rerun takes about 1.5–2 h on 8 cores.
- The scripts are the exact as-run versions with hard-coded box paths. `reproduce.sh` rewrites only the data-root path into `work/`.

## Citation

See `CITATION.cff`. Please also cite:
- the source dataset, NOAA NCEI doi:10.25921/15EY-MH50
- Palmer et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-05281-5
