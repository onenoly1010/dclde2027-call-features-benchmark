# CETACEAN-003 receipt (2026-10-16, America/Regina)
Data: DCLDE 2027 DFO_CRP pilot subset, 55 FLAC files, 3,061 annotated events (/workspace/cetacean-003/data, checksums in its CHECKSUMS.sha256).
Run: `/workspace/cetacean-001/.venv/bin/python run003.py .` (sklearn 1.9.1, seed 0, ~16 min). Local only: no uploads, sends, or commits.

## Method
- Both deployments were resampled to 16,000 Hz (WVanIsl starts at 16,384 Hz).
- Events are the human-annotated time boxes, capped at 10 s. The annotated frequency bounds were NOT used as features.
- Features come from the cetacean-002 extractor: 21 BASE features (spectral stats + 13 MFCC means), plus 65 X_ features (MFCC std, delta and delta-delta) in the "+deltas" rows. Signed log1p, then StandardScaler.
- k-means: k from 2 to 12, chosen by silhouette. HDBSCAN with min_cluster_size=10.
- Per-file normalization: robust z-score within each file. It uses only that file's own events and no labels.
- Levels:
  - species: HW vs KW, 3,030 events.
  - ecotype: NRKW/TKW/OKW, 2,432 events.
  - Excluded: KW_unk (from the ecotype level), AB and UndBio.
- Holdouts:
  - 5-fold GroupKFold by deployment+day.
  - Train on one deployment and test on the other, both directions.
  - Clusters are mapped to labels by majority vote on train, then applied to test.
  - HDBSCAN test points are assigned by 5-NN to train labels including noise. Coverage = the share of test points that are not noise.
- Floors: majority class, and clustering on duration + peak frequency only.
- Full table: table.txt. Everything else: results.json.

## Verdict
No separation of species or ecotype beyond recording conditions.
- Baseline k-means (k=2) recovers **deployment** almost perfectly: ARI 0.925 at species level, 0.979 at ecotype level. Its ARI vs species is only 0.002.
- Held out by day, every mapped variant is at the majority-class floor: species macro recall 0.500, accuracy 0.815 = majority.
- Deltas and per-file normalization remove the deployment signal (ARI_dep ≤0.08), but they don't replace it with species signal: ARI_species ≤0.02.
- Ecotype baseline day-CV ARI is 0.415. This is an artefact. The two clusters are the two deployments, and the deployments' ecotype mixes differ. Mapped accuracy (0.359) is below majority (0.536).
- Cross-deployment transfer is at chance in both directions.

## Caveats
- This is unsupervised clustering with silhouette-chosen k (almost always 2). That only tests whether the dominant structure is species. A supervised classifier on the same features was not run, so it remains untested.
- The extractor uses whole-band 0–8 kHz features on short boxes. These are dominated by background noise and recorder, not the call contour. The labels are gated by an 800 Hz high-pass detector.
- This is a pilot subset (55 files). Licence: the filename says CC BY 4.0 but the text says CC BY-SA 4.0. Fine for local use.
- Clusters are acoustic pattern groups, not meaning.
