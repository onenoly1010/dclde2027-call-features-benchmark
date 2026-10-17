# Data attribution and licensing

## What licence covers what
- **Code** (`*.py`, `reproduce.sh`, `tools/`, `code/`) is under the MIT licence. See `LICENSE`.
- **Derived data and results** (everything under `data/`, `FILE_MANIFEST.csv`, `runs/*/results.json`, `runs/*/table.txt`, `runs/*/RECEIPT.md`, `runs/*/run.log`) are under **CC BY-SA 4.0**. See `LICENSE-DATA-CC-BY-SA-4.0.txt`.
  - These files are adapted from the DCLDE 2027 annotations: subsets of the annotation rows, file lists, hashes, and metrics computed from the audio.
- **No audio is redistributed.** `reproduce.sh` downloads it from NOAA's public bucket.

## Source dataset
- **DCLDE 2027 Killer Whale and other species dataset.** Fisheries and Oceans Canada (DFO) Cetacean Research Program portion, archived at NOAA National Centers for Environmental Information (NCEI).
  - DOI: https://doi.org/10.25921/15ey-mh50
  - Bucket: `gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/` (`dfo_crp/audio/`, plus `Annotations.csv`).
- **Dataset citation (as given by the paper):** Palmer, K. & Joy, R. (2025). *DCLDE 2026: Killer whale (Orcinus orca) ecotype and other species annotations for the Detection Classification Localization and Density Estimate (DCLDE) conference in 2026.* NOAA National Centers for Environmental Information. https://doi.org/10.25921/15EY-MH50
  - The bucket path says "2027"; the paper and NCEI title say "DCLDE 2026". It is the same dataset.
- **Paper:** Palmer, K. J., Cummings, E., Dowd, M. G., et al. & Joy, R. (2025). A Public Dataset of Annotated *Orcinus orca* Acoustic Signals for Detection and Ecotype Classification. *Scientific Data*. https://doi.org/10.1038/s41597-025-05281-5
  - The paper is published under CC BY-NC-ND 4.0. No text or figures from it are reproduced here; it is only cited.
- **Data provider for the audio used here:** DFO Cetacean Research Program (J. Pilkington et al.).
- **Compilation code:** https://doi.org/10.5281/zenodo.15743033

Please cite both the NCEI DOI and Palmer et al. 2025 when using anything in this repository. Credit the DFO Cetacean Research Program as data provider.

## Licence-file mismatch in the source (unresolved)
- The bucket's licence file is named `CC-BY-4.0.txt`, but its text is **CC BY-SA 4.0**.
- NCEI's catalogue record gives only warranty and liability disclaimers.
- We therefore apply the stricter reading: derived data here is released under **CC BY-SA 4.0** (ShareAlike).
- If the data owners clarify the licence as CC BY 4.0, this choice is still compatible.
- Before reuse, check the current NCEI record.

## Changes made to the source
- Selected 300 of 9,374 DFO_CRP files, deterministically (`code/select_005.py`, `code/select_subset.py`).
- Subset and normalised annotation rows. 7 events that ran past the end of their file were dropped.
- Resampled the 16,384 Hz WVanIsl recordings to 16,000 Hz in memory, for feature extraction only.
- Computed features and classification metrics.

The data providers do not endorse this work.
