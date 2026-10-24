# Verifying `runs/cetacean-005/receipt.json`

`receipt.json` is an `evidence-receipt/1.0` record of run 005. Its schema
(`schema.json`) and a standard-library verifier (`verify_receipt.py`) are in this
folder. All paths in the receipt are relative to the repository root.

## 1. Check the receipt (seconds, no download)

```bash
git clone https://github.com/onenoly1010/dclde2027-call-features-benchmark
cd dclde2027-call-features-benchmark
python3 tools/evidence-receipt/verify_receipt.py runs/cetacean-005/receipt.json
```

This checks structure, that no paths are absolute, and that the SHA-256 of
every listed script, manifest, annotation file and output matches. It does not
rerun anything and does not prove the result is correct.

## 2. Rerun run 005 (~1.03 GB download, about 1 h on 8 cores)

```bash
./reproduce.sh --runs "005"
python3 - <<'PY'
import json
a=json.load(open('runs/cetacean-005/results.json')); b=json.load(open('work/out/cetacean-005/results.json'))
print('IDENTICAL' if a==b else 'DIFFERENT')
PY
```

Requires Python 3.13 and network access to storage.googleapis.com.
`reproduce.sh` verifies every audio file's md5/SHA-256 against
`FILE_MANIFEST.csv` and installs `requirements.lock.txt` into `work/venv`.

## What a match does and does not mean

`reproduction.status` is written by the author. A match on your own machine is
an independent reproduction; please report it (match or mismatch) as a GitHub
issue. A match shows the computation is repeatable, not that the scientific
claim is right; see `failures_and_limits` and `interpretation_note`.
