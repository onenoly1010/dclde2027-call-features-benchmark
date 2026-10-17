#!/usr/bin/env bash
# Reproduce CETACEAN-003/004/004b/005 from public NOAA audio.
# Usage: ./reproduce.sh [--local-audio DIR] [--local-annotations FILE] [--verify-only] [--runs "003 004 004b 005"]
# Downloads ~1.03 GB of FLAC (300 files) + 50 MB Annotations.csv unless --local-audio is given.
# Full rerun takes roughly 1.5-2 h on 8 cores (run 005 alone ~1 h).
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$PWD/work}"; RUNS="003 004 004b 005"; VERIFY_ONLY=0; FV_ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --local-audio) FV_ARGS+=(--local-audio "$2"); shift 2;;
  --local-annotations) FV_ARGS+=(--local-annotations "$2"); shift 2;;
  --verify-only) VERIFY_ONLY=1; shift;;
  --runs) RUNS="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
python3 tools/fetch_verify.py --manifest FILE_MANIFEST.csv --work "$WORK" "${FV_ARGS[@]}"
[ "$VERIFY_ONLY" = 1 ] && { echo "verify-only: done"; exit 0; }
if [ ! -x "$WORK/venv/bin/python" ]; then python3 -m venv "$WORK/venv"; "$WORK/venv/bin/pip" install -q -r requirements.lock.txt; fi
PY="$WORK/venv/bin/python"
for r in $RUNS; do
  OUT="$WORK/out/cetacean-$r"; mkdir -p "$OUT"
  # as-run scripts hard-code the original box paths; rewrite only the data root
  sed -e "s#/workspace/cetacean-003/data#$WORK/cetacean-003/data#g" -e "s#/workspace/cetacean-005/data#$WORK/cetacean-005/data#g" \
      "runs/cetacean-$r/run$r.py" > "$OUT/run$r.py"
  echo "== run $r -> $OUT"; ( cd "$OUT" && "$PY" "run$r.py" . 2>&1 | tee run.log )
  echo "compare: $OUT/results.json vs runs/cetacean-$r/results.json"
done
