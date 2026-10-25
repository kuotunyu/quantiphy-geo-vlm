#!/usr/bin/env bash
# Stage 13 (README "Per-question version"): wait until no other python process uses the GPU, then run one split
# of methods.geometry_iq.run and its combine. Usage: SPLIT=val bash scripts/geometry_iq_wait_run.sh
cd "$(dirname "$0")/.." || exit 1
SPLIT=${SPLIT:-val}
LOG=runs/geometry_iq_${SPLIT}.log
mkdir -p runs
echo "waiting for a free GPU ($(date '+%m/%d %H:%M:%S'))" >> "$LOG"
free=0
until [ "$free" -ge 2 ]; do
  py=$(nvidia-smi --query-compute-apps=name --format=csv,noheader | grep -ci python)
  if [ "$py" -eq 0 ]; then free=$((free+1)); else free=0; fi
  sleep 30
done
echo "start $(date '+%m/%d %H:%M:%S'); $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used" >> "$LOG"
code=0
for st in detect solve verify decide; do  # one process per stage: the GPU stages never share a CUDA context
  uv run python -m methods.geometry_iq.run --split "$SPLIT" --stage "$st" --prereg README.md >> "$LOG" 2>&1
  code=$?
  echo "stage $st exit $code ($(date '+%m/%d %H:%M:%S'))" >> "$LOG"
  [ "$code" -ne 0 ] && break
done
echo "run exit $code ($(date '+%m/%d %H:%M:%S'))" >> "$LOG"
if [ "$code" -eq 0 ]; then
  uv run python -m methods.geometry_iq.combine --split "$SPLIT" > "runs/geometry_iq_${SPLIT}_combine.log" 2>&1
  echo "combine exit $? ($(date '+%m/%d %H:%M:%S'))" >> "$LOG"
fi
echo "done $(date '+%m/%d %H:%M:%S')" >> "$LOG"
