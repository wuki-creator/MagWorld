#!/usr/bin/env bash
# Build and submit the recursive MagRSI final-panel prediction.
# VCC_TOKEN must be injected into the process environment, never written here.
set -euo pipefail

ROOT=/nfs_beijing_os/zizhuo_vcc
WORK=$ROOT/work
CONTROLS=$ROOT/controls_final
CONV=$ROOT/controls_final_conv
PRED=$ROOT/preds
LOG=$ROOT/logs/final_magrsi.log
VCC=/home/zizhuo/.local/bin/vcc
CKPT=$ROOT/ckpts/v36_s543/best.pt
BASE=$PRED/v36_s543_final.h5ad
R1=$PRED/v36_s543_final_magrsi_r1.nc
REC=$PRED/v36_s543_final_magrsi_recursive.nc
FINAL=$PRED/v36_s543_final_magrsi_r2.h5ad
PACKAGE=$PRED/v36_s543_final_magrsi_r2.vcc

mkdir -p "$PRED" "$ROOT/logs" "$ROOT/work/tmp"
export TMPDIR=$ROOT/work/tmp
: "${VCC_TOKEN:?VCC_TOKEN must be supplied in the environment}"
for context in D E F; do
  test -s "$CONTROLS/context_${context}.h5ad"
done
test -s "$CONTROLS/gene_names.csv"
test -s "$CONTROLS/pert_counts.csv"
test -s "$CONTROLS/manifest.json"
test -s "$CKPT"

{
  echo "=== final MagRSI pipeline $(date -Is) ==="
  "$VCC" --version
  "$VCC" whoami || echo "warning: account probe unavailable; continuing local inference"

  echo "=== convert controls $(date -Is) ==="
  python3 "$WORK/convert_controls_generic.py" \
    --src "$CONTROLS" --dst "$CONV" --contexts D,E,F

  echo "=== v36 final prediction $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=6 python3 -u "$WORK/v14_predict.py" \
    --ckpt "$CKPT" --top-k -1 --scale 1.0 --self-scale 1.0 --max-delta 3.0 \
    --decode-style tight --jitter-shape 200 --cells-per-target 400 \
    --controls-dir "$CONV" --contexts D,E,F --genes "$CONV/gene_names.csv" \
    --perts "$CONV/pert_counts.csv" --out "$BASE"

  echo "=== support-preserving MagRSI round 1 $(date -Is) ==="
  python3 -u "$WORK/rsi_magnet_cell.py" \
    --pred "$BASE" --out "$R1" --controls "$CONV" \
    --window 512 --alpha 0.5 --theta 0.3 --block 4000

  echo "=== recursive field-axis update $(date -Is) ==="
  python3 -u "$WORK/rsi_recurse.py" \
    --nc "$R1" --out "$REC" --initial-round 1 --rounds 1 \
    --window 512 --alpha 0.5 --theta 0.3 --block 2000

  echo "=== integer sparse H5AD conversion $(date -Is) ==="
  python3 -u "$WORK/nc_to_h5ad.py" \
    --nc "$REC" --template "$BASE" --expression-key expression_r2 \
    --out "$FINAL" --block 1000

  echo "=== official VCC validation and packaging $(date -Is) ==="
  "$VCC" prep "$FINAL" -g "$CONTROLS/gene_names.csv" \
    --perts "$CONTROLS/pert_counts.csv" -o "$PACKAGE"

  echo "=== final submission $(date -Is) ==="
  "$VCC" submit "$PACKAGE" -m "MagRSI-v36-recursive-r2-final" --wait
  echo "=== FINAL_MAGRSI_DONE $(date -Is) ==="
} >> "$LOG" 2>&1
