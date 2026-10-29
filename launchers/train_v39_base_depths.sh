#!/usr/bin/env bash
# Closed-field controls needed to separate depth effects from v39 field effects.
set -euo pipefail
ROOT=/nfs_beijing_os/zizhuo_vcc
WORK=$ROOT/work
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/ckpts"
cd "$WORK"

train_base() {
  local layers=$1
  local gpu=$2
  local name=v39_base_r${layers}_s543
  mkdir -p "$ROOT/ckpts/$name"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u train_v14_aux.py \
    --signatures "$ROOT/signatures/h1_trainval_signatures.npz" \
    --official-perts "$ROOT/h1_2025/pert_counts_Training.csv" \
    --gene-embeddings "$ROOT/embeddings/vcc_gene_embeddings_hybrid_256.npy" \
    --magworld-src /home/zizhuo/maglab_deploy/src \
    --model-module model_world_h1_v39 --model-class WorldModelH1V39Base \
    --out "$ROOT/ckpts/$name/best.pt" \
    --holdout-mode random --random-holdout 50 --epochs 60 --patience 20 \
    --batch-size 8 --learning-rate 0.0001 --weight-decay 0.0001 \
    --top-k 100 --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 \
    --lambda-kernel 0.05 --lambda-shared-bias 0.05 --lambda-de 1.0 \
    --lambda-cosine 0.5 --lambda-rank 0.1 --lambda-pds 0.05 --lambda-yield 0.15 \
    --latent-dim 64 --hidden-dim 192 --d-dir 32 --graph-k 50 \
    --n-layers "$layers" --response-rank 64 --max-delta 1.5 --context-strength 0.25 \
    --langevin-init 4.0 --kernel-powers 4 --seed 543 \
    --dgidb-features "$ROOT/embeddings/dgidb_prior_features.npy" \
    --prior-graphs "$ROOT/embeddings/dgidb_graph.npz,$ROOT/embeddings/trrust_graph.npz" \
    --init-from "$ROOT/ckpts/v4_seed337" --freeze-base \
    > "$ROOT/logs/$name.log" 2>&1
}

(train_base 1 6; train_base 8 6) & P6=$!
train_base 2 7 & P7=$!
FAILED=0
for pid in "$P6" "$P7"; do wait "$pid" || FAILED=1; done
if [ "$FAILED" -ne 0 ]; then echo V39_BASE_DEPTH_TRAINING_FAILED; exit 1; fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/nfs_beijing_os/zizhuo_vcc/ckpts')
out = {}
for layers in (1, 2, 8):
    name = f'v39_base_r{layers}_s543'
    d = json.loads((root / name / 'best.json').read_text())
    out[name] = {'best_epoch': d['best_epoch'],
                 'best_validation': d['best_validation'],
                 'training_args': d['training_args']}
Path('/nfs_beijing_os/zizhuo_vcc/results/v39_base_depth_benchmark.json').write_text(
    json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY
echo V39_BASE_DEPTH_BENCHMARK_DONE
