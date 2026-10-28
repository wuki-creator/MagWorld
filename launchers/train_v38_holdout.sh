#!/usr/bin/env bash
# Matched held-out comparison: v37 backbone versus v38 pole--field--cell model.
set -euo pipefail
ROOT=/nfs_beijing_os/zizhuo_vcc
WORK=$ROOT/work
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/ckpts/v37_random50_s543" "$ROOT/ckpts/v38_random50_s543"
cd "$WORK"

train_model() {
  local version=$1
  local gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python3 -u train_v14_aux.py \
    --signatures "$ROOT/signatures/h1_trainval_signatures.npz" \
    --official-perts "$ROOT/h1_2025/pert_counts_Training.csv" \
    --gene-embeddings "$ROOT/embeddings/vcc_gene_embeddings_hybrid_256.npy" \
    --magworld-src /home/zizhuo/maglab_deploy/src \
    --model-module "model_world_h1_v${version}" \
    --model-class "WorldModelH1V${version}" \
    --out "$ROOT/ckpts/v${version}_random50_s543/best.pt" \
    --holdout-mode random --random-holdout 50 --epochs 60 --patience 20 \
    --batch-size 8 --learning-rate 0.0001 --weight-decay 0.0001 \
    --top-k 100 --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 \
    --lambda-kernel 0.05 --lambda-shared-bias 0.05 --lambda-de 1.0 \
    --lambda-cosine 0.5 --lambda-rank 0.1 --lambda-pds 0.05 --lambda-yield 0.15 \
    --latent-dim 64 --hidden-dim 192 --d-dir 32 --graph-k 50 \
    --n-layers 3 --response-rank 64 --max-delta 1.5 --context-strength 0.25 \
    --langevin-init 4.0 --kernel-powers 4 --seed 543 \
    --dgidb-features "$ROOT/embeddings/dgidb_prior_features.npy" \
    --prior-graphs "$ROOT/embeddings/dgidb_graph.npz,$ROOT/embeddings/trrust_graph.npz" \
    --init-from "$ROOT/ckpts/v4_seed337" --freeze-base \
    > "$ROOT/logs/v${version}_random50_s543.log" 2>&1
}

train_model 37 5 & P37=$!
train_model 38 6 & P38=$!
wait "$P37"
wait "$P38"
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/nfs_beijing_os/zizhuo_vcc/ckpts')
out = {}
for version in (37, 38):
    p = root / f'v{version}_random50_s543' / 'best.json'
    d = json.loads(p.read_text())
    out[f'v{version}'] = {
        'best_epoch': d['best_epoch'],
        'best_validation': d['best_validation'],
        'baseline': d['baseline'],
        'training_args': d['training_args'],
    }
Path('/nfs_beijing_os/zizhuo_vcc/results/v38_holdout_benchmark.json').write_text(
    json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY
echo V38_HOLDOUT_BENCHMARK_DONE
