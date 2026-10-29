#!/usr/bin/env bash
# Single predefined activation control: field_mix=0.05, gate std=0.02.
set -euo pipefail
ROOT=/nfs_beijing_os/zizhuo_vcc
WORK=$ROOT/work
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/ckpts"
cd "$WORK"

train_warm() {
  local name=$1
  local class=$2
  local layers=$3
  local gpu=$4
  mkdir -p "$ROOT/ckpts/$name"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u train_v14_aux.py \
    --signatures "$ROOT/signatures/h1_trainval_signatures.npz" \
    --official-perts "$ROOT/h1_2025/pert_counts_Training.csv" \
    --gene-embeddings "$ROOT/embeddings/vcc_gene_embeddings_hybrid_256.npy" \
    --magworld-src /home/zizhuo/maglab_deploy/src \
    --model-module model_world_h1_v39 --model-class "$class" \
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

train_warm v39_warm_mix_r1_s543 WorldModelH1V39Warm 1 1 & P1=$!
train_warm v39_warm_mix_r2_s543 WorldModelH1V39Warm 2 3 & P3=$!
train_warm v39_warm_mix_r4_s543 WorldModelH1V39Warm 4 4 & P4=$!
train_warm v39_warm_static_r4_s543 WorldModelH1V39WarmNonRecursive 4 5 & P5=$!
FAILED=0
for pid in "$P1" "$P3" "$P4" "$P5"; do wait "$pid" || FAILED=1; done
if [ "$FAILED" -ne 0 ]; then echo V39_WARM_TRAINING_FAILED; exit 1; fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/nfs_beijing_os/zizhuo_vcc/ckpts')
names = ['v39_warm_mix_r1_s543', 'v39_warm_mix_r2_s543',
         'v39_warm_mix_r4_s543', 'v39_warm_static_r4_s543']
out = {}
for name in names:
    d = json.loads((root / name / 'best.json').read_text())
    out[name] = {'best_epoch': d['best_epoch'],
                 'best_validation': d['best_validation'],
                 'baseline': d['baseline'],
                 'training_args': d['training_args']}
result = {'activation': {'field_mix': 0.05, 'gate_last_weight_std': 0.02,
                         'note': 'Predefined activation control; initial function is not v37.'},
          'models': out}
Path('/nfs_beijing_os/zizhuo_vcc/results/v39_warm_ablation_benchmark.json').write_text(
    json.dumps(result, indent=2) + '\n')
print(json.dumps({n: {'epoch': v['best_epoch'],
                      'objective': v['best_validation']['objective'],
                      'metrics': v['best_validation']['metrics']}
                  for n, v in out.items()}, indent=2))
PY
echo V39_WARM_ABLATION_DONE
