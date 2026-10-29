#!/usr/bin/env bash
# Matched v39 neural field-mixture and recursive-recombination ablation.
set -euo pipefail
ROOT=/nfs_beijing_os/zizhuo_vcc
WORK=$ROOT/work
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/ckpts"
cd "$WORK"

train_model() {
  local name=$1
  local module=$2
  local class=$3
  local layers=$4
  local gpu=$5
  mkdir -p "$ROOT/ckpts/$name"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u train_v14_aux.py \
    --signatures "$ROOT/signatures/h1_trainval_signatures.npz" \
    --official-perts "$ROOT/h1_2025/pert_counts_Training.csv" \
    --gene-embeddings "$ROOT/embeddings/vcc_gene_embeddings_hybrid_256.npy" \
    --magworld-src /home/zizhuo/maglab_deploy/src \
    --model-module "$module" \
    --model-class "$class" \
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

# GPUs 0 and 2 are left untouched. Each queue runs two independent jobs at most.
(train_model v37_base_r4_s543 model_world_h1_v39 WorldModelH1V39Base 4 1; \
 train_model v39_pole_r4_s543 model_world_h1_v39 WorldModelH1V39Pole 4 1) & P1=$!
(train_model v39_mix_r1_s543 model_world_h1_v39 WorldModelH1V39 1 3; \
 train_model v39_dipole_r4_s543 model_world_h1_v39 WorldModelH1V39Dipole 4 3) & P3=$!
(train_model v39_mix_r2_s543 model_world_h1_v39 WorldModelH1V39 2 4; \
 train_model v39_local_r4_s543 model_world_h1_v39 WorldModelH1V39Local 4 4) & P4=$!
(train_model v39_mix_r4_s543 model_world_h1_v39 WorldModelH1V39 4 5; \
 train_model v39_free_r4_s543 model_world_h1_v39 WorldModelH1V39Free 4 5) & P5=$!
(train_model v39_mix_r8_s543 model_world_h1_v39 WorldModelH1V39 8 6) & P6=$!
(train_model v39_static_r4_s543 model_world_h1_v39 WorldModelH1V39NonRecursive 4 7) & P7=$!

FAILED=0
for pid in "$P1" "$P3" "$P4" "$P5" "$P6" "$P7"; do
  wait "$pid" || FAILED=1
done
if [ "$FAILED" -ne 0 ]; then
  echo V39_ABLATION_TRAINING_FAILED
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/nfs_beijing_os/zizhuo_vcc/ckpts')
names = [
    'v37_base_r4_s543',
    'v39_mix_r1_s543', 'v39_mix_r2_s543',
    'v39_mix_r4_s543', 'v39_mix_r8_s543',
    'v39_static_r4_s543', 'v39_pole_r4_s543',
    'v39_dipole_r4_s543', 'v39_local_r4_s543', 'v39_free_r4_s543',
]
out = {}
for name in names:
    d = json.loads((root / name / 'best.json').read_text())
    out[name] = {
        'best_epoch': d['best_epoch'],
        'best_validation': d['best_validation'],
        'baseline': d['baseline'],
        'training_args': d['training_args'],
    }
summary = {
    name: {
        'best_epoch': value['best_epoch'],
        'objective': value['best_validation']['objective'],
        'pearson': value['best_validation']['metrics']['pearson'],
        'cosine': value['best_validation']['metrics']['cosine'],
        'nmae': value['best_validation']['metrics']['nmae'],
        'direction_fidelity': value['best_validation']['metrics']['direction_fidelity'],
    }
    for name, value in out.items()
}
result = {'protocol': {'seed': 543, 'random_holdout': 50,
                       'note': 'Validation holdout is used for checkpoint selection, not an independent test set.'},
          'summary': summary, 'models': out}
Path('/nfs_beijing_os/zizhuo_vcc/results/v39_ablation_benchmark.json').write_text(
    json.dumps(result, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
echo V39_ABLATION_BENCHMARK_DONE
