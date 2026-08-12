# -*- coding: utf-8 -*-
"""
数据预处理：Norman 2019 (scPerturb 版) → 训练/测试张量（低内存版）
  - backed 模式 + 小行块读取，绝不整体稠密化
  - 分块计算基因方差，选 top-G 高变基因，分块构建 HVG 子矩阵
  - 单基因扰动随机划分 train/test（未见扰动泛化设定）
  - 训练样本 = 条件内 mini-bulk（bootstrap 细胞组均值），输入为对照 bootstrap 均值
输出: data/prepared.npz
"""
import numpy as np
import anndata as ad
import scipy.sparse as sp

H5AD = 'data/norman.h5ad'
N_HVG = 1200
N_TEST_SINGLE = 20
MIN_CELLS = 50
CTRL_EVAL = 400
COND_EVAL = 150
N_REP = 6
BULK_N = 64
CHUNK = 500
RNG = np.random.default_rng(42)

print('loading (backed)...', flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
n, g = adata.shape
print('shape:', adata.shape, flush=True)
pert_col = next(c for c in ['perturbation', 'condition'] if c in adata.obs.columns)
pert = adata.obs[pert_col].astype(str).values
print('perturbation col:', pert_col, '| conditions:', len(np.unique(pert)), flush=True)

X = adata.X
print('X type:', type(X).__name__, flush=True)


def get_chunk(i0, i1):
    blk = X[i0:i1]
    if sp.issparse(blk):
        blk = blk.toarray()
    return np.asarray(blk, dtype=np.float32)


# ---- 分块方差 ----
print('computing variance...', flush=True)
s1 = np.zeros(g, dtype=np.float64)
s2 = np.zeros(g, dtype=np.float64)
for i in range(0, n, CHUNK):
    blk = get_chunk(i, min(i + CHUNK, n))
    s1 += blk.sum(0)
    s2 += (blk.astype(np.float64) ** 2).sum(0)
    del blk
mean = s1 / n
var = s2 / n - mean ** 2
print('expression stats: mean range', float(mean.min()), float(mean.max()),
      '(判断已log化)' if mean.max() < 20 else '(似为原始计数)', flush=True)
hvg = np.argsort(-var)[:N_HVG]
hvg.sort()
gene_names = np.array(adata.var_names)[hvg]
del s1, s2, mean, var

# ---- 分块构建 HVG 子矩阵 ----
print('building HVG matrix...', flush=True)
Xh = np.empty((n, N_HVG), dtype=np.float32)
for i in range(0, n, CHUNK):
    blk = get_chunk(i, min(i + CHUNK, n))
    Xh[i:i + blk.shape[0]] = blk[:, hvg]
    del blk
print('HVG matrix:', Xh.shape, f'{Xh.nbytes/1e6:.0f}MB', flush=True)
adata.file.close()
del adata, X

is_ctrl = pert == 'control'
ctrl_idx = np.where(is_ctrl)[0]
conds = np.array(sorted(set(pert) - {'control'}))


def ncells(c):
    return int(np.sum(pert == c))


singles = [c for c in conds if '+' not in c and c in set(gene_names)
           and ncells(c) >= MIN_CELLS]
doubles = [c for c in conds if '+' in c and ncells(c) >= MIN_CELLS]
print(f'singles: {len(singles)}, doubles: {len(doubles)}', flush=True)

test_singles = list(RNG.choice(singles, size=N_TEST_SINGLE, replace=False))
train_singles = [c for c in singles if c not in test_singles]
test_doubles = [c for c in doubles
                if all(x in train_singles for x in c.split('+'))]
print(f'train: {len(train_singles)}, test singles: {len(test_singles)}, '
      f'test doubles: {len(test_doubles)}', flush=True)

# ---- 训练 mini-bulk 样本 ----
train_x, train_y, train_pert = [], [], []
for c in train_singles:
    idx = np.where(pert == c)[0]
    for _ in range(N_REP):
        cb = RNG.choice(ctrl_idx, size=BULK_N, replace=len(ctrl_idx) < BULK_N)
        tb = RNG.choice(idx, size=BULK_N, replace=len(idx) < BULK_N)
        train_x.append(Xh[cb].mean(0))
        train_y.append(Xh[tb].mean(0))
        train_pert.append(c)
train_x = np.stack(train_x).astype(np.float32)
train_y = np.stack(train_y).astype(np.float32)
print('train samples:', train_x.shape, flush=True)

# ---- 评测数据 ----
ctrl_eval = RNG.choice(ctrl_idx, size=CTRL_EVAL, replace=False)
out = {
    'gene_names': gene_names,
    'train_conds': np.array(train_singles),
    'train_x': train_x, 'train_y': train_y,
    'train_pert': np.array(train_pert),
    'ctrl_mean': Xh[is_ctrl].mean(0).astype(np.float32),
    'cells_ctrl': Xh[ctrl_eval].astype(np.float32),
    'test_conds': np.array(test_singles + test_doubles),
}
for c in test_singles + test_doubles:
    idx = np.where(pert == c)[0]
    take = idx if len(idx) <= COND_EVAL else RNG.choice(idx, COND_EVAL, replace=False)
    out[f'condmean::{c}'] = Xh[idx].mean(0).astype(np.float32)
    out[f'condcells::{c}'] = Xh[take].astype(np.float32)

np.savez_compressed('data/prepared.npz', **out)
print('saved data/prepared.npz', flush=True)
