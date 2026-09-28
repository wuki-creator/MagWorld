import numpy as np, h5py, json, time
from collections import defaultdict

t0 = time.time()
import os
H5 = 'NormanWeissman2019_filtered.h5ad' if os.path.exists('NormanWeissman2019_filtered.h5ad') else '/local_nvme_data/zizhuo/vc_fix/NormanWeissman2019_filtered.h5ad'
f = h5py.File(H5, 'r')
print('top keys:', list(f.keys()), flush=True)
print('obs keys:', list(f['obs'].keys()), flush=True)

# ---- 找扰动列 ----
pert_col = None
for cand in ['perturbation', 'perturbation_name', 'pert_name', 'condition', 'guide_identity']:
    if cand in f['obs']:
        pert_col = cand
        break
print('pert col:', pert_col, flush=True)

def read_cat(grp):
    """读 h5ad categorical 列 -> str array"""
    g = f['obs'][grp]
    if isinstance(g, h5py.Group) and 'categories' in g:
        cats = g['categories'][:].astype(str)
        codes = g['codes'][:]
        out = np.empty(len(codes), dtype=object)
        for i, c in enumerate(cats):
            out[codes == i] = c
        return out.astype(str)
    return g[:].astype(str)

pert = read_cat(pert_col)
genes = f['var']['_index'][:].astype(str) if '_index' in f['var'] else f['var'][list(f['var'].keys())[0]][:].astype(str)
G = len(genes)
print('cells:', len(pert), 'genes:', G, flush=True)
uniq_pert = sorted(set(pert))
print('n conditions:', len(uniq_pert), 'first 15:', uniq_pert[:15], flush=True)

# ---- 读 X (CSR) ----
data = f['X']['data'][:]; indices = f['X']['indices'][:]; indptr = f['X']['indptr'][:]
import scipy.sparse as sp
if len(indptr) == G + 1:  # CSC（按基因列存储）
    X = sp.csc_matrix((data, indices, indptr), shape=(len(pert), G)).tocsr()
else:
    X = sp.csr_matrix((data, indices, indptr), shape=(len(pert), G))
print('X loaded', X.shape, 'nnz', X.nnz, flush=True)

# ---- 条件均值签名（处理 - 对照） ----
ctrl_names = [c for c in uniq_pert if c.lower() in ('control', 'ctrl', 'non-targeting', 'nontargeting')]
print('control name:', ctrl_names, flush=True)
ctrl = ctrl_names[0]
ctrl_idx = np.where(pert == ctrl)[0]
ctrl_mean = np.asarray(X[ctrl_idx].mean(axis=0)).ravel()

cond_mean = {}
cond_n = {}
for c in uniq_pert:
    if c == ctrl: continue
    idx = np.where(pert == c)[0]
    if len(idx) < 20: continue
    cond_mean[c] = np.asarray(X[idx].mean(axis=0)).ravel() - ctrl_mean
    cond_n[c] = len(idx)
print('usable conditions:', len(cond_mean), flush=True)

# 拆分单扰动 / 组合扰动
singles = {c for c in cond_mean if '_' not in c}
combos = {c for c in cond_mean if '_' in c and len(c.split('_')) == 2}
print('singles:', len(singles), 'combos:', len(combos), flush=True)

# 高变基因（基于签名方差）
Sall = np.stack(list(cond_mean.values()))
gstd = Sall.std(axis=0)
top_idx = np.argsort(-gstd)[:2000]

def cor(a, b, idx):
    a = a[idx] - a[idx].mean(); b = b[idx] - b[idx].mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12))

# ---- E-N1: 组合加性预测（表达空间） ----
res_expr = {}
evaluated = []
for c in sorted(combos):
    a, b = c.split('_')
    if a in singles and b in singles:
        pred = cond_mean[a] + cond_mean[b]
        res_expr[c] = cor(pred, cond_mean[c], top_idx)
        evaluated.append(c)
r_expr = np.array(list(res_expr.values()))
print('additive expr-space r: %.3f +- %.3f (n=%d)' % (r_expr.mean(), r_expr.std(), len(r_expr)), flush=True)

# ---- E-N2: 噪声天花板 —— 组合条件内细胞分半 ----
rng = np.random.default_rng(0)
ceil = []
for c in evaluated[:40]:  # 取前 40 个组合估计天花板
    idx = np.where(pert == c)[0]
    if len(idx) < 60: continue
    rng.shuffle(idx)
    h = len(idx) // 2
    s1 = np.asarray(X[idx[:h]].mean(axis=0)).ravel() - ctrl_mean
    s2 = np.asarray(X[idx[h:2*h]].mean(axis=0)).ravel() - ctrl_mean
    ceil.append(cor(s1, s2, top_idx))
ceiling = dict(mean=float(np.mean(ceil)), sd=float(np.std(ceil)), n=len(ceil))
print('ceiling:', ceiling, flush=True)

# ---- E-N3: 潜在（PCA）空间加性 vs 表达空间加性 ----
# 用所有条件签名建 PCA 基，把签名投到 k 维潜在空间后再做加性预测
names = list(cond_mean.keys())
Smat = np.stack([cond_mean[n] for n in names])[:, top_idx]
Sm_c = Smat - Smat.mean(axis=0)
U, Sx, Vt = np.linalg.svd(Sm_c, full_matrices=False)
res_lat = {}
for k in [10, 20, 50]:
    B = Vt[:k].T  # G_top x k
    proj = {n: (cond_mean[n][top_idx] - Smat.mean(axis=0)) @ B for n in names}
    rs = []
    for c in evaluated:
        a, b = c.split('_')
        pred = proj[a] + proj[b] + Smat.mean(axis=0) @ B
        t = proj[c]
        p = pred - pred.mean(); tt = t - t.mean()
        rs.append(float((p * tt).sum() / (np.sqrt((p**2).sum() * (tt**2).sum()) + 1e-12)))
    res_lat[k] = dict(mean=float(np.mean(rs)), sd=float(np.std(rs)))
print('latent additivity:', json.dumps(res_lat), flush=True)

# ---- E-N4: 非加性（互作）幅度分布 ----
# synergy metric: ||obs - (a+b)|| / ||obs||
synergy = {}
for c in evaluated:
    a, b = c.split('_')
    pred = cond_mean[a] + cond_mean[b]
    obs = cond_mean[c]
    synergy[c] = float(np.sqrt(((obs - pred)[top_idx]**2).sum()) / (np.sqrt((obs[top_idx]**2).sum()) + 1e-12))
syn = np.array(list(synergy.values()))

out = dict(n_cells=int(len(pert)), n_genes=int(G), n_singles=len(singles), n_combos=len(combos),
           n_evaluated=len(evaluated),
           additive_expr=dict(mean=float(r_expr.mean()), sd=float(r_expr.std()),
                              median=float(np.median(r_expr)),
                              frac_above_06=float((r_expr > 0.6).mean())),
           ceiling=ceiling,
           latent_additivity={str(k): v for k, v in res_lat.items()},
           synergy=dict(mean=float(syn.mean()), median=float(np.median(syn)),
                        q90=float(np.quantile(syn, 0.9))),
           per_combo={c: dict(r=res_expr[c], syn=synergy[c]) for c in evaluated},
           var_explained=(Sx[:20]**2 / (Sx**2).sum()).tolist())
OUT_JSON = 'vc_results4.json' if os.path.exists('NormanWeissman2019_filtered.h5ad') else '/local_nvme_data/zizhuo/vc_fix/vc_results4.json'
with open(OUT_JSON, 'w') as fo:
    json.dump(out, fo)
print('ALL DONE %.1fs' % (time.time() - t0), flush=True)
