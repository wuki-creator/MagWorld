import numpy as np, zipfile, io, json, time
from collections import defaultdict

t0 = time.time()
NPZ = '/home/zizhuo/cross_cell_vcc_v2/data/sciplex_full_v2.npz'
z = zipfile.ZipFile(NPZ)
def load(n): return np.load(io.BytesIO(z.read(n)), allow_pickle=True)

C = load('controls.npy'); E = load('effects.npy')
cl = load('cell_line.npy').astype(str)
dr = load('drug.npy').astype(str)
ds = load('dose.npy').astype(float)
rep = np.char.replace(load('replicate.npy').astype(str), '_', '')
genes = load('genes.npy').astype(str)
G = E.shape[1]
dkey = np.round(np.log10(np.maximum(ds, 1e-9)), 3)  # 剂量取 log10 后取整避免浮点问题
print('loaded', E.shape, 'dose keys:', sorted(set(dkey)), flush=True)

def pearson_rows(A, B):
    A = A - A.mean(axis=1, keepdims=True); B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A**2).sum(axis=1) * (B**2).sum(axis=1)) + 1e-12
    return num / den

def cor(a, b, idx=None):
    if idx is not None:
        a = a[idx]; b = b[idx]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12
    return float((a * b).sum() / d)

# ---------- 0. 聚合 + 噪声天花板 ----------
key = np.array(['|'.join(t) for t in zip(cl, dr, dkey.astype(str))])
uniq = sorted(set(key))
S = np.zeros((len(uniq), G), dtype=np.float32)
meta = []
for i, k in enumerate(uniq):
    m = key == k
    S[i] = E[m].mean(axis=0)
    l, d, dd = k.split('|')
    meta.append((l, d, float(dd)))
lines = sorted(set(m[0] for m in meta))
print('aggregated:', S.shape, flush=True)

nc = []
for i, k in enumerate(uniq):
    m = np.where(key == k)[0]
    r1 = [j for j in m if rep[j] == 'rep1']; r2 = [j for j in m if rep[j] == 'rep2']
    if r1 and r2:
        nc.append(cor(E[r1].mean(axis=0), E[r2].mean(axis=0)))
noise_ceiling = dict(mean=float(np.mean(nc)), sd=float(np.std(nc)), n=len(nc))
print('noise ceiling:', noise_ceiling, flush=True)

# ---------- 1. LOCO 跨细胞系预测 ----------
ld = defaultdict(list)
for i, (l, d, dd) in enumerate(meta): ld[(l, d)].append(i)
ld_keys = sorted(ld)
M = np.zeros((len(ld_keys), G), dtype=np.float32)
mline = np.array([k[0] for k in ld_keys]); mdrug = np.array([k[1] for k in ld_keys])
for i, k in enumerate(ld_keys): M[i] = S[ld[k]].mean(axis=0)
gstd = M.std(axis=0); top_idx = np.argsort(-gstd)[:2000]

uniq_drugs = sorted(set(mdrug)); d2i = {d: i for i, d in enumerate(uniq_drugs)}
ctrl_line = {l: C[cl == l].mean(axis=0) for l in lines}

def ridge_fit(X, Y, lam=50.0):
    K = X @ X.T + lam * np.eye(X.shape[0])
    return X.T @ np.linalg.solve(K, Y)

res_loco = {}; rstore = {}
rng = np.random.default_rng(7)
for tl in lines:
    te = np.where(mline == tl)[0]; tr = np.where(mline != tl)[0]
    Ytr = M[tr]; Yte = M[te]; nte = len(te)
    mdtr = mdrug[tr]; mdte = mdrug[te]
    preds = {}
    preds['train_mean'] = np.tile(Ytr.mean(axis=0, keepdims=True), (nte, 1))
    P = np.zeros_like(Yte)
    for j in range(nte):
        m = mdtr == mdte[j]
        P[j] = Ytr[m].mean(axis=0) if m.sum() else Ytr.mean(axis=0)
    preds['cross_line_mean'] = P
    Ftr = np.zeros((len(tr), len(uniq_drugs)), dtype=np.float32)
    Fte = np.zeros((nte, len(uniq_drugs)), dtype=np.float32)
    for k, d in enumerate(mdtr): Ftr[k, d2i[d]] = 1
    for k, d in enumerate(mdte): Fte[k, d2i[d]] = 1
    W = ridge_fit(np.hstack([Ftr, np.ones((len(tr), 1))]), Ytr)
    preds['drug_onehot_ridge'] = np.hstack([Fte, np.ones((nte, 1))]) @ W
    # few-shot 适应: 20% 测试药物做校准，学每基因缩放+平移，应用到其余 80%
    cal, eva = [], []
    for d in sorted(set(mdte)):
        idx = np.where(mdte == d)[0]
        (cal if rng.random() < 0.2 else eva).extend(idx.tolist())
    cal = np.array(cal, dtype=int); eva = np.array(eva, dtype=int)
    Pfs = preds['cross_line_mean'].copy()
    if len(cal) >= 3 and len(eva) >= 3:
        Pc = preds['cross_line_mean'][cal]; Tc = M[te][cal]
        # 每基因一元回归 T ~ a*P + b
        a = ((Pc * Tc).mean(axis=0) - Pc.mean(axis=0) * Tc.mean(axis=0)) / (Pc.var(axis=0) + 1e-8)
        b = Tc.mean(axis=0) - a * Pc.mean(axis=0)
        Pfs[eva] = Pfs[eva] * a[None, :] + b[None, :]
    preds['vc_fewshot_adapted'] = Pfs
    for name, P in preds.items():
        r = pearson_rows(P[:, top_idx], Yte[:, top_idx])
        res_loco[f'{tl}|{name}'] = dict(n=nte, mean=float(r.mean()), sd=float(r.std()))
        rstore[f'{tl}|{name}'] = r
    print('LOCO done', tl, flush=True)

names = ['train_mean', 'cross_line_mean', 'drug_onehot_ridge', 'vc_fewshot_adapted']
summary_loco = {}
for name in names:
    rs = np.concatenate([rstore[f'{tl}|{name}'] for tl in lines])
    summary_loco[name] = dict(mean=float(rs.mean()), sd=float(rs.std()), n=int(rs.size))
print(json.dumps(summary_loco, indent=1), flush=True)

# 示例散点: A172 上 few-shot 适应后提升最大的药物
tl0 = 'A172'
te0 = np.where(mline == tl0)[0]
trm = mline != tl0
mdtr0 = mdrug[trm]; mdte0 = mdrug[te0]
Pclm = np.zeros((len(te0), G), dtype=np.float32)
for j in range(len(te0)):
    m = mdtr0 == mdte0[j]
    Pclm[j] = M[trm][m].mean(axis=0) if m.sum() else M[trm].mean(axis=0)
rng3 = np.random.default_rng(7)
cal, eva = [], []
for d in sorted(set(mdte0)):
    idx = np.where(mdte0 == d)[0]
    (cal if rng3.random() < 0.2 else eva).extend(idx.tolist())
cal = np.array(cal, dtype=int); eva = np.array(eva, dtype=int)
Pfs = Pclm.copy()
if len(cal) >= 3:
    Pc = Pclm[cal]; Tc = M[te0][cal]
    a = ((Pc * Tc).mean(axis=0) - Pc.mean(axis=0) * Tc.mean(axis=0)) / (Pc.var(axis=0) + 1e-8)
    b = Tc.mean(axis=0) - a * Pc.mean(axis=0)
    Pfs[eva] = Pfs[eva] * a[None, :] + b[None, :]
r_clm0 = pearson_rows(Pclm[:, top_idx], M[te0][:, top_idx])
r_fs0 = pearson_rows(Pfs[:, top_idx], M[te0][:, top_idx])
j0 = int(np.argmax(r_fs0 - r_clm0))
sub = top_idx[::4]
scatter = dict(line=tl0, drug=str(mdte0[j0]),
               r_clm=float(r_clm0[j0]), r_fs=float(r_fs0[j0]),
               true=M[te0][j0][sub].tolist(), pred_clm=Pclm[j0][sub].tolist(),
               pred_fs=Pfs[j0][sub].tolist())

# ---------- 2. 剂量轴连续插值 ----------
dose_levels = sorted(set(m[2] for m in meta))
dose_interp = {}
for hold in dose_levels[1:-1]:
    r_lin, r_mean = [], []
    for i, (l, d, dd) in enumerate(meta):
        if dd != hold: continue
        lo = [j for j in range(len(meta)) if meta[j][0] == l and meta[j][1] == d and meta[j][2] < hold]
        hi = [j for j in range(len(meta)) if meta[j][0] == l and meta[j][1] == d and meta[j][2] > hold]
        if not lo or not hi: continue
        lo_d = max(meta[j][2] for j in lo); hi_d = min(meta[j][2] for j in hi)
        Slo = S[[j for j in lo if meta[j][2] == lo_d]].mean(axis=0)
        Shi = S[[j for j in hi if meta[j][2] == hi_d]].mean(axis=0)
        w = (hold - lo_d) / (hi_d - lo_d)  # log 剂量轴上线性
        pred = (1 - w) * Slo + w * Shi
        r_lin.append(cor(pred, S[i], top_idx))
        r_mean.append(cor(S.mean(axis=0), S[i], top_idx))
    dose_interp[str(10 ** hold)] = dict(
        linear=dict(mean=float(np.nanmean(r_lin)), sd=float(np.nanstd(r_lin)), n=len(r_lin)),
        global_mean=dict(mean=float(np.nanmean(r_mean)), sd=float(np.nanstd(r_mean)), n=len(r_mean)))
print(json.dumps(dose_interp, indent=1), flush=True)

# 示例剂量曲线
strength = np.abs(S[:, top_idx]).mean(axis=1)
bi = int(np.argmax(strength)); bl, bd, bdd = meta[bi]
idxs = sorted([i for i in range(len(meta)) if meta[i][0] == bl and meta[i][1] == bd], key=lambda i: meta[i][2])
top10 = top_idx[np.argsort(-np.abs(S[bi, top_idx]))[:10]]
ex_curves = dict(line=bl, drug=bd, doses=[10 ** meta[i][2] for i in idxs],
                 gene_names=genes[top10].tolist(),
                 values=[[float(S[i, g]) for i in idxs] for g in top10])

# ---------- 3. 流形 kNN 去噪（限同一细胞系内邻居） ----------
m1 = rep == 'rep1'
X1 = E[m1]; k1 = key[m1]; cl1 = cl[m1]
truth = {key[j]: E[j] for j in np.where(rep == 'rep2')[0]}
cm2 = X1.mean(axis=0)
U, Sx, Vt = np.linalg.svd(X1 - cm2, full_matrices=False)
Z50 = (X1 - cm2) @ Vt[:50].T
rng2 = np.random.default_rng(0)
sel = rng2.choice(len(X1), size=min(400, len(X1)), replace=False)
imp_raw, imp_smooth = [], []
for j in sel:
    if k1[j] not in truth: continue
    same = np.where(cl1 == cl1[j])[0]
    Dj = ((Z50[same] - Z50[j]) ** 2).sum(axis=1)
    nbrs = same[np.argsort(Dj)[1:16]]
    t = truth[k1[j]]
    imp_raw.append(cor(X1[j], t, top_idx))
    imp_smooth.append(cor(X1[nbrs].mean(axis=0), t, top_idx))
knn_result = dict(raw=dict(mean=float(np.mean(imp_raw)), sd=float(np.std(imp_raw)), n=len(imp_raw)),
                  manifold_knn=dict(mean=float(np.mean(imp_smooth)), sd=float(np.std(imp_smooth)), n=len(imp_smooth)),
                  paired_delta=float(np.mean(np.array(imp_smooth) - np.array(imp_raw))))
print(json.dumps(knn_result, indent=1), flush=True)

# ---------- 保存 ----------
out = dict(noise_ceiling=noise_ceiling, nc_values=[float(x) for x in nc], loco=res_loco, summary_loco=summary_loco,
           dose_interp=dose_interp, ex_curves=ex_curves, knn=knn_result, scatter=scatter,
           rstore={k: v.tolist() for k, v in rstore.items()},
           n_agg=int(S.shape[0]), n_genes=int(G), lines=lines, top_genes=genes[top_idx[:30]].tolist())
with open('/tmp/vc_results2.json', 'w') as f:
    json.dump(out, f)
print('ALL DONE %.1fs' % (time.time() - t0), flush=True)
