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
dkey = np.round(np.log10(np.maximum(ds, 1e-9)), 3)
print('loaded', E.shape, flush=True)

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

# ---- 聚合到 (line, drug, dose) ----
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

# ---- 剂量平均到 (line, drug) ----
ld = defaultdict(list)
for i, (l, d, dd) in enumerate(meta): ld[(l, d)].append(i)
ld_keys = sorted(ld)
M = np.zeros((len(ld_keys), G), dtype=np.float32)
mline = np.array([k[0] for k in ld_keys]); mdrug = np.array([k[1] for k in ld_keys])
for i, k in enumerate(ld_keys): M[i] = S[ld[k]].mean(axis=0)
gstd = M.std(axis=0); top_idx = np.argsort(-gstd)[:2000]
uniq_drugs = sorted(set(mdrug)); d2i = {d: i for i, d in enumerate(uniq_drugs)}

# 各系 basal control profile
ctrl_line = {l: C[cl == l].mean(axis=0) for l in lines}

def ridge_fit(X, Y, lam=50.0):
    K = X @ X.T + lam * np.eye(X.shape[0])
    return X.T @ np.linalg.solve(K, Y)

# ============ E0: 剂量平均版噪声天花板（rep1 vs rep2, 同 (line,drug) 内剂量平均） ============
repkey = np.array(['|'.join(t) for t in zip(cl, dr, rep)])
rkd = defaultdict(lambda: defaultdict(list))  # (line,drug) -> rep -> list of row idx (in E)
for i in range(len(E)):
    l, d, r_ = cl[i], dr[i], rep[i]
    rkd[(l, d)][r_].append(i)
ceil_pairs = []
for (l, d), dd_ in rkd.items():
    reps = sorted(dd_)
    if len(reps) < 2: continue
    s1 = E[dd_[reps[0]]].mean(axis=0)
    s2 = E[dd_[reps[1]]].mean(axis=0)
    ceil_pairs.append(cor(s1, s2, top_idx))
ceiling_doseavg = dict(mean=float(np.mean(ceil_pairs)), sd=float(np.std(ceil_pairs)), n=len(ceil_pairs))
print('ceiling dose-avg', json.dumps(ceiling_doseavg), flush=True)

# ============ E1: LOCO 多方法 + 多种子少样本 + 学习曲线 ============
NSEEDS = 10
FRACS = [0.05, 0.1, 0.2, 0.4]
res = defaultdict(list)          # (method, line) -> list of r (across seeds或单次)
learn_curve = defaultdict(list)  # frac -> all r across lines/seeds

# 静态方法（与种子无关）
static_methods = ['train_mean', 'line_mean_loo', 'cross_line_mean', 'weighted_transfer', 'drug_onehot_ridge']
rstore_static = {}
for tl in lines:
    te = np.where(mline == tl)[0]; tr = np.where(mline != tl)[0]
    Ytr = M[tr]; Yte = M[te]; nte = len(te)
    mdtr = mdrug[tr]; mdte = mdrug[te]
    preds = {}
    preds['train_mean'] = np.tile(Ytr.mean(axis=0), (nte, 1))
    # line_mean 留一法: 排除目标药物本身，避免泄漏
    Plm = np.zeros_like(Yte)
    for j in range(nte):
        m2 = np.arange(nte) != j
        Plm[j] = Yte[m2].mean(axis=0)
    preds['line_mean_loo'] = Plm
    P = np.zeros_like(Yte)
    for j in range(nte):
        m = mdtr == mdte[j]
        P[j] = Ytr[m].mean(axis=0) if m.sum() else Ytr.mean(axis=0)
    preds['cross_line_mean'] = P
    # 加权迁移: 按 basal 状态相似度加权
    ctrls_tr = np.stack([ctrl_line[l] for l in mline[tr]])
    ctrl_te = ctrl_line[tl]
    # 每对 (train drug signature) 属于某系 -> 该系 basal 与 test basal 的相关
    line_sims = {}
    for l in lines:
        if l != tl:
            line_sims[l] = cor(ctrl_line[l], ctrl_te)
    Pw = np.zeros_like(Yte)
    for j in range(nte):
        m = np.where(mdtr == mdte[j])[0]
        if len(m) == 0:
            Pw[j] = Ytr.mean(axis=0); continue
        w = np.array([max(line_sims[mline[tr][i]], 0.0) + 1e-3 for i in m])
        Pw[j] = (Ytr[m] * w[:, None]).sum(axis=0) / w.sum()
    preds['weighted_transfer'] = Pw
    Ftr = np.zeros((len(tr), len(uniq_drugs)), dtype=np.float32)
    Fte = np.zeros((nte, len(uniq_drugs)), dtype=np.float32)
    for k, d in enumerate(mdtr): Ftr[k, d2i[d]] = 1
    for k, d in enumerate(mdte): Fte[k, d2i[d]] = 1
    W = ridge_fit(np.hstack([Ftr, np.ones((len(tr), 1))]), Ytr)
    preds['drug_onehot_ridge'] = np.hstack([Fte, np.ones((nte, 1))]) @ W
    for name in static_methods:
        r = pearson_rows(preds[name][:, top_idx], Yte[:, top_idx])
        rstore_static[f'{tl}|{name}'] = r
        res[(name, tl)].append(r)
    print('static done', tl, flush=True)

# 少样本适应: 多种子 + 学习曲线
fewshot_seed = defaultdict(list)  # (frac, tl, seed) -> r array
for frac in FRACS:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 + seed)
        for tl in lines:
            te = np.where(mline == tl)[0]; tr = np.where(mline != tl)[0]
            Ytr = M[tr]; Yte = M[te]
            mdtr = mdrug[tr]; mdte = mdrug[te]
            P = np.zeros_like(Yte)
            for j in range(len(te)):
                m = mdtr == mdte[j]
                P[j] = Ytr[m].mean(axis=0) if m.sum() else Ytr.mean(axis=0)
            # 按药物分组抽样校准集
            uq = sorted(set(mdte))
            perm = rng.permutation(len(uq))
            ncal = max(1, int(round(frac * len(uq))))
            cal_drugs = set(uq[i] for i in perm[:ncal])
            cal = np.array([j for j in range(len(te)) if mdte[j] in cal_drugs])
            eva = np.array([j for j in range(len(te)) if mdte[j] not in cal_drugs])
            if len(cal) >= 3 and len(eva) >= 1:
                Pc = P[cal]; Tc = Yte[cal]
                a = ((Pc * Tc).mean(axis=0) - Pc.mean(axis=0) * Tc.mean(axis=0)) / (Pc.var(axis=0) + 1e-8)
                b = Tc.mean(axis=0) - a * Pc.mean(axis=0)
                P[eva] = P[eva] * a[None, :] + b[None, :]
                r = pearson_rows(P[eva][:, top_idx], Yte[eva][:, top_idx])  # 只在未校准药物上评估
                fewshot_seed[(frac, tl, seed)] = r
                learn_curve[frac].extend(r.tolist())
        print('fewshot frac', frac, 'seed', seed, flush=True)

# 汇总少样本（frac=0.2, 所有种子）
fs20 = np.array(learn_curve[0.2])
lc_summary = {str(f): dict(mean=float(np.mean(learn_curve[f])),
                           sd=float(np.std(learn_curve[f])),
                           n=len(learn_curve[f])) for f in FRACS}
print(json.dumps(lc_summary, indent=1), flush=True)

# 静态方法汇总
summary_static = {}
for name in static_methods:
    rs = np.concatenate([rstore_static[f'{tl}|{name}'] for tl in lines])
    summary_static[name] = dict(mean=float(rs.mean()), sd=float(rs.std()), n=int(rs.size))
print(json.dumps(summary_static, indent=1), flush=True)

# ============ E2: 剂量外推（预测最高剂量 10 µM） ============
r_ext, r_ext_gm = [], []
for i, (l, d, dd) in enumerate(meta):
    if dd != 1.0: continue  # log10(10µM)=1
    lows = [j for j in range(len(meta)) if meta[j][0] == l and meta[j][1] == d and meta[j][2] < dd]
    if len(lows) < 2: continue
    lows.sort(key=lambda j: meta[j][2])
    j1, j2 = lows[-2], lows[-1]  # 最近的两个低剂量
    d1, d2 = meta[j1][2], meta[j2][2]
    slope = (S[j2] - S[j1]) / (d2 - d1)
    pred = S[j2] + slope * (dd - d2)
    r_ext.append(cor(pred, S[i], top_idx))
    r_ext_gm.append(cor(S.mean(axis=0), S[i], top_idx))
dose_extrap = dict(linear_extrap=dict(mean=float(np.mean(r_ext)), sd=float(np.std(r_ext)), n=len(r_ext)),
                   global_mean=dict(mean=float(np.mean(r_ext_gm)), sd=float(np.std(r_ext_gm)), n=len(r_ext_gm)))
print(json.dumps(dose_extrap, indent=1), flush=True)

# ============ E3: 药物检索（留一法：检索时排除目标行自身） ============
top1 = top5 = 0; tot = 0; ranks = []
for i in range(len(M)):
    d_ = mdrug[i]
    # 留一原型库: 同一药物在其他系的均值
    m = (mdrug == d_) & (np.arange(len(M)) != i)
    if m.sum() < 2: continue
    proto = {}
    for dd_ in uniq_drugs:
        mm = (mdrug == dd_) & (np.arange(len(M)) != i)
        if mm.sum() >= 2:
            proto[dd_] = M[mm].mean(axis=0)
    if d_ not in proto: continue
    Pmat = np.stack(list(proto.values())); pnames = list(proto.keys())
    Pc_ = Pmat - Pmat.mean(axis=1, keepdims=True)
    Mc_ = M[i] - M[i].mean()
    sims = Pc_ @ Mc_ / (np.sqrt((Pc_**2).sum(axis=1)) * np.sqrt((Mc_**2).sum()) + 1e-12)
    order = np.argsort(-sims)
    rank = int(np.where(np.array(pnames)[order] == d_)[0][0])
    ranks.append(rank)
    tot += 1
    if rank == 0: top1 += 1
    if rank < 5: top5 += 1
retrieval = dict(top1=top1/tot, top5=top5/tot, n=tot,
                 ranks_hist=np.histogram(ranks, bins=[0,1,5,10,50,500])[0].tolist())
print(json.dumps({k: v for k, v in retrieval.items() if k != 'ranks_hist'}, indent=1), flush=True)

# ============ E5: 签名分解 —— 系特异背景 vs 药特异残差 ============
# 每系背景 = 该系所有药签名的留一均值；残差 = 签名 - 背景
bg_var_frac = []   # 背景方差占总方差比例（每系）
res_cross_r = []   # 残差的跨系相关（同药不同系）
raw_cross_r = []   # 原始签名的跨系相关（同药不同系）
line_bg = {}
for l in lines:
    idx = np.where(mline == l)[0]
    bg = np.zeros((len(idx), G), dtype=np.float32)
    for j, ii in enumerate(idx):
        m2 = idx != ii
        bg[j] = M[idx[m2]].mean(axis=0)
    line_bg[l] = bg
    resid = M[idx] - bg
    vtot = M[idx].var(axis=0).sum()
    vbg = bg.var(axis=0).sum()
    bg_var_frac.append(float(vbg / (vtot + 1e-12)))
    # 存残差供跨系比较
    if l == lines[0]:
        resid_store = {}
    for j, ii in enumerate(idx):
        resid_store[(l, mdrug[ii])] = resid[j]
# 同药跨系: 原始 vs 残差 的相关
drug_lines = defaultdict(list)
for i in range(len(M)): drug_lines[mdrug[i]].append(i)
for d_, idxs in drug_lines.items():
    if len(idxs) < 2: continue
    for a in range(len(idxs)):
        for b in range(a+1, len(idxs)):
            i, j = idxs[a], idxs[b]
            raw_cross_r.append(cor(M[i], M[j], top_idx))
            ri = resid_store[(mline[i], d_)]; rj = resid_store[(mline[j], d_)]
            res_cross_r.append(cor(ri, rj, top_idx))
decomp = dict(bg_var_frac_mean=float(np.mean(bg_var_frac)),
              bg_var_frac_sd=float(np.std(bg_var_frac)),
              bg_var_frac_by_line=dict(zip(lines, bg_var_frac)),
              raw_cross_line_r=dict(mean=float(np.mean(raw_cross_r)), sd=float(np.std(raw_cross_r)), n=len(raw_cross_r)),
              resid_cross_line_r=dict(mean=float(np.mean(res_cross_r)), sd=float(np.std(res_cross_r)), n=len(res_cross_r)))
print(json.dumps(decomp, indent=1), flush=True)

# ============ E4: PCA 概览坐标 ============
cm = S.mean(axis=0)
U, Sx, Vt = np.linalg.svd(S[:, top_idx] - S[:, top_idx].mean(axis=0), full_matrices=False)
pc = (S[:, top_idx] - S[:, top_idx].mean(axis=0)) @ Vt[:2].T
pca_overview = dict(x=pc[:, 0].tolist(), y=pc[:, 1].tolist(),
                    line=[m[0] for m in meta], dose=[10 ** m[2] for m in meta],
                    var_explained=(Sx[:10]**2 / (Sx**2).sum()).tolist())

# ============ 保存 ============
fs20_by_line = {}
for tl in lines:
    parts = [fewshot_seed[(0.2, tl, s)] for s in range(NSEEDS) if (0.2, tl, s) in fewshot_seed and len(fewshot_seed[(0.2, tl, s)]) > 0]
    fs20_by_line[tl] = float(np.mean(np.concatenate(parts))) if parts else float('nan')
out = dict(lc_summary=lc_summary, summary_static=summary_static,
           dose_extrap=dose_extrap, retrieval=retrieval,
           ceiling_doseavg=ceiling_doseavg, decomp=decomp,
           rstore_static={k: v.tolist() for k, v in rstore_static.items()},
           fewshot20_by_line=fs20_by_line,
           pca_overview=pca_overview, lines=lines)
with open('/local_nvme_data/zizhuo/vc_fix/vc_results3.json', 'w') as f:
    json.dump(out, f)
print('ALL DONE %.1fs' % (time.time() - t0), flush=True)
