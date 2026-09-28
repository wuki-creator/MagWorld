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
G = E.shape[1]
dkey = np.round(np.log10(np.maximum(ds, 1e-9)), 3)
print('loaded', E.shape, flush=True)

def cor(a, b, idx=None):
    if idx is not None:
        a = a[idx]; b = b[idx]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12
    return float((a * b).sum() / d)

# 聚合到 (line, drug, dose) 再到 (line, drug)
key = np.array(['|'.join(t) for t in zip(cl, dr, dkey.astype(str))])
uniq = sorted(set(key))
S = np.zeros((len(uniq), G), dtype=np.float32)
meta = []
for i, k in enumerate(uniq):
    m = key == k
    S[i] = E[m].mean(axis=0)
    l, d, dd = k.split('|')
    meta.append((l, d, float(dd)))
ld = defaultdict(list)
for i, (l, d, dd) in enumerate(meta): ld[(l, d)].append(i)
ld_keys = sorted(ld)
M = np.zeros((len(ld_keys), G), dtype=np.float32)
mline = np.array([k[0] for k in ld_keys]); mdrug = np.array([k[1] for k in ld_keys])
for i, k in enumerate(ld_keys): M[i] = S[ld[k]].mean(axis=0)
gstd = M.std(axis=0); top_idx = np.argsort(-gstd)[:2000]
lines = sorted(set(mline))
uniq_drugs = sorted(set(mdrug))
drugs_per_line = {l: int((mline == l).sum()) for l in lines}
print('drugs per line:', drugs_per_line, flush=True)

# ---- 背景能量占比（逐签名）: s = bg + resid, frac = ||bg||^2 / (||bg||^2 + ||resid||^2) ----
bg_frac_energy = []
R = np.zeros_like(M)  # 残差矩阵
BG = np.zeros_like(M)
for l in lines:
    idx = np.where(mline == l)[0]
    for ii in idx:
        m2 = (mline == l) & (np.arange(len(M)) != ii)
        bg = M[m2].mean(axis=0)
        BG[ii] = bg
        R[ii] = M[ii] - bg
for ii in range(len(M)):
    e_bg = (BG[ii][top_idx]**2).sum(); e_rs = (R[ii][top_idx]**2).sum()
    bg_frac_energy.append(float(e_bg / (e_bg + e_rs + 1e-12)))
bg_energy = dict(mean=float(np.mean(bg_frac_energy)), sd=float(np.std(bg_frac_energy)),
                 by_line={l: float(np.mean([bg_frac_energy[i] for i in range(len(M)) if mline[i] == l])) for l in lines})
print('bg energy frac:', bg_energy['mean'], flush=True)

# ---- 残差上的跨系药物检索（留一） vs 原始签名检索 ----
def loo_retrieval(Xmat, top_idx):
    top1 = top5 = 0; tot = 0; ranks = []
    for i in range(len(Xmat)):
        d_ = mdrug[i]
        proto = {}
        for dd_ in uniq_drugs:
            mm = (mdrug == dd_) & (np.arange(len(Xmat)) != i)
            if mm.sum() >= 2:
                proto[dd_] = Xmat[mm].mean(axis=0)
        if d_ not in proto: continue
        Pmat = np.stack(list(proto.values())); pnames = list(proto.keys())
        Pc = Pmat[:, top_idx] - Pmat[:, top_idx].mean(axis=1, keepdims=True)
        Mc = Xmat[i][top_idx] - Xmat[i][top_idx].mean()
        sims = Pc @ Mc / (np.sqrt((Pc**2).sum(axis=1)) * np.sqrt((Mc**2).sum()) + 1e-12)
        rank = int(np.where(np.array(pnames)[np.argsort(-sims)] == d_)[0][0])
        ranks.append(rank); tot += 1
        if rank == 0: top1 += 1
        if rank < 5: top5 += 1
    return dict(top1=top1/tot, top5=top5/tot, n=tot,
                mean_rank=float(np.mean(ranks)), median_rank=float(np.median(ranks)),
                n_proto=len(set(mdrug[i] for i in range(len(Xmat)))),
                ranks=ranks)

retr_raw = loo_retrieval(M, top_idx)
retr_resid = loo_retrieval(R, top_idx)
print('retr raw:', retr_raw['top1'], retr_raw['top5'], ' resid:', retr_resid['top1'], retr_resid['top5'], flush=True)

# ---- Spearman-Brown 提升天花板（双重复均值目标的上限） ----
rkd = defaultdict(lambda: defaultdict(list))
for i in range(len(E)):
    rkd[(cl[i], dr[i])][rep[i]].append(i)
ceil1, ceil_sb = [], []
for (l, d), dd_ in rkd.items():
    reps = sorted(dd_)
    if len(reps) < 2: continue
    s1 = E[dd_[reps[0]]].mean(axis=0); s2 = E[dd_[reps[1]]].mean(axis=0)
    rho = cor(s1, s2, top_idx)
    ceil1.append(rho)
    if rho > 0:
        ceil_sb.append(float(np.sqrt(2 * rho / (1 + rho))))
ceiling_sb = dict(rep_vs_rep_mean=float(np.mean(ceil1)),
                  sb_corrected_mean=float(np.mean(ceil_sb)),
                  sb_corrected_sd=float(np.std(ceil_sb)), n=len(ceil_sb))
print('ceiling SB:', ceiling_sb, flush=True)

# ---- 少样本校准预测残差还是全签名？ 对照实验: 先减背景再校准 ----
# 在 20% 校准设定下, 对「残差」做跨系迁移+校准, 与全签名对比
rng = np.random.default_rng(7)
r_full, r_resid = [], []
for seed in range(10):
    rng = np.random.default_rng(2000 + seed)
    for tl in lines:
        te = np.where(mline == tl)[0]; tr = np.where(mline != tl)[0]
        if len(te) < 4: continue
        mdtr = mdrug[tr]; mdte = mdrug[te]
        for mat, acc in ((M, r_full), (R, r_resid)):
            Ytr = mat[tr]; Yte = mat[te]
            P = np.zeros_like(Yte)
            for j in range(len(te)):
                m = mdtr == mdte[j]
                P[j] = Ytr[m].mean(axis=0) if m.sum() else Ytr.mean(axis=0)
            uq = sorted(set(mdte))
            perm = rng.permutation(len(uq))
            ncal = max(1, int(round(0.2 * len(uq))))
            cal_drugs = set(uq[i] for i in perm[:ncal])
            cal = np.array([j for j in range(len(te)) if mdte[j] in cal_drugs])
            eva = np.array([j for j in range(len(te)) if mdte[j] not in cal_drugs])
            if len(cal) >= 3 and len(eva) >= 1:
                Pc = P[cal]; Tc = Yte[cal]
                a = ((Pc * Tc).mean(axis=0) - Pc.mean(axis=0) * Tc.mean(axis=0)) / (Pc.var(axis=0) + 1e-8)
                b = Tc.mean(axis=0) - a * Pc.mean(axis=0)
                P[eva] = P[eva] * a[None, :] + b[None, :]
                # 残差版评估: 预测残差+背景 vs 真实全签名
                if mat is R:
                    pred_full = P[eva] + BG[te][eva]
                else:
                    pred_full = P[eva]
                T = M[te][eva]
                A = pred_full[:, top_idx] - pred_full[:, top_idx].mean(axis=1, keepdims=True)
                Bm = T[:, top_idx] - T[:, top_idx].mean(axis=1, keepdims=True)
                rr = (A * Bm).sum(axis=1) / (np.sqrt((A**2).sum(axis=1) * (Bm**2).sum(axis=1)) + 1e-12)
                acc.extend(rr.tolist())
calib_cmp = dict(full_signature=dict(mean=float(np.mean(r_full)), sd=float(np.std(r_full)), n=len(r_full)),
                 residual_plus_bg=dict(mean=float(np.mean(r_resid)), sd=float(np.std(r_resid)), n=len(r_resid)))
print('calib cmp:', calib_cmp, flush=True)

out = dict(drugs_per_line=drugs_per_line, bg_energy=bg_energy,
           retr_raw={k: v for k, v in retr_raw.items() if k != 'ranks'},
           retr_resid={k: v for k, v in retr_resid.items() if k != 'ranks'},
           retr_raw_ranks=retr_raw['ranks'], retr_resid_ranks=retr_resid['ranks'],
           ceiling_sb=ceiling_sb, calib_cmp=calib_cmp)
with open('/local_nvme_data/zizhuo/vc_fix/vc_results5.json', 'w') as f:
    json.dump(out, f)
print('ALL DONE %.1fs' % (time.time() - t0), flush=True)
