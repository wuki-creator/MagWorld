import numpy as np, zipfile, io, json, time
from collections import defaultdict

t0 = time.time()
NPZ = '/home/zizhuo/cross_cell_vcc_v2/data/sciplex_full_v2.npz'
z = zipfile.ZipFile(NPZ)
def load(name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=True)

controls = load('controls.npy')
effects  = load('effects.npy')
cell_line = load('cell_line.npy').astype(str)
drug = load('drug.npy').astype(str)
dose = load('dose.npy')
genes = load('genes.npy').astype(str)
G = effects.shape[1]
print('loaded', effects.shape, flush=True)

# ---- aggregate to (line, drug) signatures, mean over dose & replicate ----
pairs = defaultdict(list); cpairs = defaultdict(list)
for i in range(len(effects)):
    pairs[(cell_line[i], drug[i])].append(effects[i])
    cpairs[cell_line[i]].append(controls[i])
lines = sorted(set(cell_line))
S = {k: np.mean(v, axis=0) for k, v in pairs.items()}
Cbasal = {l: np.mean(cpairs[l], axis=0) for l in lines}
conds = sorted(S.keys())
Smat = np.stack([S[k] for k in conds])
gstd = Smat.std(axis=0)
top = np.argsort(-gstd)[:2000]

def cor(a, b):
    a = a[top] - a[top].mean(); b = b[top] - b[top].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

# ---- decomposition: background (LOO drug mean) & residual ----
drugs_of = {l: sorted([d for (ll, d) in conds if ll == l]) for l in lines}
B = {}; R = {}
for l in lines:
    ds = drugs_of[l]
    for d in ds:
        others = [S[(l, dd)] for dd in ds if dd != d]
        b = np.mean(others, axis=0) if others else np.zeros(G)
        B[(l, d)] = b; R[(l, d)] = S[(l, d)] - b
Bline = {l: np.mean([S[(l, d)] for d in drugs_of[l]], axis=0) for l in lines}

# transfer residual / raw cross-line mean
Rtr = {}; CLM = {}
for (l, d) in conds:
    src_r = [R[(ll, d)] for ll in lines if ll != l and (ll, d) in R]
    src_s = [S[(ll, d)] for ll in lines if ll != l and (ll, d) in S]
    Rtr[(l, d)] = np.mean(src_r, axis=0) if src_r else None
    CLM[(l, d)] = np.mean(src_s, axis=0) if src_s else None

def score(predmap):
    rs = [cor(predmap[k], S[k]) for k in conds if predmap.get(k) is not None]
    return dict(mean=float(np.mean(rs)), sd=float(np.std(rs)), n=len(rs))

out = {}
out['baseline_cross_line_mean'] = score(CLM)
print('baseline CLM:', out['baseline_cross_line_mean'], flush=True)

# ---- CREST-zero: basal-similarity ridge predicts background (LOO over lines) ----
Ct = np.stack([Cbasal[l] for l in lines])[:, top]        # 7 x 2000
Bt = np.stack([Bline[l] for l in lines])                 # 7 x G
def ridge_w(x, Y, lam):
    # x: (p,) target; Y: (m, p) donor rows -> w: (m,)|x - w@Y||
    A = Y @ Y.T + lam * np.eye(Y.shape[0])
    return np.linalg.solve(A, Y @ x)

Bhat_basal = {}
Wmat = {}
for i, l in enumerate(lines):
    tr = [j for j in range(len(lines)) if j != i]
    w = ridge_w(Ct[i], Ct[tr], lam=1.0)
    Bhat_basal[l] = w @ Bt[tr]
    Wmat[l] = {lines[j]: float(w[k]) for k, j in enumerate(tr)}
out['basal_bg_cor'] = {l: cor(Bhat_basal[l], Bline[l]) for l in lines}
print('basal->bg cor:', {l: round(v, 3) for l, v in out['basal_bg_cor'].items()}, flush=True)

ZS = {k: (Bhat_basal[k[0]] + Rtr[k]) if Rtr[k] is not None else None for k in conds}
out['zeroshot_crest'] = score(ZS)
out['zeroshot_bg_only'] = score({k: Bhat_basal[k[0]] for k in conds})
print('zeroshot CREST:', out['zeroshot_crest'], flush=True)

# ---- per-gene ridge solvers (vectorized over genes) ----
def fit_genewise(A, Y, lam):
    # A: (n, k, G) design per gene; Y: (n, G) targets -> W: (k, G)
    GtG = np.einsum('nkg,nlg->klg', A, A) + lam * np.eye(A.shape[1])[:, :, None]
    GtY = np.einsum('nkg,ng->kg', A, Y)
    return np.linalg.solve(GtG.transpose(2, 0, 1), GtY.T[..., None])[:, :, 0].T  # (k, G)

def cor_pred(pred, y):
    a = pred - pred.mean(); b = y - y.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

# ---- few-shot variants (10 seeds, fractions 0.2 / 0.4) ----
def fewshot(frac, method, seeds=10, kappa=2.0, lam=1.0):
    rec = []; per_line = defaultdict(list)
    for l in lines:
        ds = drugs_of[l]
        n_cal = int(round(len(ds) * frac))
        if n_cal < 1:
            continue
        eval_ds = [d for d in ds if Rtr[(l, d)] is not None]
        for seed in range(seeds):
            rng2 = np.random.default_rng(seed * 100 + abs(hash(l)) % 100)
            cal = list(rng2.choice(ds, size=min(n_cal, len(ds)), replace=False))
            test = [d for d in eval_ds if d not in cal]
            if not test:
                continue
            b_cal = np.mean([S[(l, d)] for d in cal], axis=0)
            Yc = np.stack([S[(l, c)][top] for c in cal])               # (n_cal, 2000)
            if method == 'affine':
                Tc = np.stack([CLM[(l, c)][top] for c in cal])
                A = np.stack([Tc, np.ones_like(Tc)], axis=1)           # (n,2,G)
                W = fit_genewise(A, Yc, lam)
            elif method == 'crest_stack':
                Bg = np.broadcast_to(Bhat_basal[l][top], Yc.shape)
                Rc = np.stack([Rtr[(l, c)][top] for c in cal])
                A = np.stack([Bg, Rc, np.ones_like(Rc)], axis=1)       # (n,3,G)
                W = fit_genewise(A, Yc, lam)
            for d in test:
                if method == 'affine':
                    pred = W[0] * CLM[(l, d)][top] + W[1]
                    r = cor_pred(pred, S[(l, d)][top])
                elif method == 'crest_cal':
                    r = cor(b_cal + Rtr[(l, d)], S[(l, d)])
                elif method == 'crest_shrink':
                    b_shr = (len(cal) * b_cal + kappa * Bhat_basal[l]) / (len(cal) + kappa)
                    r = cor(b_shr + Rtr[(l, d)], S[(l, d)])
                elif method == 'crest_stack':
                    pred = W[0] * Bhat_basal[l][top] + W[1] * Rtr[(l, d)][top] + W[2]
                    r = cor_pred(pred, S[(l, d)][top])
                rec.append(r); per_line[l].append(r)
    rs = np.array(rec)
    return dict(mean=float(rs.mean()), sd=float(rs.std()), n=len(rs),
                per_line={l: float(np.mean(v)) for l, v in per_line.items()})

for frac in [0.2, 0.4]:
    for m in ['affine', 'crest_cal', 'crest_shrink', 'crest_stack']:
        out[f'{m}_{frac}'] = fewshot(frac, m)
        print(frac, m, round(out[f'{m}_{frac}']['mean'], 4), flush=True)

out['basal_weights'] = Wmat
out['runtime_s'] = time.time() - t0
json.dump(out, open('/local_nvme_data/zizhuo/vc_fix/vc_results6.json', 'w'))
print('DONE', time.time() - t0, flush=True)
