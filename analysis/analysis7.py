import numpy as np, h5py, json, time, os
import scipy.sparse as sp

t0 = time.time()
H5 = 'NormanWeissman2019_filtered.h5ad'
f = h5py.File(H5, 'r')

def read_cat(grp):
    g = f['obs'][grp]
    if isinstance(g, h5py.Group) and 'categories' in g:
        cats = g['categories'][:].astype(str)
        codes = g['codes'][:]
        out = np.empty(len(codes), dtype=object)
        for i, c in enumerate(cats):
            out[codes == i] = c
        return out.astype(str)
    return g[:].astype(str)

pert_col = next(c for c in ['perturbation', 'perturbation_name', 'pert_name', 'condition', 'guide_identity'] if c in f['obs'])
pert = read_cat(pert_col)
genes = f['var']['_index'][:].astype(str) if '_index' in f['var'] else f['var'][list(f['var'].keys())[0]][:].astype(str)
G = len(genes)
data = f['X']['data'][:]; indices = f['X']['indices'][:]; indptr = f['X']['indptr'][:]
if len(indptr) == G + 1:
    X = sp.csc_matrix((data, indices, indptr), shape=(len(pert), G)).tocsr()
else:
    X = sp.csr_matrix((data, indices, indptr), shape=(len(pert), G))
print('X', X.shape, flush=True)

uniq_pert = sorted(set(pert))
ctrl = [c for c in uniq_pert if c.lower() in ('control', 'ctrl', 'non-targeting', 'nontargeting')][0]
ctrl_mean = np.asarray(X[np.where(pert == ctrl)[0]].mean(axis=0)).ravel()

cond_mean = {}
for c in uniq_pert:
    if c == ctrl: continue
    idx = np.where(pert == c)[0]
    if len(idx) < 20: continue
    cond_mean[c] = np.asarray(X[idx].mean(axis=0)).ravel() - ctrl_mean
print('conditions:', len(cond_mean), flush=True)

singles = sorted([c for c in cond_mean if '_' not in c])
combos = sorted([c for c in cond_mean if '_' in c and len(c.split('_')) == 2])
Sall = np.stack(list(cond_mean.values()))
top_idx = np.argsort(-Sall.std(axis=0))[:2000]

def cor(a, b):
    a = a[top_idx] - a[top_idx].mean(); b = b[top_idx] - b[top_idx].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

evaluated = [c for c in combos if all(p in singles for p in c.split('_'))]
print('evaluated combos:', len(evaluated), flush=True)

# interaction residuals on top genes
Iv = {}; ADD = {}; OBS = {}; syn = {}
for c in evaluated:
    a, b = c.split('_')
    add = cond_mean[a] + cond_mean[b]
    ADD[c] = add; OBS[c] = cond_mean[c]
    Iv[c] = (cond_mean[c] - add)[top_idx]
    syn[c] = float(np.linalg.norm(Iv[c]) / (np.linalg.norm(cond_mean[c][top_idx]) + 1e-12))

# single-signature correlation for pair similarity
Ssing = np.stack([cond_mean[s][top_idx] for s in singles])
Ssing_c = Ssing - Ssing.mean(axis=1, keepdims=True)
Corr = np.corrcoef(Ssing_c)
sidx = {s: i for i, s in enumerate(singles)}

def pairsim(c1, c2):
    a1, b1 = c1.split('_'); a2, b2 = c2.split('_')
    return 0.5 * (Corr[sidx[a1], sidx[a2]] + Corr[sidx[b1], sidx[b2]])

# LOO interaction-corrected prediction
K = 5
res = {}
for c in evaluated:
    sims = sorted(((pairsim(c, o), o) for o in evaluated if o != c), key=lambda t: -t[0])
    nbr = [(s, o) for s, o in sims[:K] if s > 0]
    if nbr:
        w = np.array([s for s, _ in nbr])
        ihat = np.sum([s * Iv[o] for s, o in nbr], axis=0) / w.sum()
    else:
        ihat = np.zeros(2000)
    pred = ADD[c].copy(); pred[top_idx] += ihat
    res[c] = dict(r_add=cor(ADD[c], OBS[c]), r_crest=cor(pred, OBS[c]), syn=syn[c],
                  top_nbr=nbr[0][1] if nbr else None, top_sim=float(nbr[0][0]) if nbr else 0.0)

r_add = np.array([v['r_add'] for v in res.values()])
r_cr = np.array([v['r_crest'] for v in res.values()])
synv = np.array([v['syn'] for v in res.values()])
hi = synv > 0.75
out = dict(
    n=len(res), k=K,
    additive=dict(mean=float(r_add.mean()), sd=float(r_add.std())),
    crest_inter=dict(mean=float(r_cr.mean()), sd=float(r_cr.std())),
    delta=float((r_cr - r_add).mean()),
    wilcoxon_p=None,
    hi_subset=dict(n=int(hi.sum()),
                   additive=float(r_add[hi].mean()) if hi.any() else None,
                   crest=float(r_cr[hi].mean()) if hi.any() else None),
    lo_subset=dict(n=int((~hi).sum()),
                   additive=float(r_add[~hi].mean()),
                   crest=float(r_cr[~hi].mean())),
    per_combo={c: {k: (round(vv, 4) if isinstance(vv, float) else vv) for k, vv in v.items()} for c, v in res.items()},
)
def wilcoxon_p(x, y):
    d = x - y
    d = d[d != 0]
    n = len(d)
    ranks = np.empty(n)
    order = np.argsort(np.abs(d))
    sr = np.empty(n); sr[order] = np.arange(1, n + 1)
    # average ties
    ad = np.abs(d)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ad[order[j + 1]] == ad[order[i]]:
            j += 1
        sr[order[i:j + 1]] = sr[order[i:j + 1]].mean()
        i = j + 1
    W = float(sr[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    zval = (W - mu) / sd
    from math import erf, sqrt
    return float(2 * (1 - 0.5 * (1 + erf(abs(zval) / sqrt(2)))))

out['wilcoxon_p'] = wilcoxon_p(r_cr, r_add)
# worst additive pairs: before/after
worst = sorted(res.items(), key=lambda kv: kv[1]['r_add'])[:8]
out['worst_pairs'] = {c: dict(r_add=round(v['r_add'], 3), r_crest=round(v['r_crest'], 3),
                              syn=round(v['syn'], 2), nbr=v['top_nbr']) for c, v in worst}
json.dump(out, open('vc_report6_norman.json', 'w'))
print('additive: %.3f -> CREST-inter: %.3f (delta %.4f, wilcoxon %.2e)' % (r_add.mean(), r_cr.mean(), (r_cr - r_add).mean(), out['wilcoxon_p']), flush=True)
print('hi-subset n=%d: %.3f -> %.3f' % (hi.sum(), r_add[hi].mean(), r_cr[hi].mean()), flush=True)
print('worst:', json.dumps(out['worst_pairs'], indent=1)[:800], flush=True)
print('DONE', time.time() - t0, flush=True)
