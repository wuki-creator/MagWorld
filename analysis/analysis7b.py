import numpy as np, h5py, json, time
import scipy.sparse as sp

t0 = time.time()
f = h5py.File('NormanWeissman2019_filtered.h5ad', 'r')

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

uniq_pert = sorted(set(pert))
ctrl = [c for c in uniq_pert if c.lower() in ('control', 'ctrl', 'non-targeting', 'nontargeting')][0]
ctrl_mean = np.asarray(X[np.where(pert == ctrl)[0]].mean(axis=0)).ravel()
cond_mean = {}
for c in uniq_pert:
    if c == ctrl: continue
    idx = np.where(pert == c)[0]
    if len(idx) < 20: continue
    cond_mean[c] = np.asarray(X[idx].mean(axis=0)).ravel() - ctrl_mean

singles = sorted([c for c in cond_mean if '_' not in c])
combos = sorted([c for c in cond_mean if '_' in c and len(c.split('_')) == 2])
Sall = np.stack(list(cond_mean.values()))
top_idx = np.argsort(-Sall.std(axis=0))[:2000]

def cor(a, b):
    a = a[top_idx] - a[top_idx].mean(); b = b[top_idx] - b[top_idx].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

evaluated = [c for c in combos if all(p in singles for p in c.split('_'))]
Iv = {}; ADD = {}; OBS = {}; syn = {}
for c in evaluated:
    a, b = c.split('_')
    add = cond_mean[a] + cond_mean[b]
    ADD[c] = add; OBS[c] = cond_mean[c]
    Iv[c] = (cond_mean[c] - add)[top_idx]
    syn[c] = float(np.linalg.norm(Iv[c]) / (np.linalg.norm(cond_mean[c][top_idx]) + 1e-12))

Ssing = np.stack([cond_mean[s][top_idx] for s in singles])
Corr = np.corrcoef(Ssing - Ssing.mean(axis=1, keepdims=True))
sidx = {s: i for i, s in enumerate(singles)}

# ---- family-gated interaction transfer ----
# query (A,B): neighbours must share exactly one gene; weight = sim of the differing partners
def gated_neighbors(c, sim_thr=0.0):
    a, b = c.split('_')
    out = []
    for o in evaluated:
        if o == c: continue
        a2, b2 = o.split('_')
        if a2 == a and b2 != b:
            out.append((Corr[sidx[b], sidx[b2]], o))
        elif b2 == b and a2 != a:
            out.append((Corr[sidx[a], sidx[a2]], o))
        elif a2 == b and b2 != a:
            out.append((Corr[sidx[a], sidx[b2]], o))
        elif b2 == a and a2 != b:
            out.append((Corr[sidx[b], sidx[a2]], o))
    return [(s, o) for s, o in out if s > sim_thr]

for thr in [0.0, 0.2, 0.4]:
    res = {}
    for c in evaluated:
        nbr = sorted(gated_neighbors(c, thr), key=lambda t: -t[0])[:5]
        if nbr:
            w = np.array([s for s, _ in nbr])
            ihat = np.sum([s * Iv[o] for s, o in nbr], axis=0) / w.sum()
        else:
            ihat = np.zeros(2000)
        pred = ADD[c].copy(); pred[top_idx] += ihat
        res[c] = (cor(ADD[c], OBS[c]), cor(pred, OBS[c]))
    r_add = np.array([v[0] for v in res.values()]); r_cr = np.array([v[1] for v in res.values()])
    synv = np.array([syn[c] for c in res])
    hi = synv > 0.75
    n_cov = sum(1 for c in res if gated_neighbors(c, thr))
    print('thr=%.1f coverage=%d/131  overall %.3f->%.3f  hi(n=%d) %.3f->%.3f  lo %.3f->%.3f' % (
        thr, n_cov, r_add.mean(), r_cr.mean(), hi.sum(),
        r_add[hi].mean(), r_cr[hi].mean(), r_add[~hi].mean(), r_cr[~hi].mean()), flush=True)
    if thr == 0.2:
        worst = sorted(res.items(), key=lambda kv: kv[1][0])[:8]
        print('worst pairs thr=0.2:', {c: (round(a, 2), round(n, 2)) for c, (a, n) in worst}, flush=True)
        json.dump(dict(thr=thr, overall_add=float(r_add.mean()), overall_crest=float(r_cr.mean()),
                       hi_add=float(r_add[hi].mean()), hi_crest=float(r_cr[hi].mean()), hi_n=int(hi.sum()),
                       lo_add=float(r_add[~hi].mean()), lo_crest=float(r_cr[~hi].mean()),
                       coverage=n_cov,
                       per_combo={c: dict(r_add=round(v[0], 4), r_crest=round(v[1], 4), syn=round(syn[c], 3)) for c, v in res.items()}),
                  open('vc_report7_gated.json', 'w'))
print('DONE', time.time() - t0, flush=True)
