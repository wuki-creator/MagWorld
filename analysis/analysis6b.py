import numpy as np, zipfile, io, json, time
from collections import defaultdict

t0 = time.time()
NPZ = '/home/zizhuo/cross_cell_vcc_v2/data/sciplex_full_v2.npz'
z = zipfile.ZipFile(NPZ)
def load(name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=True)

controls = load('controls.npy'); effects = load('effects.npy')
cell_line = load('cell_line.npy').astype(str); drug = load('drug.npy').astype(str)
G = effects.shape[1]

pairs = defaultdict(list)
for i in range(len(effects)):
    pairs[(cell_line[i], drug[i])].append(effects[i])
lines = sorted(set(cell_line))
S = {k: np.mean(v, axis=0) for k, v in pairs.items()}
conds = sorted(S.keys())
Smat = np.stack([S[k] for k in conds])
top = np.argsort(-Smat.std(axis=0))[:2000]
drugs_of = {l: sorted([d for (ll, d) in conds if ll == l]) for l in lines}

CLM = {}
for (l, d) in conds:
    src = [S[(ll, d)] for ll in lines if ll != l and (ll, d) in S]
    CLM[(l, d)] = np.mean(src, axis=0) if src else None

def cor_pred(pred, y):
    a = pred - pred.mean(); b = y - y.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

def fit_genewise(A, Y, lam):
    GtG = np.einsum('nkg,nlg->klg', A, A) + lam * np.eye(A.shape[1])[:, :, None]
    GtY = np.einsum('nkg,ng->kg', A, Y)
    return np.linalg.solve(GtG.transpose(2, 0, 1), GtY.T[..., None])[:, :, 0].T

def run(n_cal, method, seeds=20, lam=1.0):
    rec = []; per_line = defaultdict(list)
    for l in lines:
        ds = drugs_of[l]
        if n_cal < 1 or n_cal >= len(ds):
            continue
        eval_ds = [d for d in ds if CLM[(l, d)] is not None]
        for seed in range(seeds):
            rng2 = np.random.default_rng(seed * 37 + abs(hash(l)) % 97)
            cal = list(rng2.choice(ds, size=n_cal, replace=False))
            test = [d for d in eval_ds if d not in cal]
            if not test:
                continue
            Yc = np.stack([S[(l, c)][top] for c in cal])
            Tc = np.stack([CLM[(l, c)][top] for c in cal])
            if method == 'offset':          # slope fixed to 1: per-gene offset = mean(s_cal - t_cal)
                off = (Yc - Tc).mean(axis=0)
                W = None
            else:                            # affine per-gene
                A = np.stack([Tc, np.ones_like(Tc)], axis=1)
                W = fit_genewise(A, Yc, lam)
            for d in test:
                if method == 'offset':
                    pred = CLM[(l, d)][top] + off
                else:
                    pred = W[0] * CLM[(l, d)][top] + W[1]
                r = cor_pred(pred, S[(l, d)][top])
                rec.append(r); per_line[l].append(r)
    rs = np.array(rec)
    return dict(mean=float(rs.mean()), sd=float(rs.std()), n=len(rs),
                per_line={l: round(float(np.mean(v)), 4) for l, v in per_line.items()})

out = {}
for n_cal in [1, 2, 3, 5]:
    for m in ['offset', 'affine']:
        out[f'{m}_n{n_cal}'] = run(n_cal, m)
        print(m, n_cal, round(out[f'{m}_n{n_cal}']['mean'], 4), flush=True)
# also fractional regimes for reference
for l_check in ['A172']:
    pass
out['offset_n1']['per_line'] = out['offset_n1']['per_line']
json.dump(out, open('/local_nvme_data/zizhuo/vc_fix/vc_results6b.json', 'w'))
print('DONE', time.time() - t0, flush=True)
