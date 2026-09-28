import numpy as np, zipfile, io, json, time

t0 = time.time()
NPZ = '/home/zizhuo/cross_cell_vcc_v2/data/sciplex_full_v2.npz'
z = zipfile.ZipFile(NPZ)
def load(name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=True)

controls = load('controls.npy')      # (570, 18533)
effects  = load('effects.npy')       # (570, 18533)
cell_line = load('cell_line.npy').astype(str)
drug = load('drug.npy').astype(str)
dose = load('dose.npy')
genes = load('genes.npy').astype(str)
print('loaded', controls.shape, effects.shape, 'lines=', np.unique(cell_line).tolist(), flush=True)
print('effect stats: mean=%.4f std=%.4f' % (effects.mean(), effects.std()), flush=True)
print('n drugs:', len(np.unique(drug)), 'doses:', np.unique(dose)[:10], flush=True)

G = effects.shape[1]
# top variable genes across signatures
gstd = effects.std(axis=0)
top_idx = np.argsort(-gstd)[:2000]

def pearson_rows(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A**2).sum(axis=1) * (B**2).sum(axis=1)) + 1e-12
    return num / den

def ridge_fit(X, Y, lam=10.0):
    # dual ridge: W = X.T (X X.T + lam I)^-1 Y
    K = X @ X.T + lam * np.eye(X.shape[0])
    A = np.linalg.solve(K, Y)
    return X.T @ A

results = {}
r_curves = {}
lines = sorted(np.unique(cell_line))
uniq_drugs = sorted(np.unique(drug))
d2i = {d: i for i, d in enumerate(uniq_drugs)}

for tl in lines:
    te = cell_line == tl
    tr = ~te
    Xtr_ctrl = controls[tr]; Ytr = effects[tr]
    Xte_ctrl = controls[te]; Yte = effects[te]
    nte = te.sum()
    # baseline 1: train mean
    pred_mean = np.tile(Ytr.mean(axis=0, keepdims=True), (nte, 1))
    # baseline 2: cross-line mean of same drug (fallback to train mean)
    pred_clm = np.zeros_like(Yte)
    for j, i in enumerate(np.where(te)[0]):
        m = (drug[tr] == drug[i])
        pred_clm[j] = Ytr[m].mean(axis=0) if m.sum() > 0 else Ytr.mean(axis=0)
    # model A: drug one-hot ridge
    Ftr = np.zeros((tr.sum(), len(uniq_drugs))); Fte = np.zeros((nte, len(uniq_drugs)))
    for k, i in enumerate(np.where(tr)[0]): Ftr[k, d2i[drug[i]]] = 1
    for k, i in enumerate(np.where(te)[0]): Fte[k, d2i[drug[i]]] = 1
    W = ridge_fit(np.hstack([Ftr, np.ones((tr.sum(),1))]), Ytr, lam=50.0)
    pred_oh = np.hstack([Fte, np.ones((nte,1))]) @ W
    # model B (VirtualCell-style): drug one-hot + basal-state PCs (from controls)
    cmean = Xtr_ctrl.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xtr_ctrl - cmean, full_matrices=False)
    npc = 32
    PCtr = (Xtr_ctrl - cmean) @ Vt[:npc].T
    PCte = (Xte_ctrl - cmean) @ Vt[:npc].T
    Ftr2 = np.hstack([Ftr, PCtr, np.ones((tr.sum(),1))])
    Fte2 = np.hstack([Fte, PCte, np.ones((nte,1))])
    W2 = ridge_fit(Ftr2, Ytr, lam=50.0)
    pred_vc = Fte2 @ W2

    for name, P in [('train_mean', pred_mean), ('cross_line_mean', pred_clm),
                    ('drug_onehot_ridge', pred_oh), ('vc_state_conditioned', pred_vc)]:
        r_all = pearson_rows(P, Yte)
        r_top = pearson_rows(P[:, top_idx], Yte[:, top_idx])
        key = f'{tl}|{name}'
        results[key] = dict(n=int(nte),
                            r_all_mean=float(r_all.mean()), r_all_sd=float(r_all.std()),
                            r_top_mean=float(r_top.mean()), r_top_sd=float(r_top.std()))
        r_curves[key] = r_top
    print('done line', tl, 'n=', nte, flush=True)

# 汇总
summary = {}
for name in ['train_mean', 'cross_line_mean', 'drug_onehot_ridge', 'vc_state_conditioned']:
    rs = np.concatenate([r_curves[f'{tl}|{name}'] for tl in lines])
    summary[name] = dict(mean=float(rs.mean()), sd=float(rs.std()), n=int(rs.size))

# 例子散点：取 A172 中 vc_state_conditioned 中位相关的签名
tl = 'A172' if 'A172' in lines else lines[0]
te = np.where(cell_line == tl)[0]
key = f'{tl}|vc_state_conditioned'
r = r_curves[key]
j = int(np.argsort(r)[len(r)//2])
# 重新算预测以取出该签名的预测向量
trm = cell_line != tl
Ftr = np.zeros((trm.sum(), len(uniq_drugs)))
for k, i in enumerate(np.where(trm)[0]): Ftr[k, d2i[drug[i]]] = 1
Xtr_ctrl = controls[trm]; Ytr = effects[trm]
cmean = Xtr_ctrl.mean(axis=0)
U, S, Vt = np.linalg.svd(Xtr_ctrl - cmean, full_matrices=False)
PCtr = (Xtr_ctrl - cmean) @ Vt[:32].T
Ftr2 = np.hstack([Ftr, PCtr, np.ones((trm.sum(),1))])
W2 = ridge_fit(Ftr2, Ytr, lam=50.0)
i = te[j]
Fex = np.zeros((1, len(uniq_drugs))); Fex[0, d2i[drug[i]]] = 1
PCex = (controls[i:i+1] - cmean) @ Vt[:32].T
pred_ex = np.hstack([Fex, PCex, np.ones((1,1))]) @ W2
true_ex = effects[i]
ex_drug = drug[i]
sub = top_idx[::4]  # 500 个基因
np.savez_compressed('/tmp/vc_results.npz',
                    scatter_true=true_ex[sub], scatter_pred=pred_ex[0][sub],
                    top_genes=genes[top_idx[:50]])
with open('/tmp/vc_results.json', 'w') as f:
    json.dump(dict(per_line=results, summary=summary,
                   example=dict(line=tl, drug=str(ex_drug), r=float(r[j])),
                   doses=np.unique(dose).astype(str).tolist(),
                   n_genes=int(G), n_sig=int(effects.shape[0])), f, indent=1)
print('ALL DONE in %.1fs' % (time.time() - t0), flush=True)
print(json.dumps(summary, indent=1))
