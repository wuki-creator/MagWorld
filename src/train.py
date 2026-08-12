# -*- coding: utf-8 -*-
"""
训练与评测：基线 vs 磁力场模型
用法: python3 train.py --magnet 0 --epochs 6 --tag baseline
      python3 train.py --magnet 1 --epochs 6 --tag magnet
评测: VCC2025 指标 MAE / DES / PDS（另附"无变化"平凡基线作参照）
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

from model import MagFieldPerturbModel
from metrics import mae_score, des_score, pds_score

torch.set_num_threads(3)


def load_prepared(path='data/prepared.npz'):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    return d


class CondHelper:
    def __init__(self, d):
        self.gene_names = list(d['gene_names'])
        self.g2i = {g: i for i, g in enumerate(self.gene_names)}

    def pert_idx(self, cond):
        return torch.tensor([self.g2i[g] for g in cond.split('_')],
                            dtype=torch.long)


def evaluate(model, d, helper):
    """在全部测试条件上计算 MAE / DES / PDS，并附平凡基线（预测=对照）。"""
    model.eval()
    ctrl = d['ctrl_mean']
    test_conds = list(d['test_conds'])
    preds, trues, cells_by_cond = {}, {}, {}
    x = torch.tensor(ctrl[None, :], dtype=torch.float32)
    with torch.no_grad():
        for c in test_conds:
            pidx = [helper.pert_idx(c)]
            yhat = model(x, pidx)[0].numpy()
            preds[c] = yhat
            trues[c] = d[f'condmean::{c}']
            cells_by_cond[c] = d[f'condcells::{c}']

    mae = mae_score(np.stack([preds[c] for c in test_conds]),
                    np.stack([trues[c] for c in test_conds]))
    des, des_detail = des_score(preds, trues, ctrl, cells_by_cond,
                                d['cells_ctrl'])
    pred_d = np.stack([preds[c] - ctrl for c in test_conds])
    true_d = np.stack([trues[c] - ctrl for c in test_conds])
    pds1, pdspct = pds_score(pred_d, true_d)

    # 平凡基线：预测 = 对照（delta = 0）
    naive = {c: ctrl.copy() for c in test_conds}
    mae_nv = mae_score(np.stack([naive[c] for c in test_conds]),
                       np.stack([trues[c] for c in test_conds]))
    des_nv, _ = des_score(naive, trues, ctrl, cells_by_cond, d['cells_ctrl'])
    # 平凡基线 delta 全 0，PDS 无意义 → 用随机打乱近似时记 0/NaN
    return {'MAE': mae, 'DES': des, 'PDS@1': pds1, 'PDS_pct': pdspct,
            'naive_MAE': mae_nv, 'naive_DES': des_nv}, des_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--magnet', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--de_weight', type=float, default=1.0,
                    help='top-50 DE 基因的损失附加权重')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='model')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_prepared()
    helper = CondHelper(d)
    G = len(helper.gene_names)
    print(f'genes={G} magnet={args.magnet} train_samples={d["train_x"].shape[0]}')

    model = MagFieldPerturbModel(G, d_model=args.d_model, n_layers=args.layers,
                                 use_magnet=bool(args.magnet))
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params: {nparam/1e3:.0f}k')

    X = torch.tensor(d['train_x'])
    Y = torch.tensor(d['train_y'])
    conds = list(d['train_pert'])
    ctrl = torch.tensor(d['ctrl_mean'])
    # 每个训练条件的 top-50 DE 基因掩码（由训练 bulk 估计）
    de_masks = {}
    for c in set(conds):
        ybar = Y[[i for i, cc in enumerate(conds) if cc == c]].mean(0)
        top = torch.argsort(-(ybar - ctrl).abs())[:50]
        m = torch.ones(G)
        m[top] = 1.0 + args.de_weight
        de_masks[c] = m

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = X.shape[0]
    steps_per_ep = int(np.ceil(n / args.batch))
    t0 = time.time()
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        model.train()
        for s in range(steps_per_ep):
            b = perm[s * args.batch:(s + 1) * args.batch]
            xb, yb = X[b], Y[b]
            pidx = [helper.pert_idx(conds[i]) for i in b]
            w = torch.stack([de_masks[conds[i]] for i in b])
            pred = model(xb, pidx)
            loss = (w * (pred - yb).abs()).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f'epoch {ep+1}/{args.epochs} loss={tot/steps_per_ep:.4f} '
              f'elapsed={time.time()-t0:.0f}s', flush=True)

    torch.save(model.state_dict(), f'results/{args.tag}.pt')
    metrics, des_detail = evaluate(model, d, helper)
    print(json.dumps(metrics, indent=2))
    with open(f'results/{args.tag}_metrics.json', 'w') as f:
        json.dump({'metrics': metrics,
                   'des_per_cond': [(c, float(v), int(n_)) for c, v, n_ in des_detail],
                   'args': vars(args)}, f, indent=2)
    print(f'saved results/{args.tag}_metrics.json')


if __name__ == '__main__':
    main()
