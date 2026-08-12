# -*- coding: utf-8 -*-
"""
融合损失训练：MAE(加权) + DES 导向加权 + PDS 对比损失（对齐 xTrimoSCPerturb 的融合损失思路）
  PDS 损失 = InfoNCE：batch 内，预测的扰动 delta 应与对应的真实 delta 相近、
  与其他条件的真实 delta 相远（余弦相似度 / 温度 τ）。
用法: python3 train_fused.py --magnet 1 --epochs 40 --lambda_pds 1.0 --tag world_magnet_fused
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

from model_world import WorldModel
from train import load_prepared, CondHelper, evaluate

torch.set_num_threads(3)


def pds_contrastive_loss(pred, y, x_ctrl, tau=0.2):
    """pred, y, x_ctrl: (B, G)。返回 InfoNCE 损失。"""
    dp = F.normalize(pred - x_ctrl, dim=-1)          # 预测 delta
    dt = F.normalize(y - x_ctrl, dim=-1)             # 真实 delta
    sim = dp @ dt.T / tau                            # (B, B)
    labels = torch.arange(sim.shape[0], device=sim.device)
    # 双向：pred→true 与 true→pred
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--magnet', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)   # 更大 batch → 更多负样本
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_z', type=int, default=64)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--de_weight', type=float, default=1.0)
    ap.add_argument('--lambda_pds', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='world_fused')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_prepared()
    helper = CondHelper(d)
    G = len(helper.gene_names)
    print(f'genes={G} magnet={args.magnet} lambda_pds={args.lambda_pds} '
          f'batch={args.batch}', flush=True)

    model = WorldModel(G, d_z=args.d_z, n_steps=args.steps,
                       use_magnet=bool(args.magnet))
    print(f'params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k', flush=True)

    X = torch.tensor(d['train_x'])
    Y = torch.tensor(d['train_y'])
    conds = list(d['train_pert'])
    ctrl = torch.tensor(d['ctrl_mean'])
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
        tot_mae, tot_pds = 0.0, 0.0
        model.train()
        for s in range(steps_per_ep):
            b = perm[s * args.batch:(s + 1) * args.batch]
            if len(b) < 4:        # 负样本太少则跳过
                continue
            xb, yb = X[b], Y[b]
            pidx = [helper.pert_idx(conds[i]) for i in b]
            w = torch.stack([de_masks[conds[i]] for i in b])
            pred = model(xb, pidx)
            l_mae = (w * (pred - yb).abs()).mean()
            l_pds = pds_contrastive_loss(pred, yb, xb, args.tau)
            loss = l_mae + args.lambda_pds * l_pds
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_mae += l_mae.item(); tot_pds += l_pds.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'epoch {ep+1}/{args.epochs} mae={tot_mae/steps_per_ep:.4f} '
                  f'pds={tot_pds/steps_per_ep:.4f} elapsed={time.time()-t0:.0f}s',
                  flush=True)

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
