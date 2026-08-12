# -*- coding: utf-8 -*-
"""
世界模型训练与评测（复用 train.py 的数据加载与 VCC 评测）
用法: python3 train_world.py --magnet 1 --epochs 40 --tag world_magnet
      python3 train_world.py --magnet 0 --epochs 40 --tag world_plain
"""
import argparse, json, time
import numpy as np
import torch

from model_world import WorldModel
from train import load_prepared, CondHelper, evaluate

torch.set_num_threads(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--magnet', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_z', type=int, default=64)
    ap.add_argument('--steps', type=int, default=4, help='rollout 步数 K')
    ap.add_argument('--de_weight', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='world')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_prepared()
    helper = CondHelper(d)
    G = len(helper.gene_names)
    print(f'genes={G} magnet={args.magnet} K={args.steps} '
          f'train_samples={d["train_x"].shape[0]}', flush=True)

    model = WorldModel(G, d_z=args.d_z, n_steps=args.steps,
                       use_magnet=bool(args.magnet))
    print(f'params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k',
          flush=True)

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
        if (ep + 1) % 5 == 0 or ep == 0:
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
