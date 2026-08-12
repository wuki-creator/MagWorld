# -*- coding: utf-8 -*-
"""
对比学习两阶段实验
  --mode ssl     : SimCLR 式自监督对比预训练（同条件两个 mini-bulk 视图为正样本对）
                   预训练编码器 → 冻结初始化后用标准 MAE 微调整个世界模型
  --mode latent  : 潜空间有监督对比（SupCon）：对比损失作用在 rollout 终态 z_K 上，
                   同条件为正、异条件为负（P×K 条件采样保证 batch 内有正样本）
用法:
  python3 train_cl.py --mode ssl    --magnet 1 --tag wm_ssl
  python3 train_cl.py --mode latent --magnet 1 --lambda_cl 0.05 --tag wm_latentcl
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world import WorldModel
from train import load_prepared, CondHelper, evaluate

torch.set_num_threads(3)


def info_nce(z1, z2, tau=0.2):
    """z1, z2: (B, d) 两视图。双向 InfoNCE。"""
    a = F.normalize(z1, dim=-1)
    b = F.normalize(z2, dim=-1)
    sim = a @ b.T / tau
    lab = torch.arange(sim.shape[0])
    return 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))


def supcon_loss(z, labels, tau=0.2):
    """Khosla et al. 有监督对比损失。z: (B, d)，labels: (B,) 条件 ID。"""
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.T / tau                                   # (B, B)
    B = z.shape[0]
    mask_self = torch.eye(B, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -1e9)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = []
    for i in range(B):
        if pos[i].any():
            loss.append(-logprob[i][pos[i]].mean())
    return torch.stack(loss).mean() if loss else torch.tensor(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['ssl', 'latent'], required=True)
    ap.add_argument('--magnet', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--ssl_epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_z', type=int, default=64)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--de_weight', type=float, default=1.0)
    ap.add_argument('--lambda_cl', type=float, default=0.05)
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='wm_cl')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_prepared()
    helper = CondHelper(d)
    G = len(helper.gene_names)
    X = torch.tensor(d['train_x'])
    Y = torch.tensor(d['train_y'])
    conds = list(d['train_pert'])
    uniq = sorted(set(conds))
    cond2id = {c: i for i, c in enumerate(uniq)}
    labels_all = torch.tensor([cond2id[c] for c in conds])
    by_cond = {c: [i for i, cc in enumerate(conds) if cc == c] for c in uniq}
    ctrl = torch.tensor(d['ctrl_mean'])

    model = WorldModel(G, d_z=args.d_z, n_steps=args.steps,
                       use_magnet=bool(args.magnet))
    print(f'mode={args.mode} magnet={args.magnet} '
          f'params={sum(p.numel() for p in model.parameters())/1e3:.0f}k', flush=True)

    # ---------- 阶段一（仅 ssl 模式）：SimCLR 预训练编码器 ----------
    if args.mode == 'ssl':
        proj = nn.Sequential(nn.Linear(args.d_z, args.d_z), nn.GELU(),
                             nn.Linear(args.d_z, args.d_z))
        opt = torch.optim.Adam(list(model.encoder.parameters()) +
                               list(proj.parameters()), lr=args.lr)
        t0 = time.time()
        for ep in range(args.ssl_epochs):
            # 每轮：每条件随机取 2 个不同视图组成对
            rng = np.random.default_rng(ep)
            pairs = []
            for c in uniq:
                idxs = by_cond[c]
                if len(idxs) >= 2:
                    i, j = rng.choice(idxs, size=2, replace=False)
                    pairs.append((i, j))
            rng.shuffle(pairs)
            tot = 0.0; nb = 0
            model.train()
            for s in range(0, len(pairs), args.batch):
                chunk = pairs[s:s + args.batch]
                if len(chunk) < 4:
                    continue
                v1 = torch.stack([X[i] for i, _ in chunk])
                v2 = torch.stack([X[j] for _, j in chunk])
                z1 = proj(model.encoder(v1))
                z2 = proj(model.encoder(v2))
                loss = info_nce(z1, z2, args.tau)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            if (ep + 1) % 10 == 0:
                print(f'ssl epoch {ep+1}/{args.ssl_epochs} nce={tot/nb:.4f} '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
        del proj
        print('SSL 预训练完成，进入微调', flush=True)

    # ---------- 阶段二：预测任务训练（latent 模式加 SupCon） ----------
    de_masks = {}
    for c in uniq:
        ybar = Y[by_cond[c]].mean(0)
        top = torch.argsort(-(ybar - ctrl).abs())[:50]
        m = torch.ones(G)
        m[top] = 1.0 + args.de_weight
        de_masks[c] = m

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if args.mode == 'latent':
            # P×K 采样：8 个条件 × 2 视图，保证 batch 内有正样本
            rng = np.random.default_rng(1000 + ep)
            batches = []
            cs = rng.choice(uniq, size=len(uniq), replace=False)
            for s in range(0, len(cs) - 7, 8):
                batch_idx = []
                for c in cs[s:s + 8]:
                    batch_idx += list(rng.choice(by_cond[c], size=2,
                                                 replace=len(by_cond[c]) < 2))
                batches.append(batch_idx)
        else:
            perm = np.random.permutation(len(X))
            batches = [perm[s:s + args.batch] for s in
                       range(0, len(X), args.batch)]
        tot_mae, tot_cl = 0.0, 0.0; nb = 0
        for b in batches:
            if len(b) < 4:
                continue
            b = list(b)
            xb, yb = X[b], Y[b]
            pidx = [helper.pert_idx(conds[i]) for i in b]
            w = torch.stack([de_masks[conds[i]] for i in b])
            if args.mode == 'latent':
                zK = model.latent(xb, pidx)
                pred = model.decoder(zK)
                l_cl = supcon_loss(zK, labels_all[b], args.tau)
            else:
                pred = model(xb, pidx)
                l_cl = torch.tensor(0.0)
            l_mae = (w * (pred - yb).abs()).mean()
            loss = l_mae + args.lambda_cl * l_cl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_mae += l_mae.item(); tot_cl += float(l_cl); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'epoch {ep+1}/{args.epochs} mae={tot_mae/nb:.4f} '
                  f'cl={tot_cl/nb:.4f} elapsed={time.time()-t0:.0f}s', flush=True)

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
