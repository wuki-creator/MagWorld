#!/usr/bin/env python3
# rsi_recurse.py — self-evolving recursive MagRSI.
# Round 1: field axis = control baseline magnetization (--controls).
# Round t>1: field axis = global mean profile of x_{t-1} (evolving magnetization),
#            recomputed each round (pre-pass). x_t = x_{t-1} - gate*detrend(x_{t-1}).
# Tracks convergence delta_t = mean|x_t - x_{t-1}| and damp_ratio per round.
# Output: NetCDF with one expression variable per round: expression_r1, r2, ...
import argparse, json, time
import numpy as np
import h5py

def rollmean(a, W):
    B, G = a.shape
    c = np.cumsum(np.pad(a, ((0,0),(1,0)), mode='constant'), axis=1)
    idx = np.arange(G)
    lo = np.clip(idx - W//2, 0, None)
    hi = np.clip(idx + W - W//2, None, G)
    return (c[:, hi] - c[:, lo]) / (hi - lo)[None, :]

def rsi_wilder(d, W):
    up = np.maximum(d, 0.0); dn = np.maximum(-d, 0.0)
    au = up[:, :W].mean(axis=1, keepdims=True)
    ad = dn[:, :W].mean(axis=1, keepdims=True)
    alpha = 1.0 / W
    B, G = d.shape
    rsi = np.empty((B, G), dtype=np.float32)
    rsi[:, W-1] = (100.0 - 100.0 / (1.0 + au / (ad + 1e-12)))[:, 0]
    for g in range(W, G):
        au = au + alpha * (up[:, g:g+1] - au)
        ad = ad + alpha * (dn[:, g:g+1] - ad)
        rsi[:, g] = (100.0 - 100.0 / (1.0 + au / (ad + 1e-12)))[:, 0]
    rsi[:, :W] = 50.0
    return rsi

def _inv(order):
    v = np.empty_like(order); v[order] = np.arange(order.size); return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nc', required=True)
    ap.add_argument('--controls', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=3,
                    help='number of additional recursive rounds to compute')
    ap.add_argument('--initial-round', type=int, default=0,
                    help='round already represented by --nc (0=base, 1=MagRSI output)')
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--theta', type=float, default=0.3)
    ap.add_argument('--block', type=int, default=4000)
    a = ap.parse_args()
    if a.initial_round < 0 or a.rounds < 1:
        ap.error('--initial-round must be >=0 and --rounds must be >=1')
    t0 = time.time()
    f = h5py.File(a.nc, 'r')
    expr = f['expression']
    N, G = expr.shape
    print(f'recurse {N}x{G} initial_round={a.initial_round} additional_rounds={a.rounds}', flush=True)

    # The control-derived axis is needed only for round 1. Later rounds
    # recompute their axis from the input prediction itself.
    if a.initial_round == 0 and a.controls:
        import glob as _glob
        csum = np.zeros(G, dtype=np.float64); cn = 0
        for cf in sorted(_glob.glob(a.controls.rstrip('/') + '/*.h5ad')):
            c = h5py.File(cf, 'r'); cx = c['X']
            if isinstance(cx, h5py.Group):
                cd, ci, cp = cx['data'], cx['indices'], cx['indptr']
                n = cp.shape[0] - 1
                for s2 in range(0, n, 4000):
                    e2 = min(s2 + 4000, n)
                    p0, p1 = int(cp[s2]), int(cp[e2])
                    nnz2 = np.diff(cp[s2:e2+1])
                    xx = np.zeros((e2 - s2, G), dtype=np.float32)
                    xx[np.repeat(np.arange(e2 - s2), nnz2), ci[p0:p1].astype(np.int64)] = cd[p0:p1].astype(np.float32)
                    csum += np.log1p(xx).sum(axis=0); cn += e2 - s2
            else:
                for s2 in range(0, cx.shape[0], 4000):
                    csum += np.log1p(cx[s2:s2+4000].astype(np.float32)).sum(axis=0); cn += min(4000, cx.shape[0]-s2)
            c.close()
        order = np.argsort(csum / max(cn, 1), kind='stable')
    else:
        order = np.arange(G)

    out = h5py.File(a.out, 'w')
    for k in ('gene', 'cell_id', 'context', 'target_gene'):
        out.create_dataset(k, data=f[k][:])
    out.attrs['Conventions'] = 'CF-1.8'
    out.attrs['title'] = 'MagRSI self-evolving recursion'
    out.attrs['params'] = json.dumps({'rounds': a.rounds, 'initial_round': a.initial_round,
                                      'window': a.window, 'alpha': a.alpha, 'theta': a.theta})
    CH = min(N, max(1, 2**22 // (G * 4)))
    log = []
    cur = expr
    for rd in range(a.initial_round + 1, a.initial_round + a.rounds + 1):
        # pre-pass: global mean for evolving axis (round>=2)
        if rd >= 2:
            acc = np.zeros(G, dtype=np.float64)
            for s in range(0, N, a.block):
                acc += cur[s:s+a.block].sum(axis=0)
            order = np.argsort(acc / N, kind='stable')
            print(f'round {rd}: field axis re-derived from round {rd-1} mean', flush=True)
        ds = out.create_dataset(f'expression_r{rd}', shape=(N, G), dtype='float32',
                                chunks=(CH, G), compression='gzip', compression_opts=4, fillvalue=0.0)
        inv = _inv(order)
        dsum = msum = delta_sum = value_sum = 0.0
        for s in range(0, N, a.block):
            e = min(s + a.block, N)
            xlog = np.log1p(cur[s:e].astype(np.float32))[:, order]
            original = np.expm1(xlog)[:, inv]
            base = rollmean(xlog, a.window)
            r = xlog - base
            rsi = rsi_wilder(np.diff(r, axis=1), a.window)
            gate = a.alpha * np.maximum(np.abs((rsi - 50.0) / 50.0) - a.theta, 0.0)
            corrected = xlog.copy()
            corrected[:, 1:] = base[:, 1:] + r[:, 1:] * (1.0 - gate)
            # Preserve the observed/predicted support: no de novo nonzero genes.
            corrected[xlog == 0.0] = 0.0
            xn = np.expm1(corrected)[:, inv].clip(min=0.0)
            ds[s:e] = xn
            delta_sum += float(np.abs(xn - original).sum())
            value_sum += float(np.abs(original).sum())
            dsum += float(gate.sum()); msum += float(np.abs(r[:, 1:]).sum())
        dr = dsum / max(msum, 1e-9)
        relative_delta = delta_sum / max(value_sum, 1e-9)
        cur = ds
        print(f'round {rd} done {time.time()-t0:.0f}s damp_ratio={dr:.4f} '
              f'relative_delta={relative_delta:.6f}', flush=True)
        log.append({'round': rd, 'damp_ratio': dr, 'relative_delta': relative_delta})
    out.attrs['round_log'] = json.dumps(log)
    out.close(); f.close()
    print('DONE', time.time() - t0, flush=True)

if __name__ == '__main__':
    main()
