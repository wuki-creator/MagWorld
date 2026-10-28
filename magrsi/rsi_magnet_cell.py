#!/usr/bin/env python3
# rsi_magnet_cell.py — MagRSI: RSI momentum gating for magnetic-field cell response.
# X is CSR sparse counts (uint32) -> work in log1p space -> damp -> expm1 -> counts.
#   1. r = x_log - rollmean_W(x_log)      (local field baseline along gene axis)
#   2. RSI_W(r)                            (Wilder; >50 local up-run, <50 down-run)
#   3. gate = alpha * relu(|RSI-50|/50 - theta)   (overbought/oversold -> overextended)
#   4. x'_log = x_log - gate * r           (mean reversion, Langevin-style saturation)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--theta', type=float, default=0.3)
    ap.add_argument('--block', type=int, default=8000)
    ap.add_argument('--controls', default='')
    ap.add_argument('--max-cells', type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()

    f = h5py.File(a.pred, 'r')
    g = f['X']
    data, indices, indptr = g['data'], g['indices'], g['indptr']
    G = f['var']['gene_name'].shape[0]
    N = indptr.shape[0] - 1
    if a.max_cells > 0: N = min(N, a.max_cells)
    print(f'pred CSR {N}x{G} nnz_total={data.shape[0]}', flush=True)

    def cat(col):
        d = f['obs'][col]
        if isinstance(d, h5py.Dataset): return d[:N].astype(str)
        return d['categories'][:].astype(str)[d['codes'][:N]]
    ctx = cat('context'); tgt = cat('target_gene')
    cellid = f['obs']['_index'][:N].astype(str)
    genes = f['var']['gene_name'][:].astype(str)

    # ---- field axis: genes sorted by control baseline magnetization ----
    order = None
    if a.controls:
        import glob as _glob
        csum = np.zeros(G, dtype=np.float64); cn = 0
        for cf in sorted(_glob.glob(a.controls.rstrip('/') + '/*.h5ad')):
            c = h5py.File(cf, 'r')
            cx = c['X']
            if isinstance(cx, h5py.Group):  # CSR
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
        print(f'field axis from {a.controls} ({cn} control cells)', flush=True)
    if order is None:
        order = np.arange(G)
    inv = np.empty_like(order); inv[order] = np.arange(G)

    out = h5py.File(a.out, 'w')
    out.attrs['Conventions'] = 'CF-1.8'
    out.attrs['title'] = 'MagRSI-improved magnetic-field cell perturbation prediction'
    out.attrs['source_model'] = 'v36_s543_p3 (causal transformer + masked graph prior)'
    out.attrs['method'] = 'MagRSI momentum mean-reversion gate (log1p space)'
    out.attrs['params'] = json.dumps({'window': a.window, 'alpha': a.alpha, 'theta': a.theta,
                                      'gate': "x' = x - alpha*relu(|RSI-50|/50-theta)*(x-rollmean(x))"})
    out.attrs['history'] = time.strftime('%Y-%m-%d %H:%M:%S') + ' rsi_magnet_cell.py'
    out.create_dataset('gene', data=genes.astype(h5py.string_dtype('utf-8')))
    out.create_dataset('cell_id', data=cellid.astype(h5py.string_dtype('utf-8')))
    out.create_dataset('context', data=ctx.astype(h5py.string_dtype('utf-8')))
    out.create_dataset('target_gene', data=tgt.astype(h5py.string_dtype('utf-8')))
    chunk_rows = min(N, max(1, 2**22 // (G * 4)))
    expr = out.create_dataset('expression', shape=(N, G), dtype='float32',
                              chunks=(chunk_rows, G), compression='gzip', compression_opts=4,
                              fillvalue=0.0)
    expr.attrs['long_name'] = 'improved predicted expression counts (MagRSI)'
    expr.dims[0].attach_scale(out['cell_id']); expr.dims[1].attach_scale(out['gene'])

    W = a.window
    damp_sum = mag_sum = cnt = 0.0
    for s in range(0, N, a.block):
        e = min(s + a.block, N)
        B = e - s
        p0, p1 = int(indptr[s]), int(indptr[e])
        nnz = np.diff(indptr[s:e+1])
        x = np.zeros((B, G), dtype=np.float32)
        rows = np.repeat(np.arange(B), nnz)
        x[rows, indices[p0:p1].astype(np.int64)] = data[p0:p1].astype(np.float32)
        xlog = np.log1p(x)[:, order]                    # reorder to field axis
        base = rollmean(xlog, W)
        r = xlog - base
        rsi = rsi_wilder(np.diff(r, axis=1), W)          # B x (G-1)
        gate = a.alpha * np.maximum(np.abs((rsi - 50.0) / 50.0) - a.theta, 0.0)
        corrected = xlog.copy()
        corrected[:, 1:] = base[:, 1:] + r[:, 1:] * (1.0 - gate)
        # Smoothing across a sparse count profile must not invent expression at
        # structurally zero entries; doing so also breaches the VCC density cap.
        corrected[xlog == 0.0] = 0.0
        corrected = corrected[:, inv]                    # map back to panel order
        expr[s:e] = np.expm1(corrected).clip(min=0.0)
        damp_sum += float(gate.sum()); mag_sum += float(np.abs(r[:, 1:]).sum()); cnt += r[:, 1:].size
        print(f'[{e}/{N}] {time.time()-t0:.0f}s damp_ratio={damp_sum/max(mag_sum,1e-9):.4f}', flush=True)
    out.close(); f.close()
    print('DONE', time.time() - t0, flush=True)

if __name__ == '__main__':
    main()
