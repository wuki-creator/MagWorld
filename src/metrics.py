# -*- coding: utf-8 -*-
"""
VCC2025 (Arc Institute Virtual Cell Challenge) 三大指标的忠实近似实现：
  - MAE : 预测与真实 bulk 表达的平均绝对误差
  - DES : Differential Expression Score，真实差异基因被正确找回（含方向）的比例
  - PDS : Perturbation Discrimination Score，预测 profile 能否在候选扰动中
          辨认出真实的扰动（top-1 命中率 + 平均百分位排名）
"""
import numpy as np
from scipy import stats


def mae_score(y_pred, y_true):
    """y_pred, y_true: (n_cond, G) bulk 表达。返回所有条件的平均 MAE。"""
    return float(np.mean(np.abs(y_pred - y_true)))


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # 单调化
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def de_score_single(y_pred, y_true, ctrl_mean, cells_cond, cells_ctrl, fdr=0.05):
    """
    单条件 DES：
      真实 DE 基因 = 该条件细胞 vs 对照细胞的 Welch t 检验，BH-FDR < fdr
      预测 DE 基因 = |y_pred - ctrl| 排名前 |DE_true| 的基因
      DES = 方向一致的真阳性数 / |DE_true|
    cells_cond: (n1, G) 该条件的单细胞矩阵；cells_ctrl: (n0, G) 对照细胞矩阵
    """
    t, p = stats.ttest_ind(cells_cond, cells_ctrl, axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    q = _bh_fdr(p)
    de_true = np.where(q < fdr)[0]
    if len(de_true) == 0:
        return np.nan, 0
    true_dir = np.sign(y_true[de_true] - ctrl_mean[de_true])
    delta_pred = y_pred - ctrl_mean
    top_pred = np.argsort(-np.abs(delta_pred))[: len(de_true)]
    pred_set = set(top_pred.tolist())
    hit = 0
    for g, d in zip(de_true, true_dir):
        if g in pred_set and np.sign(delta_pred[g]) == d:
            hit += 1
    return hit / len(de_true), len(de_true)


def des_score(preds, trues, ctrl_mean, cells_by_cond, cells_ctrl, fdr=0.05):
    """对所有测试条件求平均 DES。返回 (mean_DES, per_cond列表)。"""
    vals, detail = [], []
    for c in preds:
        v, n_de = de_score_single(preds[c], trues[c], ctrl_mean,
                                  cells_by_cond[c], cells_ctrl, fdr)
        if not np.isnan(v):
            vals.append(v)
        detail.append((c, v, n_de))
    return float(np.mean(vals)) if vals else np.nan, detail


def pds_score(pred_deltas, true_deltas):
    """
    pred_deltas, true_deltas: (n_cond, G) 相对对照的 delta 矩阵（条件顺序一致）
    对每个预测 delta，用 Pearson 相关在所有真实 delta 中排名：
      PDS@1 = 真实条件排第 1 的比例
      PDS_pct = 真实条件的平均百分位排名（1 最好）
    """
    P = pred_deltas - pred_deltas.mean(1, keepdims=True)
    T = true_deltas - true_deltas.mean(1, keepdims=True)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    sim = Pn @ Tn.T                                        # (n, n)
    n = sim.shape[0]
    top1, pct = 0, []
    for i in range(n):
        rank = int(np.sum(sim[i] > sim[i, i])) + 1         # 1-based 排名
        if rank == 1:
            top1 += 1
        pct.append(1 - (rank - 1) / (n - 1) if n > 1 else 1.0)
    return float(top1 / n), float(np.mean(pct))
