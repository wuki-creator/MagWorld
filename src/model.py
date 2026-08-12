# -*- coding: utf-8 -*-
"""
MagField-Perturb: scFoundation 风格的基因-Token Transformer 扰动响应模型
------------------------------------------------------------------------
基线 (use_magnet=False) 与磁力场模型 (use_magnet=True) 结构/参数量级相同，
唯一区别是注意力 logits 上是否叠加"磁场相互作用"偏置。

磁力场概念（物理类比）：
  - 每个基因 g 有磁荷 charge_g、磁极向量 pole_g、磁导率 perm_g，全部由
    共享基因嵌入 gene_emb_g 线性分解 + 可学习残差构成：
        charge_g = w_c·gene_emb_g + c_g
        pole_g   = W_p·gene_emb_g + p_g
        perm_g   = sigmoid(w_m·gene_emb_g + m_g)
    因此未见过的测试扰动基因也能通过其基因嵌入获得合理的磁参数（可迁移）。
  - 被扰动基因 s 是"磁源"：场强 S_s = softplus(charge_s) 被激活；
  - 场对基因 j 的方向作用 F_{s->j} = cos(pole_s, pole_j)（磁力有方向性，
    可增强/抑制）；
  - 注意力偏置 B_ij = sum_{s in P} S_s · F_{s->j} · perm_i
    —— 扰动基因的磁场重新调制所有基因两两间的信息流。
    不同扰动基因因 (charge, pole) 不同而产生不同的"磁力反应"。

扰动效应嵌入同样因子化：pert_vec_g = W_e·gene_emb_g + e_g（未见基因靠 W_e 迁移）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MagneticFieldBias(nn.Module):
    def __init__(self, d_model, d_pole=32, init_scale=0.02):
        super().__init__()
        self.d_pole = d_pole
        self.charge_lin = nn.Linear(d_model, 1)       # 共享部分：由基因嵌入决定磁荷
        self.pole_lin = nn.Linear(d_model, d_pole)    # 共享部分：由基因嵌入决定磁极
        self.perm_lin = nn.Linear(d_model, 1)         # 共享部分：由基因嵌入决定磁导率
        self.charge_bias = nn.Parameter(torch.tensor(-0.5))  # 让初始场偏弱

    def forward(self, gene_emb, charge_res, pole_res, perm_res, pert_idx):
        """
        gene_emb: (G, d_model) 共享基因嵌入（detach 后的，作为磁参数的结构先验）
        charge_res/pole_res/perm_res: 模型中持有的可学习残差参数
        pert_idx: LongTensor (n_pert,) 被扰动基因索引
        """
        charge = self.charge_lin(gene_emb).squeeze(-1) + charge_res        # (G,)
        pole = self.pole_lin(gene_emb) + pole_res                          # (G, d_pole)
        perm = torch.sigmoid(self.perm_lin(gene_emb).squeeze(-1) + perm_res)

        S = F.softplus(charge[pert_idx] + self.charge_bias)                # (n_pert,)
        pole_src = F.normalize(pole[pert_idx], dim=-1)
        pole_all = F.normalize(pole, dim=-1)
        F_dir = pole_all @ pole_src.T                                      # (G, n_pert)
        field_on_j = F_dir @ S                                             # (G,)
        B = torch.outer(perm, field_on_j)                                  # (G, G)
        return B


class GeneTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.mag_gate = nn.Parameter(torch.tensor(1.0))  # 磁场注入强度（每层）

    def _attn(self, x, mag_bias):
        B_, N, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B_, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                      # (B,H,N,dh)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mag_bias is not None:
            attn = attn + self.mag_gate * mag_bias.unsqueeze(1)   # (B,1,N,N) 广播到 heads
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        return self.out((attn @ v).transpose(1, 2).reshape(B_, N, D))

    def forward(self, x, mag_bias=None):
        x = x + self._attn(x, mag_bias)
        x = x + self.ff(self.ln2(x))
        return x


class MagFieldPerturbModel(nn.Module):
    """
    输入: 对照表达 x_ctrl (B, G)；扰动基因索引 pert_gene_idx（每样本）
    输出: 预测扰动后表达 = x_ctrl + delta
    """

    def __init__(self, n_genes, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, d_pole=32, use_magnet=False):
        super().__init__()
        self.use_magnet = use_magnet
        self.n_genes = n_genes
        self.d_model = d_model
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        # 扰动效应嵌入（因子化：共享线性 + 每基因残差）
        self.pert_lin = nn.Linear(d_model, d_model)
        self.pert_res = nn.Parameter(torch.randn(n_genes, d_model) * 0.02)
        self.layers = nn.ModuleList([
            GeneTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 1))
        if use_magnet:
            self.magnet = MagneticFieldBias(d_model, d_pole)
            self.charge_res = nn.Parameter(torch.randn(n_genes) * 0.02)
            self.pole_res = nn.Parameter(torch.randn(n_genes, d_pole) * 0.02)
            self.perm_res = nn.Parameter(torch.randn(n_genes) * 0.02)

    def pert_vector(self, gene_idx):
        """扰动效应向量：共享嵌入变换 + 残差；多基因扰动取和。"""
        base = self.pert_lin(self.gene_emb.weight) + self.pert_res       # (G, D)
        return base[gene_idx]                                            # (n_pert, D)

    def forward(self, x_ctrl, pert_gene_idx):
        B_, G = x_ctrl.shape
        tok = self.gene_emb.weight.unsqueeze(0).expand(B_, G, -1).contiguous()
        tok = tok + self.value_proj(x_ctrl.unsqueeze(-1))
        for b in range(B_):
            idx = pert_gene_idx[b]
            tok[b, idx] = tok[b, idx] + self.pert_vector(idx).sum(0, keepdim=True)

        mag_bias = None
        if self.use_magnet:
            g_emb = self.gene_emb.weight.detach()
            mag_bias = torch.stack([
                self.magnet(g_emb, self.charge_res, self.pole_res,
                            self.perm_res, pert_gene_idx[b])
                for b in range(B_)
            ])                                                           # (B, G, G)

        x = tok
        for layer in self.layers:
            x = layer(x, mag_bias)
        delta = self.delta_head(self.ln_f(x)).squeeze(-1)
        return x_ctrl + delta

    # ---- 可解释性工具 ----
    def field_strength(self):
        """返回每个基因的磁荷与场强（仅磁力场模型）。"""
        assert self.use_magnet
        g_emb = self.gene_emb.weight.detach()
        charge = (self.magnet.charge_lin(g_emb).squeeze(-1) + self.charge_res).detach()
        return charge, F.softplus(charge + self.magnet.charge_bias)
