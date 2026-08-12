# -*- coding: utf-8 -*-
"""
MagWorld: 潜空间世界模型 + 磁力场转移动力学
--------------------------------------------
范式（对齐 AlphaCell / Chreode / PD-scWorld 的细胞世界模型思路）：
    x (表达) --Encoder--> z_0 (潜状态)
    z_{k+1} = z_k + eta_k · drift(z_k, field)   ← 世界模型的"转移规则"，K 步 rollout
    z_K --Decoder--> ŷ (预测的扰动后表达)

扰动基因 = 磁源，在潜空间激发"磁场"：
    - 场向量 h = Σ_{s∈P} S_s · normalize(pole_s)
        S_s = softplus(charge_s)：场强（磁荷经 softplus 激活）
        pole_s ∈ R^{d_z}：场在潜空间的方向（不同扰动基因 → 不同方向的场）
    - 磁化率 χ(z) = sigmoid(W_s z + b)：状态依赖的"介质响应"，
      决定当前细胞状态在场中"受力"的大小与方向
    - 漂移 drift = χ(z) ⊙ h + 状态-场交互修正项

charge / pole 均由共享基因嵌入分解 + 每基因残差构成（未见扰动基因可迁移）。

world-plain 消融版：动力学结构相同，但场向量 h 换成普通的扰动嵌入和
（无 softplus 场强、无单位磁极方向、无磁化率结构），检验"磁场物理"本身的贡献。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MagneticLatentField(nn.Module):
    """由扰动基因集合产生潜空间场向量 h（磁力场版）。"""

    def __init__(self, d_model, d_z, init_scale=0.02):
        super().__init__()
        self.charge_lin = nn.Linear(d_model, 1)
        self.pole_lin = nn.Linear(d_model, d_z)
        self.charge_bias = nn.Parameter(torch.tensor(-0.5))

    def forward(self, gene_emb, charge_res, pole_res, pert_idx):
        charge = self.charge_lin(gene_emb).squeeze(-1) + charge_res
        pole = self.pole_lin(gene_emb) + pole_res
        S = F.softplus(charge[pert_idx] + self.charge_bias)        # (n_pert,)
        dirs = F.normalize(pole[pert_idx], dim=-1)                 # (n_pert, d_z)
        h = (dirs * S.unsqueeze(-1)).sum(0)                        # (d_z,) 场叠加
        return h


class WorldModel(nn.Module):
    def __init__(self, n_genes, d_model=64, d_z=64, d_hidden=256,
                 n_steps=4, use_magnet=True):
        super().__init__()
        self.use_magnet = use_magnet
        self.n_steps = n_steps
        self.gene_emb = nn.Embedding(n_genes, d_model)
        # 扰动嵌入（plain 版用；magnet 版也保留作残差交互项的输入）
        self.pert_lin = nn.Linear(d_model, d_z)
        self.pert_res = nn.Parameter(torch.randn(n_genes, d_z) * 0.02)

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_z)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_genes)
        )
        # 磁化率（状态依赖介质响应）；plain 版退化为普通状态门控
        self.suscept = nn.Linear(d_z, d_z)
        # 状态-场交互修正
        self.interact = nn.Sequential(
            nn.Linear(2 * d_z, d_z), nn.GELU(), nn.Linear(d_z, d_z)
        )
        # 每步的步长（可学习标量）
        self.etas = nn.Parameter(torch.full((n_steps,), 0.5))

        if use_magnet:
            self.magnet = MagneticLatentField(d_model, d_z)
            self.charge_res = nn.Parameter(torch.randn(n_genes) * 0.02)
            self.pole_res = nn.Parameter(torch.randn(n_genes, d_z) * 0.02)

    def field_vector(self, pert_idx):
        """magnet: 磁场叠加；plain: 普通扰动嵌入和。"""
        if self.use_magnet:
            return self.magnet(self.gene_emb.weight,
                               self.charge_res, self.pole_res, pert_idx)
        base = self.pert_lin(self.gene_emb.weight) + self.pert_res
        return base[pert_idx].sum(0)

    def latent(self, x_ctrl, pert_gene_idx):
        """编码 + rollout，返回终态 z_K（不经过解码器）。"""
        z = self.encoder(x_ctrl)                                   # (B, d_z)
        h = torch.stack([self.field_vector(idx) for idx in pert_gene_idx])
        for k in range(self.n_steps):
            chi = torch.sigmoid(self.suscept(z))                   # (B, d_z) 磁化率
            drift = chi * h + self.interact(torch.cat([z, h], dim=-1))
            z = z + self.etas[k] * drift
        return z

    def forward(self, x_ctrl, pert_gene_idx):
        return self.decoder(self.latent(x_ctrl, pert_gene_idx))

    # ---- 可解释性 ----
    def rollout(self, x_ctrl, pert_gene_idx):
        """返回逐步潜状态轨迹，用于分析。"""
        z = self.encoder(x_ctrl)
        h = torch.stack([self.field_vector(idx) for idx in pert_gene_idx])
        traj = [z.detach().clone()]
        with torch.no_grad():
            for k in range(self.n_steps):
                chi = torch.sigmoid(self.suscept(z))
                drift = chi * h + self.interact(torch.cat([z, h], dim=-1))
                z = z + self.etas[k] * drift
                traj.append(z.detach().clone())
        return torch.stack(traj)                                   # (K+1, B, d_z)
