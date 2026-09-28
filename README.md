# MagWorld: 磁力场世界模型——单细胞扰动响应预测

> **MagWorld** 是一个物理学启发的细胞世界模型（world model）：把被扰动的基因视为**磁源**，
> 在潜空间激发"磁场"，驱动细胞状态的 K 步演化，从而预测扰动后的转录组表达。
> 融合损失设计对齐 **xTrimoSCPerturb**（Arc Institute Virtual Cell Challenge 2025 冠军方案）的思路。

本仓库发布 **xTrimo 融合损失训练线中的最佳模型** `world_magnet_fused005`
（权重 + 评测明细 + 完整训练/推理代码），对应论文见 `paper/MagWorld_ICLR2027.pdf`。

---

## 最佳模型 (Best Checkpoint)

`weights/world_magnet_fused005.pt`：MagWorld 世界模型架构（磁力场开启），
以 **DE 加权 MAE + λ·InfoNCE（λ=0.05）** 融合损失训练。
在 Norman 2019 Perturb-seq 基准（92 个未见测试条件，VCC2025 评测协议）上：

| MAE ↓ | DES ↑ | PDS@1 ↑ | PDS pct ↑ |
|---|---|---|---|
| **0.0820** | **0.506** | **0.489** | **0.903** |

xTrimo 融合损失系列（6 个配置）完整对比——本模型在幅度精度（MAE/DES）与
扰动辨识（PDS）上取得最佳平衡，λ=0.05 位于 MAE–PDS 权衡前沿的最优点：

| 模型 | MAE ↓ | DES ↑ | PDS@1 ↑ | PDS pct ↑ |
|---|---|---|---|---|
| **world_magnet_fused005（本仓库）** | **0.0820** | **0.506** | 0.489 | **0.903** |
| world_magnet_fused02 | 0.0895 | 0.471 | **0.543** | 0.897 |
| world_magnet_fused (λ=1.0) | 0.0995 | 0.410 | 0.467 | 0.911 |
| world_plain_fused005 | 0.0904 | 0.497 | 0.391 | 0.890 |
| world_plain_fused02 | 0.1036 | 0.413 | 0.511 | 0.898 |
| world_plain_fused (λ=1.0) | 0.1180 | 0.371 | 0.478 | 0.884 |

（`world_plain_*` 为消融版：场向量换为普通扰动嵌入和，用于检验"磁场物理"本身的贡献。）

---

## 算法原理 (Algorithm)

### 1. 问题设定

给定对照细胞表达谱 $x_{\mathrm{ctrl}} \in \mathbb{R}^{G}$ 与扰动集合 $P$
（一个或两个被扰动基因），预测扰动后的伪bulk表达谱 $\hat{y} \in \mathbb{R}^{G}$。
评测遵循 VCC2025 协议：held-out 单基因扰动与组合双基因扰动。

### 2. 磁力场参数化：每个基因都是潜在的磁源

每个基因 $g$ 携带三个磁参数，均由**共享基因嵌入** $e_g \in \mathbb{R}^{d}$
分解 + 每基因自由残差构成：

$$
c_g = w_c^{\top} e_g + \tilde{c}_g \quad (\text{磁荷} \to \text{场强}), \qquad
\mathbf{p}_g = W_p\, e_g + \tilde{\mathbf{p}}_g \quad (\text{磁极} \to \text{场方向}), \qquad
\mu_g = \sigma(w_m^{\top} e_g + \tilde{m}_g) \quad (\text{磁导率} \to \text{磁化率})
$$

共享投影 $(w_c, W_p, w_m)$ 在所有训练扰动上学习，因此**从未被扰动过的基因**
也能通过其嵌入获得合理的磁参数（残差只为见过的基因精修）——这是模型泛化到
未见扰动的关键。被扰动基因 $s \in P$ 成为磁源，场强为
$S_s = \mathrm{softplus}(c_s + \beta_0)$（$\beta_0$ 初值让训练初期场保持微弱）。

### 3. MagWorld：场驱动的潜空间世界模型

编码器把对照表达压入潜状态 $z_0 = \mathrm{Enc}(x_{\mathrm{ctrl}}) \in \mathbb{R}^{d_z}$，
随后进行 $K$ 步 rollout 再解码。各磁源的场按**叠加原理**合成潜空间场向量：

$$
h = \sum_{s \in P} S_s \, \bar{\mathbf{p}}_s, \qquad \bar{\mathbf{p}}_s = \mathbf{p}_s / \lVert \mathbf{p}_s \rVert_2
$$

状态在场中漂移，"受力"大小由**状态依赖的磁化率** $\chi(z) = \sigma(W_s z + b)$ 调制：

$$
z_{k+1} = z_k + \eta_k \Big[ \chi(z_k) \odot h + f_{\mathrm{int}}(z_k, h) \Big],
\qquad \hat{y} = \mathrm{Dec}(z_K)
$$

其中 $f_{\mathrm{int}}$ 是小型状态-场交互网络，$\eta_k$ 为可学习步长。

**为什么磁场能表达上位性（epistasis）？** 组合扰动对应场叠加 $h = h_A + h_B$；
由于 $\chi$ 依赖不断演化的状态、且 $f_{\mathrm{int}}$ 把 $z$ 与 $h$ 耦合，组合响应
**不是**单扰动响应的线性相加——这为组合扰动的非加性效应提供了物理机制。
实验中磁力场对组合扰动的提升最大（MAE −12.7%，Wilcoxon $p = 8.4\times10^{-5}$）。

### 4. 融合损失：幅度精度 + 输出空间对比辨识（xTrimoSCPerturb 思路）

VCC2025 把"表达幅度精度"（MAE/DES）与"扰动方向辨识"（PDS）分开计分，
xTrimoSCPerturb 的冠军方案将两类目标融合。本模型采用：

$$
\mathcal{L} = \underbrace{\sum_{i,g} w_g^{(c_i)} \big| \hat{y}_{i,g} - y_{i,g} \big|}_{\text{幅度（DE 加权 MAE）}}
\;+\; \lambda_{\mathrm{pds}}\,
\underbrace{\mathcal{L}_{\mathrm{NCE}}\big(\{\Delta\hat{y}_i\}, \{\Delta y_i\}\big)}_{\text{方向辨识（双向 InfoNCE）}}
$$

- $w_g^{(c)}$：上调条件 $c$ 的 top-50 差异表达基因权重，让模型聚焦生物学上最重要的基因；
- $\Delta\hat{y} = \hat{y} - x_{\mathrm{ctrl}}$、$\Delta y = y - x_{\mathrm{ctrl}}$ 为**输出空间的扰动 delta**；
- InfoNCE 把预测 delta 拉向对应真实 delta、推离 batch 内其他条件的 delta
  （余弦相似度，温度 $\tau = 0.2$，双向 cross-entropy）。

冠军配置 $\lambda_{\mathrm{pds}} = 0.05$：轻量对比项把 PDS@1 从 0.348 提升到 0.489，
同时几乎不损失幅度精度——过大的 $\lambda$ 则沿权衡前沿牺牲 MAE 换 PDS（见上表）。

---

## 快速开始 (Inference)

```python
import torch
from src.model_world import WorldModel

# 1293 个基因（Norman 2019 filtered），潜维 64，4 步 rollout，磁力场开启
model = WorldModel(n_genes=1293, d_z=64, n_steps=4, use_magnet=True)
model.load_state_dict(torch.load('weights/world_magnet_fused005.pt', weights_only=True))
model.eval()

# x_ctrl: (1, 1293) 对照细胞 log1p(CP10k) 表达；pert_gene_idx: 被扰动基因的索引
with torch.no_grad():
    y_hat = model(x_ctrl, [torch.tensor([gene_idx])])   # 组合扰动传入多个索引
```

## 复现训练 (Reproduce)

```bash
pip install -r requirements.txt
# 1) 下载 Norman 2019 数据（scPerturb, Zenodo record 10044268）
#    NormanWeissman2019_filtered.h5ad → data/norman.h5ad
# 2) 预处理（流式低内存，见 src/data_prep.py）生成 data/prepared.npz
# 3) 训练最佳模型（xTrimo 融合损失，λ=0.05）
python src/train_fused.py --magnet 1 --epochs 40 --lambda_pds 0.05 --batch 16 --tag world_magnet_fused005
```

## 文件结构

- `src/model_world.py` — **核心架构**：潜空间世界模型 + 磁力场动力学（含 `world_plain` 消融开关）
- `src/model.py` — 基因-Token Transformer + 磁力场注意力偏置（另一架构变体，用于物理先验的跨架构消融）
- `src/train_fused.py` — xTrimo 式融合损失训练脚本（**本模型**）
- `src/train_world.py` / `src/train.py` / `src/train_cl.py` — 基线与对照训练脚本
- `src/metrics.py` — VCC2025 指标：MAE / DES / PDS
- `src/data_prep.py` — 数据预处理（Norman 2019 h5ad → npz）
- `weights/world_magnet_fused005.pt` — 最佳模型权重
- `weights/world_magnet_fused005_metrics.json` — 评测明细（含逐条件 DES）
- `paper/MagWorld_ICLR2027.pdf` — 完整论文（方法、实验、阴性结果）
- `analysis/` — **Nature Communications 投稿配套分析代码**：sci-Plex（GEO GSE225775）与
  Norman 2019（Zenodo 10044268）全部图表与统计量的复现脚本、缓存结果与 Figures 1–6，
  详见 `analysis/README.md`

## 引文 (Citation)

如果使用本模型或代码，请引用：

```bibtex
@misc{magworld2026,
  title        = {MagWorld: A Magnetic-Field World Model for Single-Cell
                  Perturbation Response Prediction},
  author       = {{MagWorld Team}},
  year         = {2026},
  note         = {ICLR 2027 submission, under review. Code and checkpoints:
                  \url{https://github.com/wuki-creator/MagWorld}}
}
```

**关键参考文献**（本工作直接建立于其上的）：

```bibtex
@misc{xtrimo2025,
  title        = {xTrimoSCPerturb: Winning solution of the Virtual Cell Challenge 2025},
  author       = {{BioMap Research}},
  howpublished = {Technical report},
  year         = {2025}
}

@misc{vcc2025,
  title        = {Virtual Cell Challenge 2025},
  author       = {{Arc Institute}},
  howpublished = {\url{https://virtualcellchallenge.org}},
  year         = {2025}
}

@article{norman2019,
  title   = {Exploring genetic interaction manifolds constructed from rich
             single-cell phenotypes},
  author  = {Norman, Thomas M. and Horlbeck, Max A. and Replogle, Joseph M.
             and others},
  journal = {Science},
  volume  = {365}, number = {6455}, pages = {786--793}, year = {2019}
}

@article{infonce2018,
  title   = {Representation learning with contrastive predictive coding},
  author  = {van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol},
  journal = {arXiv preprint arXiv:1807.03748}, year = {2018}
}

@inproceedings{dreamer2019,
  title     = {Learning latent dynamics for planning from pixels},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {2555--2565}, year = {2019}
}
```

## 许可 (License)

代码 MIT；训练数据来自 scPerturb / Norman 2019（CC-BY-4.0），请遵循原始数据许可。
