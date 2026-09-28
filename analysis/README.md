# NC Submission Analysis Code

Analysis code reproducing all figures and statistics of the MagWorld
manuscript submitted to *Nature Communications*
(*A magnetic-field world model for single-cell perturbation response prediction*).

## Datasets

| Dataset | Source | Used by |
|---|---|---|
| sci-Plex (A549/K562/MCF7, 188 drugs × 4 doses) | GEO [GSE225775](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775), processed pseudo-bulk signatures `sciplex_full_v2.npz` | `analysis1.py`–`analysis3.py`, `analysis5.py` |
| Norman 2019 Perturb-seq | scPerturb, [Zenodo record 10044268](https://zenodo.org/records/10044268) (`NormanWeissman2019_filtered.h5ad`, 111,445 cells, 105 single + 131 dual perturbations) | `analysis4.py` |

Raw data files are **not** included in this repository (size); download them
from the sources above. `dl_norman.py` fetches the Norman 2019 h5ad.

## Pipeline

Run order (Python ≥ 3.8; numpy, scipy, h5py, matplotlib, python-docx):

```bash
python dl_norman.py        # download Norman 2019 h5ad from Zenodo
python analysis1.py        # sci-Plex baseline: within-line prediction, ceilings
python analysis2.py        # cross-line transfer baselines
python analysis3.py        # LOCO transfer, line-mean decomposition, learning curves
python analysis4.py        # Norman 2019: additivity of dual perturbations
python analysis5.py        # residual retrieval & background energy fraction
python make_figs_nc.py     # render Figures 1–6 into figures/
python build_nc2.py        # assemble manuscript docx (optional)
```

Key results are cached in `results/vc_results*.json` so that
`make_figs_nc.py` can run without re-executing the full pipeline.

## Headline numbers (as reported in the manuscript)

- sci-Plex within-line ceilings: 0.41 / 0.60 (top variable genes / all genes), shuffled baseline 0.846
- Leave-one-cell-line-out (LOCO) transfer: r = 0.135; line-mean leave-one-out: 0.763
- Background (housekeeping) energy fraction of perturbation signatures: 62%
- Residual retrieval: top-1 14.1% / top-5 52.9% (raw: 3.5% / 15.3%)
- Few-shot fine-tuning: 20% data → r = 0.711; 40% → 0.781
- Dose interpolation r = 0.79 / 0.81; extrapolation 0.628 vs 0.417 baseline
- Norman 2019 dual perturbations: additive model r = 0.912 vs ceiling 0.927; latent-space additivity 0.73–0.75; median interaction score 0.42
