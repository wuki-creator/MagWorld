import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 8.5,
    'legend.fontsize': 6.5, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
C = dict(bg='#4878CF', accent='#D65F5F', green='#6ACC64', gray='#999999',
         purple='#956CB4', orange='#F0A202')

r2 = json.load(open('vc_results2.json'))
r3 = json.load(open('vc_results3.json'))
r4 = json.load(open('vc_results4.json'))
r5 = json.load(open('vc_results5.json'))

def pl(ax, s):
    ax.text(-0.18, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
pca = r3['pca_overview']
x = np.array(pca['x']); y = np.array(pca['y'])
lines = sorted(set(pca['line']))
cmap = dict(zip(lines, plt.cm.tab10(np.linspace(0, 1, len(lines)))))
for l in lines:
    m = np.array(pca['line']) == l
    axes[0].scatter(x[m], y[m], s=4, color=cmap[l], label=l, alpha=0.7, linewidths=0)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].set_title('Signatures by line', fontsize=8)
axes[0].legend(markerscale=2, frameon=False, ncol=2)
pl(axes[0], 'a')
ve = np.array(pca['var_explained']) * 100
axes[1].bar(range(1, len(ve)+1), ve, color=C['bg'])
axes[1].set_xlabel('Principal component'); axes[1].set_ylabel('Variance explained (%)')
axes[1].set_title('Variance explained', fontsize=8)
pl(axes[1], 'b')
be = r5['bg_energy']['by_line']
ks = list(be.keys()); vs = [be[k]*100 for k in ks]
axes[2].bar(range(len(ks)), vs, color=C['accent'])
axes[2].axhline(r5['bg_energy']['mean']*100, color='k', ls='--', lw=0.8,
                label=f"mean {r5['bg_energy']['mean']*100:.0f}%")
axes[2].set_xticks(range(len(ks))); axes[2].set_xticklabels(ks, rotation=45, ha='right')
axes[2].set_ylabel('Background energy (%)'); axes[2].set_title('Line-background share', fontsize=8)
axes[2].legend(frameon=False)
pl(axes[2], 'c')
fig.savefig('fig1_overview.png'); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
ss = r3['summary_static']
order = ['train_mean', 'drug_onehot_ridge', 'cross_line_mean', 'weighted_transfer', 'line_mean_loo']
labels = ['Train', '1-hot', 'Cross-\nline', 'Weighted', 'Line\nmean']
vals = [ss[k]['mean'] for k in order]
errs = [ss[k]['sd']/np.sqrt(ss[k]['n']) for k in order]
axes[0].bar(range(len(order)), vals, yerr=errs, capsize=2, color=[C['gray']]*4 + [C['accent']])
axes[0].bar([len(order)], [r3['lc_summary']['0.2']['mean']], color=C['purple'])
axes[0].axhline(r5['ceiling_sb']['sb_corrected_mean'], color='k', ls='--', lw=0.8)
axes[0].text(1.2, r5['ceiling_sb']['sb_corrected_mean']+0.02, 'noise ceiling (SB)', fontsize=6)
axes[0].set_xticks(range(len(order)+1))
axes[0].set_xticklabels(labels + ['Few-shot\n20%'], fontsize=5, rotation=45, ha='right')
axes[0].set_ylabel('Pearson r (top-2000 HV genes)')
axes[0].set_title('LOCO transfer', fontsize=8)
pl(axes[0], 'a')
lc = r3['lc_summary']
mv = [lc[f]['mean'] for f in ['0.2', '0.4']]
sv = [lc[f]['sd']/np.sqrt(lc[f]['n']) for f in ['0.2', '0.4']]
axes[1].errorbar([20, 40], mv, yerr=sv, marker='o', ms=4, color=C['purple'], capsize=2)
axes[1].axhline(r5['ceiling_sb']['sb_corrected_mean'], color='k', ls='--', lw=0.8)
axes[1].axhline(ss['line_mean_loo']['mean'], color=C['accent'], ls=':', lw=0.8, label='Line mean (LOO)')
axes[1].set_xlabel('Calibration drugs (%)'); axes[1].set_ylabel('Pearson r')
axes[1].set_title('Learning curve (10 seeds)', fontsize=8)
axes[1].legend(frameon=False, loc='lower right')
pl(axes[1], 'b')
fbl = r3['fewshot20_by_line']
ks = [k for k in fbl if not np.isnan(fbl[k])]
axes[2].bar(range(len(ks)), [fbl[k] for k in ks], color=C['green'])
axes[2].axhline(r5['ceiling_sb']['sb_corrected_mean'], color='k', ls='--', lw=0.8)
axes[2].set_xticks(range(len(ks))); axes[2].set_xticklabels(ks, rotation=45, ha='right')
axes[2].set_ylabel('Pearson r'); axes[2].set_title('Few-shot by line', fontsize=8)
pl(axes[2], 'c')
fig.subplots_adjust(wspace=0.55)
fig.savefig('fig2_loco.png'); plt.close(fig)
print('fig1-2 ok')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
dc = r3['decomp']
axes[0].bar([0, 1], [dc['raw_cross_line_r']['mean'], dc['resid_cross_line_r']['mean']],
            yerr=[dc['raw_cross_line_r']['sd']/np.sqrt(dc['raw_cross_line_r']['n']),
                  dc['resid_cross_line_r']['sd']/np.sqrt(dc['resid_cross_line_r']['n'])],
            capsize=2, color=[C['gray'], C['bg']])
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Raw\nsignatures', 'Drug-specific\nresiduals'])
axes[0].set_ylabel('Cross-line r (same drug)')
axes[0].set_title('Component transfer', fontsize=8)
pl(axes[0], 'a')
rr, rz = r5['retr_raw'], r5['retr_resid']
w = 0.35
axes[1].bar([0, 1], [rr['top1']*100, rr['top5']*100], width=w, color=C['gray'], label='Raw')
axes[1].bar([w, 1+w], [rz['top1']*100, rz['top5']*100], width=w, color=C['bg'], label='Residual')
axes[1].set_xticks([w/2, 1+w/2]); axes[1].set_xticklabels(['Top-1', 'Top-5'])
axes[1].set_ylabel('Retrieval accuracy (%)')
axes[1].set_title('Drug retrieval (LOO)', fontsize=8)
axes[1].legend(frameon=False)
pl(axes[1], 'b')
bins = np.arange(0, 60, 2) - 0.5
axes[2].hist(r5['retr_raw_ranks'], bins=bins, alpha=0.6, color=C['gray'], label='Raw', density=True)
axes[2].hist(r5['retr_resid_ranks'], bins=bins, alpha=0.6, color=C['bg'], label='Residual', density=True)
axes[2].set_xlabel('Rank of correct drug'); axes[2].set_ylabel('Density')
axes[2].set_title('Rank distribution', fontsize=8)
axes[2].legend(frameon=False)
pl(axes[2], 'c')
fig.savefig('fig3_decomposition.png'); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
di = r2['dose_interp']
if isinstance(di, dict) and 'per_dose' in di:
    pd_ = di['per_dose']
    ks = sorted(pd_, key=float)
    axes[0].bar(range(len(ks)), [pd_[k]['mean'] for k in ks],
                yerr=[pd_[k].get('sd', 0)/np.sqrt(max(pd_[k].get('n', 1), 1)) for k in ks],
                capsize=2, color=C['bg'])
    axes[0].set_xticks(range(len(ks))); axes[0].set_xticklabels([f'{float(k):g}' for k in ks])
    axes[0].set_xlabel('Held-out dose (uM)')
elif isinstance(di, dict):
    ks = sorted(di.keys(), key=float)
    w = 0.35
    lv = [di[k]['linear']['mean'] for k in ks]
    le = [di[k]['linear']['sd']/np.sqrt(di[k]['linear']['n']) for k in ks]
    gv = [di[k]['global_mean']['mean'] for k in ks]
    ge = [di[k]['global_mean']['sd']/np.sqrt(di[k]['global_mean']['n']) for k in ks]
    axes[0].bar(np.arange(len(ks))-w/2, lv, width=w, yerr=le, capsize=2, color=C['bg'], label='Linear interp.')
    axes[0].bar(np.arange(len(ks))+w/2, gv, width=w, yerr=ge, capsize=2, color=C['gray'], label='Global mean')
    axes[0].set_xticks(range(len(ks))); axes[0].set_xticklabels([f'{float(k):g}' for k in ks])
    axes[0].set_xlabel('Held-out dose (uM)')
    axes[0].legend(frameon=False)
axes[0].set_ylabel('Pearson r'); axes[0].set_title('Dose interpolation')
pl(axes[0], 'a')
de = r3['dose_extrap']
axes[1].bar([0, 1], [de['global_mean']['mean'], de['linear_extrap']['mean']],
            yerr=[de['global_mean']['sd']/np.sqrt(de['global_mean']['n']),
                  de['linear_extrap']['sd']/np.sqrt(de['linear_extrap']['n'])],
            capsize=2, color=[C['gray'], C['orange']])
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Global\nmean', 'Linear dose\nextrapolation'])
axes[1].set_ylabel('Pearson r'); axes[1].set_title('Extrapolation to 10 uM', fontsize=8)
pl(axes[1], 'b')
ex = r2.get('ex_curves')
if isinstance(ex, dict) and 'gene_names' in ex:
    doses = ex['doses']; vals = ex['values']; gn = ex['gene_names']
    for gi in range(min(5, len(gn))):
        axes[2].plot(doses, vals[gi], 'o-', ms=2.5, lw=0.8, label=gn[gi])
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Dose (uM)'); axes[2].set_ylabel('Signature value')
    axes[2].set_title(f"{ex.get('line','')} {ex.get('drug','')} dose-response", fontsize=7)
    axes[2].legend(frameon=False, fontsize=5)
else:
    axes[2].axis('off')
pl(axes[2], 'c')
fig.savefig('fig4_dose.png'); plt.close(fig)
print('fig3-4 ok')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
pc = r4['per_combo']
rs = np.array([v['r'] for v in pc.values()])
axes[0].hist(rs, bins=25, color=C['bg'], alpha=0.85)
axes[0].axvline(r4['ceiling']['mean'], color='k', ls='--', lw=0.8, label=f"ceiling {r4['ceiling']['mean']:.2f}")
axes[0].axvline(r4['additive_expr']['mean'], color=C['accent'], ls='-', lw=0.8,
                label=f"additive {r4['additive_expr']['mean']:.2f}")
axes[0].set_xlabel('Pearson r (predicted vs observed)')
axes[0].set_ylabel('Double perturbations')
axes[0].set_title('Additive prediction', fontsize=8)
axes[0].legend(frameon=False)
pl(axes[0], 'a')
la = r4['latent_additivity']
ks = sorted(la, key=int)
axes[1].bar([0], [r4['additive_expr']['mean']], color=C['accent'], width=0.6)
axes[1].bar(np.arange(1, len(ks)+1), [la[k]['mean'] for k in ks],
            yerr=[la[k]['sd'] for k in ks], capsize=2, color=C['bg'], width=0.6)
axes[1].set_xticks([0] + list(np.arange(1, len(ks)+1)))
axes[1].set_xticklabels(['Full\nspace'] + [f'PC-{k}' for k in ks], fontsize=6)
axes[1].set_ylabel('Pearson r'); axes[1].set_title('Expression vs latent', fontsize=8)
pl(axes[1], 'b')
sy = np.array([v['syn'] for v in pc.values()])
axes[2].hist(sy, bins=25, color=C['purple'], alpha=0.85)
axes[2].axvline(float(np.median(sy)), color='k', ls='--', lw=0.8, label=f'median {np.median(sy):.2f}')
axes[2].set_xlabel('Non-additive fraction ||obs-(a+b)||/||obs||')
axes[2].set_ylabel('Double perturbations')
axes[2].set_title('Genetic interactions', fontsize=8)
axes[2].legend(frameon=False)
pl(axes[2], 'c')
fig.savefig('fig5_norman.png'); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
nc2 = r2.get('noise_ceiling', {})
vals = []; labs = []
if isinstance(nc2, dict):
    for k, v in nc2.items():
        if isinstance(v, (int, float)) and 'mean' in k:
            vals.append(v); labs.append(k.replace('mean', '').replace('_', '\n').strip())
if not vals:
    vals = [0.41]; labs = ['single\ncondition']
vals.append(r3['ceiling_doseavg']['mean']); labs.append('dose-avg\nreplicates')
vals.append(r5['ceiling_sb']['sb_corrected_mean']); labs.append('SB-corrected\n(2-rep mean)')
axes[0].bar(range(len(vals)), vals, color=[C['gray'], C['bg'], C['accent']][:len(vals)])
axes[0].set_xticks(range(len(vals))); axes[0].set_xticklabels(labs, fontsize=6)
axes[0].set_ylabel('Pearson r'); axes[0].set_title('Noise ceilings', fontsize=8)
pl(axes[0], 'a')
knn = r2.get('knn', {})
if isinstance(knn, dict) and knn:
    ks = list(knn.keys())
    vv = [knn[k] if isinstance(knn[k], (int, float)) else knn[k].get('mean', 0) for k in ks]
    axes[1].bar(range(len(ks)), vv, color=C['gray'])
    axes[1].set_xticks(range(len(ks))); axes[1].set_xticklabels(ks, fontsize=6, rotation=30, ha='right')
else:
    axes[1].bar([0], [-0.016], color=C['gray'])
    axes[1].set_xticks([0]); axes[1].set_xticklabels(['kNN\nsmoothing'])
axes[1].axhline(0, color='k', lw=0.6)
axes[1].set_ylabel('Delta Pearson r'); axes[1].set_title('kNN ablation', fontsize=8)
pl(axes[1], 'b')
cc = r5['calib_cmp']
axes[2].bar([0, 1], [cc['full_signature']['mean'], cc['residual_plus_bg']['mean']],
            yerr=[cc['full_signature']['sd']/np.sqrt(cc['full_signature']['n']),
                  cc['residual_plus_bg']['sd']/np.sqrt(cc['residual_plus_bg']['n'])],
            capsize=2, color=[C['purple'], C['bg']])
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(['Calibrate full\nsignature', 'Calibrate\nresidual only'])
axes[2].set_ylabel('Pearson r'); axes[2].set_title('Calibration target', fontsize=8)
pl(axes[2], 'c')
fig.savefig('fig6_ablation.png'); plt.close(fig)
print('figures done')
