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
C = dict(bg='#4878CF', accent='#D65F5F', green='#6ACC64', gray='#999999', purple='#956CB4')

d6 = json.load(open('vc_results6.json'))
d6b = json.load(open('vc_results6b.json'))
dn = json.load(open('vc_report6_norman.json'))

def pl(ax, s):
    ax.text(-0.18, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

# a: calibration ladder
ns = [0, 1, 2, 3, 5]
aff = [d6['zeroshot_crest']['mean']] + [d6b[f'affine_n{k}']['mean'] for k in [1, 2, 3, 5]]
aff_e = [d6['zeroshot_crest']['sd'] / np.sqrt(d6['zeroshot_crest']['n'])] + \
        [d6b[f'affine_n{k}']['sd'] / np.sqrt(d6b[f'affine_n{k}']['n']) for k in [1, 2, 3, 5]]
off = [np.nan] + [d6b[f'offset_n{k}']['mean'] for k in [1, 2, 3, 5]]
axes[0].errorbar(ns, aff, yerr=aff_e, marker='o', ms=4, color=C['purple'], capsize=2, label='CREST affine')
axes[0].plot(ns[1:], off[1:], marker='s', ms=3, color=C['gray'], ls='--', label='Offset only')
axes[0].axhline(0.846, color='k', ls='--', lw=0.8)
axes[0].text(2.6, 0.856, 'noise ceiling (SB)', fontsize=6)
axes[0].axhline(0.763, color=C['accent'], ls=':', lw=0.8)
axes[0].text(2.6, 0.72, 'line-mean oracle', fontsize=6, color=C['accent'])
axes[0].set_xlabel('Calibration drugs in target line')
axes[0].set_ylabel('Pearson r (top-2000 HV genes)')
axes[0].set_title('Calibration ladder', fontsize=8)
axes[0].set_xticks(ns)
axes[0].legend(frameon=False, loc='lower right')
pl(axes[0], 'a')

# b: zero-shot diagnosis: basal->background correlation per line
bg = d6['basal_bg_cor']
ks = list(bg.keys())
axes[1].bar(range(len(ks)), [bg[k] for k in ks], color=C['bg'])
axes[1].axhline(np.mean(list(bg.values())), color='k', ls='--', lw=0.8,
                label=f"mean {np.mean(list(bg.values())):.2f}")
axes[1].set_xticks(range(len(ks)))
axes[1].set_xticklabels(ks, rotation=45, ha='right', fontsize=6)
axes[1].set_ylabel('Pearson r')
axes[1].set_title('Basal state cannot predict\nline background (zero-shot fails)', fontsize=8)
axes[1].legend(frameon=False)
pl(axes[1], 'b')

# c: Norman interaction transfer: additive vs kNN-corrected
pc = dn['per_combo']
ra = np.array([v['r_add'] for v in pc.values()])
rc = np.array([v['r_crest'] for v in pc.values()])
axes[2].scatter(ra, rc, s=6, color=C['gray'], alpha=0.6, linewidths=0)
lim = [-0.4, 1.05]
axes[2].plot(lim, lim, color='k', lw=0.7, ls='--')
axes[2].set_xlim(lim); axes[2].set_ylim(lim)
axes[2].set_xlabel('Additive model r')
axes[2].set_ylabel('Interaction-transfer r')
axes[2].set_title('Interaction transfer (Norman, n=131)', fontsize=8)
axes[2].text(-0.35, 0.9, 'no systematic gain\n(Wilcoxon p = 0.91)', fontsize=6.5)
pl(axes[2], 'c')

fig.subplots_adjust(wspace=0.45)
fig.savefig('fig7_crest.png')
plt.close(fig)
print('fig7 ok')
