"""Render existing fresh-cohort records; never runs a PDE or training job."""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'manuscript_tools/evidence'
FILES = ['fresh_s0b.json', 'fresh_s1b.json', 'fresh_s2.json']
records = [json.loads((EVIDENCE / f).read_text()) for f in FILES]
widths = [64, 96, 128, 192]
for d in records:
    assert d['cases'] == 128 and d['ladder_configurations'] == 450
for i in [0, 2]:
    assert all(m['timing']['gate_passed'] for m in records[i]['meshes'].values())

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout='constrained')
data = {}
for kind, label, colour, marker in [('head', 'Nonlinear head', '#2563a5', 'o'),
                                     ('span', 'Linear span', '#c26932', 's')]:
    errors = np.array([[d['meshes']['128']['arms'][f'r{r}_{kind}']['stats']['evolved_worst'] * 100
                        for r in widths] for d in records])
    ratios = np.array([[records[i]['meshes']['128']['arms'][f'r{r}_{kind}']['any_mesh_solve']['ratio']
                        for r in widths] for i in [0, 2]])
    for ax, values in zip(axes, [errors, ratios]):
        med = np.median(values, axis=0)
        ax.errorbar(widths, med, yerr=[med-values.min(0), values.max(0)-med],
                    marker=marker, color=colour, label=label, capsize=3, linewidth=1.5)
    data[kind] = {'worst_error_percent_by_seed': errors.tolist(),
                  'speedup_by_admitted_seed': ratios.tolist()}
for ax in axes:
    ax.set_yscale('log')
    ax.set_xticks(widths)
    ax.set_xlabel('Spatial-bank width')
    ax.grid(axis='y', which='major', color='#dddddd', linewidth=.6)
axes[0].set_ylabel('Worst-case relative error (%)')
axes[0].set_title('(a) Same bank, different dynamics', loc='left', fontsize=10)
axes[0].legend(frameon=False, fontsize=9, loc='best')
axes[1].set_ylabel('Matched solver time / model time')
axes[1].set_title('(b) Own-accuracy solver comparison', loc='left', fontsize=10)
axes[1].axhline(1, color='#666666', linewidth=1, linestyle='--')
axes[1].set_ylim(.025, 12)
out = ROOT / 'paper_revision/figs/F8_ns_head_span.pdf'
fig.savefig(out, metadata={'CreationDate': None, 'ModDate': None})
fig.savefig(out.with_suffix('.png'), dpi=180)
plt.close(fig)
data['widths'] = widths
data['sources'] = {f: hashlib.sha256((EVIDENCE/f).read_bytes()).hexdigest() for f in FILES}
(EVIDENCE/'figure_data.json').write_text(json.dumps(data, indent=2)+'\n')
print(out)
