"""Worst-case error against time per query on test64 (256 mesh, A100 80GB), from reports/summary.json.

  fig.py reports/summary.json reports/fig-error-time.png
"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s = json.loads(open(sys.argv[1]).read())
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
groups = [('paper (stored rows)', '#1f4e9c', 'o', [(r['ms'], r['worst'], r['name'].replace('paper: ', '').split(' (')[0]) for r in s['paper_rows'] if r['name'].startswith('paper')]),
          ('full-order solver', '#555555', 's', [(r['ms'], r['worst'], 'full-order solver') for r in s['paper_rows'] if r['name'].startswith('full')]),
          ('CROM-style (online residual solve)', '#c0392b', '^', [(r['ms'], r['worst'], '') for r in s['test_rows'] if r['name'].startswith('CROM')]),
          ('CoLoRA-style hyper-network (given the 5 parameters)', '#1e8449', 'D', [(r['ms'], r['worst'], '') for r in s['test_rows'] if r['name'].startswith('CoLoRA-style k')]),
          ('CoLoRA-style decoder, online residual solve', '#b9770e', 'v', [(r['ms'], r['worst'], '') for r in s['test_rows'] if r['name'].startswith('CoLoRA-style decoder')])]
for label, col, mk, pts in groups:
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=col, marker=mk, s=42, label=label, zorder=3)
    for x, y, t in pts:
        if t:
            ax.annotate(t, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=7, color=col)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('time per query, ms (median, batch 1, A100 80GB)'); ax.set_ylabel('worst-case error over 64 test cases, %')
ax.set_title('Burgers 2D, 256 mesh, test cohort: neural-field baselines and the stored paper rows', fontsize=9)
ax.grid(True, which='both', lw=.3, alpha=.5); ax.legend(fontsize=7, loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[2])
