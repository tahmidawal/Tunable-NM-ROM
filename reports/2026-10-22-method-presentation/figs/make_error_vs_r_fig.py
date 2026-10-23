"""Error against bank width r (and against mesh), CN-R, 16-case subset, with r labelled as values (128, 256, 384, 512).
Same data and series as jcp-p1-scaling make_report.py plot 3 (error_vs_r_and_mesh_CNR.png); only the axis labelling changes.
Source: worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/checks/analysis.json (configurations[...]['e0_median_sub16'])."""
import json, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/checks/analysis.json'
cf = json.load(open(SRC))['configurations']
MESHES, RS, SER = (256, 1024, 4096), (128, 256, 384, 512), ['#2a78d6', '#eb6834', '#1baf7a']
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
rows = {}
for i, m in enumerate(MESHES):
    rs = [r for r in RS if f'{m}|R{r}|CNR' in cf]
    ys = [100 * cf[f'{m}|R{r}|CNR']['e0_median_sub16'] for r in rs]
    rows[m] = dict(zip(rs, ys))
    axs[0].plot(rs, ys, '-o', color=SER[i], lw=2, ms=8, label=f'mesh {m}²')
for r, y in rows[1024].items():
    axs[0].annotate(f'{y:.2f} %', (r, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10, color='#333')
axs[0].set_ylim(0.2, 0.78); axs[0].set_xticks(RS); axs[0].set_xticklabels([str(r) for r in RS])
axs[0].set_xlabel('bank width r (number of shapes used)'); axs[0].set_ylabel('median error (%)')
axs[0].set_title('Error against r (CN-R, 16-case subset)', loc='left'); axs[0].legend(frameon=False)
for i, r in enumerate((128, 256, 384)):
    ms_ = [m for m in MESHES if f'{m}|R{r}|CNR' in cf]
    axs[1].plot(ms_, [100 * cf[f'{m}|R{r}|CNR']['e0_median_sub16'] for m in ms_], '-o', color=SER[i], lw=2, ms=8, label=f'r = {r}')
axs[1].set_xscale('log', base=2); axs[1].set_xticks(MESHES); axs[1].set_xticklabels([f'{m}²' for m in MESHES])
axs[1].set_xlabel('mesh'); axs[1].set_ylabel('median error (%)'); axs[1].set_title('Error against mesh, per r', loc='left'); axs[1].legend(frameon=False)
for a in axs:
    a.grid(alpha=.3); [a.spines[s].set_visible(False) for s in ('top', 'right')]
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'error_vs_r.png'), dpi=150)
json.dump(dict(source=SRC, scheme='CNR', population='16-case subset', median_error_pct=rows), open(os.path.join(HERE, 'error_vs_r.json'), 'w'), indent=1)
print(rows)
