"""Publish only retained actual-grid trajectory visualizations."""
from pathlib import Path
import datetime,hashlib,json,shutil
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parents[1]
source=H/'results/b5visual/output'
retention=json.loads((H/'manifests/retention-b5visual.json').read_text())
assert retention['local_and_remote_verified'] and retention['terminal_state']=='COMPLETED'
r=json.loads((source/'result.json').read_text())
assert r['status']=='complete_saved_trajectory_visualization' and not r['final_test_opened']
assert r['new_pde_solves']==0 and r['field_shape']==[4096,4096]
canonical=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports')
slug='2026-11-19-tree-embedding-solve-evolution'
assets=canonical/(slug+'-assets');assets.mkdir(exist_ok=True)
manifest=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for p in sorted(source.rglob('*.png')):
 if p.name.endswith('_4096.png'):
  with Image.open(p) as im:assert im.size==(4096,4096)
 q=assets/p.relative_to(source);q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
 manifest.append(dict(file=str(q),sha256=sha(q),bytes=q.stat().st_size))
for p in source.rglob('*.npy'):
 a=np.load(p,mmap_mode='r');assert a.shape==(4096,4096) and a.dtype==np.float64
 assert any(v['file']=='output/'+str(p.relative_to(source)) for v in retention['files'])
z=assets/'full-resolution-images.zip';shutil.copy2(source/z.name,z)
manifest.append(dict(file=str(z),sha256=sha(z),bytes=z.stat().st_size))
parts=[f'# Burgers solve evolution and full-resolution spatial errors\n\n'
 'Saved, qualified B4 PDE trajectories decoded on actual 4096 × 4096 grids. '
 'These are time-evolved solves, not independently fitted reference snapshots. '
 'The reference is a numerically qualified spectral solution, not an analytic exact solution. '
 'Error maps compare each model with that reference; a reference error against exact truth is unavailable.\n\n'
 f'[Download all full-resolution field and signed-error PNGs]({slug}-assets/full-resolution-images.zip). '
 'Each standalone PNG has one pixel per grid node. Overview figures are reduced for display. '
 'The underlying float64 arrays remain in the checksum-verified run output.\n\n'
 'Our model here is the admitted direct local-coefficient tree. The nonlinear-child PDE route failed solver convergence and is not substituted into these plots. '
 'The global control is the POD–Fourier model, not a reproduced native neural baseline. '
 'Field colors share one scale within each case; signed-error colors share another scale across both methods and all times. '
 'Negative errors are blue, positive errors red. Rows at time zero show initialization.\n']
html=['<!doctype html><meta charset="utf-8"><title>Full-resolution Burgers fields</title>',
 '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}img{max-width:100%}</style>',
 '<h1>Full-resolution Burgers solve fields</h1><p>Every individual image is 4096 × 4096 pixels. Open a link and zoom to inspect native pixels. Field and error color scales are shown in each overview.</p>']
for case in r['cases']:
 label=case['case'];m=case['metrics'];selection='Representative pulse' if label=='dev000' else 'Worst direct-tree development case'
 parts.extend([f'## {selection}: {label}\n',
  f'![Reference, direct tree, global control and signed errors over time]({slug}-assets/{label}-evolution.png)\n',
  f'![Full-grid errors over time]({slug}-assets/{label}-errors.png)\n',
  '|Time|Tree relative L2|Global relative L2|Tree maximum absolute error|Global maximum absolute error|\n|---:|---:|---:|---:|---:|'])
 for j,t in enumerate(r['times']):
  a=m['tree_direct24'];b=m['global_stable960']
  parts.append(f'|{t:.2f}|{100*a["relative_L2"][j]:.6f}%|{100*b["relative_L2"][j]:.6f}%|{a["max_absolute_error"][j]:.8f}|{b["max_absolute_error"][j]:.8f}|')
 parts.append(f'\nThe reference convergence allowance is {100*m["tree_direct24"]["reference_allowance"]:.8f} percentage points. '
  f'The largest final-field discrepancy from the originally retained B4 output is '
  f'{max(x["final_retained_grid_relative_change"] for x in m.values()):.3e} in the current-reference norm.\n')
 html.append(f'<h2>{selection}: {label}</h2><img src="{label}-evolution.png"><table><tr><th>Time</th><th>Reference</th><th>Our tree</th><th>Global</th><th>Tree error</th><th>Global error</th></tr>')
 for j,t in enumerate(r['times']):
  html.append(f'<tr><th>{t:.2f}</th>')
  for method,kind in [('reference','field'),('tree_direct24','field'),('global_stable960','field'),('tree_direct24','signed_error'),('global_stable960','signed_error')]:
   html.append(f'<td><a href="{label}/{method}_{kind}_t{j}_4096.png">4096 × 4096</a></td>')
  html.append('</tr>')
 html.append('</table>')
parts.append(f'\n[Browse all individual full-resolution images]({slug}-assets/index.html).\n\n'
 '## Glossary\n\n'
 '- **Relative L2 error:** the norm of the model-minus-reference field divided by the reference norm at the same time; computed using all grid entries.\n'
 '- **Signed spatial error:** model value minus reference value at each location, in field units.\n'
 '- **Maximum absolute error:** largest pointwise discrepancy anywhere on the complete grid.\n'
 '- **Reference allowance:** empirical numerical convergence uncertainty, not an exact-error certificate.\n'
 '- **Direct tree:** retained global parent plus independently adjustable local correction coefficients.\n'
 '- **Global control:** one spatial basis formed from training POD directions and prescribed Fourier completion.\n')
(assets/'index.html').write_text('\n'.join(html))
text='\n'.join(parts)+'\n';(canonical/(slug+'.md')).write_text(text);(H/'reports'/(slug+'.md')).write_text(text)
manifest.append(dict(file=str(assets/'index.html'),sha256=sha(assets/'index.html')))
(H/'manifests/solve-evolution-publication.json').write_text(json.dumps(dict(
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_result_sha256=sha(source/'result.json'),
 raw_retention='manifests/retention-b5visual.json',report=str(canonical/(slug+'.md')),files=manifest),indent=2)+'\n')
with Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md').open('a') as f:
 f.write('\n\n### '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' — Full-resolution saved PDE evolution published\n\n'
  f'Run b5visual is retained and verified. Published {len(r["cases"])} cases across {len(r["times"])} saved times, '
  'qualified reference/direct tree/global fields and signed errors at actual4096-square resolution, plus current-reference L2 and maximum-error curves. '
  'No new solves or fitting; final grids reproduce the earlier B4 outputs and all dense-grid errors agree with saved physical metrics. '
  f'Report: reports/{slug}.md. Float64 arrays and raw images have verified local/remote retention. Final-test solutions remain unopened.\n')
print(canonical/(slug+'.md'))
