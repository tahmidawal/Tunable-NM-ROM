"""Generate compact main-paper experiments from frozen accepted run snapshots."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
E=HERE/'evidence/main-experiments-2026-09-20'
SOURCES={
 'burgers':('worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d007/summary.json','3e988887f590898d7186611ab884eb2a06ecd070','accepted final; same-grid only'),
 'poisson':('worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/replay07/archive/out/result.json','ed016f3f','accepted development; final pending'),
 'heat':('worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/extra03/archive/out/result.json','0abc0f14e4bd692ccfcdb616081780b179ab9547','audited development snapshot; CG measurement pending'),
 'ns':('worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/confirmation06b/paper_summary.json','a4404fe9cd45f05d5b2782ad00bcc2b451f9b5b1','accepted development; final pending'),
}
def digest(b):return hashlib.sha256(b).hexdigest()
def refresh():
 E.mkdir(parents=True,exist_ok=True);manifest={}
 for key,(path,commit,status) in SOURCES.items():
  raw=(ROOT/path).read_bytes();(E/f'{key}.json').write_bytes(raw)
  manifest[key]=dict(path=path,retention_commit=commit,status=status,sha256=digest(raw))
 (E/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def table(name,headers,rows,align):
 lines=[r'% Generated from hash-pinned accepted run records; do not edit.',r'\small',r'\begin{tabular}{@{}'+align+'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
 if name.startswith(('TR_3d_linear','TR_3d_nonlinear')):
  groups=['Poisson3D (development)','Heat3D (development)'] if name.startswith('TR_3d_linear') else ['Burgers3D (final)','NS3D (development)']
  lines.insert(4, ' & '+r' \multicolumn{3}{c}{'+groups[0]+'} & '+r'\multicolumn{3}{c}{'+groups[1]+'}'+r' \\')
 lines+=[' & '.join(row)+r' \\' for row in rows];lines += [r'\bottomrule',r'\end{tabular}']
 (HERE/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n')
 (HERE/'tables-md'/f'{name}.md').write_text('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows))
def aggregate(data,kind):
 groups=defaultdict(list)
 for row in data['invocations']:
  if row['intervals']==32:groups[row['method']].append(row)
 out={}
 for method,rr in groups.items():
  assert all(r['finite'] for r in rr)
  values=[r['same_grid_error'] if kind=='poisson' else r['same_grid']['current_evolved'] for r in rr]
  times=[r['device_ms'] for r in rr]
  # Maximum over every retained case and repeat; CG failures cannot disappear.
  out[method]=dict(error=100*max(values),ms=statistics.median(times),calls=len(rr),cases=len({r['case'] for r in rr}),repetitions_ms=times,nonstationary=sum(not r['stationary'] if kind=='poisson' else r['nonstationary_solves']>0 for r in rr))
 return out
if '--refresh' in sys.argv:refresh()
manifest=json.loads((E/'manifest.json').read_text());data={}
for k,v in manifest.items():
 b=(E/f'{k}.json').read_bytes();assert digest(b)==v['sha256'];data[k]=json.loads(b)
assert data['burgers']['final_evaluation']['complete']['complete']
assert data['burgers']['final_evaluation']['primary_seed_index']==0
assert all(x['passed'] for x in data['burgers']['final_evaluation']['replay_audits'])
for kind in ('poisson','heat'):
 assert data[kind]['complete'] and data[kind]['backend']=='gpu' and data[kind]['x64']
assert data['burgers']['checksums_passed'] and data['burgers']['remote_directory_removed']
B={r['method']:dict(error=100*r['worst_evolved'],ms=r['median_ms'],cases=r['cases'],repetitions_ms=r['timing_repetitions_ms']) for r in data['burgers']['rows']}
P=aggregate(data['poisson'],'poisson');H=aggregate(data['heat'],'heat')
N={r['method']:dict(error=r['same_grid_evolved_worst_percent'],ms=r['gpu_ms'],cases=r['cases'],nonstationary=r['nonstationary_cases']) for r in data['ns']['rows']}
controls={'burgers':'fom_n33_dt0.01_nt1e-02_lt5e-01','poisson':'cg_identity_plain_rtol1e-02','heat':None,'ns':'fom_dt0.01'}
# Representative frozen settings, not per-case or final-truth selections.
methods={
 'burgers':['rom_q0','rom_q192','pod_256','fno3d','unet3d','deeponet3d','transolver3d'],
 'poisson':['nmrom_K16_q0_dense','nmrom_K16_q96_dense','pod128_galerkin','fno3d_w24_m8','unet3d_w16','deeponet3d_r128_w16_podinit','transolver3d_w48_s32'],
 'heat':['nmrom_K32_q0_dense','nmrom_K32_q96_dense','pod128_exact','fno3d_w16_m6','unet3d_w8','deeponet3d_r128_w16','transolver3d_w48_s32'],
 'ns':['nmrom_pca64_free_q0','nmrom_pca64_free_q256','pod_galerkin_320','fno3d_increment_projected','unet3d_increment_projected','deeponet3d_increment_projected','transolver3d_increment_projected']}
labels=['NM-ROM, uncorrected','NM-ROM, corrected','POD','FNO','U-Net','DeepONet','Transolver']
D=dict(burgers=B,poisson=P,heat=H,ns=N);selected={}
for kind,lookup in D.items():
 selected[kind]=[]
 for m in methods[kind]+([controls[kind]] if controls[kind] else []):
  r=dict(lookup[m],method=m,job=data[kind].get('job_id'),status=manifest[kind]['status'])
  r['speedup']=lookup[controls[kind]]['ms']/r['ms'] if controls[kind] else None
  selected[kind].append(r)
def vals(kind,i):
 r=selected[kind][i];return [f"{r['error']:.3f}",f"{r['ms']:.3f}",f"{r['speedup']:.3g}"+r'$\times$' if r['speedup'] is not None else '---']
for name,kinds,names in [('TR_3d_linear',['poisson','heat'],['Poisson3D (dev.)','Heat3D (dev.)']),('TR_3d_nonlinear',['burgers','ns'],['Burgers3D (final)','NS3D (dev.)'])]:
 rows=[[label]+vals(kinds[0],i)+vals(kinds[1],i) for i,label in enumerate(labels)]
 row=['FOM']
 for k in kinds:row+=vals(k,7) if controls[k] else ['---']*3
 rows.append(row)
 table(name,['Method']+[x for _ in kinds for x in ['Error (\\%)','GPU ms','$S$']],rows,'lrrrrrr')
 table(name+'_fom_only',['Method']+[x for _ in kinds for x in ['Error (\\%)','GPU ms','$S$']],rows[:2]+rows[-1:],'lrrrrrr')
# Independent Burgers seed results retained beside final primary, not pooled.
seed={r['method']:r for r in data['burgers']['seed1']['rows']}
seedrows=[]
for m,label in zip(methods['burgers']+[controls['burgers']],labels+['FOM']):
 r=B[m];r1=seed[m]
 seedrows.append([label,f"{r['error']:.3f}",f"{r['ms']:.3f}",f"{100*r1['worst_evolved']:.3f}",f"{r1['median_ms']:.3f}"])
table('TR_b3d_seeds',['Method','Seed 0 error (\\%)','Seed 0 ms','Seed 1 error (\\%)','Seed 1 ms'],seedrows,'lrrrr')
(HERE/'tables/main-experiments-provenance.json').write_text(json.dumps(dict(sources=manifest,selected=selected,controls=controls,definitions='Worst same-grid relative L2; B/NS initial-normalised evolved, H current-normalised evolved; P steady. Median GPU query time. One fixed FOM denominator per PDE. Heat CG absent, no speedup inferred.'),indent=2)+'\n')
print('Generated two main 3D comparison tables; all inputs hash verified; no live training inputs.')
