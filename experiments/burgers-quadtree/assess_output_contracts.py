"""Audit supplemental output contracts against qualified primary coefficients."""
from pathlib import Path
import subprocess,json,numpy as np
H=Path(__file__).resolve().parent;job='c1outputcontracts';r=json.loads((H/f'results/{job}/output/result.json').read_text());assert r['complete'] and len(r['rows'])==40
script=r'''
from pathlib import Path
import json,hashlib,numpy as np,functools
ns=Path('/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117');job='c1outputcontracts';r=json.loads((ns/job/'output/result.json').read_text());manifests={}
def verify(job,name):
 if job not in manifests:manifests[job]=dict(x.split('  ',1)[::-1] for x in (ns/job/'OUTPUTS.sha256').read_text().splitlines())
 p=ns/job/'output'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==manifests[job]['output/'+name];return p
rows=[]
for x in r['rows']:
 fam=x['family'];case=x['case'];primary='c1compresscost' if fam=='A' else 'c1bcomplete';d=np.load(ns/('c1data' if fam=='A' else 'c1familyb')/'output'/f'case{case+(128 if fam=="A" else 32)}.npz');norm=float(d['initial_norm'])
 for contract,methods in x['times'].items():
  for name in methods:
   original=('compressed384' if fam=='A' else 'tree768') if name=='local' else name+('' if fam=='A' else '_dt0.00625');a=np.load(verify(job,f'{fam}_{contract}_{name}_case{case}.npz'));b=np.load(verify(primary,f'{original}_case{case}_block0_rep0.npz'))['coefficients'];coef=a['coefficients'];assert coef.shape==b.shape;delta=float(np.linalg.norm((coef-b).reshape(len(coef),-1),axis=1).max()/(2*norm));shapes=json.loads(str(a['output_shapes_json']));expected=[[4,4096,4096],[4096,4096]] if contract.startswith('five') else [[4096,4096]];assert shapes==expected;assert np.isfinite(a['output_norms']).all() and np.all(a['output_norms']>0)
   rows.append(dict(family=fam,case=case,contract=contract,name=name,coefficient_change=delta,shape_pass=True))
print(json.dumps(rows))
'''
audit=json.loads(subprocess.check_output(['ssh','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=script.encode()));assert max(x['coefficient_change'] for x in audit)<1e-10
primary={fam:json.loads((H/f'evidence/{name}-assessment.json').read_text()) for fam,name in [('A','c1compresscost'),('B','c1bcomplete')]};rows=[];rng=np.random.default_rng(2026111819)
for fam in ('A','B'):
 cases=[x for x in r['rows'] if x['family']==fam];count=len(cases);sample=rng.integers(0,count,(10000,count))
 for contract in ('five_resident','final_host','five_host'):
  methods=[]
  for name in cases[0]['times'][contract]:
   original=('compressed384' if fam=='A' else 'tree768') if name=='local' else name+('' if fam=='A' else '_dt0.00625');p=next(x for x in primary[fam]['methods'] if x['name']==original);times=np.array([x['times'][contract][name] for x in cases]);methods.append(dict(name=name,qualified=p['qualified'],worst_error_plus_uncertainty=p['worst_error_plus_uncertainty'],median_seconds=float(np.median(times)),per_case_block_medians=np.median(times,axis=2).tolist(),first_case_cold_seconds=cases[0]['cold'][contract+':'+name]))
  baseline=min([x for x in methods if x['name']!='local' and x['qualified']],key=lambda x:x['median_seconds']);rom=next(x for x in methods if x['name']=='local');ratio=np.asarray(rom['per_case_block_medians'])/np.asarray(baseline['per_case_block_medians']);ci=np.quantile(np.median(ratio[sample],axis=1),[.025,.975],axis=0);median=np.median(ratio,axis=0);fraction=np.mean(ratio<1,axis=0);q90=np.quantile(ratio,.9,axis=0)
  rows.append(dict(family=fam,contract=contract,methods=methods,baseline=baseline['name'],median_time_ratio=median.tolist(),paired_case_bootstrap_95=ci.tolist(),fraction_faster=fraction.tolist(),ratio_q90=q90.tolist(),useful_speed_pass=bool(rom['qualified'] and np.all(ci[1]<1) and np.all(fraction>=.75) and np.all(q90<=1.1)),substantial_speed_pass=bool(rom['qualified'] and np.all(ci[1]<1) and np.all(fraction>=.75) and np.all(q90<=1.1) and np.all(median<=.5)),cache_setup_seconds=cases[0]['setup_seconds']))
result=dict(rows=rows,coefficient_audit=audit,maximum_coefficient_change=max(x['coefficient_change'] for x in audit),contract='same qualified primary physical coefficients and fixed decoder, all dense shapes verified; supplemental development output/transfer contracts; no unseen confirmation')
(H/'evidence/c1outputcontracts-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='methods'} for x in rows],indent=2))
