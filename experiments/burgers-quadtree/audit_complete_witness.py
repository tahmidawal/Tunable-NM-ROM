"""Checksum-verified remote NumPy dense-field audit; download only plotting samples."""
from pathlib import Path
import subprocess,sys,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;job=sys.argv[1];assert job in ('c1compresscost','c1bcomplete','c3cost');target=H/f'evidence/{job}-witness.npz'
script=r'''
from pathlib import Path
import numpy as np,json,hashlib,sys,importlib.util
job=JOB;root=Path('/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117')/job;case=0 if job=='c1bcomplete' else 9;p=root/'output'/f'final4096_case{case}.npz';manifest=dict(line.split('  ',1)[::-1] for line in (root/'OUTPUTS.sha256').read_text().splitlines());digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==manifest['output/'+p.name];d=np.load(p);ref=d['reference'];row=next(x for x in json.loads((root/'output/result.json').read_text())['rows'] if x['case']==case)
if job!='c1bcomplete':
 ph=np.load(root.parent/'c1data/output/case137.npz')['parameters'];x=np.arange(4096)/4095;gx=np.exp(-(x-ph[0])**2/(2*ph[2]**2));gy=np.exp(-(x-ph[1])**2/(2*ph[2]**2));gx[[0,-1]]=0;gy[[0,-1]]=0;norm=ph[3]*np.linalg.norm(gx)*np.linalg.norm(gy);mapping={k:k for k in d.files if k!='reference'};local='compressed384' if job=='c1compresscost' else 'neural';baseline='spectral48'
else:
 path=root/'experiments/burgers-quadtree/family_b.py';spec=importlib.util.spec_from_file_location('FB',path);FB=importlib.util.module_from_spec(spec);spec.loader.exec_module(FB);ph=json.loads((root.parent/'c1familyb/output/parameters.json').read_text())['dev'][case];norm=np.linalg.norm(FB.initial(4095,ph));mapping={'tree768':'tree768','uniform1152':'uniform1152','spectral48':'spectral48_dt0.00625'};local='tree768';baseline='spectral48'
errors={};samples={'reference':ref[::8,::8]}
for key,name in mapping.items():
 field=d[key];assert field.shape==ref.shape==(4096,4096);err=float(np.linalg.norm(field-ref)/norm);assert np.isclose(err,row['errors'][name][-1],rtol=1e-9,atol=1e-12);errors[name]=err;samples[key+'_error']=(field-ref)[::8,::8]
wrong=float(np.linalg.norm(d[local]-ref)/np.linalg.norm(ref));assert not np.isclose(wrong,errors[mapping[local]],rtol=1e-3)
audit=dict(job=job,case=case,sha256=digest,shape=list(ref.shape),numpy_errors=errors,wrong_final_normalization=wrong,negative_control_pass=True,display_stride=8,local=local,baseline=baseline)
np.savez_compressed(sys.stdout.buffer,audit_json=json.dumps(audit),**samples)
'''.replace('JOB',repr(job))
with target.open('wb') as f:subprocess.run(['ssh','tufts-login','OPENBLAS_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=script.encode(),stdout=f,check=True)
d=np.load(target);audit=json.loads(str(d['audit_json']));(H/f'evidence/{job}-witness-audit.json').write_text(json.dumps(audit,indent=2)+'\n');fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained');local=audit['local'];baseline=audit['baseline'];limit=float(np.max(np.abs(d[local+'_error'])))
for ax,key,title in zip(axes,['reference',local+'_error',baseline+'_error'],['Reference',local+' error',baseline+' error']):
 data=d[key];kw={} if key=='reference' else dict(vmin=-limit,vmax=limit);im=ax.imshow(data.T,origin='lower',extent=(0,1,0,1),cmap='viridis' if key=='reference' else 'RdBu_r',**kw);fig.colorbar(im,ax=ax,shrink=.8);ax.set(title=title,xlabel='x',ylabel='y')
fig.suptitle(f"{job}: actual4096 output at t=0.25, development case {audit['case']}; full-grid NumPy audit")
(H/'figures').mkdir(exist_ok=True);fig.savefig(H/f'figures/{job}-witness.png',dpi=160);print(json.dumps(audit))
