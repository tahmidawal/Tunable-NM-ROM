"""Collect compact, invocation-linked evidence; CPU only, no numerical solves.

Archived cluster files are read in place. Never modify or delete shared archives.
"""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
WT = ROOT / 'worktrees'
DATA = HERE / 'data'
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(value, allow_nan=True).encode(), mtime=0))


# This program also runs locally. It only reads files and prints compact JSON.
WORKER = r'''
import hashlib, json, io
from pathlib import Path
import numpy as np
REQ = __REQUEST__
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
store=Path(REQ['store'])
assert sha(store/'MANIFEST.sha256')==REQ['reference_manifest_sha256']
manifest={p:h for h,p in (line.split(None,1) for line in (store/'MANIFEST.sha256').read_text().splitlines() if line.strip())}
status=json.loads((store/'STATUS.json').read_text())
refs={}
def reference(cid):
 if cid not in refs:
  coh,c=cid.split('|'); assert coh=='dev6', 'Only development references are read here'
  prefix=f'2d/{coh}/ST/c{int(c):03d}'
  for suffix in ['.json','.npz']:
   assert sha(store/(prefix+suffix))==manifest[prefix+suffix]
  meta=json.loads((store/(prefix+'.json')).read_text())
  with np.load(store/(prefix+'.npz')) as z: f=np.array(z[meta['final'][2]])
  assert meta['metric_id']=='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1'
  assert meta['outcome']=='accepted' and status['dimension_status']['2']=='validated'
  refs[cid]=(f,meta,manifest[prefix+'.npz'],manifest[prefix+'.json'])
 return refs[cid]
result_path=Path(REQ['result'])
assert not REQ.get('expected_result_sha256') or sha(result_path)==REQ['expected_result_sha256']
doc=json.loads(result_path.read_text())
assert doc['backend']=='gpu' and doc['complete']
out={'result':doc,'result_path':str(result_path),'result_sha256':sha(result_path),'subjects':{}}
for lb,sub in doc['subjects'].items():
 p=result_path.parent/('S_'+lb.replace('|','_')+'.npz')
 if not p.exists(): continue
 raw_sha=sha(p)
 expected=REQ.get('expected_npz',{}).get(p.name)
 assert not expected or raw_sha==expected, str(p)
 with np.load(p,allow_pickle=False) as z:
  if 'f257' not in z.files: continue
  ids=[str(c) for c in z['case_ids']]
  if ids!=[f'dev6|{i}' for i in range(6)]: continue
  field=z['f257']; scores={}
  if 'T_W' in z.files:
   tix={str(c):j for j,c in enumerate(z['T_case_ids'])}
   assert all(np.array_equal(z['W'][j],z['T_W'][tix[c]]) for j,c in enumerate(ids))
   by={r['case']:r for r in sub['certification']}
   assert all(str(z['T_ident'][tix[c]])==by[c]['ident'] for c in ids)
  for j,cid in enumerate(ids):
   ref,meta,rsha,msha=reference(cid)
   n0=meta['judge']['n0']
   assert field[j].shape==ref.shape and np.isfinite(field[j]).all()
   et=np.linalg.norm((field[j]-ref).reshape(6,-1),axis=1)/n0
   scores[cid]={'e':float(max(et[1:])),'e_t':et.tolist(),'eps':meta['judge']['eps_ref'],
     'reference_sha256':rsha,'reference_meta_sha256':msha,'reference_eligible':True}
  metrics={}
  for key in ['raw','unweighted','cert','gamma','gamma_lit','jcond','it','tolratio','scale']:
   if key in z.files:
    a=z[key]
    if a.dtype.kind in 'if' and a.ndim and len(a)==6:
     metrics[key]={c:{'median':float(np.median(a[j])),'max':float(np.max(a[j])),
       'sum':float(np.sum(a[j])),'finite':bool(np.isfinite(a[j]).all())} for j,c in enumerate(ids)}
  meta={k:np.asarray(z[k]).item() for k in ['Rp','dim','M','mesh','dt','steps','scheme','form','rule','kind','policy','square'] if k in z.files and np.asarray(z[k]).ndim==0}
  out['subjects'][lb]={'npz_path':str(p),'npz_sha256':raw_sha,'expected_manifest_match':bool(expected),
   'scores':scores,'metrics':metrics,'meta':meta,'tol':z['tol'].tolist() if 'tol' in z.files else None,
   'invocation_link':'timed solver reproduces scoring trajectory identity; saved f257 is that trajectory',
   'head_T_W_equals_W': 'T_W' in z.files}
print(json.dumps(out,allow_nan=True))
'''


def run_worker(req, remote=False):
    code = WORKER.replace('__REQUEST__', repr(req))
    command = ['ssh', 'tufts-login', PY, '-'] if remote else ['/home/tahmid/Dev/.venv/bin/python', '-']
    p = subprocess.run(command, input=code, text=True, capture_output=True, check=True,
                       env=__import__('os').environ | {'OPENBLAS_NUM_THREADS':'1'})
    return json.loads(p.stdout)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ref = WT/'2026-10-06-jcp-references/experiments/jcp-references/store'
    for lane, w in [('jcp-heads','2026-10-18-jcp-heads'),('jcp-qualify','2026-10-18-jcp-qualify')]:
        base=WT/w/'experiments'/lane
        for mesh in [256,1024,4096]:
            target=DATA/f'{lane}-{mesh}-timed.json.gz'
            if target.exists(): continue
            attempt=base/'runs'/f'time{mesh}'
            listing={p:h for h,p in (line.split(None,1) for line in (attempt/'archive.sha256-listing').read_text().splitlines() if line.strip())}
            entry=next(p for p in listing if p.endswith('/result.json') and '/output/' in p)
            req={'result':f'/cluster/tufts/paralab/tawal01/burgers-archive/{lane}/time{mesh}/'+entry.removeprefix('./'),
                 'store':'/cluster/tufts/paralab/tawal01/jcp-ref/store',
                 'reference_manifest_sha256':sha(ref/'MANIFEST.sha256'),
                 'expected_result_sha256':listing[entry],
                 'expected_npz':{Path(p).name:h for p,h in listing.items() if p.endswith('.npz') and '/S_' in p}}
            print('Collecting',lane,mesh,flush=True)
            dump(target,run_worker(req,remote=True))
    for lane,w,rel in [
      ('dims','2026-10-18-jcp-dims','experiments/jcp-dims/runs/t2/slim/output/dtime_t2/result.json'),
      ('p1','2026-10-17-jcp-p1-scaling','experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time/result.json')]:
        target=DATA/f'{lane}-timed.json.gz'
        if target.exists(): continue
        print('Collecting',lane,flush=True)
        req={'result':str(WT/w/rel),'store':str(ref),'reference_manifest_sha256':sha(ref/'MANIFEST.sha256')}
        dump(target,run_worker(req))
    sources={
      'heads-export':WT/'2026-10-18-jcp-heads/experiments/jcp-heads/runs/EXPORT.json',
      'dims-export':WT/'2026-10-18-jcp-dims/experiments/jcp-dims/runs/EXPORT.json',
      'dims-accuracy':WT/'2026-10-18-jcp-dims/experiments/jcp-dims/checks/analysis-2d.json',
      'qualify-export':WT/'2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/EXPORT.json',
      'qualify-analysis':WT/'2026-10-18-jcp-qualify/experiments/jcp-qualify/checks/analysis.json',
      'p1-export':WT/'2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/EXPORT.json',
      'p1-analysis':WT/'2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/checks/analysis.json',
      'frontier3d':ROOT/'reports/2026-10-26-burgers3d-refinement/data/frozen-frontier.json.gz',
      'mesh3d':ROOT/'reports/2026-10-26-burgers3d-refinement/data/frozen-mesh.json.gz',
      'solver3d':ROOT/'reports/2026-10-26-burgers3d-refinement/data/lane-solver-export.json',
      'values3d':ROOT/'reports/2026-10-26-burgers3d-refinement/VALUES.json',
      'inventory3d':ROOT/'reports/2026-10-26-burgers3d-refinement/data/inventory.json.gz',
    }
    provenance=json.loads((HERE/'SOURCES.json').read_text()) if (HERE/'SOURCES.json').exists() else {}
    for name,p in sources.items():
        b=p.read_bytes();doc=json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
        dump(DATA/f'{name}.json.gz',doc)
        provenance[name]={'source':str(p),'sha256':sha(p),'retained':f'data/{name}.json.gz'}
    (HERE/'SOURCES.json').write_text(json.dumps(provenance,indent=2)+'\n')
    target=DATA/'historical3d-work.json.gz'
    if not target.exists():
        print('Collecting 3D invocation work',flush=True)
        inventory=json.loads(gzip.decompress(sources['inventory3d'].read_bytes()))
        selected=[o for o in inventory['observations'] if o['lane']=='dims' and o['cohort']=='val64'
                  and o['n_scored']==64 and o.get('selection_allowed')]
        bypath={}
        for o in selected:bypath.setdefault(o['result'],[]).append(o)
        out=[]
        for path,obs in bypath.items():
            raw=json.loads(Path(path).read_text()); raw_sha=sha(path)
            bykey={}
            for row in raw.get('rows',[]):
                if 'tag' in row and 'spec' in row:
                    bykey.setdefault(row['tag']+'|'+row['spec'],{})[str(row['case'])]=row
            for o in obs:
                rows=bykey.get(o['config_key'],{})
                if sorted(map(int,rows))!=list(range(64)):continue
                scorepath=Path(o['source']);scores=json.loads(scorepath.read_text())['scores'][o['config_key']]
                assert sorted(map(int,scores))==list(range(64))
                cases={}
                for cid,row in rows.items():
                    st=row.get('steps',{})
                    if not isinstance(st,dict) or not any(k in st for k in ['it','step_it','q']):break
                    iterations=st.get('it',st.get('step_it',st.get('q')))
                    gamma=st.get('gn',st.get('step_gn',st.get('gam')))
                    cases[cid]={'e':scores[cid]['e'],'eps':scores[cid]['eps_ref'],
                      'iterations':sum(iterations),'residual_median':float(np.median(st['rn'])) if 'rn' in st else None,
                      'gamma_max':float(max(gamma)) if gamma else None,'valid':row['valid']}
                if len(cases)!=64:continue
                out.append({'identity':o['identity'],'key':o['config_key'],'attempt':o['attempt'],
                  'job_id':o['job_id'],'source_result':path,'result_sha256':raw_sha,
                  'source_scores':str(scorepath),'scores_sha256':sha(scorepath),'cases':cases,
                  'qualification_evidence':o['qualification_evidence'],'qualification_scope':o['qualification_scope']})
        dump(target,out)


if __name__=='__main__':main()
