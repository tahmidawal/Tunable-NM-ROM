"""Rescore preserved 3D screening timings; pure NumPy CPU analysis, no new solves.

Use blockwise physical-lattice Gram products in f64 and compare against direct field
differences on one timed trajectory per bank. Preserve coefficients, references,
minimal model parameters, products and source hashes for independent replay.
"""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import pickle
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
ROOT=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANE=ROOT/'worktrees/2026-10-18-jcp-dims/experiments/jcp-dims'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):p.write_bytes(gzip.compress(json.dumps(o).encode(),mtime=0))


def split_large(p):
    if p.stat().st_size<=48*1024*1024:return []
    payload=p.read_bytes();parts=[]
    for i,start in enumerate(range(0,len(payload),32*1024*1024)):
        q=p.with_name(p.name+f'.part{i:02d}');q.write_bytes(payload[start:start+32*1024*1024]);parts.append(q)
    assert hashlib.sha256(b''.join(q.read_bytes() for q in parts)).hexdigest()==hashlib.sha256(payload).hexdigest()
    p.unlink()
    return [{'file':q.name,'sha256':sha(q)} for q in parts]


def features(p,T,X):
    ang=2*np.pi*(X@p['freq'])
    h=np.concatenate((np.sin(ang),np.cos(ang)),axis=1)
    for w,b in p['net'][:-1]:
        a=h@w+b;h=a*(.5*(1+np.tanh(.5*a)))
    w,b=p['net'][-1]
    return ((64*p['scale']*np.prod(X*(1-X),axis=1))[:,None]*(h@w+b))@T


def main():
    shutil.copyfile(LANE/'vendor/quad3d/npcore.py',DATA/'independent_bank3d.py')
    base=LANE/'runs/t3s/slim/output/d3run_t3s'
    raw=json.loads((base/'result.json').read_text());assert raw['complete'] and raw['backend']=='gpu'
    panel=raw['timing']['screen'];assert panel['valid']
    Z=np.load(base/'W_t3s.npz')
    subjects={x['B'] for x in panel['invocations'] if not x['B'].startswith('T1|')}
    specs={x['name']:x for x in raw['config']['settings']}
    refdir=ROOT/'worktrees/2026-10-06-jcp-references/experiments/jcp-references'
    import sys
    sys.path.insert(0,str(refdir));import refstore
    refs=[];metas=[]
    for c in range(4):
        f,meta=refstore.load(3,'val64',c,'ST');assert refstore.reference_status(3,'val64',c)['eligible']
        refs.append(f);metas.append(meta)
    R=np.asarray(refs).reshape(24,-1);norm=np.sum(R*R,axis=1).reshape(4,6)
    np.savez_compressed(DATA/'timed3d-reference-fields.npz',fields=np.asarray(refs))
    cases=np.arange(1,64)/64
    X=np.stack(np.meshgrid(cases,cases,cases,indexing='ij'),-1).reshape(-1,3)
    models={'M2':LANE/'vendor/quad3d/inputs/model_M2/bank.pkl',
            'W1024':LANE/'inputs/model_W1024/bank.pkl'}
    source_banks={};products={};direct_checks={};coef_save={}
    for bank,model in models.items():
        b=pickle.loads(model.read_bytes());p=b['params'];T=b['rotation']
        dim=512 if bank=='M2' else 1024;T=T[:,:dim]
        assert all(a.dtype==np.float64 for a in [p['freq'],T])
        print('Scoring bank',bank,'dimension',dim,flush=True)
        pp={'freq':p['freq'],'scale':np.asarray(p['scale']),'T':T}
        for i,(w,bias) in enumerate(p['net']):pp[f'w{i}']=w;pp[f'b{i}']=bias
        np.savez_compressed(DATA/f'timed3d-model-{bank}.npz',**pp)
        source_banks[bank]={'source':str(model),'sha256':sha(model),'minimal_replay':f'timed3d-model-{bank}.npz'}
        source_banks[bank]['parts']=split_large(DATA/f'timed3d-model-{bank}.npz')
        gram=np.zeros((dim,dim));cross=np.zeros((24,dim))
        relevant=sorted(k for k in subjects if specs[k.split('|')[0]]['bank']==bank)
        control=relevant[0];C=Z[control][0]
        direct=np.zeros(6)
        for start in range(0,len(X),8192):
            end=min(start+8192,len(X));G=features(p,T,X[start:end])
            gram+=G.T@G;cross+=R[:,start:end]@G
            delta=C@G[:,:C.shape[-1]].T-R[:6,start:end]
            direct+=np.sum(delta*delta,axis=1)
        np.savez_compressed(DATA/f'timed3d-products-{bank}.npz',gram=gram,cross=cross,ref_norm_squared=norm)
        products[bank]=(gram,cross.reshape(4,6,dim))
        rc=C.shape[-1]
        quadratic=np.einsum('ti,ij,tj->t',C,gram[:rc,:rc],C)-2*np.sum(C*cross[:6,:rc],axis=1)+norm[0]
        rel=np.max(np.abs(np.sqrt(np.maximum(quadratic,0))-np.sqrt(direct))/metas[0]['judge']['n0'])
        assert rel<1e-10,(bank,rel)
        direct_checks[bank]={'subject':control,'case':0,'max_error_absolute_difference':float(rel),'passed':True}
    out={'source_result':str(base/'result.json'),'source_result_sha256':sha(base/'result.json'),
      'source_coefficients':str(base/'W_t3s.npz'),'source_coefficients_sha256':sha(base/'W_t3s.npz'),
      'raw_timing':raw,'reference_manifest_sha256':refstore.store_version(),
      'reference_metadata':metas,'models':source_banks,'direct_checks':direct_checks,'subjects':{}}
    for key in sorted(subjects):
        setting,rule=key.split('|')[:2];spec=specs[setting];A=np.asarray(Z[key]);r=A.shape[-1]
        ids=[int(v) for v in Z['cases|'+key]];assert ids==list(range(4))
        assert A.dtype==np.float64
        gram,cross=products[spec['bank']];g=gram[:r,:r];q=cross[:,:,:r]
        e2=np.einsum('cti,ij,ctj->ct',A,g,A)-2*np.sum(A*q,axis=-1)+norm
        assert np.min(e2)>=-1e-10*np.max(norm)
        et=np.sqrt(np.maximum(e2,0))/np.array([m['judge']['n0'] for m in metas])[:,None]
        ff=[v for v in panel['invocations'] if v['B']==key]
        assert len(ff)==8 and all(v['hashes_ok'] for v in ff)
        links={str(c):raw['link_evidence'][key+'#'+str(c)] for c in ids}
        assert all(v['identity'] and v['valid'] and v['fields_finite'] for v in links.values())
        coef_save[key]=A
        out['subjects'][key]={'setting':setting,'rule':rule,'bank':spec['bank'],
          'r':r,'M':raw['settings_M'][setting]['M'],
          'scores':{str(c):{'e':float(max(et[c,1:])),'e_t':et[c].tolist(),
             'eps':metas[c]['judge']['eps_ref']} for c in ids},
          'coefficient_entry':key,'timing_links':links,'timing':ff}
    np.savez_compressed(DATA/'timed3d-coefficients.npz',**coef_save)
    dump(DATA/'timed3d-screen.json.gz',out)
    sources=json.loads((HERE/'SOURCES.json').read_text())
    sources['screen3d']={'source':str(base/'result.json'),'sha256':sha(base/'result.json'),
       'retained':'data/timed3d-screen.json.gz','coefficients_sha256':sha(base/'W_t3s.npz'),
       'transformation':'f64 physical-lattice Gram rescoring; timed-output identity links checked; direct field checks retained'}
    (HERE/'SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Saved',len(out['subjects']),'timed and rescored subjects',direct_checks,flush=True)


if __name__=='__main__':main()
