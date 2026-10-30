"""Exact immutable optimizer restart closure; no automatic job or budget authorization."""
from pathlib import Path
import hashlib,json,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(directory,state,arrays):
    started=time.monotonic();out=Path(directory).resolve();state=dict(state);references=set()
    def absolute(stem):return str((out/stem).resolve())
    if state.get('best'):
        state['best']=dict(state['best'],checkpoint=absolute(state['best']['checkpoint']));references.add(state['best']['checkpoint'])
    state['curvature_state_pairs']=[[absolute(a),absolute(b)] for a,b in state['curvature_state_pairs']]
    for pair in state['curvature_state_pairs']:references.update(pair)
    if state.get('last_measured_checkpoint'):
        state['last_measured_checkpoint']=absolute(state['last_measured_checkpoint']);references.add(state['last_measured_checkpoint'])
    closure={}
    for stem in sorted(references):
        record=Path(stem+'.json');payload=Path(stem+'.npz');r=json.loads(record.read_text())
        if sha(payload)!=r['array_sha256']:raise ValueError('checkpoint payload differs')
        closure[stem]={'record_sha256':sha(record),'array_sha256':sha(payload)}
    np.savez_compressed(out/'CONTINUATION.npz',**arrays)
    state.update(schema='ns-obstacle-optimizer-continuation-v2',checkpoint_closure=closure,array_sha256=sha(out/'CONTINUATION.npz'),config_sha256=sha(out/'CONFIG.json'),resume_requires_separate_exact_release=True,resume_loader_implemented=True)
    state['elapsed_seconds']+=time.monotonic()-started
    state['elapsed_boundary']='includes payload/closure hashing; excludes final receipt JSON write'
    (out/'CONTINUATION.json').write_text(json.dumps(state,indent=2)+'\n')
    return state

def load(directory,expected_bindings,expected_receipt_sha256):
    out=Path(directory).resolve();p=out/'CONTINUATION.json'
    if sha(p)!=expected_receipt_sha256:raise ValueError('released continuation differs')
    state=json.loads(p.read_text())
    if state['schema']!='ns-obstacle-optimizer-continuation-v2' or sha(out/'CONFIG.json')!=state['config_sha256'] or json.loads((out/'CONFIG.json').read_text())!=expected_bindings:raise ValueError('restart source/data/recipe differs')
    if sha(out/'CONTINUATION.npz')!=state['array_sha256']:raise ValueError('restart array differs')
    for stem,row in state['checkpoint_closure'].items():
        if sha(stem+'.json')!=row['record_sha256'] or sha(stem+'.npz')!=row['array_sha256']:raise ValueError('restart history differs')
    with np.load(out/'CONTINUATION.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    if set(arrays)!={'theta','gradient','adam_m','adam_v'} or any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('invalid optimizer arrays')
    shape=arrays['theta'].shape
    if len(shape)!=1 or any(a.shape!=shape for a in arrays.values()):raise ValueError('optimizer layout differs')
    pairs=[]
    for before,after in state['curvature_state_pairs']:
        if before not in state['checkpoint_closure'] or after not in state['checkpoint_closure']:raise ValueError('unbound curvature')
        with np.load(before+'.npz') as a,np.load(after+'.npz') as b:s=b['theta']-a['theta'];y=b['gradient']-a['gradient']
        sy=float(np.dot(s,y))
        if not np.isfinite(sy) or sy<=0:raise ValueError('invalid curvature pair')
        pairs.append((s,y,1/sy))
    if state['best'] and state['best']['checkpoint'] not in state['checkpoint_closure']:raise ValueError('unbound best checkpoint')
    return state,arrays,pairs
