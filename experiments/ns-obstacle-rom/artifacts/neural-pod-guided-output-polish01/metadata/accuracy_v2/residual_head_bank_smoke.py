"""Synthetic CPU serialization and live-evaluation contract; no trained inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
from accuracy_v2.spatial_bank import initialize
from accuracy_v2.independent_residual_heads import initialize_heads,evaluate_live
from accuracy_v2.residual_head_bank import ResidualHeadBank

def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    parent=initialize(rank=4,width=8,depth=2,seed=123);parent.save(out/'synthetic-parent.npz');raw=(out/'synthetic-parent.npz').read_bytes();parent_sha=hashlib.sha256(raw).hexdigest()
    heads=initialize_heads();bank=ResidualHeadBank(raw,parent_sha,heads,{'synthetic':True});bank.save(out/'synthetic-composite.npz');loaded=ResidualHeadBank.load(out/'synthetic-composite.npz')
    x=np.array([[.4,.1],[.7,.2],[1.4,.3],[0.,.1],[.8,0.],[.8,.41],[.25,.2]],dtype=np.float64)
    pv,pg=parent.evaluate(x);v,g=loaded.evaluate(x);old_v,old_g=evaluate_live(parent,heads,x)
    rng=np.random.default_rng(456);moved=[]
    for head in heads:
        h=list(head);w,b=h[-1];h[-1]=(rng.normal(size=w.shape)*.01,rng.normal(size=b.shape)*.01);moved.append(h)
    changed=ResidualHeadBank(raw,parent_sha,moved,{'synthetic':True});changed.save(out/'synthetic-moved.npz');roundtrip=ResidualHeadBank.load(out/'synthetic-moved.npz');cv,cg=changed.evaluate(x);rv,rg=roundtrip.evaluate(x)
    np.savez_compressed(out/'arrays.npz',points=x,parent_values=pv,parent_gradients=pg,initial_values=v,initial_gradients=g,original_helper_values=old_v,original_helper_gradients=old_g,moved_values=cv,moved_gradients=cg,reloaded_values=rv,reloaded_gradients=rg)
    assert np.array_equal(pv,v) and np.array_equal(pg,g) and np.array_equal(v,old_v) and np.array_equal(g,old_g)
    assert np.array_equal(cv,rv) and np.array_equal(cg,rg)
    blocked=roundtrip.values_blocked(x,3)
    np.savez_compressed(out/'blocked-arrays.npz',blocked_values=blocked,full_values=rv)
    assert np.allclose(blocked,rv,rtol=1e-13,atol=1e-14)
    assert loaded.parent_bytes==raw and all(not a.flags.writeable for layer in loaded.parent.layers for a in layer)
    for prefix in (1,2,4):
        a,b=roundtrip.evaluate(x,prefix);assert np.array_equal(a,rv[:,:,:prefix]) and np.array_equal(b,rg[:,:,:,:prefix])
    bad=[list(h) for h in heads];w,b=bad[0][0];bad[0][0]=(w.astype(np.float32),b)
    try:ResidualHeadBank(raw,parent_sha,bad,{})
    except ValueError:pass
    else:raise AssertionError('non-f64 head accepted')
    result={'scope':'synthetic CPU-only serialization/live evaluation; no actual parent/TRAIN/JAX','initial_exact_parent':True,'roundtrip_exact':True,'blocked_value_max_abs_gap':float(np.max(np.abs(blocked-rv))),'blocked_arrays_sha256':hashlib.sha256((out/'blocked-arrays.npz').read_bytes()).hexdigest(),'prefix_exact':True,'embedded_parent_exact':True,'parent_arrays_readonly':True,'non_f64_rejected':True,'arrays_sha256':hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest()}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    import sys
    run(sys.argv[1])
