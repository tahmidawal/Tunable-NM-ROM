"""Tiny synthetic NumPy-only formula fixture, no trained model or data."""
from pathlib import Path
import json,hashlib
import numpy as np
from accuracy_v2.spatial_bank import initialize
from accuracy_v2.independent_residual_heads import initialize_heads,evaluate_heads,evaluate_live

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    points=np.array([[.4,.1],[.7,.2],[1.4,.3],[0.,.1],[.8,0.],[.8,.41],[.25,.2]],dtype=np.float64)
    parent=initialize(rank=4,width=8,depth=2,seed=123)
    heads=initialize_heads();pv,pg=parent.evaluate(points);iv,ig=evaluate_live(parent,heads,points)
    assert np.array_equal(pv,iv) and np.array_equal(pg,ig)
    rng=np.random.default_rng(456)
    moved=[]
    for h in heads:
        h=list(h);w,b=h[-1];h[-1]=(rng.normal(size=w.shape)*.01,rng.normal(size=b.shape)*.01);moved.append(tuple(h))
    v,g=evaluate_heads(tuple(moved),points);fd=np.zeros_like(g);eps=1e-6
    for j in range(2):
        plus=points.copy();minus=points.copy();plus[:,j]+=eps;minus[:,j]-=eps
        fd[:,:,j,:]=(evaluate_heads(tuple(moved),plus)[0]-evaluate_heads(tuple(moved),minus)[0])/(2*eps)
    arrays={'points':points,'parent_values':pv,'parent_gradients':pg,'initial_values':iv,'initial_gradients':ig,'moved_values':v,'moved_gradients':g,'finite_difference_gradients':fd}
    for i,h in enumerate(moved):
        for j,(w,b) in enumerate(h):arrays[f'head{i}_W{j}']=w;arrays[f'head{i}_b{j}']=b
    np.savez_compressed(output/'arrays.npz',**arrays)
    gap=float(np.max(np.abs(g-fd)));assert gap<2e-7
    assert np.max(np.abs(v[3:]))<1e-14
    n=sum(w.size+b.size for h in heads for w,b in h);assert n==22792
    result={'scope':'synthetic NumPy formula only; no actual model/TRAIN/DEV/VAL, JAX or timing claim','initial_parent_exact':True,'four_independent_heads':True,'trainable_parameters':n,'gradient_max_abs_fd_gap':gap,'boundary_value_max_abs':float(np.max(np.abs(v[3:]))),'arrays_sha256':hashlib.sha256((output/'arrays.npz').read_bytes()).hexdigest()}
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':
    import sys
    run(sys.argv[1])
