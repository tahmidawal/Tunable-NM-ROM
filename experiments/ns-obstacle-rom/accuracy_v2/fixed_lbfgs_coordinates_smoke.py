"""Tiny synthetic CPU proof of coordinate algebra; no actual fit or model."""
from pathlib import Path
import hashlib,json
import numpy as np
from accuracy_v2.fixed_lbfgs_coordinates import direction,physical,gradient_z,diagonal

def original_direction(g,pairs):
    q=g.copy();alphas=[]
    for s,y in reversed(pairs):rho=1/np.dot(s,y);a=rho*np.dot(s,q);alphas.append(a);q-=a*y
    if pairs:q*=np.dot(pairs[-1][0],pairs[-1][1])/np.dot(pairs[-1][1],pairs[-1][1])
    for (s,y),a in zip(pairs,reversed(alphas)):q+=s*(a-np.dot(y,q)/np.dot(s,y))
    return -q

def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(12345);n=7
    m=rng.normal(size=(n,n));A=m.T@m+np.eye(n);b=rng.normal(size=n);anchor=rng.normal(size=n);z=rng.normal(size=n);D=np.geomspace(.2,5.,n);theta=physical(z,anchor,D);g=A@theta+b
    pairs=[]
    for _ in range(3):s=rng.normal(size=n);pairs.append((s,A@s))
    dp,metrics=direction(g,pairs,D);gz=gradient_z(g,D);zpairs=[(s/D,D*y) for s,y in pairs];dz=original_direction(gz,zpairs);mapped=D*dz
    identity,_=direction(g,pairs,np.ones(n));original=original_direction(g,pairs);eps=1e-6;fd=np.zeros(n)
    def fun(v):t=physical(v,anchor,D);return .5*t@A@t+b@t
    for i in range(n):e=np.zeros(n);e[i]=eps;fd[i]=(fun(z+e)-fun(z-e))/(2*eps)
    arrays={'A':A,'b':b,'anchor':anchor,'z':z,'D':D,'theta':theta,'gradient_theta':g,'gradient_z':gz,'gradient_z_fd':fd,'physical_direction':dp,'mapped_z_direction':mapped,'identity_direction':identity,'original_direction':original}
    for i,(s,y) in enumerate(pairs):arrays[f's{i}']=s;arrays[f'y{i}']=y
    np.savez_compressed(out/'arrays.npz',**arrays)
    r={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(Path(__file__).with_name('fixed_lbfgs_coordinates.py').read_bytes()).hexdigest(),'scope':'synthetic CPU-only coordinate algebra; no chosen production scale/model/optimizer execution','arrays_sha256':hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest(),'mapped_direction_max_gap':float(np.max(np.abs(dp-mapped))),'slope_gap':float(abs(g@dp-gz@dz)),'gradient_chain_fd_max_gap':float(np.max(np.abs(gz-fd))),'identity_scale_max_gap':float(np.max(np.abs(identity-original))),'curvature_inner_product_gaps':[float(abs(s@y-(s/D)@(D*y))) for s,y in pairs],'production_hidden_coordinates':22272,'production_output_coordinates':520,'metrics':metrics}
    (out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
    assert np.allclose(dp,mapped,rtol=1e-12,atol=1e-12) and np.allclose(gz,fd,rtol=1e-7,atol=1e-7) and np.allclose(identity,original,rtol=1e-14,atol=1e-14)
    assert np.array_equal(physical(np.zeros(n),anchor,D),anchor) and len(diagonal(1.,1.))==22792
    print(json.dumps(r,indent=2))
if __name__=='__main__':
    import sys
    run(sys.argv[1])
