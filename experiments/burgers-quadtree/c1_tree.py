"""Initial-only quadtree allocation span screen, frozen trained local features."""
from campaign import *
from owned_inputs import verified
import family_b as FB
import quadtree as QT,specialized as S,local_bank as B,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--family',choices=['A','B'],default='A');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local' if a.family=='A' else 'c1bbank','network.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1data' if a.family=='A' else 'c1bbank','pod.npz'))['basis']);dev=draw()[1][:32] if a.family=='A' else json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];points=jnp.asarray(E.coords(128,True));rows=[];blocks={};models={}
for c in set(QT.leaves(range(4))+QT.leaves([])):
 patch=c.patch();F=jnp.concatenate([B.features(points,[patch],32,'poly'),S.features(p,points,[patch],32)],axis=1);blocks[c]=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F)
recipes=[(128,0,32),(128,1,16),(128,2,16),(128,1,32),(128,2,32),(128,4,32),(64,1,32),(64,2,32)]
for case,ph in enumerate(dev[:32]):
 order=QT.initial_order(E.initial(4095,ph) if a.family=='A' else FB.initial(4095,ph));d=np.load(verified('c1data' if a.family=='A' else 'c1familyb',f'case{128+case if a.family=="A" else 32+case}.npz'));raw=np.asarray(d['coefficients']);tail=raw.copy();tail[:,:127,:127]=0;tailnorm=np.linalg.norm(tail.reshape(11,-1),axis=1)/(2*float(d['initial_norm']));Y=jnp.asarray(raw[:,:127,:127]).reshape(11,-1).T/(2*float(d['initial_norm']))
 for r,n,k in recipes:
  cells=QT.leaves(order[:n]);QT.validate(cells);key=(r,k,cells)
  name=f'r{r}_k{k}_split'+''.join(map(str,sorted(order[:n])))
  if key not in models:
   idx=list(range(k))+list(range(32,32+k));A=jnp.concatenate([V[:,:r]]+[blocks[c][:,idx] for c in cells],axis=1);Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];models[key]=Q
   np.savez_compressed(o/(name+'.npz'),basis=Q,singular_values=s,physical_to_orthogonal=vh[keep].T/s[keep],cells=np.array([(c.depth,c.i,c.j) for c in cells]))
  Q=models[key];err=np.sqrt(np.asarray(jnp.linalg.norm(Y-Q@(Q.T@Y),axis=0))**2+tailnorm**2);rows.append(dict(case=case,coarse=r,refined_parents=n,k=k,columns=r+len(cells)*2*k,model=name,per_time=err,evolved=float(err[[2,4,6,8,10]].max())))
 save(o/'result.json',dict(meta=meta(),rows=rows));print(case,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,family=a.family,contract='complete adaptive-depth leaf partitions chosen from initial field only; representation, not PDE solve'))
