from campaign import *
from owned_inputs import verified
from spectral_cost import make,PJ
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[];queries={m:make(output_mode=m) for m in ['sine','bilinear256']};rng=np.random.default_rng(2026111707)
for i,ph in enumerate(dev[:8]):
 initial=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(initial);ref=jnp.asarray(np.load(verified('dense01',f'dev{i}_final4096.npz'))['field'])
 for mode,q in queries.items():v,cold=timer(lambda:q(initial,ph[4]))
 times={m:[] for m in queries};errors={m:[] for m in queries};coeffs={};fields={}
 for rep in range(5):
  for mode in rng.permutation(list(queries)):
   E.burn(.4);v,sec=timer(lambda:queries[mode](initial,ph[4]));c,field=v
   times[mode].append(sec);errors[mode].append(float(jnp.linalg.norm(field-ref)/jnp.linalg.norm(initial)));coeffs[mode]=c;fields[mode]=field
   np.savez_compressed(o/f'{mode}_case{i}_rep{rep}.npz',coefficients=c,final_norm=float(jnp.linalg.norm(field)))
 # Qualify the interpolation extension at EVERY required evolved time against sine output.
 diffs=[]
 for c in coeffs['sine'][1:]:
  small=PJ.synth_mesh(c,256);x=jnp.arange(4096,dtype=jnp.float64)*256/4095
  from jax.scipy.ndimage import map_coordinates
  coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1);interp=map_coordinates(small,coords,order=1,mode='nearest').reshape(4096,4096)
  exact=PJ.synth_mesh(c,4095);diffs.append(float(jnp.linalg.norm(interp-exact)/jnp.linalg.norm(initial)))
 rows.append(dict(case=i,times=times,final_errors=errors,interpolation_differences=diffs,output_shape=list(fields['sine'].shape)))
 if i==0:np.savez_compressed(o/'final4096_case0.npz',sine=fields['sine'],bilinear=fields['bilinear256'])
 save(o/'result.json',dict(meta=meta(),rows=rows));print(i,times,errors,flush=True)
