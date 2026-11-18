from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();o=Path(args.out);o.mkdir(parents=True,exist_ok=True)
r=dict(meta=meta(),fixtures=[],rows=[])
# Complete weak trial/test space, rotated coordinates: parity with independent inherited solver.
N=8 if args.smoke else 16;ph=E.params_draw(17,1)[0];b=jnp.asarray(P.ic_coefs_2d(ph,N,128));Q,_=jnp.linalg.qr(jax.random.normal(jax.random.key(18),((N-1)**2,(N-1)**2),dtype=jnp.float64))
W,eigs,rot=C.diffusion_basis(Q,N);a0=W.T@b.ravel()/2
red=C.reduced_trajectory(N,.0025)(a0,ph[4],W,eigs);bc=(2*(red@W.T)).reshape(11,N-1,N-1)
full=C.full_trajectory(N,.0025)(b,ph[4]);direct=P.make_ps(N,2)(b,ph[4],.0025,P.lattice_S(N,32))['lat']
fields=np.asarray(jax.vmap(lambda c:P.synth_lattice(c,P.lattice_S(N,32),2))(bc[::2]));err=float(np.linalg.norm(fields-direct)/np.linalg.norm(direct))
assert err<1e-10,err
r['fixtures'].append(dict(complete_weak_space_relative_error=err,coefficient_parity=float(jnp.max(jnp.abs(bc-full)))))
if args.smoke:save(o/'result.json',r);raise SystemExit
_,dev=draw()
for N in (32,64,128):
 for dt in (.005,.0025,.00125):
  solver=C.full_trajectory(N,dt)
  for i,ph in enumerate(dev[:8]):
   b=jnp.asarray(P.ic_coefs_2d(ph,N,256));v,sec=timer(lambda:solver(b,ph[4]));ref=np.load(verified('reference01',f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511)
   ref=np.pad(ref,((0,0),(1,1),(1,1)))[:,::2,::2]
   f=np.asarray(jax.vmap(lambda b:P.synth_lattice(b,P.lattice_S(N,256),2))(v[::2]))
   error=np.linalg.norm((f-ref).reshape(6,-1),axis=1)/np.linalg.norm(ref[0])
   np.savez_compressed(o/f'N{N}_dt{dt}_case{i}.npz',coefficients=v,error=error)
   r['rows'].append(dict(N=N,dt=dt,case=i,error=error,seconds=sec,finite=bool(np.isfinite(v).all())));save(o/'result.json',r)
r['complete']=True;save(o/'result.json',r)
