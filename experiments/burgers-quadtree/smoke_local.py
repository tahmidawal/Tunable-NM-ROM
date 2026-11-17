from campaign import *
import local_bank as B
import snapshots
out=B.smoke()
f=snapshots.make(N=8,dt=.0025,M=8);ph=E.params_draw(1,1)[0];b=jnp.asarray(P.ic_coefs_2d(ph,8,16));v=f(b,ph[4]);assert v.shape==(11,9,9) and np.isfinite(v).all()
ref=P.make_ps(8,2)(b,ph[4],.0025,P.lattice_S(8,8))['lat'];assert np.allclose(v[::2],ref,atol=1e-12)
out['eleven_time_parity']=float(np.max(np.abs(np.asarray(v[::2])-ref)));save('smoke/local_bank.json',out)
