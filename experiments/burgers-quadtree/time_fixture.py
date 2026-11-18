"""Exact-time plumbing fixture, separate from any scientific performance result."""
from campaign import *
import continuous as C
def check(out,require_gpu=False):
 if require_gpu:assert jax.default_backend()=='gpu'
 v=jnp.array([.2]);G=jnp.ones((1,1));Hq=jnp.ones((1,1))*.1;e=jnp.array([2.]);nu=.05
 old=C.flux_trajectory(.00625)(v,nu,G,Hq,e);new=C.flux_trajectory(.00625,.05)(v,nu,G,Hq,e);assert np.allclose(old[::2],new,rtol=1e-13,atol=1e-14)
 y=C.flux_trajectory(.01,.05)(v,nu,G,Hq,e);t=np.linspace(0,.25,6);exact=.2*np.exp(-.1*t)/(1-.2*(1-np.exp(-.1*t)));err=float(np.max(np.abs(np.asarray(y)[:,0]-exact)));assert err<1e-12
 try:C.flux_trajectory(.01)
 except AssertionError:pass
 else:raise AssertionError('nondividing step accepted')
 save(out,dict(meta=meta(),analytic_max_error=err,old_new_interval_equivalence=True,nondividing_step_rejected=True));print('exact-time fixture passed',err,flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--gpu',action='store_true');a=ap.parse_args();check(a.out,a.gpu)
