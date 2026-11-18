"""Small bounded GPU fixtures, not a scientific experiment."""
import json, argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import run_mechanism as M

@jax.jit
def affine(x):
    a=jnp.array([[1.,0.],[0.,2.],[1.,1.]])
    b=jnp.array([2.,-1.,.2])
    return a@x-b,a

x,info=M.solve(affine,np.zeros(2),(),1e-8,100)
r,j=affine(jnp.asarray(x))
assert info['reason']=='stationarity' and M.stationarity(np.asarray(r),np.asarray(j))<=1e-8
xround,roundinfo=M.solve(affine,np.zeros(2),(),1e-14,100)
assert roundinfo['reason']=='stationarity' and roundinfo['gradient']<=1e-14
_,cap=M.solve(affine,np.zeros(2),(),1e-15,0)
assert cap['reason']=='budget'
assert M.stationarity(np.zeros(3),np.eye(3))==0
ck=np.load(M.HERE/'assets/head_k8_n64.npz')
p={key:jnp.asarray(ck['p_'+key]) for key in ('W1','b1','W2','b2','W3','b3','Ws')}
z=jnp.asarray(ck['Z'][0])
ad=np.asarray(jax.jacfwd(M.H.head_apply,argnums=1)(p,z));analytic=np.asarray(M.H.head_jacobian(p,z))
head_gap=float(np.linalg.norm(ad-analytic)/np.linalg.norm(ad));assert head_gap<1e-12
n=8;r=4
G,_lam,ids=M.SR.R.test_modes(n,r);a=np.arange(1,r+1,dtype=np.float64)
geom=M.F.geometry(n)
gram=np.asarray(M.tangent_gram(jnp.asarray(G),jnp.asarray(a),jnp.eye(r),geom))
field=(jnp.asarray(G)@jnp.asarray(a)).reshape(3,n,n,n)
B=np.asarray(jax.jacfwd(lambda w:M.SR.shift_field((jnp.asarray(G)@w[:r]).reshape(3,n,n,n),w[r:]*n).ravel())(jnp.r_[a,jnp.zeros(3)]))
gram_gap=float(np.linalg.norm(gram-B.T@B/n**3)/np.linalg.norm(gram));assert gram_gap<1e-12
spec=M.physical_spectrum(np.eye(r+3),gram);assert spec['excluded']>0
rng=np.random.default_rng(9);rows=20
A=jnp.asarray(rng.normal(size=(rows,r)));T=jnp.asarray(rng.normal(size=(rows,r,r)));lam=jnp.arange(1,rows+1,dtype=jnp.float64);Dd=jnp.asarray(rng.normal(size=(3,rows,r)))
prepare,ev=M.make_eval(.02,n,r,2,'span');ops=prepare(.01,A,T,lam,Dd)
check=M.action_check(ev,np.r_[a,.001,.002,.003],(jnp.asarray(a*.9),ops,p),31)
assert check['minimum']<1e-7
pred=M.reconstruct(jnp.asarray(G),jnp.asarray(a),jnp.array([.1,.2,.3]))
metrics=M.field_metrics(pred,pred,geom);assert float(metrics[0])==0
report={'backend':jax.default_backend(),'dtype':str(z.dtype),'affine_solve':{k:v for k,v in info.items() if k!='history'},'cap_reason':cap['reason'],'head_jacobian_relative':head_gap,'physical_tangent_gram_relative':gram_gap,'tangent_retained_rank':spec['rank'],'tangent_excluded':spec['excluded'],'analytic_residual_action':check,'all_passed':True,'roundoff_tight_affine':roundinfo}
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=M.HERE/'fixture_results.json');args=ap.parse_args()
M.dump(args.out,report)
print(json.dumps(report),flush=True)
