from common import *
import data, decoder as dec, geometry as geo, fit

key=jax.random.PRNGKey(2026111851);pg=dec.init_global(key);pl=dec.init_local(key)
ph=dec.init_head(key,16,128,64);ch=dec.init_head(key,4,16,32)
xy,w=data.grid(12,True);x=jnp.asarray(xy);w=jnp.asarray(w)
cells=geo.uniform(1);geo.valid(cells)
G=dec.global_features(pg,x);B=dec.local_matrix(pl,x,cells)
z=jax.random.normal(key,(16,),dtype=jnp.float64)*.1
root=G@dec.head(ph,z)
extended=dec.field(jnp.concatenate([G,B],axis=1),ph,ch,jnp.concatenate([z,jnp.zeros(16)]),4)
assert float(jnp.max(jnp.abs(root-extended)))==0
np.testing.assert_array_equal(dec.head(ch,jnp.zeros(4)),jnp.zeros(16))
J=jax.jacfwd(lambda c:dec.local_features(pl,x,cells[0])@dec.head(ch,c))(jnp.zeros(4))
s=np.linalg.svd(np.asarray(J)*np.sqrt(np.asarray(w))[:,None],compute_uv=False)
assert s[-1]>1e-8
point=jnp.array([.27,.19]);cell=(2,1,0)
fun=lambda a:dec.local_features(pl,a[None],cell)[0,0]
ad=np.asarray(jax.grad(fun)(point));eps=1e-6
fd=np.array([(float(fun(point+jnp.eye(2)[a]*eps))-float(fun(point-jnp.eye(2)[a]*eps)))/(2*eps) for a in range(2)])
np.testing.assert_allclose(ad,fd,atol=1e-8,rtol=1e-5)
wall=jnp.array([[0.,.4],[1.,.4],[.4,0.],[.4,1.]])
assert float(jnp.max(jnp.abs(dec.local_features(pl,wall,cells[0]))))==0
coef=jax.random.normal(key,(2,16),dtype=jnp.float64);target=coef@B[:,:16].T
C,pred,sval=fit.linear_fit(B[:,:16],target,w)
qr=np.linalg.lstsq(np.asarray(B[:,:16])*np.sqrt(np.asarray(w))[:,None],
                    np.asarray(target).T*np.sqrt(np.asarray(w))[:,None],rcond=1e-11)[0].T
np.testing.assert_allclose(pred,qr@np.asarray(B[:,:16]).T,atol=1e-11)
save(HERE/'smoke/decoder.json',dict(meta=meta(),zero_child_field=0.,child_zero_singular_values=s,
     derivatives_ad=ad,derivatives_fd=fd,boundary_pass=True,direct_fit_qr_pass=True,
     status='passed_initialization_fixture; trained checkpoints require repeated audits'))
print('decoder smoke passed',flush=True)
