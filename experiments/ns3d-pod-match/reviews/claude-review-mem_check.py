import sys, time, numpy as np
W="/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-ns3d-pod-match/experiments"
for e in ("ns3d","ns3d-shift","ns3d-grok","ns2d","separable-decoder","ns3d-coordnet","ns3d-pod-match"): sys.path.insert(0, f"{W}/{e}")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import coordnet as CN, train_bank2 as TB
kind=sys.argv[1]; nt=48; R=64; Qn=8192
arch=dict(width=512,depth=4,kmax=4,band_kcut=10,band_weight=1.0)
sf=CN.make_sample_fn(nt)
loss,_=TB.make_objective(kind,arch,nt,R,sf)
p=jax.eval_shape(lambda: CN.init_params(jax.random.PRNGKey(0),R,512,4,4))
S=jax.ShapeDtypeStruct; f=jnp.float64; d=3*nt**3
data=dict(pts=S((nt**3,3),f),Y=S((d,512),f))
if kind=="pod_regress":
    data.update(U=S((d,R),f),w=S((R,),f),Q=S((Qn,3),f),Fq=S((Qn,3,R),f),dFq=S((3,Qn,3,R),f),nF=S((),f),ndF=S((),f))
t=time.time()
c=jax.jit(jax.value_and_grad(loss)).lower(p["layers"],p["B"],data).compile()
m=c.memory_analysis()
print(kind, "temp GB %.1f arg GB %.2f out GB %.2f"%(m.temp_size_in_bytes/1e9,m.argument_size_in_bytes/1e9,m.output_size_in_bytes/1e9), "%.0fs"%(time.time()-t))
import optax
lopt = optax.lbfgs(memory_size=20, linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=30))
def lchunk(layers, state, B, data):
    fn = lambda q: loss(q, B, data)
    vg = optax.value_and_grad_from_state(fn)
    def body(carry, _):
        q, st = carry
        v, g = vg(q, state=st)
        upd, st = lopt.update(g, st, q, value=v, grad=g, value_fn=fn)
        q = optax.apply_updates(q, upd)
        ls = optax.tree_utils.tree_get(st, "info")
        return (q, st), (v, ls.num_linesearch_steps)
    return jax.lax.scan(body, (layers, state), None, length=25)
st=jax.eval_shape(lopt.init, p["layers"])
t=time.time()
c=jax.jit(lchunk).lower(p["layers"],st,p["B"],data).compile()
m=c.memory_analysis()
print("lbfgs", kind, "temp GB %.1f arg GB %.2f"%(m.temp_size_in_bytes/1e9,m.argument_size_in_bytes/1e9), "%.0fs"%(time.time()-t))
