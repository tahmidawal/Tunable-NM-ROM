"""Small CPU check of compact runtime patches against full-axis contractions."""
from common import *
import geometry as geo,tensor_metric as TM,packed_fields as PF,compact_fields as CF,discrete_weak as W
rng=np.random.default_rng(114)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
rotation=np.linalg.qr(rng.normal(size=(36,3)))[0];metric=TM.make_geometry(6)
defects=[]
trees=[geo.uniform(1)+geo.children((1,0,0)),geo.uniform(1)+geo.children((1,1,1)),
       geo.uniform(1)+geo.children((1,0,0))+geo.children((2,0,0))+geo.children((3,0,0))+geo.children((1,1,1))]
for cells in trees:
    geo.valid(cells);depth=max(c[0] for c in cells)
    _,tr,_,_=metric(jnp.asarray(basis),jnp.asarray(cells),*TM.rule(panels=64,order=16,N=16))
    c=jnp.asarray(rng.normal(size=2+len(cells)*3)*.1);v=jnp.asarray(rng.normal(size=len(c)))
    ix,lam=W.selected_modes(64,64)
    p=dict(packed=PF.prepare_stencil(basis,cells,tr,rotation,64,32),indices=jnp.asarray(ix))
    q=dict(compact=CF.prepare_stencil(basis,cells,tr,rotation,64,32,depth=depth,node_cap=len(cells)),indices=jnp.asarray(ix))
    f=PF.fields(c,p);g=CF.fields(c,q)
    defects.append(float(jnp.max(jnp.abs(f-g))))
    defects.append(float(jnp.max(jnp.abs(jax.jvp(lambda z:PF.fields(z,p),(c,),(v,))[1]-jax.jvp(lambda z:CF.fields(z,q),(c,),(v,))[1]))))
    defects.append(float(jnp.max(jnp.abs(PF.tested_advection(64,32)(c,p)-CF.tested_advection(64,32)(c,q)))))
    for n in [65,66]:
        axis=np.linspace(0,1,n)[None]
        a=PF.center(c,dict(packed=PF.prepare(basis,cells,tr,rotation,axis,axis)))
        b=CF.center(c,dict(compact=CF.prepare(basis,cells,tr,rotation,axis,axis,depth=depth,node_cap=len(cells))))
        defects.append(float(jnp.max(jnp.abs(a-b))))
        boundary=jnp.concatenate([b[0],b[-1],b[:,0],b[:,-1]])
        assert float(jnp.max(jnp.abs(boundary)))<1e-12
assert max(defects)<1e-9,defects
print(json.dumps(dict(status='passed',max_absolute_defect=max(defects),geometries=len(trees),maximum_depth=4,
    checked='fields, derivatives, signed upwind and65/66-square actual outputs',
    scope='CPU fixture only; no GPU equivalence or speed admission'),indent=2))
