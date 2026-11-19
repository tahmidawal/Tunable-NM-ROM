"""Sub-minute CPU fixture: a full cell catalogue versus fresh tree assembly."""
from common import *
import geometry as geo, tensor_metric as TM, discrete_weak as W, packed_fields as PF
import cell_catalogue as CC
from build_local_cache import expand_matrix

rng=np.random.default_rng(71)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
rotation=np.linalg.qr(rng.normal(size=(36,2)))[0]
rule=TM.rule(panels=16,order=24,N=16)
catalogue=CC.build(basis,rotation,L=64,Q=32,tests=64,depth=2,output_points=65,rule=rule)
metric=TM.make_geometry(6)
maximum={k:0. for k in ['gram','tested_mass','field','field_jvp','advection','output']}
trees=[geo.uniform(1)+geo.children((1,i,j)) for i,j in [(0,0),(1,1)]]
trees.append(geo.uniform(2))
for cells in trees:
    selected=CC.select(catalogue,cells)
    full,tr,_,_=metric(jnp.asarray(basis),jnp.asarray(cells),*rule)
    T=expand_matrix(jnp.asarray(rotation),len(cells),2)
    gram=T.T@full@T
    fresh,_=W.prepare(basis,cells,tr,rotation,64,64,gram)
    fresh['packed']=PF.prepare_stencil(basis,cells,tr,rotation,64,32)
    c=jnp.asarray(rng.normal(size=len(gram))*.02);c=c.at[0].set(.3)
    v=jnp.asarray(rng.normal(size=len(gram)))
    ca=selected['cache']
    maximum['gram']=max(maximum['gram'],float(jnp.max(jnp.abs(gram-ca['gram']))))
    maximum['tested_mass']=max(maximum['tested_mass'],float(jnp.max(jnp.abs(fresh['A']-ca['A']))))
    f0=PF.fields(c,fresh);f1=PF.fields(c,ca)
    maximum['field']=max(maximum['field'],float(jnp.max(jnp.abs(f1-f0))))
    j0=jax.jvp(lambda q:PF.fields(q,fresh),(c,),(v,))[1]
    j1=jax.jvp(lambda q:PF.fields(q,ca),(c,),(v,))[1]
    maximum['field_jvp']=max(maximum['field_jvp'],float(jnp.max(jnp.abs(j1-j0))))
    adv=PF.tested_advection(64,32)
    maximum['advection']=max(maximum['advection'],float(jnp.max(jnp.abs(adv(c,fresh)-adv(c,ca)))))
    axis=np.linspace(0,1,65)[None]
    direct_output={'packed':PF.prepare(basis,cells,tr,rotation,axis,axis)}
    maximum['output']=max(maximum['output'],float(jnp.max(jnp.abs(PF.center(c,direct_output)-PF.center(c,selected['output_cache'])))))
    assert np.allclose(tr,selected['transforms'],rtol=0,atol=1e-11)
# Adding a sibling group without changing existing coefficients preserves field.
small=geo.uniform(1);large=trees[0]
c=jnp.asarray(rng.normal(size=2+len(small)*2)*.02)
zero_change=float(jnp.max(jnp.abs(PF.center(c,CC.select(catalogue,small)['cache'])-
    PF.center(jnp.pad(c,(0,8)),CC.select(catalogue,large)['cache']))))
assert max(maximum.values())<1e-9 and zero_change<1e-13,(maximum,zero_change)
print(json.dumps(dict(status='passed',max_absolute_defects=maximum,zero_child_field_change=zero_change,
                     all_depth2_cells=len(catalogue['cells']),different_geometries=len(trees),
                     scope='CPU fixture only; no scientific GPU or runtime admission'),indent=2))
