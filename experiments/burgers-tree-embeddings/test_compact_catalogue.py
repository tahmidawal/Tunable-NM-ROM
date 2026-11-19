"""Sub-minute catalogue subset and complete-query equivalence fixture."""
from common import *
import geometry as geo,tensor_metric as TM,cell_catalogue as CC
import packed_fields as PF,compact_fields as CF
from direct_query import FixedQuery
rng=np.random.default_rng(219)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
rotation=np.linalg.qr(rng.normal(size=(36,2)))[0]
options=dict(L=64,Q=32,tests=64,depth=2,output_points=65,rule=TM.rule(panels=16,order=24,N=16))
packed=CC.build(basis,rotation,**options)
compact=CC.build(basis,rotation,**options,field_kernel='compact')
maximum=0.
for cells in [[],geo.uniform(1),geo.uniform(1)+geo.children((1,1,0)),list(reversed(geo.uniform(2)))]:
    a=CC.select(packed,cells);b=CC.select(compact,cells)
    c=jnp.asarray(rng.normal(size=2+len(cells)*2)*.01)
    for left,right in [(PF.fields(c,a['cache']),CF.fields(c,b['cache'])),
                       (PF.center(c,a['output_cache']),CF.center(c,b['output_cache']))]:
        maximum=max(maximum,float(jnp.max(jnp.abs(left-right))))
    assert np.array_equal(a['cache']['A'],b['cache']['A'])
    assert np.array_equal(a['cache']['gram'],b['cache']['gram'])
assert maximum<1e-10,maximum
initial=lambda xy:.15*jnp.sin(jnp.pi*xy[...,0])*jnp.sin(jnp.pi*xy[...,1])
queries=[FixedQuery(basis,rotation,64,32,64,.025,catalogue=cat,nodes=8,depth=2,
                    output_points=65,reuse_normal=True) for cat in [packed,compact]]
runs=[query(initial,outputs=5,components=False) for query in queries]
assert all(r['complete'] and len(r['step_status'])==10 for r in runs)
delta=max(float(jnp.linalg.norm(a-b)/jnp.linalg.norm(a)) for a,b in zip(runs[0]['fields'],runs[1]['fields']))
assert delta<1e-9,delta
print(json.dumps(dict(status='passed',subset_max_absolute_field_defect=maximum,
    complete_query_max_relative_field_change=delta,subset_geometries=4,root_only_checked=True,
    returned_steps=[len(r['step_status']) for r in runs],outputs_per_invocation=5,
    scope='CPU fixture only; no GPU qualification or speed claim'),indent=2))
