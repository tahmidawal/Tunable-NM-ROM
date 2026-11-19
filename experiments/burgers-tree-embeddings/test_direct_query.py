"""CPU fixture for complete query accounting and actual output contracts."""
from common import *
import cell_catalogue as CC, tensor_metric as TM
from direct_query import FixedQuery
rng=np.random.default_rng(102)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
rotation=np.linalg.qr(rng.normal(size=(36,2)))[0]
cat=CC.build(basis,rotation,64,32,64,depth=2,output_points=65,rule=TM.rule(panels=16,order=24,N=16))
initial=lambda xy:.15*jnp.sin(jnp.pi*xy[...,0])*jnp.sin(jnp.pi*xy[...,1])
ordinary=FixedQuery(basis,rotation,64,32,64,.025,catalogue=cat,nodes=8,depth=2,output_points=65)
reused=FixedQuery(basis,rotation,64,32,64,.025,catalogue=cat,nodes=8,depth=2,output_points=65,reuse_normal=True)
a=ordinary(initial,outputs=1);b=reused(initial,outputs=5)
assert a['complete'] and b['complete'],(a['failure'],b['failure'])
assert len(a['fields'])==1 and len(b['fields'])==5
assert all(v.shape==(65,65) and v.dtype==jnp.float64 for v in a['fields']+b['fields'])
delta=float(jnp.linalg.norm(a['fields'][-1]-b['fields'][-1])/jnp.linalg.norm(a['fields'][-1]))
assert delta<1e-10,delta
for v in [a,b]:
    parts=sum(v[k] for k in ['setup_seconds','initial_fit_seconds','solving_seconds','requested_reconstruction_seconds'])
    assert 0<=parts<=v['total_query_seconds'] and v['total_query_seconds']-parts<.1
    assert len(v['step_status'])==10 and v['coefficients'].shape[0]==6
print(json.dumps(dict(status='passed',relative_final_field_change=delta,outputs=[len(a['fields']),len(b['fields'])],
    returned_steps=[len(a['step_status']),len(b['step_status'])],complete_accounting_boundary=True,
    scope='CPU fixture only;65-square fields; no scientific accuracy or GPU performance claim'),indent=2))
