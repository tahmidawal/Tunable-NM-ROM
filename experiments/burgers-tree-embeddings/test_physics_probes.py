"""Small analytic diffusion, observable defect and zero-transfer fixture."""
from common import *
import geometry as geo,cell_catalogue as CC,tensor_metric as TM,packed_fields as PF
from physics_probes import PhysicsProbes
from adaptive_actions import transfer,candidates,admissible_trial
rng=np.random.default_rng(831)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
U=np.linalg.qr(rng.normal(size=(36,3)))[0]
cat=CC.build(basis,U,128,64,128,depth=2,output_points=65,rule=TM.rule(panels=16,order=24,N=16))
probes=PhysicsProbes(cat,32,64)
empty=probes.select([]);old=jnp.array([1e-9,0.]);dt=.01;nu=.005
lam=8*128**2*np.sin(np.pi/(2*128))**2
exact=old*(1-.5*dt*nu*lam)/(1+.5*dt*nu*lam)
good=probes.step(exact,empty,old,empty,dt,nu)
bad=probes.step(exact*1.02,empty,old,empty,dt,nu)
assert good['global_score']<1e-9 and bad['global_score']>.019,(good,bad)
cells=geo.uniform(1);larger=cells+geo.children((1,1,0))
c=jnp.asarray(rng.normal(size=2+len(cells)*2)*.01)
c2=transfer(c,cells,2,larger,3,2)
small=probes.select(cells,rank=2);large=probes.select(larger,rank=3)
field_change=float(jnp.max(jnp.abs(PF.fields(c,small)-PF.fields(c2,large))))
r1=probes.step(c,small,c,small,dt,nu);r2=probes.step(c2,large,c,small,dt,nu)
score_change=abs(r1['global_score']-r2['global_score'])
assert field_change<1e-12 and score_change<1e-12,(field_change,score_change)
known=lambda xy:2e-9*jnp.sin(jnp.pi*xy[...,0])*jnp.sin(jnp.pi*xy[...,1])
initial=probes.initial(old,empty,known);assert initial['global_score']<1e-12
leaves=geo.leaves(cells);scores={v:float(i+1) for i,v in enumerate(leaves)}
actions=candidates(cells,2,2,scores,depth=2,max_variables=40,next_rank=3,
    split_costs={v:1. for v in leaves},rank_cost=2.)
assert len(actions)==4 and [a['kind'] for a in actions]==['split','split','rank','global_fallback']
assert actions[0]['leaf']==leaves[-1]
assert admissible_trial(1.,.8,1.,2.,solver_converged=True,finite=True)
assert not admissible_trial(1.,.81,1.,2.,solver_converged=True,finite=True)
assert not admissible_trial(1.,.1,1.,2.,solver_converged=False,finite=True)
print(json.dumps(dict(status='passed',analytic_diffusion_defect=good['global_score'],
    perturbed_defect=bad['global_score'],zero_transfer_field_change=field_change,
    zero_transfer_probe_change=score_change,initial_mismatch=initial['global_score'],
    candidate_cap_checked=True,scope='CPU fixture only; no calibrated threshold or adaptive PDE admission'),indent=2))
