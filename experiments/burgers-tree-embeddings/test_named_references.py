"""Named smooth/pulse definitions and reference integrator agree with verified primary code."""
from common import *
from neural import named_references as N
import data,references

spec=json.loads((HERE/'neural/named_cases.json').read_text())['burgers']
xy,w=data.grid(12,True)
definitions=spec['cases'];params=data.fixtures()
errors=[]
for case,p in [(definitions[0],params[1]),(definitions[1],params[0])]:
    err=float(jnp.max(jnp.abs(N.initial(jnp.asarray(xy),case)-data.initial(xy,p))))
    assert err<1e-14;errors.append(err)
    b=N.initial_coefficients(case,16,8);old=jnp.asarray(data.initial_coefficients(p,16))
    assert float(jnp.linalg.norm(b-old)/jnp.linalg.norm(old))<1e-10
b=jnp.asarray(data.initial_coefficients(params[0],16))
u=N.trajectory(16,.000625,.005)(b);v=references.trajectory(16,.000625)(b)
change=float(jnp.linalg.norm(u-v)/jnp.linalg.norm(v));assert change<1e-13
print(json.dumps(dict(status='passed',primary_initial_max_changes=errors,primary_integrator_change=change,
    case_count=len(definitions),viscosities=spec['proposed_viscosities'])))
