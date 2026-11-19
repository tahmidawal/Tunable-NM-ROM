"""Sub-minute optimizer fixtures, not a scientific benchmark."""
from common import *
import quasi_newton as B,coupled_child as C,discrete_weak as W
from test_coupled_child import setup


def main():
    # An independent nonlinear objective with a known minimizer tests curvature
    # updates and backtracking, rather than merely repeating the PDE code.
    fun=lambda x:(1-x[0])**2+100*(x[1]-x[0]**2)**2
    solve=jax.jit(lambda x:B.minimize(fun,x,jnp.eye(2),lambda g,x:jnp.max(jnp.abs(g)),
        lambda new,old:jnp.linalg.norm(new-old),120,1e-8,1e-10))
    result=solve(jnp.array([-1.2,1.]));x,loss,g,H,used,accepted,evaluations,update,done=result
    assert float(jnp.max(jnp.abs(x-1)))<1e-7 and float(jnp.max(jnp.abs(g)))<1e-8,result
    assert int(evaluations)>2*int(used)+1,'Fixture must include rejected trials'
    assert float(jnp.linalg.eigvalsh(H)[0])>0
    L,ca,field,z=setup()
    lm,_=W.make_solver(field,L,theta=.5,coefficient_map=C.coefficients,tested_mass_jacobian=C.tested_mass_jacobian)
    qn,_=B.make_solver(field,L,theta=.5,coefficient_map=C.coefficients,tested_mass_jacobian=C.tested_mass_jacobian)
    zl,sl=lm(z,ca,.0025,.005,1e-8,1e-10);zb,sb=qn(z,ca,.0025,.005,1e-8,1e-10)
    assert float(sl[1])<1e-8 and float(sb[1])<1e-8,(sl,sb)
    diff=C.coefficients(zb,ca)-C.coefficients(zl,ca);ref=C.coefficients(zl,ca)
    change=float(jnp.sqrt(diff@ca['gram']@diff/(ref@ca['gram']@ref)))
    assert change<1e-7,change
    print('PASS independent Rosenbrock curvature/Armijo and unchanged nonlinear CN solution',
        dict(rosenbrock_iterations=int(used),objective_evaluations=int(evaluations),stationarity=float(sb[1]),physical_change=change),flush=True)


if __name__=='__main__':main()
