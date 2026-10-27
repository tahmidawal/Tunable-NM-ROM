"""Independent symbolic differentiation of the prospective manufactured control.

This verifies algebra only; it is not an FE run, convergence test, or reference gate.
"""
from pathlib import Path
import json
import sympy as s

x,y,z,R,L,nu=s.symbols('x y z R L nu', real=True, nonzero=True)
u=s.Matrix([y*y/R**2+z/R,z*z/R**2+x/L,x*x/L**2+y/R])
p=x/L+y*y/R**2-z/R
xyz=(x,y,z)
G=u.jacobian(xyz)
lap=s.Matrix([sum(s.diff(ui,t,2) for t in xyz) for ui in u])
pressure_gradient=s.Matrix([s.diff(p,t) for t in xyz])
manualG=s.Matrix([[0,2*y/R**2,1/R],[1/L,0,2*z/R**2],[2*x/L**2,1/R,0]])
manualLap=s.Matrix([2/R**2,2/R**2,2/L**2])
manualGp=s.Matrix([1/L,2*y/R**2,-1/R])
div=s.trace(G)
assert s.simplify(div)==0
assert s.simplify(G-manualG)==s.zeros(3,3)
assert s.simplify(lap-manualLap)==s.zeros(3,1)
assert s.simplify(pressure_gradient-manualGp)==s.zeros(3,1)
conv=s.simplify(G*u)
assert all(s.simplify(conv[i])!=0 for i in range(3))
force=s.simplify(conv+pressure_gradient-nu*lap)
stress=-p*s.eye(3)+nu*(G+G.T)
divstress=s.Matrix([sum(s.diff(stress[i,j],xyz[j]) for j in range(3)) for i in range(3)])
assert s.simplify(divstress-(-pressure_gradient+nu*lap))==s.zeros(3,1)
report={"scope":"symbolic algebra only; not FE or numerical validation","velocity":[str(v) for v in u],"pressure":str(p),"gradient":[[str(v) for v in G.row(i)] for i in range(3)],"divergence":str(div),"convection":[str(v) for v in conv],"forcing":[str(v) for v in force],"stress_divergence":[str(v) for v in divstress],"manual_gradient_laplacian_pressure_gradient_match":True,"nonzero_convection_all_components":True,"symbolic_sphere_force_divided_by_volume":[str(v.subs({x:0,y:0,z:0})) for v in divstress]}
Path(__file__).with_name('MANUFACTURED.symbolic.json').write_text(json.dumps(report,indent=2)+'\n')
print('manufactured symbolic identities: verified (no FE execution)')
