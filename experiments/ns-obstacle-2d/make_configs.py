"""Derive frozen smoke cases from saved authoritative campaign constants."""
import hashlib
import json
from pathlib import Path
import sympy as sp

root=Path(__file__).resolve().parent
source=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle/campaign.json')
raw=source.read_bytes();(root/'campaign-A2.json').write_bytes(raw)
c=json.loads(raw);b=c['case2d']
base=dict(U=b['inlet_mean'],D=2*b['radius'],nu=b['benchmark_nu'],H=b['height'],xmin=b['xmin'],xmax=b['xmax'],
          center=b['obstacle_center'],radius=b['radius'],outlet='gradient',
          velocity_order=3,pressure_order=2,geometry_order=3,maxh=.08,obstacle_h=.02,
          assembly_bonus_order=6,diagnostic_order=12,threads=16,seed=c['seeds']['mesh'],
          newton_max=30,nonlinear_relative_tolerance=1e-10,linear_relative_tolerance=1e-10,
          residual_floor=1.0,damping_sequence=[1,.5,.25,.125,.0625,.03125,.015625],armijo=1e-4,
          inverse_backend='umfpack',max_elements=20000,max_dofs=100000,max_nnz=12000000,
          factor_fill_estimate=100,memory_stop_gib=48,
          jacobian_steps=[1e-4,1e-5,1e-6],jacobian_state_perturbation=.1,
          campaign_sha256=hashlib.sha256(raw).hexdigest(),design_sha256=c['design_sha256'],
          release='G1_smoke_only_not_reference_admission',precision='float64')
for kind in ['channel','manufactured','dfg']:
 d=dict(base,kind=kind)
 if kind=='channel': d['maxh']=.16
 (root/'configs'/f'{kind}-smoke.json').write_text(json.dumps(d,indent=2)+'\n')
# Independent symbolic witness: differentiate with SymPy, not NGSolve's CF calculus.
x,y=sp.symbols('x y');U=sp.Rational(str(base['U']));H=sp.Rational(str(base['H']))
xmin=sp.Rational(str(base['xmin']));L=sp.Rational(str(base['xmax']))-xmin;nu=sp.Rational(str(base['nu']))
X=(x-xmin)/L;Y=y/H
psi=U*H*(X**2*Y+X*Y**2+sp.Rational(1,4)*X**3*Y**3)
uvec=sp.Matrix([sp.diff(psi,y),-sp.diff(psi,x)]);p=U**2*(X+Y+X*Y)
grad=uvec.jacobian([x,y]);lap=sp.Matrix([sp.diff(a,x,2)+sp.diff(a,y,2) for a in uvec])
f=grad*uvec-nu*lap+sp.Matrix([sp.diff(p,x),sp.diff(p,y)])
div=sp.simplify(sp.diff(uvec[0],x)+sp.diff(uvec[1],y));assert div==0
samples=[]
for point in [[.11,.08],[.7,.17],[1.2,.29],[2.0,.35]]:
 sub={x:point[0],y:point[1]}
 samples.append({'point':point,'u':[float(a.subs(sub)) for a in uvec], 'p':float(p.subs(sub)),
                 'gradient':[float(a.subs(sub)) for a in grad],'forcing':[float(a.subs(sub)) for a in f]})
theta=sp.symbols('theta',real=True)
radius=sp.Rational(str(base['radius']));cx,cy=map(lambda z:sp.Rational(str(z)),base['center'])
normal=sp.Matrix([sp.cos(theta),sp.sin(theta)])
force_exact={}
for name,stress in [('gradient',nu*grad-p*sp.eye(2)),('physical_symmetric',nu*(grad+grad.T)-p*sp.eye(2))]:
 tr=stress.subs({x:cx+radius*sp.cos(theta),y:cy+radius*sp.sin(theta)})*normal*radius
 force_exact[name]=[float(sp.integrate(sp.expand(z),(theta,0,2*sp.pi))) for z in tr]
w={'obstacle_force':force_exact,'force_normal':'outward from solid into fluid','configuration':{k:base[k] for k in ['U','nu','H','xmin','xmax']},'sympy_version':sp.__version__,
   'divergence_symbolic':str(div),'velocity':[str(a) for a in uvec], 'pressure':str(p),
   'forcing':[str(sp.simplify(a)) for a in f], 'advection':[str(sp.simplify(a)) for a in grad*uvec],
   'samples':samples,'status':'symbolic_control_witness_not_measured_solver_result'}
(root/'manufactured_witness.json').write_text(json.dumps(w,indent=2)+'\n')
