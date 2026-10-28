"""Generate prospective post-ladder controls from retained h1 counts; no execution."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[1]
p=B/'artifacts/g2-ladder01/results/dfg_h1/result.json';r=json.loads(p.read_text())
with np.load(p.parent/'full_rule.npz',allow_pickle=False) as a:nq=len(a['weights'])
nnz=max(a['nnz'] for a in json.loads((p.parent/'memory_checks.json').read_text()))
rows=[]
for name,hscale,pscale,qscale,domain in (
 ('h2_expected',4.,1.,1.,1.),('h3_if_needed',16.,1.,1.,1.),
 ('h2_p4_g4',4.,(16+4.5)/(9+2),1.,1.),('h2_p3_g4',4.,1.,1.,1.),
 ('h2_solver_100x',4.,1.,1.,1.),('h2_quadrature_plus4',4.,1.,(9/7)**2,1.),
 ('h2_upstream_extension',4.,1.,1.,2.4/2.2),('h2_downstream_extension',4.,1.,1.,2.),
 ('h2_symmetric_outlet_long_domain',4.,1.,1.,2.)):
 counts={'elements':round(r['geometry']['element_count']*hscale*domain),
         'dofs':round(r['geometry']['ndof']*hscale*domain*pscale),
         'volume_points':round(nq*hscale*domain*qscale)}
 caps={'elements':20000,'dofs':100000,'volume_points':500000}
 rows.append({'case':name,'projected':counts,'producer_cap_violations':[k for k in caps if counts[k]>caps[k]],
              'ROM_point_cap_violation':counts['volume_points']>100000})
record={'status':'source_only_prospective_not_released','measured_h1_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'measured_h1':{'cells':r['geometry']['element_count'],'dofs':r['geometry']['ndof'],'points':nq,'nnz':nnz,
                       'peak_rss_bytes':r['peak_rss_bytes'],'reference_pipeline_seconds':r['elapsed_seconds']},
        'projection_assumptions':'h^-2 and domain-length mesh scaling; asymptotic p DOF scaling; these are not actual counts or memory guarantees',
        'cases':rows,
        'next_decisions':['Wait for actual ladder02 h2 and complete common-rule comparisons; no extrapolated admission',
          'If h2 fails physical gates or two successive differences, review refined h ladder and producer capacity using actual h2 counts before submission',
          'Run p4/g4 and p3/g4, 100x solver tightening, +4 quadrature and domain controls on the selected qualified h level',
          'Preserve symmetric-traction long outlet as separate BVP, never substitute into DFG reference',
          'Qualify the full registered parameter family; Re20 alone does not admit other Reynolds numbers',
          'Use one binary mesh and layout for training/dev/validation with explicit exact hashes; export one certified common Stokes lift and test immutable reuse natively',
          'Resolve downstream ROM capacity separately; do not weaken producer quadrature or coarsen reference to fit old adapter caps'],
        'admission_evidence':['per-case geometry/trace/divergence/mass/residual/force/benchmark diagnostics',
          'two successive h field/force differences and p plus independent geometry comparison',
          'solver/quadrature/domain comparisons with exact pressure poles',
          'all constituents of conditional reference uncertainty with unresolved/nondecreasing cases rejected',
          'source-bound saved-array independent review and complete checksum-backed retention'],
        'physical_admission':False,'training_released':False}
(B/'g2/NEXT-CONTROLS.DRAFT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(rows,indent=2))
