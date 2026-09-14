"""Bounded schema/oracle/query smoke, not a reference-accuracy measurement."""
from pathlib import Path
import sys
import data as d
import diagnose

jax,e=d.gpu_modules()
out=d.make_output(Path(__file__).parent/'runs/diagnosis-smoke-reference01')
p=e.params_draw(d.case_seed('calibration',0),1)[0]
query,dense=d.make_solver(e,64,.05,64)
fields,it,rn,seconds=d.solve(jax,e,query,dense,64,p)
record=d.save_case(out,d.case_record('calibration',0),fields,p,64,dict(role='smoke only'))
d.write_json(out/'index.json',dict(pde='burgers',provenance=d.provenance(jax),records=[record]))
sys.argv=['diagnose.py','--reference-index',str(out/'index.json'),'--out',str(out.parent/'diagnosis-smoke01'),'--intervals','64','--cases','1','--reps','1','--smoke']
diagnose.main()
