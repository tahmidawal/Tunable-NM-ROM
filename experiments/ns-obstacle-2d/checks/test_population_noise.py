"""Small synthetic audit of common-rule/noise arithmetic and malformed input rejection."""
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B));from immutable_rule import array_digest
PYTHON='/home/tahmid/Dev/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x))
class NoiseTest(unittest.TestCase):
 def test_formula_and_integrity(self):
  with tempfile.TemporaryDirectory()as t:
   t=Path(t);common=t/'common';fresh=t/'fresh';common.mkdir();fresh.mkdir();points=np.array([[.3,.1],[.4,.2],[.5,.3]],dtype='float64');w=np.ones(3);original=t/'original.npz';np.savez(original,points=points,weights=w)
   descriptor={'source':str(original),'sha256':sha(original),'point_count':3};plan=t/'plan.json';dump(plan,{'common_rule':descriptor});base={'complete':True,'Re':15,'split':'train','states':{},'complete_common_rule':descriptor};replay={**base,'states':{}}
   for folder in (common,fresh):np.savez(folder/'common-rule.npz',points=points,weights=w)
   spaces=dict(p4_g4_h1=1.,p4_g4_h2=.1,p4_g4_h3=.01,p3_g3_h3=.02,p3_g4_h3=.015)
   for space,value in spaces.items():
    for variant,shift in [('baseline',0.),('solver_100x',1e-8),('quadrature_plus4',2e-8)]:
     key=space+'_'+variant;fields={'velocity':np.full((3,2),1+value+shift),'pressure':np.full(3,.2+value+shift)};force=[5+value+shift,.01+value+shift];file=common/(key+'.npz');np.savez(file,**fields)
     r={'field_file':file.name,'field_sha256':sha(file),'field_content_sha256':{k:array_digest(v)for k,v in fields.items()},'solution_sha256':'a'*64,'force_order16':force,'force_replay_order16':force if variant=='baseline'else None,'other_force_study':{'physical_symmetric_order20':{'coefficient':force}}};base['states'][key]=r
     if variant=='baseline':(fresh/file.name).write_bytes(file.read_bytes());replay['states'][key]=r
   dump(common/'readout.json',base);dump(fresh/'readout.json',replay)
   command=[PYTHON,str(B/'g2/population_noise.py'),'--common',str(common),'--fresh-replay',str(fresh),'--plan',str(plan),'--original-rule',str(original),'--out',str(t/'noise.json')]
   result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);out=json.loads((t/'noise.json').read_text());m=out['metrics']['pressure_scaled'];self.assertAlmostEqual(m['h_fine'],.09/.04);self.assertAlmostEqual(m['delta'],2*max(m['h_fine'],m['p'])+m['solver']+m['quadrature']);self.assertTrue(m['resolved_h']['decreasing']);self.assertFalse(out['admission'])
   np.savez(common/'common-rule.npz',points=points,weights=w*2);result=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn('original immutable rule',result.stderr)
   np.savez(common/'common-rule.npz',points=points,weights=w);key='p4_g4_h1_baseline';file=common/(key+'.npz');fields={'velocity':np.ones((3,2),dtype='float32'),'pressure':np.ones(3)};np.savez(file,**fields);base['states'][key].update(field_sha256=sha(file),field_content_sha256={k:array_digest(v)for k,v in fields.items()});dump(common/'readout.json',base);result=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn('finite f64',result.stderr)
if __name__=='__main__':unittest.main()
