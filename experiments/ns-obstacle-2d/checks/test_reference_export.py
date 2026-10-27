"""Pure array/polynomial/schema checks, not a native FE or G2 execution."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reference_export as rex
import shared_lift

class ExportControls(unittest.TestCase):
    def test_obstacle_count_keeps_original_rule(self):
        points=np.array([[2.,0.],[.5,0.],[1.,0.]])
        weights=np.array([.2,.3,.5]);original=weights.copy()
        distance,record=rex.obstacle_quadrature(points,weights,{'kind':'dfg','center':[0.,0.],'radius':1.,'D':2.})
        np.testing.assert_array_equal(distance,[1.,-.5,0.]);np.testing.assert_array_equal(weights,original)
        self.assertEqual(record['candidate_excluded_count'],1);self.assertAlmostEqual(record['candidate_excluded_weight'],.3)
        self.assertEqual(record['actually_removed_count'],0);self.assertFalse(record['no_obstacle_quadrature'])
    def test_polynomial_max_and_bound(self):
        p=3;nodes=.5*(1-np.cos(np.pi*np.arange(p+1)/p));probe=np.linspace(0,1,2*p+3)
        def field(x):return np.column_stack([x*(1-x),np.zeros_like(x)])
        record=rex.polynomial_trace_bound(nodes,field(nodes),probe,field(probe),p,1.)
        self.assertAlmostEqual(record['maximum_estimate'],.25,12)
        self.assertGreaterEqual(record['coefficient_norm_upper_bound_with_roundoff'],.25)
        self.assertLess(record['coefficient_norm_upper_bound_with_roundoff'],.251)
    def test_unverified_polynomial_degree_rejected(self):
        nodes=np.array([0.,1.]);probe=np.linspace(0,1,5)
        def field(x):return np.column_stack([x*x,np.zeros_like(x)])
        with self.assertRaises(ArithmeticError):rex.polynomial_trace_bound(nodes,field(nodes),probe,field(probe),1,1.)
    def test_layout_matches_reviewed_adapter_algorithm(self):
        source=subprocess.check_output(['git','show','97847a479:experiments/ns-obstacle-rom/fe_adapter.py'],cwd=ROOT,text=True)
        reviewed=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='layout_record')
        ours=next(n for n in ast.parse(Path(rex.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='layout_record')
        self.assertEqual(ast.dump(ours,include_attributes=False),ast.dump(reviewed,include_attributes=False))
        elements=[types.SimpleNamespace(nr=4,type='ET.TRIG'),types.SimpleNamespace(nr=1,type='ET.TRIG')]
        mesh=types.SimpleNamespace(dim=2,Elements=lambda vb:iter(elements))
        space=types.SimpleNamespace(ndof=3,FreeDofs=lambda:[False,True,True],GetDofNrs=lambda el:[el.nr,0,2])
        record,digest=rex.layout_record(mesh,space,space,types.SimpleNamespace(VOL='volume'))
        self.assertEqual([el['number'] for el in record['elements']],[4,1])
        self.assertEqual(digest,hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    def test_reload_metric_rejects_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):rex.relative_max_record(np.zeros(2),np.zeros(3))
        with self.assertRaises(ValueError):rex.relative_max_record([np.nan],[0.])
        record=rex.relative_max_record([2.],[1.],scale=4.)
        self.assertEqual(record,{'numerator':1.,'denominator':4.,'value':.25})
    def test_g1_cannot_enable_g2_controls(self):
        with self.assertRaises(ValueError):rex.enabled({'reference_export_controls':True,'release':'G1_smoke_only_not_reference_admission'})
    def test_common_stokes_split_and_query_scaling(self):
        c=np.array([2.,3.,.02,.03]);u,p=shared_lift.split_common_lift(c,2,.01)
        np.testing.assert_array_equal(u,[2.,3.]);np.testing.assert_allclose(p,[2.,3.])
        cached=shared_lift.CommonStokesLift();cached.velocity=u;cached.pressure_per_nu=p
        state=cached.state_for_nu(.02);np.testing.assert_allclose(state,[2.,3.,.04,.06])
        state[:]=0;np.testing.assert_array_equal(c,[2.,3.,.02,.03]);np.testing.assert_array_equal(u,[2.,3.])
        with self.assertRaises(ValueError):cached.state_for_nu(float('nan'))
        with self.assertRaises(ValueError):shared_lift.split_common_lift(c.astype('float32'),2,.01)

if __name__=='__main__':unittest.main()
