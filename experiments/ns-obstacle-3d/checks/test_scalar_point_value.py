"""Pure scalar-conversion checks; no NGSolve/native FE import or execution.

Extract the exact production function AST so importing the solver is unnecessary.
"""
import ast
import math
from pathlib import Path
import unittest
import numpy as np

source=Path(__file__).resolve().parents[1]/'reference3d.py'
tree=ast.parse(source.read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='scalar_point_value')
namespace={'np':np,'math':math}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
extract=namespace['scalar_point_value']

class ScalarPointContract(unittest.TestCase):
    def test_real_scalar_and_one_element_tuple_array(self):
        for raw in (3.0,np.float64(3),3,(3.0,),np.array([3.0]),np.array([[3.0]]),np.array(3.0)):
            with self.subTest(raw=repr(raw)):
                value,record=extract(raw)
                self.assertEqual(value,3.0)
                self.assertEqual(record['return_size'],1)
                self.assertEqual(record['return_shape'],list(np.asarray(raw).shape))
                self.assertEqual(record['return_type'],type(raw).__module__+'.'+type(raw).__qualname__)
    def test_wrong_sizes_rejected(self):
        for raw in ((),(1.0,2.0),np.array([]),np.ones((2,2))):
            with self.subTest(raw=repr(raw)):
                with self.assertRaises(ValueError):extract(raw)
    def test_nonfinite_rejected(self):
        for raw in (float('nan'),float('inf'),(-float('inf'),),np.array([float('nan')])):
            with self.subTest(raw=repr(raw)):
                with self.assertRaises(ValueError):extract(raw)
    def test_nonreal_or_unsupported_container_rejected(self):
        for raw in ('3',True,3+0j,np.array([3],dtype=object),[3.0],{'x':3.0}):
            with self.subTest(raw=repr(raw)):
                with self.assertRaises((TypeError,ValueError)):extract(raw)

if __name__=='__main__':unittest.main()
