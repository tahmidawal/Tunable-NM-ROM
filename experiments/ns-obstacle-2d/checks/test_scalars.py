"""Scalar contract unit checks; no native NGSolve import or FE work."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scalar_contract',ROOT/'scalars.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ScalarContract(unittest.TestCase):
    def setUp(self):module._observations.clear()
    def test_valid_scalars_and_singletons(self):
        for value in [2.,2,np.float64(2.),np.int64(2),np.array(2.),np.array([2.]),(2.,)]:
            with self.subTest(value=repr(value)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.scalar_value(value,'valid'),2.)
    def test_rejects_other_shapes_and_types(self):
        for value in [(),(1.,2.),((2.,),),np.array([[2.]]),np.ones((1,1,1)),np.empty(0),[2.],True,np.bool_(False),2+0j,np.array([2+0j]),'2',np.array(['2']),np.array([2],dtype=object),np.ma.array([2.],mask=[True])]:
            with self.subTest(value=repr(value)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):module.scalar_value(value,'invalid')
    def test_rejects_nonfinite(self):
        for value in [float('nan'),float('inf'),-float('inf'),(float('nan'),),np.array([float('inf')])]:
            with self.subTest(value=repr(value)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):module.scalar_value(value,'nonfinite')
    def test_records_tuple_shape_and_rejection(self):
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertEqual(module.scalar_value((2.,),'wrapped_divergence_point'),2.)
            self.assertEqual(module.scalar_value((3.,),'wrapped_divergence_point'),3.)
            with self.assertRaises(ValueError):module.scalar_value((2.,3.),'wrapped_divergence_point')
        records=[json.loads(line)['scalar_return_contract'] for line in log.getvalue().splitlines()]
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]['python_type'],'builtins.tuple')
        self.assertEqual(records[0]['shape'],[1]);self.assertTrue(records[0]['accepted'])
        self.assertEqual(records[1]['shape'],[2]);self.assertFalse(records[1]['accepted'])
        self.assertEqual(module.scalar_observations(),records)

if __name__=='__main__':unittest.main()
