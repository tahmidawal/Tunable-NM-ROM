"""Pure independent examples for physical-volume norms and no pressure shifting."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'g2'))
from compare_fields import field_metrics


class CommonNorms(unittest.TestCase):
    def test_nonuniform_volume_and_absolute_pressure(self):
        # Domain volume four; constant reference velocity has norm six.
        weights=np.array([1.,3.]);reference=np.array([[3.,0.],[3.,0.]])
        value=field_metrics(reference+np.array([0.,4.]),np.array([2.,2.]),
                            reference,np.zeros(2),weights,U=2.)
        self.assertAlmostEqual(value['velocity_relative']['numerator'],8.)
        self.assertAlmostEqual(value['velocity_relative']['denominator'],6.)
        self.assertAlmostEqual(value['pressure_scaled']['value'],.5)
        self.assertFalse(value['pressure_shift_fitted'])

    def test_zero_reference_and_nonpositive_weights_fail(self):
        u=np.ones((2,2));p=np.zeros(2)
        with self.assertRaises(ValueError):field_metrics(u,p,np.zeros_like(u),p,[1.,1.],1.)
        with self.assertRaises(ValueError):field_metrics(u,p,u,p,[1.,0.],1.)


if __name__=='__main__':unittest.main()
