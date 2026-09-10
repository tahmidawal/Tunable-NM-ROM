"""Stored/shared and original/self-contained fields preserve identical data."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from fields import file_sha, save_fields, save_reference


class FieldTests(unittest.TestCase):
    def test_lossless_shared_and_self_contained(self):
        grid = SimpleNamespace(n=4, bx='dirichlet', shape=(3, 3))
        u = np.arange(27, dtype=np.float64).reshape(3, 3, 3)/19
        v, ut, vt = u+.1, u-.2, u+.3
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ref, digest = save_reference(out, 'truth', (ut, vt), grid, 1.07)
            self.assertEqual(file_sha(out/ref), digest)
            shared, kind = save_fields(out, 'shared', u, v, (ut, vt), grid, 1.07, True, True)
            self.assertEqual(kind, 'full_grid_with_shared_truth')
            old, kind = save_fields(out, 'old', u, v, (ut, vt), grid, 1.07, True)
            self.assertEqual(kind, 'self_contained_full_grid')
            with np.load(out/shared) as prediction, np.load(out/ref) as truth, np.load(out/old) as original:
                self.assertNotIn('truth_u', prediction)
                for name in ('u', 'v'):
                    np.testing.assert_array_equal(prediction[name], original[name])
                for name in ('truth_u', 'truth_v'):
                    np.testing.assert_array_equal(truth[name], original[name])
                for name in ('intervals', 'boundary', 'speed'):
                    np.testing.assert_array_equal(prediction[name], truth[name])


if __name__ == '__main__':
    unittest.main()
