"""Controls for failed physical states and undefined phase diagnostics."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from fresh_fom import Grid
from fresh_models import head_init
from fresh_evaluate import field_metrics,physical_metrics_finite,reconstruction_metrics,physical_summary,stats


class FreshEvaluationTests(unittest.TestCase):
    def test_overflow_physical_energy_rejected(self):
        grid=Grid(4)
        g=jnp.ones((9,1))
        values=field_metrics(jnp.full((2,1),1e200),jnp.zeros((2,1)),g,jnp.zeros((2,9)),jnp.zeros((2,9)),grid,1.,1.,1.,1.)
        self.assertFalse(physical_metrics_finite({k:np.asarray(v) for k,v in values.items()}))
        self.assertTrue(np.all(np.isfinite(np.asarray(values["rom_mean"]))))

    def test_deficient_and_nan_jacobians_are_failures(self):
        for linear in (np.zeros((4,2)),np.full((4,2),np.nan)):
            p,f=head_init(jax.random.PRNGKey(1),linear,np.zeros(4),1.,"quadratic")
            pj={"a":np.ones((1,1,4)),"b":np.ones((1,1,4)),"u_floor_squared":np.zeros((1,1)),"v_floor_squared":np.zeros((1,1)),"u_scale":np.ones(1),"v_scale":np.ones(1)}
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
                result,_=reconstruction_metrics(p,f,np.zeros((1,2)),pj,"quadratic",Path(temp))
            self.assertEqual(result["rank_failures"],1)

    def test_vanished_prediction_has_undefined_phase(self):
        grid=Grid(8)
        mm={"rom_modal_u":np.zeros((4,3)),"rom_modal_v":np.zeros((4,3)),"truth_modal_u":np.ones((4,3)),"truth_modal_v":np.zeros((4,3)),"rom_wall_strip":np.zeros(4),"truth_wall_strip":np.ones(4),"displacement_error":np.ones(4),"velocity_error":np.zeros(4),"energy_state_error":np.ones(4),"rom_mean":np.zeros(4),"truth_mean":np.ones(4),"rom_energy":np.zeros(4),"truth_energy":np.ones(4)}
        summary=physical_summary(mm,grid,1.,.05)
        self.assertIsNone(summary["max_defined_modal_phase_error"])
        self.assertEqual(summary["vanished_mode_observations"],12)
        self.assertFalse(bool(np.any(mm["modal_phase_defined"])))
        self.assertEqual(stats([.01,np.nan])["outliers"],1)


if __name__=="__main__":
    print({"jax_backend":jax.default_backend(),"x64":jax.config.jax_enable_x64},flush=True)
    unittest.main()
