"""Synthetic array-only retention checks; no real solve or gate amendment."""
from pathlib import Path
import tempfile,json
from types import SimpleNamespace
import numpy as np
import neural_diagnostic2d as d

class Model:
    def __init__(self,fail=False):
        self.fail=fail;self.operator=SimpleNamespace(residual=lambda *a,**k:np.zeros(2),terms=lambda *a:{'a':np.ones(2)})
        self.rule=SimpleNamespace(direct=lambda *a:np.zeros(2),weights=np.ones(2))
    def query(self,re,contract,operator,interpolate):
        assert operator is self.operator and interpolate is False and contract=='full_velocity_pressure_fields'
        if self.fail:raise ArithmeticError('synthetic query failure')
        return {'evidence':{'state':np.zeros(2)},'outputs':{'velocity':np.zeros((2,2)),'pressure':np.zeros(2)},'work':{'status':'stationary'}}
    def reconstruct(self,*a,**k):return {'velocity':np.zeros((2,2)),'pressure':np.zeros(2),'velocity_gradient':np.zeros((2,2,2))}
    def force_output(self,*a):return np.zeros(2)


def main():
    original_score=d.pilot.score_state;original_jac=d.pilot.jacobian_check
    try:
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def score(*args):
                assert (out/'bank_r8_16.875_raw_invocation.json').exists()
                return {'accuracy_targets_passed':True}
            d.pilot.score_state=score;d.pilot.jacobian_check=lambda *a:{'passed':True}
            row=d.diagnose(Model(),16.875,{}, {},out,lambda:None)
            assert row['diagnostic_numerical_targets_passed'] and row['eligible_accuracy'] is False and row['physical_admission'] is False
            row=d.diagnose(Model(True),23.125,{}, {},out,lambda:None)
            assert row['eligible_accuracy'] is False and 'failure' in row
            raw=json.loads((out/'bank_r8_23.125_raw_invocation.json').read_text())['record'];assert raw['state_available'] is False
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def fail_score(*args):
                assert (out/'bank_r8_16.875_raw_invocation.json').exists()
                raise ValueError('synthetic scoring failure')
            d.pilot.score_state=fail_score
            row=d.diagnose(Model(),16.875,{}, {},out,lambda:None)
            assert 'diagnostic_failure' in row and row['eligible_accuracy'] is False
    finally:d.pilot.score_state=original_score;d.pilot.jacobian_check=original_jac
    print('PASS: synthetic retention ordering and failures; diagnostic target pass never opens eligibility')

if __name__=='__main__':main()
