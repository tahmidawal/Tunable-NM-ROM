"""Synthetic NumPy check, no saved model/teacher/native/JAX execution."""
import json
import numpy as np
from accuracy_v2.spatial_bank import initialize
from accuracy_v2.additive_residual import initialize_correction,evaluate_correction_sum

def check():
    rng=np.random.default_rng(17);x=np.column_stack([rng.uniform(.35,2.1,23),rng.uniform(.03,.38,23)])
    parent=initialize(width=7,depth=2,seed=13);pv,pg=parent.evaluate(x);layers=initialize_correction(width=9,depth=2,seed=19)
    v,g=evaluate_correction_sum(pv,pg,layers,x);assert np.array_equal(v,pv) and np.array_equal(g,pg)
    w,b=layers[-1];layers[-1]=(rng.normal(size=w.shape)*.01,b)
    v,g=evaluate_correction_sum(pv,pg,layers,x);errors=[]
    for axis in range(2):
        dx=np.zeros_like(x);dx[:,axis]=1e-6
        pp,dp=parent.evaluate(x+dx);pm,dm=parent.evaluate(x-dx)
        vp=evaluate_correction_sum(pp,dp,layers,x+dx)[0];vm=evaluate_correction_sum(pm,dm,layers,x-dx)[0]
        errors.append(float(np.max(np.abs((vp-vm)/2e-6-g[:,:,axis,:]))))
    assert max(errors)<1e-6
    boundary=np.array([[0.,.1],[.5,0.],[.5,.41],[.25,.2]])
    bp,bg=parent.evaluate(boundary);bv,_=evaluate_correction_sum(bp,bg,layers,boundary)
    assert np.max(np.abs(bv))<1e-13
    return {'passed':True,'synthetic_only':True,'initial_parent_values_and_gradients_bit_identical':True,'same_four_online_columns':True,'finite_difference_max_errors':errors,'boundary_value_max':float(np.max(np.abs(bv))),'JAX_execution_performed':False,'training_or_actual_model_selected':False}
if __name__=='__main__':print(json.dumps(check(),indent=2))
