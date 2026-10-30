"""Synthetic NumPy algebra tests; no native/model/teacher evaluation."""
import json
import numpy as np
from accuracy_v2.boundary_traction_objective import traction,teacher_normalization,block_traction_objective

def check():
    rng=np.random.default_rng(9);n=19;r=4;angles=np.linspace(0,2*np.pi,n,endpoint=False);normal=np.c_[np.cos(angles),np.sin(angles)];w=np.full(n,1/n);s=np.array([6.,.35,.025,.002]);teacher=rng.normal(size=(n,2,2,r));teacher[...,3]=0.;predicted=teacher+rng.normal(size=teacher.shape)*.03
    stats=teacher_normalization(teacher,normal,w,s);den=stats['global_training_traction_energy'];loss,parts=block_traction_objective(predicted,teacher,normal,w,s,den)
    assert block_traction_objective(teacher,teacher,normal,w,s,den)[0]==0
    split=sum(block_traction_objective(predicted[a:b],teacher[a:b],normal[a:b],w[a:b],s,den)[0] for a,b in [(0,7),(7,n)]);assert abs(loss-split)<1e-15
    q=np.linalg.qr(rng.normal(size=(9,r)))[0].T;c=s[:,None]*q
    errors=np.einsum('nir,rs->sni',traction(predicted-teacher,normal),c);truth=np.einsum('nir,rs->sni',traction(teacher,normal),c)
    ensemble=np.einsum('n,sni,sni->',w,errors,errors)/np.einsum('n,sni,sni->',w,truth,truth);assert abs(ensemble-loss)<1e-15
    integrated=np.sum(s*s*np.sum(parts['integrated_traction_error_by_component_mode']**2,axis=0));assert integrated<=w.sum()*parts['weighted_numerator']*(1+1e-12)
    direction=rng.normal(size=predicted.shape);h=1e-6
    fd=(block_traction_objective(predicted+h*direction,teacher,normal,w,s,den)[0]-block_traction_objective(predicted-h*direction,teacher,normal,w,s,den)[0])/(2*h)
    exact=2*np.einsum('n,r,nir,nir->',w,s*s,traction(predicted-teacher,normal),traction(direction,normal))/den
    assert abs(fd-exact)<1e-9
    return {'passed':True,'synthetic_only':True,'ensemble_gap':abs(ensemble-loss),'block_gap':abs(loss-split),'directional_derivative_gap':abs(fd-exact),'zero_mode_norm_supported':True,'no_lambda_parent_or_trainer_selected':True,'no_physical_pass_claim':True}
if __name__=='__main__':print(json.dumps(check(),indent=2))
