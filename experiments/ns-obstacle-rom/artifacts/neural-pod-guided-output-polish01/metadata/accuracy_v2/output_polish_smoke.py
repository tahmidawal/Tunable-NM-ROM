"""Synthetic-only retained affine-row/QR fixture; never opens campaign inputs."""
from pathlib import Path
import json,hashlib
import numpy as np
from accuracy_v2.fixed_hidden_output_polish import output_features,design,volume_rows,boundary_rows,qr_append,solve_factor
from accuracy_v2.spatial_bank import evaluate_layers


def run(out):
    out=Path(out);out.mkdir(exist_ok=False);rng=np.random.default_rng(31002026)
    n=31;x=np.c_[rng.uniform(.3,2.,n),rng.uniform(.05,.36,n)]
    hidden=[(rng.normal(size=(21,64))*.1,rng.normal(size=64)*.1),(rng.normal(size=(64,64))*.1,rng.normal(size=64)*.1)]
    output=(rng.normal(size=(64,2))*.1,rng.normal(size=2)*.1);delta=rng.normal(size=130)*.01
    v,g=output_features(hidden,x);V,G=design(v,.1*g)
    before=evaluate_layers(hidden+[output],x)
    changed=(output[0]+delta[:128].reshape(64,2),output[1]+delta[128:]);after=evaluate_layers(hidden+[changed],x)
    value_gap=(after[0]-before[0])[...,0]-V@delta;gradient_gap=.1*(after[1]-before[1])[...,0]-G@delta
    e=rng.normal(size=(n,2));eg=rng.normal(size=(n,2,2));w=rng.uniform(.1,.8,n);mw=.3;nv=2.;ng=3.;lam=1.
    normal=rng.normal(size=(n,2));normal/=np.linalg.norm(normal,axis=1)[:,None]
    A,b=volume_rows(v,.1*g,e,eg,w,mw,nv,ng,lam);T,t=boundary_rows(v,.1*g,eg,normal,w,.7,4.,1.)
    dv=e+V@delta;dg=eg+G@delta;traction=np.einsum('nij,nj->ni',dg+dg.swapaxes(1,2),normal)
    direct=mw*np.sum(w*(np.sum(dv*dv,axis=1)/nv+np.sum(dg*dg,axis=(1,2))/ng+(dg[:,0,0]+dg[:,1,1])**2/ng))+(.7**2/4.)*np.sum(w*np.sum(traction*traction,axis=1))
    matrix=np.r_[A,T];right=np.r_[b,t];row_value=np.sum((matrix@delta-right)**2)
    def direct_at(d):
        ev=e+V@d;gg=eg+G@d;tt=np.einsum('nij,nj->ni',gg+gg.swapaxes(1,2),normal)
        return mw*np.sum(w*(np.sum(ev*ev,axis=1)/nv+np.sum(gg*gg,axis=(1,2))/ng+(gg[:,0,0]+gg[:,1,1])**2/ng))+(.7**2/4.)*np.sum(w*np.sum(tt*tt,axis=1))
    step=1e-5;fd=np.array([(direct_at(delta+np.eye(130)[i]*step)-direct_at(delta-np.eye(130)[i]*step))/(2*step) for i in range(130)])
    row_gradient=2*matrix.T@(matrix@delta-right)
    rank_matrix=rng.normal(size=(37,130));rank_matrix[:,1]=rank_matrix[:,0];rank_matrix[:,2]=0.;rank_right=rng.normal(size=37);rank_factor=None
    for start in range(0,37,7):rank_factor=qr_append(rank_factor,rank_matrix[start:start+7],rank_right[start:start+7])
    rank_solved=solve_factor(rank_factor,1e-12);rank_dense=np.linalg.lstsq(rank_matrix,rank_right,rcond=1e-12)
    factor=None
    for start in range(0,len(matrix),73):factor=qr_append(factor,matrix[start:start+73],right[start:start+73])
    solved=solve_factor(factor,1e-12);dense=np.linalg.lstsq(matrix,right,rcond=1e-12)
    arrays={'value_error':e,'gradient_error':eg,'weights':w,'normal':normal,'fd_gradient':fd,'row_gradient':row_gradient,'rank_matrix':rank_matrix,'rank_right':rank_right,'rank_factor':rank_factor,'rank_solution':rank_solved['delta'],'rank_dense_solution':rank_dense[0],'points':x,'output_W':output[0],'output_b':output[1],'delta_probe':delta,'value_gap':value_gap,'gradient_gap':gradient_gap,'value_design':V,'gradient_design':G,'matrix':matrix,'right':right,'factor':factor,'solution':solved['delta'],'dense_solution':dense[0],'spectrum':solved['singular_values'],'row_objective':np.asarray(row_value),'direct_objective':np.asarray(direct)}
    for i,(ww,bb) in enumerate(hidden):arrays[f'hiddenW{i}']=ww;arrays[f'hiddenb{i}']=bb
    path=out/'arrays.npz';np.savez_compressed(path,**arrays)
    result={'synthetic_only':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'spatial_source_sha256':hashlib.sha256(Path(__file__).with_name('spatial_bank.py').read_bytes()).hexdigest(),'normalization':{'mode_weight':mw,'value_norm':nv,'gradient_norm':ng,'trace_weight':lam,'sigma':.7,'global_traction_energy':4.,'traction_weight':1.},'helper_sha256':hashlib.sha256(Path(__file__).with_name('fixed_hidden_output_polish.py').read_bytes()).hexdigest(),'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'FD_gradient_gap':float(np.max(np.abs(fd-row_gradient))),'rank_deficient_rank':rank_solved['numerical_rank'],'rank_deficient_delta_gap':float(np.max(np.abs(rank_solved['delta']-rank_dense[0]))),'max_value_gap':float(np.max(np.abs(value_gap))),'max_gradient_gap':float(np.max(np.abs(gradient_gap))),'objective_gap':abs(row_value-direct),'QR_solution_objective':float(np.sum((matrix@solved['delta']-right)**2)),'dense_solution_objective':float(np.sum((matrix@dense[0]-right)**2)),'rank_QR':solved['numerical_rank'],'rank_dense':int(dense[2]),'actual_TRAIN_access':False,'GPU_execution':False}
    result['passed']=bool(result['FD_gradient_gap']<1e-7 and result['rank_deficient_rank']<130 and result['rank_deficient_delta_gap']<1e-10 and result['max_value_gap']<1e-12 and result['max_gradient_gap']<1e-12 and result['objective_gap']<1e-10 and abs(result['QR_solution_objective']-result['dense_solution_objective'])<1e-8 and result['rank_QR']==result['rank_dense'])
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');assert result['passed'];return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out),indent=2))
