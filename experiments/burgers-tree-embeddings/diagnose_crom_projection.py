"""Native CROM Gauss--Newton reconstruction diagnosis on a fixed dev subset."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import argparse,sys
from common import *
from neural.build_data import resolve
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/'b6cromdeps/output/result.json',ns);r=json.loads(rp.read_text());folder=rp.parent
    native=resolve(ns/'b6native/output/result.json',ns).parent
    sys.path[:0]=[str(folder/'python-packages'),str(folder/'crom-port'),str(native/'sources/crom_online/online')]
    import torch,cromnet
    from NonlinearSolver import NonlinearSolver
    torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
    path=folder/'crom_selected.pt';assert digest(path)==r['selected_sha256']
    state=torch.load(path,map_location='cuda',weights_only=False);fmt=state['format'];stats=state['statistics']
    model=cromnet.CROMnet(fmt,stats,torch.zeros((1,fmt['npoints'],3)),1e-4,4,32,64,6,4,False,True,.3,30,False,[1.,.5],[100,100],'explicit')
    model.setup('fit');model=model.cuda().double();model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    dp=folder/'matched_data.npz';assert digest(dp)==r['matched_data_sha256'];d=dict(np.load(dp))
    xy=torch.as_tensor(d['xy'],device='cuda');dev=torch.as_tensor(d['dev'],device='cuda')
    class Problem:
        def updateStateSample(self,z,decoder,points):return model.decoder(torch.cat([z.reshape(1,32).expand(len(points),-1),points],1))
        def getJacobianSample(self,z,decoder,points):
            inp=torch.cat([z.reshape(1,32).expand(len(points),-1),points],1)
            J,_=model.decoder.computeJacobianFullAnalytical(inp);return J[:,0,:32]
    problem=Problem();solver=NonlinearSolver(problem,None,1e-10,1e-10)
    indices=[11*c+j for c in [0,3,13,43] for j in [0,5,10]]
    parent=json.loads(resolve(ns/'b6parent/output/coordinate_data.json',ns).read_text())
    xf,wf=data.grid(256,True);points=torch.as_tensor(xf,device='cuda');wf=torch.as_tensor(wf,device='cuda')
    result=dict(meta=meta,status='running',selected_sha256=digest(path),rows=[],final_test_opened=False,
        classification='Unmodified native Gauss--Newton with float64 author decoder; offline projection diagnostic only',pde_comparator_admitted=False)
    saved={}
    for i in indices:
        with torch.no_grad():z=model.encoder(dev[i:i+1,:,None]).reshape(1,1,32)
        source=parent['groups']['dev']['sources'][i//11];src=resolve(source['file'],ns);assert digest(src)==source['sha256']
        b=jnp.asarray(np.load(src)['coefficients'][i%11:i%11+1]);truth=torch.as_tensor(np.asarray(data.synth_grid(b,jnp.asarray(np.unique(xf[:,0])))).reshape(-1).copy(),device='cuda')
        errors=[];fits=[]
        for it in [0,20,50]:
            if it:z=solver.solve(z,dev[i,:,None],xy,step_size=1,max_iters=it,print_info=False)
            pred=problem.updateStateSample(z,None,points)[:,0]
            error=torch.sqrt(torch.sum(wf*(pred-truth)**2))/d['dev_norm'][i]
            errors.append(float(error));fits.append(z.cpu().numpy())
        J=problem.getJacobianSample(z,None,xy);res=problem.updateStateSample(z,None,xy)-dev[i,:,None]
        scales=torch.linalg.vector_norm(J,dim=0).clamp(min=1e-14)
        grad=float(torch.max(abs((J.T@res)[:,0]/scales))/torch.linalg.vector_norm(dev[i]))
        row=dict(index=i,case=source['case'],time=float(data.TIMES[i%11]),independent_errors=errors,scaled_stationarity=grad)
        result['rows'].append(row);saved[str(i)]=np.stack(fits);save(out/'result.json',result);print('native_crom_projection',row,flush=True)
    np.savez_compressed(out/'native_projection_codes.npz',**saved)
    result.update(status='complete',worst_encoded=max(v['independent_errors'][0] for v in result['rows']),
        worst_projected=max(v['independent_errors'][-1] for v in result['rows']),representation_admitted=False)
    save(out/'result.json',result)


if __name__=='__main__':main()
