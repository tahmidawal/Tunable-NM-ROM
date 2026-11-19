"""Pinned CROM networks/optimizer with a verified manual-loop and float64 port.

This trains only; it does not claim a reproduced 2D Burgers evolution algorithm.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import argparse,sys,subprocess,shutil,difflib,importlib.util
from common import *
from neural.build_data import resolve
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=200);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);native=resolve(ns/'b6native/output/result.json',ns)
    evidence=json.loads(native.read_text());assert evidence['crom_native_online_fixture']['passed']
    result=dict(meta=metadata,status='isolated_crom_dependencies',source_commit=evidence['sources']['crom_offline']['commit'],
        classification='Pinned author networks and optimizer; manual training-loop and float64 port, not yet an eligible PDE comparator',
        final_test_opened=False,history=[],latent=32,width=128,epochs=a.epochs)
    save(out/'result.json',result)
    packages=out/'python-packages'
    # Never resolve or replace the shared environment's JAX/Torch dependencies.
    with open(out/'dependency-install.log','w') as log:
        try:
            installed=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(packages),
                'pytorch-lightning==1.9.5','torchmetrics==0.11.4','lightning-utilities==0.8.0','nvidia-ml-py==12.560.30','tqdm==4.67.1'],
                stdout=log,stderr=subprocess.STDOUT,timeout=120)
            if installed.returncode:raise RuntimeError('Isolated dependency installation failed')
        except Exception as e:
            result.update(status='stopped_dependency_failure',error=repr(e));save(out/'result.json',result);return
    source=native.parent/'sources/crom_offline';port=out/'crom-port';port.mkdir()
    hashes={v['path']:v['sha256'] for v in evidence['sources']['crom_offline']['files']}
    patch=[]
    for name in ['cromnet.py','simulation.py','util.py']:
        assert digest(source/name)==hashes[name]
        original=(source/name).read_text();modified=original.replace('.float()', '.double()') if name=='cromnet.py' else original
        (port/name).write_text(modified)
        patch.extend(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=name,tofile=name+'.float64'))
    (out/'precision-port.patch').write_text(''.join(patch))
    sys.path[:0]=[str(packages),str(port),str(native.parent/'sources/crom_online/online')]
    try:
        import torch,cromnet
        from Decoder import Decoder
        from NonlinearSolver import NonlinearSolver
    except Exception as e:
        result.update(status='stopped_native_import_failure',error=repr(e));save(out/'result.json',result);return
    torch.set_default_dtype(torch.float64);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    result['torch']=dict(version=torch.__version__,device=torch.cuda.get_device_name(),dtype=str(torch.get_default_dtype()))
    # Same trajectories/times/references, uniformly sampled as required by native
    # CROM's unweighted MSE and convolutional field encoder.
    parent=resolve(ns/'b6parent/output/coordinate_data.json',ns);record=json.loads(parent.read_text())
    axis=np.linspace(0,1,128);xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    arrays={};synth=jax.jit(data.synth_grid)
    for group in ['train','dev']:
        values=[];norms=[]
        for row in record['groups'][group]['sources']:
            path=resolve(row['file'],ns);assert digest(path)==row['sha256'];b=np.load(path)['coefficients']
            u=np.asarray(synth(jnp.asarray(b),jnp.asarray(axis))).reshape(11,-1)
            values.append(u);norms.append(np.linalg.norm(b.reshape(11,-1),axis=1)/2)
        arrays[group]=np.concatenate(values);arrays[group+'_norm']=np.concatenate(norms)
    np.savez_compressed(out/'matched_data.npz',xy=xy,**arrays)
    result['matched_data_sha256']=digest(out/'matched_data.npz')
    fmt=dict(i_dim=2,o_dim=1,npoints=len(xy),data_path='matched-owned-training-references')
    stats=dict(mean_x=xy.mean(0),std_x=xy.std(0),min_x=xy.min(0),max_x=xy.max(0),
               mean_q=np.array([arrays['train'].mean()]),std_q=np.array([arrays['train'].std()]))
    model=cromnet.CROMnet(fmt,stats,torch.zeros((1,len(xy),3)),1e-4,4,32,64,6,4,
                         False,True,.3,30,False,[1.,.5],[100,100],'explicit')
    model.setup('fit');model=model.cuda().double()
    assert all(p.dtype==torch.float64 for p in model.parameters())
    assert all(b.dtype==torch.float64 for b in model.buffers())
    optimizer_list,scheduler_list=model.configure_optimizers();optimizer=optimizer_list[0];scheduler=scheduler_list[0]
    tx=torch.as_tensor(xy,device='cuda');train=torch.as_tensor(arrays['train'],device='cuda')
    dev=torch.as_tensor(arrays['dev'],device='cuda');rng=np.random.default_rng(2026111953)
    def inputs(q):return torch.cat([q[:,:,None],tx[None].expand(len(q),-1,-1)],2)
    # Verify manual objective and all parameter gradients against author training_step.
    q=train[:2];batch=dict(encoder_input=inputs(q),q=q[:,:,None])
    authored=model.training_step(batch,0)['loss'];ga=torch.autograd.grad(authored,tuple(model.parameters()))
    manual=1000*model.criterion(model(inputs(q))[0],q[:,:,None]);gb=torch.autograd.grad(manual,tuple(model.parameters()))
    gradient_error=max(float(torch.max(abs(x-y))) for x,y in zip(ga,gb));assert gradient_error<1e-12
    probe=torch.randn((19,34),device='cuda');analytic,_=model.decoder.computeJacobianFullAnalytical(probe)
    autodiff=torch.func.vmap(torch.func.jacrev(lambda x:model.decoder(x[None])[0]))(probe)
    jacobian_error=float(torch.max(abs(analytic-autodiff)));assert jacobian_error<1e-10
    result['port_fixtures']=dict(training_gradient_max_difference=gradient_error,native_analytic_jacobian_difference=jacobian_error)
    indices=[11*c+j for c in [0,3,13,43] for j in [0,5,10]]
    best=np.inf;stale=0
    for epoch in range(1,a.epochs+1):
        model.train();order=rng.permutation(len(train));losses=[]
        for lo in range(0,len(order),4):
            q=train[order[lo:lo+4]];optimizer.zero_grad(set_to_none=True)
            predicted=model(inputs(q))[0];loss=1000*model.criterion(predicted,q[:,:,None])
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        if epoch%10:continue
        model.eval();errors=[]
        with torch.no_grad():
            for i in indices:
                predicted=model(inputs(dev[i:i+1]))[0][0,:,0]
                error=torch.linalg.norm(predicted-dev[i])/torch.linalg.norm(dev[i]);errors.append(float(error))
        row=dict(epoch=epoch,mean_native_loss=float(np.mean(losses)),encoded_subset_worst=max(errors),encoded_subset_errors=errors)
        torch.save(dict(model=model.state_dict(),format=fmt,statistics=stats,epoch=epoch),out/f'crom_epoch{epoch:04d}.pt')
        result['history'].append(row);print('native_crom_training',row,flush=True)
        if max(errors)<best:
            significant=max(errors)<best*.99;stale=0 if significant else stale+1;best=max(errors)
            shutil.copy2(out/f'crom_epoch{epoch:04d}.pt',out/'crom_selected.pt');result['selected_epoch']=epoch
        else:stale+=1
        result['status']='native_network_training';save(out/'result.json',result)
        if epoch>=60 and stale>=5:break
    result.update(status='complete_native_network_training_diagnostic',selected_sha256=digest(out/'crom_selected.pt'),
        representation_admitted=False,pde_comparator_admitted=False,
        remaining='Native reconstruction projection, independent scoring, actual4096 output and verified 2D viscous dynamics adapter')
    save(out/'result.json',result)


if __name__=='__main__':main()
