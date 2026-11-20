"""Actual trained CROM derivative/BC/formulation checks, never PDE admission."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import argparse,sys,traceback,hashlib
from common import *
from neural.build_data import resolve
from neural import expanded_cases as E


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--model-job',default='b8cromboundary')
    a=ap.parse_args();assert a.model_job.replace('-','').isalnum()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    result=dict(status='preparing',passed=False,final_test_opened=False,development_solution_fields_opened=False,
        representation_admitted=False,pde_comparator_admitted=False,
        scope='GPU Float64 trained derivatives/continuous boundaries and tiny formulation equivalence only; no PDE trajectory',
        classification='Pinned CROM network/native Decoder plus continuous-boundary scalar2D new-PDE port; not unchanged published reproduction')
    save(out/'result.json',result)
    try:
        assert os.environ.get('SLURM_JOB_ID');result['meta']=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
        rp=resolve(ns/a.model_job/'output/result.json',ns);r=json.loads(rp.read_text())
        assert r['status']=='complete_native_expanded_training_diagnostic' and not r['final_test_opened']
        mp=HERE/'manifests'/f'retention-{a.model_job}.json';m=json.loads(mp.read_text())
        assert m['local_and_remote_verified'] and m['terminal_state']=='COMPLETED'
        retained={row['file']:row['sha256'] for row in m['files']}
        def verified(name):
            p=rp.parent/name;assert digest(p)==retained['output/'+name];return p
        assert digest(rp)==retained['output/result.json']
        cp=verified('crom_selected.pt');assert digest(cp)==r['selected_sha256']
        dp=verified('matched_data.npz');sp=verified('matched_sources.json')
        assert digest(dp)==r['matched_data_sha256'] and digest(sp)==r['matched_sources_sha256']
        sources=json.loads(sp.read_text());groups=sources['groups']
        assert len(groups['train'])==160 and len(groups['dev'])==52
        primary=resolve(ns/'b6parent/output/coordinate_data.json',ns);assert digest(primary)==sources['primary_record_sha256']
        primary_record=json.loads(primary.read_text())
        for group,oldcount in [('train',128),('dev',44)]:
            rows=groups[group];assert len({v['case'] for v in rows})==len(rows)
            assert [(v['case'],v['sha256']) for v in rows[:oldcount]]==[(v['case'],v['sha256']) for v in primary_record['groups'][group]['sources']]
            assert [v['case'] for v in rows[oldcount:]]==[v['case'] for v in E.definitions(group)]
        assert not {v['case'] for v in groups['train']}&{v['case'] for v in groups['dev']}
        fingerprint=hashlib.sha256(json.dumps({g:[(v['case'],v['sha256']) for v in rows] for g,rows in groups.items()},sort_keys=True).encode()).hexdigest()
        native=resolve(ns/'b6native/output/result.json',ns);ev=json.loads(native.read_text())
        assert digest(native)==r['native_source_result_sha256']
        offline_hash={v['path']:v['sha256'] for v in ev['sources']['crom_offline']['files']}
        port=rp.parent/'crom-port'
        for name in ['cromnet.py','simulation.py','util.py']:
            p=native.parent/'sources/crom_offline'/name;assert digest(p)==offline_hash[name]
            expected=p.read_text().replace('.float()','.double()') if name=='cromnet.py' else p.read_text()
            assert (port/name).read_text()==expected;verified('crom-port/'+name)
        online=native.parent/'sources/crom_online/online'
        online_hash={v['path']:v['sha256'] for v in ev['sources']['crom_online']['files']}
        assert digest(online/'Decoder.py')==online_hash['online/Decoder.py']
        deps=resolve(ns/'b6cromdeps/output/python-packages',ns)
        dm=json.loads((HERE/'manifests/retention-b6cromdeps.json').read_text());assert dm['local_and_remote_verified']
        for v in dm['files']:
            prefix='output/python-packages/'
            if v['file'].startswith(prefix) and not v['file'].endswith('.pyc'):
                assert digest(deps/v['file'][len(prefix):])==v['sha256']
        assert digest(HERE/'baselines/crom_boundary.py')==r['boundary_port_sha256']
        sys.dont_write_bytecode=True;sys.path[:0]=[str(deps),str(port),str(online)]
        import torch,cromnet
        from Decoder import Decoder
        from baselines.crom_loader import load_native_online
        from baselines.crom_scalar2d import ScalarBurgers2D,structured_nodes
        torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.is_available()
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert state['config']==r['config'] and state['config']['boundary_envelope'] is True
        assert state['config']['source_fingerprint']==fingerprint and state['epoch']==r['selected_epoch']
        with np.load(dp) as v:
            xy=v['xy'];train=v['train'];assert train.shape==(1760,16384)
            axis=np.linspace(0,1,128);expected_xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
            assert np.array_equal(xy,expected_xy)
            stats=dict(mean_x=xy.mean(0),std_x=xy.std(0),min_x=xy.min(0),max_x=xy.max(0),
                       mean_q=np.array([train.mean()]),std_q=np.array([train.std()]))
            for key,value in stats.items():assert np.allclose(value,state['statistics'][key],rtol=1e-13,atol=1e-14),key
            initial_fields=train[[0,33,1408,1419]].copy()
        del train
        model,decoder,classification=load_native_online(state,cromnet.CROMnet,Decoder,device='cuda')
        latent=classification['latent']
        with torch.no_grad():codes=model.encoder(torch.as_tensor(initial_fields,device='cuda')[:,:,None]).reshape(4,latent)
        codes=torch.cat([torch.zeros((1,latent),device='cuda'),codes]);assert torch.isfinite(codes).all()
        result.update(status='checking_trained_model',model_job=a.model_job,selected_epoch=state['epoch'],
            checkpoint_sha256=digest(cp),training_result_sha256=digest(rp),retention_sha256=digest(mp),
            normalization_verified=True,source_fingerprint=fingerprint,loader=classification,
            adapter_hashes={name:digest(HERE/'baselines'/name) for name in ['crom_loader.py','crom_boundary.py','crom_scalar2d.py']},
            probe_codes='Zero plus encoder codes from train000/train003/b8_train000/b8_train001 initial fields; no held-out solutions',
            checks=[],boundary_strips=[])
        save(out/'result.json',result)
        failure_count=0
        def retain_nonfinite(context,values):
            nonlocal failure_count
            failure_count+=1
            path=out/f'nonfinite_{failure_count:03d}.npz'
            np.savez_compressed(path,**{key:value.detach().cpu().numpy() for key,value in values.items()})
            result.setdefault('nonfinite_failures',[]).append(dict(context=context,file=path.name,sha256=digest(path)))
            # Raw NaN/Inf remain in the binary evidence, never in JSON scalars.
            raise FloatingPointError('Nonfinite numerical evidence retained: '+context)
        def error(actual,expected):
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                retain_nonfinite('derivative_or_formulation_comparison',dict(actual=actual,expected=expected))
            measure=torch.max(abs(actual-expected))/(1+torch.max(abs(expected)))
            if not bool(torch.isfinite(measure)):
                retain_nonfinite('nonfinite_derived_error',dict(actual=actual,expected=expected,measure=measure))
            return float(measure)
        nodes=structured_nodes(8,16384,device='cuda')
        problem=ScalarBurgers2D(decoder,latent,lambda p:torch.zeros((len(p),1),device='cuda'),nodes,intervals=16384)
        points=torch.tensor([[.17,.23],[.43,.61],[.73,.37],[.54,.81]],device='cuda')
        generator=torch.Generator(device='cuda').manual_seed(2026112003)
        directions=torch.randn((3,latent),generator=generator,device='cuda');directions/=torch.linalg.vector_norm(directions,dim=1,keepdim=True)
        for number,z in enumerate(codes):
            inp=torch.cat([z[None].expand(len(points),-1),points],1)
            analytical,value=model.decoder.computeJacobianFullAnalytical(inp)
            automatic=torch.func.vmap(torch.func.jacrev(lambda v:model.decoder(v[None])[0]))(inp)
            ad_error=error(analytical,automatic)
            fd_errors=[]
            for column in range(latent+2):
                delta=torch.zeros_like(inp);delta[:,column]=1e-6
                fd=(model.decoder(inp+delta)-model.decoder(inp-delta))/(2e-6)
                fd_errors.append(error(analytical[:,:,column],fd))
            rhsj=problem.rhs_code_jacobian(z);rhs_rows=[]
            center=problem.field(z,problem.points)[:,0]
            for d in directions:
                attempts=[]
                for epsilon in [1e-3,3e-4,1e-4]:
                    plus,minus=z+epsilon*d,z-epsilon*d
                    same=bool(torch.equal(center>0,problem.field(plus,problem.points)[:,0]>0) and torch.equal(center>0,problem.field(minus,problem.points)[:,0]>0))
                    fd=(problem.getResidualSample(plus)-problem.getResidualSample(minus))[:,0]/(2*epsilon)
                    attempts.append(dict(epsilon=epsilon,same_upwind_branch=same,relative_error=error(rhsj@d,fd)))
                # Need two independently sized stable perturbations, not one lucky epsilon.
                accepted=[v for v in attempts if v['same_upwind_branch'] and v['relative_error']<=5e-4]
                rhs_rows.append(dict(attempts=attempts,passed=len(accepted)>=2))
            row=dict(code=number,full_input_analytic_autodiff_error=ad_error,
                code_fd_worst=max(fd_errors[:latent]),spatial_fd_worst=max(fd_errors[latent:]),rhs_directional=rhs_rows)
            row['passed']=ad_error<=1e-9 and max(fd_errors)<=1e-6 and all(v['passed'] for v in rhs_rows)
            result['checks'].append(row);save(out/'result.json',result)
            for L in [4096,8192,16384]:
                tangent=torch.tensor([.17,.43,.71],device='cuda');h=1/L
                boundary=[];strip1=[];strip2=[]
                for side in range(4):
                    def sidepoints(distance):
                        fixed=torch.full_like(tangent,distance if side<2 else 1-distance)
                        return torch.stack([fixed,tangent] if side%2==0 else [tangent,fixed],1)
                    boundary.append(sidepoints(0.));strip1.append(sidepoints(h));strip2.append(sidepoints(2*h))
                bp=torch.cat(boundary);one=torch.cat(strip1);two=torch.cat(strip2)
                def direct(p):return model.decoder(torch.cat([z[None].expand(len(p),-1),p],1))[:,0]
                bval=direct(bp);u1,u2=direct(one),direct(two)
                bj=model.decoder.computeJacobianFullAnalytical(torch.cat([z[None].expand(len(bp),-1),bp],1))[0][:,0,:latent]
                # Boundary derivative gives a numerical scale for the first strip; no physics target.
                fullj=model.decoder.computeJacobianFullAnalytical(torch.cat([z[None].expand(len(bp),-1),bp],1))[0][:,0,latent:]
                slopes=torch.cat([fullj[0:3,0],fullj[3:6,1],-fullj[6:9,0],-fullj[9:12,1]])
                strip_arrays=dict(boundary_points=bp,first_strip_points=one,second_strip_points=two,
                    boundary_values=bval,first_strip_values=u1,second_strip_values=u2,
                    boundary_code_jacobian=bj,boundary_spatial_jacobian=fullj,inward_normal_derivative=slopes)
                strip_path=out/f'boundary_code{number:02d}_L{L}.npz'
                np.savez_compressed(strip_path,**{key:value.detach().cpu().numpy() for key,value in strip_arrays.items()})
                strip_artifact=dict(code=number,intervals=L,file=strip_path.name,sha256=digest(strip_path))
                result.setdefault('boundary_strip_artifacts',[]).append(strip_artifact)
                if not all(bool(torch.isfinite(value).all()) for value in strip_arrays.values()):
                    result.setdefault('nonfinite_failures',[]).append(dict(context='boundary_strip',**strip_artifact))
                    raise FloatingPointError(f'Nonfinite boundary strip retained: code{number}, L{L}')
                secant=error(u1/h,slopes)
                strip=dict(code=number,intervals=L,boundary_max=float(abs(bval).max()),boundary_code_jacobian_max=float(abs(bj).max()),
                    first_strip_max=float(abs(u1).max()),second_strip_max=float(abs(u2).max()),
                    normal_secant_relative_error=secant,artifact=strip_path.name,artifact_sha256=strip_artifact['sha256'])
                strip['passed']=strip['boundary_max']<=1e-12 and strip['boundary_code_jacobian_max']<=1e-12 and secant<=.02
                result['boundary_strips'].append(strip)
            save(out/'result.json',result)
        # Tiny coarse-grid *identity* check only, never a replacement PDE solve.
        L=32;axis=torch.linspace(0,1,L+1,device='cuda');mesh=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        small=ScalarBurgers2D(decoder,latent,lambda p:torch.zeros((len(p),1),device='cuda'),structured_nodes(31,L,device='cuda'),intervals=L)
        z=codes[1];field=small.field(z,mesh).reshape(L+1,L+1)
        c=field[1:-1,1:-1];xm,xp=field[:-2,1:-1],field[2:,1:-1];ym,yp=field[1:-1,:-2],field[1:-1,2:]
        dense=.005*L**2*(xm+xp+ym+yp-4*c)-c*L*(torch.where(c>0,c-xm,xp-c)+torch.where(c>0,c-ym,yp-c))
        actual=small.getResidualSample(z)[:,0];identity=error(actual,dense.reshape(-1))
        small.q_sample=torch.full_like(small.q_sample,123.)
        target=small.explicit_target(z,1e-5)[:,0];target_error=error(target,c.reshape(-1)+1e-5*dense.reshape(-1))
        result['formulation_identity']=dict(intervals=32,role='Tiny algebraic fixture, not scientific/coarse PDE rollout',
            dense_stencil_relative_error=identity,explicit_target_refresh_error=target_error,passed=identity<=1e-12 and target_error<=1e-12)
        np.savez_compressed(out/'trained_probe_codes.npz',codes=codes.detach().cpu().numpy())
        result.update(status='complete_trained_online_qualification',probe_codes_sha256=digest(out/'trained_probe_codes.npz'),
            passed=all(v['passed'] for v in result['checks']) and all(v['passed'] for v in result['boundary_strips']) and result['formulation_identity']['passed'],
            remaining='All-case projection/physical scoring/actual4096 representation qualification before scientific PDE solve; native projection return is not convergence')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:save(out/'result.json',result)


if __name__=='__main__':main()
