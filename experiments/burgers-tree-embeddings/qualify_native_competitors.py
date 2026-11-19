"""Bounded native-source/environment qualification, without pretrained/data downloads."""
import argparse,urllib.request,subprocess,tarfile,importlib.util,sys,signal
from common import *

SOURCES={'crom_offline':('dc3042/CROM_offline_training','39333f76ccf878dacc9ae425c185b76ed4e72c5c'),
         'crom_online':('peterchencyc/CROM_online_deployment','ccb8a6603318492442e32f9d254d420b63c86419'),
         'snf':('CMU-CBML/NeuralROMs.jl','35c55e44cdcc7eb902c6c0b6311fb6c524271f91')}


def fetch(url):
    with urllib.request.urlopen(url,timeout=45) as f:return f.read()


def source(out,key,repo,rev):
    root=out/'sources'/key;root.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1'))
    records=[]
    for item in tree['tree']:
        name=item['path'];p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        if item['type']!='blob':continue
        if not (p.suffix in ('.py','.jl','.toml') or p.name.lower().startswith(('readme','license'))):continue
        content=fetch(f'https://raw.githubusercontent.com/{repo}/{rev}/{name}')
        q=root/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(content)
        records.append(dict(path=name,sha256=digest(q)))
    return dict(repo=repo,commit=rev,root=str(root),files=records,
                supplied_weights_or_solution_data_downloaded=False)


def command(args,out,env=None,timeout=120):
    with open(out,'w') as f:
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            p.wait(timeout=timeout)
            return dict(returncode=p.returncode,log=Path(out).name)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
            return dict(returncode=None,timed_out=True,log=Path(out).name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    result=dict(meta=preflight(),status='native_dependency_qualification',sources={},final_test_opened=False)
    for key,(repo,rev) in SOURCES.items():
        try:result['sources'][key]=source(out,key,repo,rev)
        except Exception as e:result['sources'][key]=dict(error=repr(e))
        save(out/'result.json',result)
    # Execute the authors' unchanged online decoder Jacobian and GN solver on a
    # deterministic analytic fixture, with our double-precision problem adapter.
    import torch
    torch.set_default_dtype(torch.float64);assert torch.cuda.is_available()
    result['torch']=dict(version=torch.__version__,device=torch.cuda.get_device_name(),dtype='float64')
    try:
        root=Path(result['sources']['crom_online']['root'])/'online'
        def load(name):
            spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        class Net(torch.nn.Module):
            def forward(self,x):return (x[:,0]+x[:,1]*x[:,2]+.1*x[:,0]*x[:,1]*x[:,2]**2)[:,None]
        dec=load('Decoder').Decoder(Net().cuda().double(),None,None)
        x=torch.linspace(-1,1,64,device='cuda');truth=torch.tensor([.3,-.2],device='cuda').reshape(1,1,2)
        class Problem:
            def inp(self,z):return torch.cat([z.reshape(1,2).expand(len(x),2),x[:,None]],1)
            def updateStateSample(self,z,decoder,sample):return decoder.forward(self.inp(z)).reshape(1,-1,1)
            def getJacobianSample(self,z,decoder,sample):return decoder.jacobianPart(self.inp(z),2,'fir').reshape(-1,2)
        problem=Problem();J=problem.getJacobianSample(truth,dec,None)
        expected=torch.stack([1+.1*truth.flatten()[1]*x*x,x+.1*truth.flatten()[0]*x*x],1)
        solver=load('NonlinearSolver').NonlinearSolver(problem,dec,1e-12,1e-14)
        fitted=solver.solve(torch.zeros_like(truth),problem.updateStateSample(truth,dec,None),None,max_iters=50,print_info=False)
        err=float(torch.max(abs(fitted-truth)));jac=float(torch.max(abs(J-expected)))
        assert err<1e-9 and jac<1e-12
        result['crom_native_online_fixture']=dict(passed=True,code_error=err,jacobian_error=jac,
            scope='Native GN and decoder Jacobian only. Author Burgers example is 1D forced inviscid; a documented 2D viscous adapter is still required.')
    except Exception as e:result['crom_native_online_fixture']=dict(passed=False,error=repr(e))
    offline=out/'sources/crom_offline'
    env=dict(os.environ,PYTHONPATH=str(offline))
    result['crom_offline_import']=command([sys.executable,'-c','import cromnet; print("native_offline_import_pass")'],out/'crom_offline_import.log',env)
    save(out/'result.json',result)
    try:
        # Official Julia binary, verified against the official version manifest.
        version='1.10.10';manifest=fetch('https://julialang-s3.julialang.org/bin/versions.json')
        (out/'julia-versions.json').write_bytes(manifest)
        files=json.loads(manifest)[version]['files']
        selected=next(v for v in files if v['triplet']=='x86_64-linux-gnu' and v['extension']=='tar.gz')
        archive=out/'julia.tar.gz'
        with urllib.request.urlopen(selected['url'],timeout=120) as r,open(archive,'wb') as f:
            import shutil
            shutil.copyfileobj(r,f)
        assert digest(archive)==selected['sha256']
        with tarfile.open(archive) as tf:tf.extractall(out/'runtime',filter='data')
        binary=out/'runtime'/('julia-'+version)/'bin/julia'
        snf=out/'sources/snf';depot=out/'julia-depot';depot.mkdir(exist_ok=True)
        env=dict(os.environ,JULIA_DEPOT_PATH=str(depot),JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_NUM_THREADS='4',GKSwstype='100')
        result['julia']=dict(version=version,binary_sha256=digest(binary),archive_sha256=digest(archive),
            original_manifest_present=(snf/'Manifest.toml').exists())
        result['snf_instantiate']=command([str(binary),'--startup-file=no','--project='+str(snf),
            '-e','using Pkg; Pkg.instantiate()'],out/'snf_instantiate.log',env,timeout=900)
        if result['snf_instantiate'].get('returncode')==0:
            result['snf_import']=command([str(binary),'--startup-file=no','--project='+str(snf),
                '-e','using NeuralROMs, CUDA; @assert CUDA.functional(); println("native_snf_gpu_import_pass")'],out/'snf_import.log',env,timeout=300)
        if (snf/'Manifest.toml').exists():result['resolved_manifest_sha256']=digest(snf/'Manifest.toml')
    except Exception as e:result['snf_environment_error']=repr(e)
    result.update(status='complete_native_environment_diagnostic',trained_comparators=False,
                  eligible_speed_comparison=False)
    save(out/'result.json',result)


if __name__=='__main__':main()
