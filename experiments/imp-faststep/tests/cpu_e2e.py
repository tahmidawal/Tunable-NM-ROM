"""CPU end-to-end development run (not a verdict) at L = 256: a small POD fixture (8 training trajectories, ranks
32/64), an LSPG-ECSW native artifact (r = 32, k8), the ecsw job, then fsrun on 3 cases with every block enabled."""
import json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.append(str(HERE.parent / 'jcp-hr-baselines'))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import hrcore as H, build as BD
W = Path(os.environ.get('WORK', '/tmp/imp-faststep-work/e2e'))
tdir = W / 'tune' ; (tdir / 'artifacts').mkdir(parents=True, exist_ok=True)
L = 256
t0 = time.time()
if not (tdir / 'artifacts' / 'Phi.npz').exists():
    fam = H.Family(2, 'train')
    S, info = BD.snapshots(2, L, fam, list(range(8)))
    Phi, sig, *_ = BD.pod(S, 64)
    Phi = np.asarray(Phi)
    C = S @ Phi
    trust = BD.trust_radii(C, [32, 64], 2)
    np.savez(tdir / 'artifacts' / 'Phi.npz', Phi=Phi, ranks=np.array([32, 64]), trust=np.array([trust[32], trust[64]]),
             constructible=np.array([32, 64]))
    states, _ = BD.ecsw_states(2, L, fam, None)
    G, b, cand, _ = BD.ecsw_contrib(states, jnp.asarray(Phi[:, :32]), 2, L)
    rec, _ = BD.ecsw_weights(G, b, cand, 32, BD.ECSW_KAPPA_PATH, max_seconds=60)
    np.savez(tdir / 'artifacts' / 'HR~ecsw~32~k8.npz', nodes=rec[8]['nodes'], weights=rec[8]['weights'])
    print('fixture', round(time.time() - t0, 1), flush=True)
PY = sys.executable
cfg_e = dict(mesh=L, tune_dir=str(tdir), ranks=[32, 64], allow_cpu_smoke=True)
(W / 'ecsw.json').write_text(json.dumps(cfg_e))
if not (W / 'ecsw_out' / 'COMPLETE').exists():
    subprocess.run([PY, str(HERE / 'ecswjob.py'), '--config', str(W / 'ecsw.json'), '--out', str(W / 'ecsw_out')], check=True)
print('ecsw', round(time.time() - t0, 1), flush=True)
import hashlib
shaf = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
cfg = dict(attempt='cpu', mesh=L, e0_store_version='3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0',
           e0_metric_id='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1', edges=[0, 128], cases=[['dev6', 0], ['dev6', 1], ['val32', 0]], allow_cpu_smoke=True,
           ranks=[dict(Rp=128, rule='fib4181', qp='fib6765')], arms=['L-CNR', 'G-CN', 'G-E4a', 'G-E4b', 'G-R1'],
           controls=os.environ.get('CONTROLS', '1') == '1', order=True,
           pod=dict(tune_dir=str(tdir), gecsw=str(W / 'ecsw_out' / 'GECSW_r{r}.npz'), ranks=[32, 64], kappas=[2, 4, 8, 16],
                    dts=[.0125, .025], native=['ecsw|32|k8|CNR|0.005|common|central'],
                    phi_sha256=shaf(tdir / 'artifacts' / 'Phi.npz'),
                    native_sha256={'HR~ecsw~32~k8.npz': shaf(tdir / 'artifacts' / 'HR~ecsw~32~k8.npz')}),
           timing=dict(reps=1, burn=.0, seed=1, anchor='L-CNR|128'))
(W / 'fs.json').write_text(json.dumps(cfg))
import shutil
shutil.rmtree(W / 'fs_out', ignore_errors=True)
subprocess.run([PY, str(HERE / 'fsrun.py'), '--config', str(W / 'fs.json'), '--out', str(W / 'fs_out')], check=True)
print('done', round(time.time() - t0, 1))
