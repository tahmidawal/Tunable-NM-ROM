import sys, time, numpy as np, scipy.linalg as sla
sys.path.insert(0, '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines')
import subprocess, importlib.util
# pinned pre-A17 implementation (lane commit c57098e00, the algorithm that produced old_algorithm_t.log)
src = subprocess.check_output(['git', 'show', 'c57098e00:experiments/jcp-hr-baselines/nnls.py'])
open('nnls_preA17.py', 'wb').write(src)
spec = importlib.util.spec_from_file_location('nnls_preA17', 'nnls_preA17.py')
NN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN)
z = np.load('G.npz'); G, b = z['G'], z['b']
cn = np.linalg.norm(G, axis=0); print('col norm range', cn.min(), cn.max(), (cn == 0).sum())
for name, GG in (('raw', G), ('colnorm', G / np.where(cn > 0, cn, 1.))):
    t = time.time()
    o = NN.greedy_nnls(GG, b, supports=(32, 64, 128, 256), cap=256, max_iter=8000)
    print(name, o['reason'], 'banned', o['banned'], 'its', len(o['path']), 'time %.1f' % (time.time() - t),
          {k[1]: (v and (v['size'], round(v['relres'], 5))) for k, v in o['records'].items()})
