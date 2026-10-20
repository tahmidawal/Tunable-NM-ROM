import sys, time, numpy as np
sys.path.insert(0, '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines')
import nnls as NN
z = np.load('G.npz'); G, b = z['G'], z['b']
t = time.time()
o = NN.greedy_nnls(G, b, supports=(32, 64, 128), cap=128, max_iter=3000, frac=0.0,
                   log=lambda m: print(m, '%.0fs' % (time.time() - t), flush=True))
print(o['reason'], 'banned', o['banned'], 'its', len(o['path']), {k[1]: (v and (v['size'], round(v['relres'], 6))) for k, v in o['records'].items()})
