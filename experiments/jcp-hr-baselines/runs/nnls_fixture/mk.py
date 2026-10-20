import sys, time, numpy as np
sys.path.insert(0, '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines')
import hrcore as H, pipeline as PL, build as BD
d, L = 2, 256
cell = PL.Cell(d, L, log=print)
fam = H.Family(2, 'train')
S, Phi, sig, trust = cell.build_basis(fam, list(range(24)), 48, [16, 48])
cell.set_basis(Phi, trust)
cell.ecsw_states(fam)
G, b, cand, info = BD.ecsw_contrib(cell.states, cell.Phi[:, :16], d, L)
np.savez('G.npz', G=np.asarray(G), b=np.asarray(b), cand=cand)
print(G.shape)
