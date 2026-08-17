"""Exact ||u_T||/||u_0|| for the 16 TEST_SEED heat trajectories, computed in the
discrete sine basis (the FOM's backward-Euler operator is diagonal there), so the
decay quoted in README.md is reproducible without a GPU.

  /home/tahmid/Dev/.venv/bin/python tools/heat_decay.py
"""
import numpy as np

n, dt, T, m = 64, 0.005, 50, 16
dx = 1.0 / (n - 1)
kk = np.arange(1, n - 1)
lam1 = (4.0 / dx ** 2) * np.sin(np.pi * kk / (2 * (n - 1))) ** 2
LAM = lam1[:, None] + lam1[None, :]
x = np.linspace(0, 1, n)[1:-1]
S = np.sin(np.pi * np.outer(x, kk))
S = S / np.linalg.norm(S, axis=0, keepdims=True)
rng = np.random.default_rng(1)                      # TEST_SEED = SEED + 1 = 1
cx, cy = rng.uniform(.15, .85, m), rng.uniform(.15, .85, m)
w, a = rng.uniform(.05, .20, m), rng.uniform(1, 10, m)
kap = np.exp(rng.uniform(np.log(.01), np.log(.5), m))
X, Y = np.meshgrid(x, x, indexing="ij")
rat = []
for i in range(m):
    C = S.T @ (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2) / (2 * w[i] ** 2))) @ S
    g = (1.0 + dt * kap[i] * LAM) ** -1
    rat.append(np.linalg.norm(C * g ** T) / np.linalg.norm(C))
rat = np.array(rat)
print("||u_50||/||u_0||  min %.2e  median %.2e  max %.2e  (worst-case decay %.0fx)"
      % (rat.min(), np.median(rat), rat.max(), 1.0 / rat.min()))
