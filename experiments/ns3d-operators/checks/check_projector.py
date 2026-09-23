"""Local check: ops3d.Projector equals ns3d_fom.project_field (float64), and is idempotent."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / 'ns3d'))
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import torch
import ns3d_fom as F
import ops3d as O
out = {}
for n in (16, 32):
    rng = np.random.default_rng(0)
    u = rng.standard_normal((2, 3, n, n, n))
    ref = np.stack([np.asarray(F.project_field(jnp.asarray(x), F.geometry(n))) for x in u])
    P = O.Projector(n)
    got = P(torch.as_tensor(u, device='cuda')).cpu().numpy()
    twice = P(P(torch.as_tensor(u, device='cuda'))).cpu().numpy()
    truth = F.initial(n, F.parameters(202609201, 1)[0])
    keep = P(torch.as_tensor(truth[None], device='cuda')).cpu().numpy()[0]
    out[n] = dict(vs_ns3d_fom=float(np.abs(got - ref).max() / np.abs(ref).max()),
                  idempotent=float(np.abs(twice - got).max() / np.abs(got).max()),
                  truth_invariant=float(np.abs(keep - truth).max() / np.abs(truth).max()))
print(json.dumps(out))
assert all(v['vs_ns3d_fom'] < 1e-12 and v['idempotent'] < 1e-12 and v['truth_invariant'] < 1e-12 for v in out.values())
