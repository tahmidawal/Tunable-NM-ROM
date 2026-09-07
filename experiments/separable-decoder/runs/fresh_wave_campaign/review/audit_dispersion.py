"""Independent closed-form sine audit of saved reflective verification fields.

Imports no wave implementation and generates no new PDE trajectory. Evaluates
exact linear modal propagation from already saved initial samples to distinguish
RK4 temporal error from spatial-frequency error. Continuum-frequency propagation
of a sampled interpolant is a diagnostic, not certified continuum truth.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.fft import dstn

HERE = Path(__file__).resolve().parent
RAW = HERE.parent/'verify01/cluster/out/verification'
r = json.loads((RAW/'result.json').read_text())
arrays = np.load(RAW/'reference_arrays.npz')
rows = []
for pidx in (0, 1):
    for n in (32, 64, 128):
        source = next(x for x in r['actual_family'] if x.get('bc')=='dirichlet' and x.get('parameter_index')==pidx and x.get('n')==n and 'parameters' in x)
        c = source['parameters'][5]
        prefix = f'dirichlet_{pidx}_{n}'
        uu, vv = arrays[prefix+'_u'], arrays[prefix+'_v']
        u0, v0, ut, vt = [dstn(q, type=1, norm='ortho') for q in (uu[0], vv[0], uu[-1], vv[-1])]
        modes = np.arange(1, n)
        discrete_one = 4*n*n*np.sin(np.pi*modes/(2*n))**2
        continuum_one = (np.pi*modes)**2
        wd = c*np.sqrt(discrete_one[:, None]+discrete_one[None, :])
        wc = c*np.sqrt(continuum_one[:, None]+continuum_one[None, :])
        denominator = np.sum(v0*v0+(wd*u0)**2)
        row = dict(parameter_index=pidx, n_intervals=n, time=2.4)
        for name, omega in [('semidiscrete', wd), ('continuum_frequency_interpolant', wc)]:
            co, si = np.cos(omega*2.4), np.sin(omega*2.4)
            ur = u0*co+v0*si/omega
            vr = -omega*u0*si+v0*co
            row[name+'_relative_energy_error'] = float(np.sqrt(np.sum((vt-vr)**2+(wd*(ut-ur))**2)/denominator))
        rows.append(row)
result = dict(source=str(RAW/'reference_arrays.npz'), source_sha256=hashlib.sha256((RAW/'reference_arrays.npz').read_bytes()).hexdigest(),
              all_saved_arrays_finite=all(bool(np.all(np.isfinite(arrays[k]))) for k in arrays.files), rows=rows,
              interpretation='Exact semidiscrete sine evolution nearly agrees with saved RK4 fields. The much larger continuum-frequency difference isolates spatial dispersion; sampled continuum interpolation is not a mesh-converged truth certificate.')
(HERE/'dispersion-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows:
    print(row)
