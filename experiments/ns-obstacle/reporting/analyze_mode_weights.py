"""Read-only arithmetic on retained modal training metrics, not a flow solve."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for key in ('config', 'replay', 'output'):
    p.add_argument('--'+key, type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
r = json.loads(a.replay.read_text())
s = np.asarray(c['singular_values'])
w = np.asarray(c['mode_weights'])
nv = np.asarray(c['value_norms_squared'])
ng = np.asarray(c['gradient_norms_squared'])
ev = np.asarray(r['metrics']['value_relative_squared_by_mode'])
eg = np.asarray(r['metrics']['gradient_relative_squared_by_mode'])
loss = w*(ev+eg)
np.testing.assert_allclose(loss.sum(), r['objective'], rtol=1e-12)
def shares(x):
    return (x/x.sum()).tolist()
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
out = {
    'config_sha256': sha(a.config), 'replay_sha256': sha(a.replay),
    'generator_sha256': sha(Path(__file__)), 'singular_values': s.tolist(),
    'mode_weights': w.tolist(), 'modal_objective_share': shares(loss),
    'gradient_fraction_of_objective': float(np.dot(w, eg)/loss.sum()),
    'retained_correction_value_energy_share': shares(s*s*nv),
    'retained_correction_gradient_energy_share': shares(s*s*ng),
    'fixed_POD_coefficient_value_error_energy_share': shares(s*s*nv*ev),
    'fixed_POD_coefficient_gradient_error_energy_share': shares(s*s*ng*eg),
    'fixed_POD_coefficient_retained_correction_relative_value_error': float(np.sqrt(np.dot(s*s*nv, ev)/np.sum(s*s*nv))),
    'fixed_POD_coefficient_retained_correction_relative_gradient_error': float(np.sqrt(np.dot(s*s*ng, eg)/np.sum(s*s*ng))),
    'relative_to_mode1_weight_amplification_against_sigma_squared': ((w/w[0])/(s*s/s[0]**2)).tolist(),
    'limitations': [
        'Original POD coefficient row orthogonality cancels cross-mode terms only over the full training ensemble.',
        'These fixed-coefficient retained-four-mode correction calculations exclude the lift and truncated modes.',
        'They are not actual solved fields, best-fit projection errors, individual Reynolds cases, or development/validation errors.',
        'Loss allocation may create optimization competition in a shared network, but this arithmetic does not prove causal harm.',
        'No current candidate, objective, data, or model was changed.'
    ]
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k: out[k] for k in ('modal_objective_share', 'retained_correction_value_energy_share')}, indent=2))
