"""Screen one frozen zero-code tangent decoder; no retraining or PDE claims."""
import argparse
from common import *
from training_io import read_checkpoint, checkpoint, summary
from build_local_cache import cohort
import data, geometry as geo, local_polynomial as LP, tensor_metric as TM
import shared_child_head as H, linearized_child as LC


def main():
    ap = argparse.ArgumentParser()
    for key in ['out', 'model', 'cache', 'growth', 'dev']:
        ap.add_argument('--'+key, required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = preflight()
    mp, cp, gp = Path(a.model), Path(a.cache), Path(a.growth)
    mr = json.loads((mp/'result.json').read_text())
    gr = json.loads((gp/'result.json').read_text())
    cr = json.loads((cp/'result.json').read_text())
    assert mr['status'] == 'complete_shared_child_representation'
    assert gr['status'] == 'complete_frozen_child_growth' and gr['representation_gate']
    assert digest(mp/'shared_child.pkl') == mr['model_sha256'] == gr['model_sha256']
    assert digest(gp/'growth_fits.pkl') == gr['fit_sha256']
    model = read_checkpoint(mp/'shared_child.pkl')
    assert digest(cp/'result.json') == model['cache_sha256']
    assert digest(cp/'global_basis.npz') == gr['basis_sha256']
    basis = jnp.asarray(np.load(cp/'global_basis.npz')['coefficients'])
    old = {r['case']:r for r in read_checkpoint(gp/'growth_fits.pkl')}
    refs = {r['case']:(p,r) for p,r in cohort(a.dev)}
    assert len(cr['cases']['dev']) == 44 and model['latent'] == 16 and len(basis) == 128
    metric, rule = TM.make_geometry(6), TM.rule()
    x, w = data.grid(128, True)
    saved, errors = [], []
    result = dict(meta=meta, status='frozen_tangent_screen_running', nodes=64, latent=16,
                  active_variables=1152, decoder='frozen zero-code tangent; direct coefficients',
                  head_weights_changed=False, model_sha256=mr['model_sha256'],
                  basis_sha256=gr['basis_sha256'], growth_sha256=digest(gp/'result.json'),
                  cache_sha256=digest(cp/'result.json'),
                  allocation='same initial-callable swept64 prescription as frozen nonlinear growth',
                  reference_assisted_projection=True, PDE_solve=False, final_test_opened=False,
                  stationarity_tolerance=1e-8, eigenvalue_relative_floor=1e-10, cases=[])
    for row in cr['cases']['dev']:
        label = row['case']; folder, ref = refs[label]
        initial = lambda xy:data.initial(xy, ref['parameters'])
        cells = LP.callable_tree(x, w, initial, 64, 4, 'swept')
        geo.valid(cells)
        assert np.array_equal(cells, old[label]['cells'])
        gram, tr, mx, my = metric(basis, jnp.asarray(cells), *rule)
        assert np.allclose(tr, old[label]['transforms'], rtol=0, atol=1e-10)
        mapping = LC.zero_map(model['params']['head'], jnp.asarray(model['scale']), cells)
        # The frozen policy and derivative map are complete before opening truth.
        refpath = folder/ref['levels'][-1]['file']
        assert digest(refpath) == ref['levels'][-1]['sha256']
        bfield = jnp.asarray(np.load(refpath)['coefficients'])
        norm2 = jnp.sum(bfield*bfield, axis=(1,2))*.25
        rhs = TM.rhs(basis, bfield, mx, my)
        fit = LC.fit(gram, rhs, norm2, mapping)
        finite = bool(fit['finite']) and bool(jnp.all(jnp.isfinite(fit['error'])))
        ev = np.asarray(fit['normalized_gram_eigenvalues'])
        S, _, _, _, _ = H.schur(gram, rhs, norm2)
        g, h = H.geometry(cells)
        zero = H.zero_audit(model['params']['head'], jnp.asarray(model['scale']), g, h, S)
        station = np.asarray(fit['stationarity'])
        passed_fit = finite and np.all(np.isfinite(station)) and np.max(station) <= 1e-8
        directions = zero['independent_directions'] == 1024 and zero['zero_child_max'] < 1e-13
        conditioning = np.all(np.isfinite(ev)) and ev[0] > ev[-1]*1e-10
        e = np.asarray(fit['error'])
        errors.append(e)
        saved.append(dict(case=label, cells=cells, transforms=np.asarray(tr),
                          child_map=np.asarray(mapping), coefficients=np.asarray(fit['coefficients']),
                          codes=np.asarray(fit['codes'])))
        result['cases'].append(dict(case=label, error=e, stationarity=station,
            normalized_gram_eigenvalues=ev, finite=finite, fitted=bool(passed_fit),
            zero_jacobian=zero, conditioning_pass=bool(conditioning),
            useful_independent_directions=bool(directions),
            reference_sha256=ref['levels'][-1]['sha256']))
        save(out/'result.json', result)
        print('tangent_projection', label, float(np.max(e)), float(np.max(station)), bool(conditioning), flush=True)
    result['projection'] = summary(np.stack(errors))
    result['fit_sha256'] = checkpoint(out/'tangent_fits.pkl', saved)
    ii, tt = np.unravel_index(np.nanargmax(np.asarray(errors)), np.asarray(errors).shape)
    witness = saved[ii]; folder, ref = refs[witness['case']]
    truthcoef = jnp.asarray(np.load(folder/ref['levels'][-1]['file'])['coefficients'][tt])
    axis = np.linspace(0,1,4096)
    pred = LP.field_grid(basis, jnp.asarray(witness['coefficients'][tt]), witness['cells'], witness['transforms'], 6, axis)[0]
    truth = data.synth_grid(truthcoef, axis)
    result['actual4096_witness'] = dict(case=witness['case'], time=float(data.TIMES[tt]),
        error=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)), shape=[4096,4096])
    np.savez_compressed(out/'actual4096.npz', prediction=np.asarray(pred), reference=np.asarray(truth), case=witness['case'], time=data.TIMES[tt])
    fit_gate = all(r['fitted'] and r['conditioning_pass'] and r['useful_independent_directions'] for r in result['cases'])
    dense = []
    if fit_gate and max(np.max(errors), result['actual4096_witness']['error']) <= .005:
        axis = np.linspace(0,1,2048)
        for v in saved:
            folder, ref = refs[v['case']]
            bb = jnp.asarray(np.load(folder/ref['levels'][-1]['file'])['coefficients'])
            ee = []
            for t in range(11):
                pred = LP.field_grid(basis,jnp.asarray(v['coefficients'][t]),v['cells'],v['transforms'],6,axis)[0]
                truth = data.synth_grid(bb[t],axis)
                ee.append(float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)))
            dense.append(ee)
        result['dense2048'] = summary(np.asarray(dense))
        result['max_scoring_change'] = float(np.max(np.abs(np.asarray(dense)-np.asarray(errors))))
    result['representation_gate'] = bool(dense) and fit_gate and max(np.max(dense),result['actual4096_witness']['error']) <= .005 and result['max_scoring_change'] <= .0002
    result['decision'] = ('Admit this separate direct tangent representation to a later bounded PDE diagnostic'
                          if result['representation_gate'] else 'Stop this tangent ablation; retain the nonlinear representation and its scoped PDE solver failure')
    result['status'] = 'complete_frozen_tangent_screen'
    save(out/'result.json',result)


if __name__ == '__main__':
    main()
