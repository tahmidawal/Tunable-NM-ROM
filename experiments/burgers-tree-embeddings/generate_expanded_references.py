"""B8 ridge/two-pulse ladders and matched coordinate cache, separate by cohort."""
import argparse
from common import *
import data
from neural import expanded_cases as E, named_references as N
from references import SETTINGS, relative_difference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cohort', required=True, choices=['train', 'dev'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int)
    a = ap.parse_args()
    definitions = E.definitions(a.cohort, a.start, a.count)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = dict(cohort=a.cohort, start=a.start, count=len(definitions), nu=E.NU,
                definition_source_sha256=digest(HERE/'neural/expanded_cases.py'),
                driver_sha256=digest(Path(__file__)), definitions=definitions,
                settings=[list(x) for x in SETTINGS])
    metadata = preflight()
    rp = out/'result.json'
    if rp.exists():
        result = json.loads(rp.read_text())
        assert result['spec'] == spec, 'Refuse to reuse output for changed cohort/configuration/source'
        result.setdefault('resumptions', []).append(metadata)
    else:
        result = dict(meta=metadata, spec=spec, status='expanded_reference_ladders_running',
                      cohort=a.cohort, cases=[], final_test_opened=False,
                      role='training' if a.cohort == 'train' else 'held-out development, never training',
                      normalization='current-time reference physical L2',
                      independent_discretization_crosscheck='pending', admitted_reference=False)
    save(rp, result)
    for definition in definitions:
        label = definition['case']
        existing = next((row for row in result['cases'] if row['case'] == label), None)
        if existing and existing.get('complete', False):
            for level in existing['levels']:
                assert digest(out/level['file']) == level['sha256']
            continue
        row = existing or dict(definition, levels=[], complete=False)
        if existing is None:
            result['cases'].append(row)
        b0 = N.initial_coefficients(definition['definition'], 512)
        refined = N.initial_coefficients(definition['definition'], 512, 64)
        initial_change = float(jnp.linalg.norm(b0-refined)/jnp.linalg.norm(refined))
        assert initial_change < 1e-10, (label, initial_change)
        row['initial_quadrature_change'] = initial_change
        save(rp, result)
        levels = []
        for level_index, (n, dt) in enumerate(SETTINGS):
            if level_index < len(row['levels']):
                record = row['levels'][level_index]
                assert (record['N'], record['dt']) == (n, dt)
                path = out/record['file']; assert digest(path) == record['sha256']
                with np.load(path) as payload:
                    values = payload['coefficients']
            else:
                values, seconds = timer(lambda: N.trajectory(n, dt, E.NU)(b0[:n-1, :n-1]))
                values = np.asarray(values); assert np.isfinite(values).all()
                path = out/f'{label}_N{n}_dt{dt:.8f}.npz'
                # Per-level atomic output plus the manifest enables an interrupted job to resume.
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, coefficients=values, times=data.TIMES, nu=E.NU,
                                    case=label, cohort=a.cohort,
                                    definition_json=json.dumps(definition['definition'], sort_keys=True))
                tmp.replace(path)
                row['levels'].append(dict(N=n, dt=dt, file=path.name, sha256=digest(path),
                                          bytes=path.stat().st_size, seconds=seconds))
                save(rp, result)
                print('expanded_reference', label, n, dt, seconds, flush=True)
            assert values.shape == (11, n-1, n-1) and np.isfinite(values).all()
            levels.append(values)
        s0 = relative_difference(levels[0], levels[1])
        s1 = relative_difference(levels[1], levels[4])
        t0 = relative_difference(levels[2], levels[3])
        t1 = relative_difference(levels[3], levels[4])
        changes = []
        for n in [257, 513]:
            axis = jnp.linspace(0., 1., n)
            hi = data.synth_grid(jnp.asarray(levels[4]), axis)
            lo = data.synth_grid(jnp.asarray(levels[1]), axis)
            changes.append(np.asarray(jnp.linalg.norm((hi-lo).reshape(11, -1), axis=1)
                                      / jnp.linalg.norm(hi.reshape(11, -1), axis=1)))
        scoring = np.abs(changes[1]-changes[0]); allowance = s1+t1+scoring
        row.update(spatial_previous=s0, spatial_last=s1, temporal_previous=t0,
                   temporal_last=t1, scoring_change=scoring, allowance=allowance,
                   provisional_ladder_pass=bool(max(allowance) <= .001 and max(scoring) <= .0002),
                   independent_discretization_crosscheck='pending', admitted_reference=False,
                   complete=True)
        save(rp, result)
    # The cache contains exactly this chunk/cohort; reference coefficients remain retained.
    xy, w = data.grid(128, True); encoder_xy, _ = data.grid(16, True)
    fields, encoded, norms, labels = [], [], [], []
    for row in result['cases']:
        record = row['levels'][-1]; path = out/record['file']
        assert row['complete'] and digest(path) == record['sha256']
        with np.load(path) as payload:
            b = payload['coefficients']
        y = np.asarray(data.synth_grid(jnp.asarray(b), jnp.asarray(np.unique(xy[:, 0])))).reshape(11, -1)
        enc = np.asarray(data.synth_grid(jnp.asarray(b), jnp.asarray(np.unique(encoder_xy[:, 0])))).reshape(11, -1)
        norm = np.linalg.norm(b.reshape(11, -1), axis=1)/2
        discrepancy = float(np.max(np.abs(np.sqrt(np.sum(y*y*w, axis=1))/norm-1)))
        assert discrepancy <= .0002, (row['case'], 'coordinate grid underresolved', discrepancy)
        row['coordinate_norm_change'] = discrepancy
        fields.append(y); encoded.append(enc); norms.append(norm); labels.append(row['case'])
    cache = out/'coordinate_data.npz'; tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, xy=xy, w=w, encoder_xy=encoder_xy, y=np.concatenate(fields),
                        encoder=np.concatenate(encoded), norm=np.concatenate(norms),
                        case_ids=np.repeat(labels, 11), times=np.tile(data.TIMES, len(labels)),
                        cohort=a.cohort)
    tmp.replace(cache)
    result.update(status='complete_expanded_ladders_pending_independent_crosscheck',
                  all_ladders_pass=all(row['provisional_ladder_pass'] for row in result['cases']),
                  coordinate_data=dict(file=cache.name, sha256=digest(cache), labels=labels,
                                       snapshots=11*len(labels), quadrature=128, encoder_grid=16))
    save(rp, result)


if __name__ == '__main__':
    main()
