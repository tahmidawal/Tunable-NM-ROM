"""Small CPU fixture for B8 split isolation, deterministic chunks and boundary fields."""
import json
import numpy as np
from neural import expanded_cases as E, named_references as N
from common import jnp


def main():
    train = E.definitions('train'); dev = E.definitions('dev')
    assert len(train) == 32 and len(dev) == 8
    assert E.definitions('train', 8, 8) == train[8:16]
    assert E.definitions('dev') == dev
    assert {row['family'] for row in train} == {'ridge', 'two_pulse'}
    assert sum(row['family'] == 'ridge' for row in train) == 16
    assert sum(row['family'] == 'ridge' for row in dev) == 4
    # Exclude human-readable names: compare actual field definitions across splits.
    signature = lambda r: json.dumps(r['definition']['gaussians'], sort_keys=True)
    assert not ({signature(r) for r in train} & {signature(r) for r in dev})
    assert not ({r['case'] for r in train} & {r['case'] for r in dev})
    try:
        E.definitions('final_test')
    except PermissionError:
        pass
    else:
        raise AssertionError('Sealed final cohort was exposed')
    for args in [('train', -1, 2), ('dev', 7, 2), ('train', 0, 0)]:
        try:
            E.definitions(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid chunk accepted')
    axis = np.linspace(0, 1, 17)
    boundary = np.concatenate([np.column_stack([axis, np.zeros_like(axis)]),
                               np.column_stack([axis, np.ones_like(axis)]),
                               np.column_stack([np.zeros_like(axis), axis]),
                               np.column_stack([np.ones_like(axis), axis])])
    max_boundary = 0.
    for row in train+dev:
        value = np.asarray(N.initial(jnp.asarray(boundary), row['definition']))
        max_boundary = max(max_boundary, float(np.max(np.abs(value))))
        assert value.dtype == np.float64 and max_boundary < 1e-28
        assert row['nu'] == .005 and row['definition']['background'] == .15
        g = row['definition']['gaussians']
        if row['family'] == 'ridge':
            assert len(g) == 1 and .18 <= g[0]['sigma_t'] <= .28
            assert .025 <= g[0]['sigma_n'] <= .04 and -60 <= g[0]['angle_degrees'] <= -30
        else:
            assert len(g) == 2 and all(x['sigma_t'] == x['sigma_n'] for x in g)
    print(json.dumps(dict(status='passed',train_cases=len(train),dev_cases=len(dev),
                          final_test_guarded=True,max_boundary=max_boundary,
                          independent_seeds=E.SEEDS)))


if __name__ == '__main__':
    main()
