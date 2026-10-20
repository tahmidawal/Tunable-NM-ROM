"""Timing-panel validity shared by c2grid.py and c3run.py (DESIGN A2.11, A4.4 as amended by A5.3; code audit 2 item 5).

A panel is valid iff (i) every invocation's A and B output hashes are non-missing and equal to the accuracy phase, and
(ii) for EVERY subject the median over its invocations of t_A2 / t_A1 lies in [1/1.1, 1.1] (subject-level drift, as the
inherited 3D panel gates its worst subject), (iii) every paired ratio is finite and positive. The worst subject drift,
the spread and the per-subject medians are reported. `python timing.py` runs the injected-fault self-test.
"""
from __future__ import annotations

import numpy as np

BAND = (1 / 1.1, 1.1)


def panel_validity(inv):
    by = {}
    for x in inv:
        by.setdefault(x['B'], []).append(x['drift'])
    sub = {k: float(np.median(v)) for k, v in by.items()}
    worst = max((max(v, 1 / v) for v in sub.values()), default=float('nan'))
    hok = bool(inv) and all(x.get('hashes_ok') for x in inv)
    fin = bool(inv) and all(np.isfinite(x['ratio']) and x['ratio'] > 0 for x in inv)
    dok = bool(sub) and all(BAND[0] <= v <= BAND[1] for v in sub.values())
    return dict(hashes_ok=hok, ratios_finite=fin, subject_drift=sub, worst_subject_drift=worst,
                drift_q05=float(np.quantile([x['drift'] for x in inv], .05)) if inv else None,
                drift_q95=float(np.quantile([x['drift'] for x in inv], .95)) if inv else None,
                valid=bool(hok and fin and dok))


def _selftest():
    rng = np.random.default_rng(0)
    base = [dict(B=f's{j}', drift=float(np.exp(rng.normal(0, .02))), ratio=1., hashes_ok=True) for j in range(20) for _ in range(6)]
    assert panel_validity(base)['valid']
    bad = [dict(x, drift=x['drift'] * 1.2) if x['B'] == 's3' else x for x in base]      # one subject drifts by 20 %
    assert not panel_validity(bad)['valid'], 'injected subject drift not detected'
    pooled_median = float(np.median([x['drift'] for x in bad]))
    assert BAND[0] <= pooled_median <= BAND[1], 'the pooled median would have hidden the fault (demonstration)'
    hb = [dict(x, hashes_ok=False) if i == 7 else x for i, x in enumerate(base)]
    assert not panel_validity(hb)['valid']
    print('timing self-test passed: injected 20 % subject drift detected; pooled median', round(pooled_median, 4), 'would not')


if __name__ == '__main__':
    _selftest()
