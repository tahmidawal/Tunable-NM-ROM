"""Summarize immutable returned measurements; never select validation settings."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = HERE / 'results/val01/output'
    data = json.loads((root / 'EXPORT.json').read_text())
    assert data['phase'] == 'val'
    rows = data['cases']
    assert len(rows) == 93 * 64 * 2 * 2
    assert all(r['status'] == 'complete' for r in rows)
    tables, comparisons = [], []
    for lc in (16, 32):
        for source in ('A', 'B'):
            panel = [r for r in rows if r['coarse_intervals'] == lc and r['source'] == source]
            by_config = {}
            for r in panel:
                by_config.setdefault(r['config'], []).append(r)
            selected = {r['descriptor']['category']: r['config'] for r in panel if r['selected']}
            names = list(dict.fromkeys(['linear', 'cubic_notaknot', 'sine_full', *selected.values()]))
            for name in names:
                rr = sorted(by_config[name], key=lambda r: r['case'])
                assert [r['case'] for r in rr] == list(range(64))
                vals = np.array([r['error'] for r in rr])
                times = {key: float(np.median([r['timing_median'][key] for r in rr]))
                         for key in rr[0]['timing_median']}
                tables.append(dict(coarse_intervals=lc, source=source, config=name,
                    selected=rr[0]['selected'], median=float(np.median(vals)),
                    p90=float(np.quantile(vals, .9)), worst=float(vals.max()),
                    timing_median_of_case_medians_seconds=times,
                    max_input_sensitivity=max(r['input_sensitivity_max'] for r in rr),
                    max_physical_amplification=max(r['physical_amplification'] for r in rr)))
            bank = sorted(by_config[selected['bank']], key=lambda r: r['case'])
            for other in names:
                if other == selected['bank']:
                    continue
                base = sorted(by_config[other], key=lambda r: r['case'])
                be = np.array([r['error'] for r in bank]); oe = np.array([r['error'] for r in base])
                eps = np.array([r['reference_eps'] for r in bank])
                threshold = np.full(64, 1e-14) if source == 'A' else 2 * eps
                delta = oe - be
                ratio = be / oe
                wins = int(np.count_nonzero(delta > threshold)); losses = int(np.count_nonzero(delta < -threshold))
                comparisons.append(dict(coarse_intervals=lc, source=source, bank=selected['bank'], baseline=other,
                    baseline_is_selected_ordinary=other == selected['ordinary'],
                    cases=64, ratio_of_cohort_medians=float(np.median(be) / np.median(oe)),
                    median_paired_ratio=float(np.median(ratio)), p90_paired_ratio=float(np.quantile(ratio, .9)),
                    worst_paired_ratio=float(ratio.max()), wins=wins, losses=losses, ties=64-wins-losses,
                    classification='fixed-array numerical' if source == 'A' else 'conditional two-epsilon full-field',
                    worst_error_worsening=float(be.max() - oe.max()),
                    worst_error_resolved_worsening=bool(be.max() > oe.max() + threshold.max()),
                    median_paired_device_cost_ratio=float(np.median([
                        b['timing_median']['device_query_seconds'] / o['timing_median']['device_query_seconds']
                        for b, o in zip(bank, base)]))))
    primary = next(c for c in comparisons if c['coarse_intervals'] == 16 and c['source'] == 'B'
                   and c['baseline_is_selected_ordinary'])
    gate = dict(ratio_at_most_point8=primary['ratio_of_cohort_medians'] <= .8,
                resolved_win_fraction_at_least_point75=primary['wins'] / primary['cases'] >= .75,
                no_resolved_worst_worsening=not primary['worst_error_resolved_worsening'])
    ledger = []
    for attempt in ('smk01', 'prep01', 'dev02', 'val01'):
        lines = (HERE / 'results' / attempt / 'SACCT.txt').read_text().splitlines()
        master = next(line.split('|') for line in lines if '.' not in line.split('|')[0])
        if master[0].isdigit() and master[1] == 'COMPLETED':
            job, state, exitcode, seconds = master[:4]
        else:
            raise ValueError(f'Unexpected accounting schema: {attempt}: {master}')
        assert state == 'COMPLETED' and exitcode == '0:0'
        ledger.append(dict(attempt=attempt, job_id=int(job), gpu_seconds=int(seconds)))
    result = dict(export_sha256=digest(root / 'EXPORT.json'),
                  frozen_selection_sha256=digest(root / 'selection.json'),
                  measurements=tables, bank_comparisons=comparisons,
                  primary_screen_gate=gate, primary_screen_gate_pass=all(gate.values()),
                  gpu_ledger=ledger, gpu_seconds=sum(r['gpu_seconds'] for r in ledger),
                  scope='Frozen dev choices only. A is fixed-array reconstruction; B uses conditional reference resolution. No validation selection, dynamics or sealed cases. Timings are within one H200 allocation, warm per field, with setup excluded and reported separately in plans/setup.json.')
    out = HERE / 'results/SCREEN_SUMMARY.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(out)


if __name__ == '__main__':
    main()
