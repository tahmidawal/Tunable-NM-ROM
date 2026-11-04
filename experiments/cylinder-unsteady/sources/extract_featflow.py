"""Extract DFG 2D-2 (Re=100) comparison values from the Featflow 2014 official logs with cycles.py.

Run with the local venv by absolute path; reads only the extracted zip under
featflow_2014_official/raw and writes dfg2d2_reference.json next to this file.
Columns: bdforces -> (step, t, bdc, C_D, C_L); pointvalues -> (step, t, then
repeated x y type deriv value); the first two values are p(a1), p(a2).
"""
import hashlib
import json
from pathlib import Path

import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles as cyc  # noqa: E402  byte copy of the reference lane's cycles.py (one frozen procedure)
LOGS = HERE / 'featflow_2014_official/raw/logs/results_cc2d_bench2_referencenew'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = {'source_page': 'https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html',
           'source_page_sha256': sha(HERE / 'dfg_benchmark2_re100.html'),
           'zip': 'https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench2new_2d/20141124_dfg_bench2_official.zip',
           'zip_sha256': sha(HERE / 'featflow_2014_official/20141124_dfg_bench2_official.zip'),
           'discretisation': 'Q2/P1disc, Crank-Nicolson, levels 2-6, dt index 1,2,4',
           'normalisation': 'C = 2F/(U_mean^2 L), U_mean=1, L=0.1',
           'levels': {}}
    for lv in range(2, 7):
        for dt in (1, 2, 4):
            f = LOGS / f'bdforces_q2_lv{lv}_dt{dt}'
            g = LOGS / f'pointvalues_q2_lv{lv}_dt{dt}'
            a = np.loadtxt(f, comments='#')
            b = np.loadtxt(g, comments='#')
            t, cd, cl = a[:, 1], a[:, 3], a[:, 4]
            if not np.allclose(b[:, 1], t):
                raise ValueError('force/pressure time mismatch')
            dp = b[:, 6] - b[:, 11]
            recs = cyc.cycles(t, cd, cl, dp)
            rec = cyc.summarize(recs)
            rec['per_cycle'] = recs
            rec.update({'file_sha256': {f.name: sha(f), g.name: sha(g)}, 'dt': float(np.median(np.diff(t))),
                        't_window': [float(t[0]), float(t[-1])]})
            out['levels'][f'lv{lv}_dt{dt}'] = rec
    best = out['levels']['lv6_dt4']
    out['finest'] = {k: best[k]['mean'] for k in ('CD_max', 'CL_max', 'St', 'dp_at_t0_plus_half_period', 'dp_1996_t0_at_CLmax_plus_half_period', 'period')}
    # Schaefer & Turek (1996), NNFM 52, Table 4 (2D-2): published acceptance ranges.
    out['schaefer_turek_1996_ranges'] = {
        'CD_max': [3.22, 3.24], 'CL_max': [0.99, 1.01], 'St': [0.295, 0.305],
        'dp_at_t0_plus_half_period': [2.46, 2.50],
        'dp_convention': 't0 at the C_L maximum (1996 definition), i.e. dp_1996_t0_at_CLmax_plus_half_period',
        'provenance': 'Table 4 lower/upper bound rows, verified against the hosted compilation sources/karlin_compilation/dfg_benchmark_results.pdf (page 13)',
        'compilation_sha256': sha(HERE / 'karlin_compilation/dfg_benchmark_results.pdf')}
    (HERE / 'dfg2d2_reference.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['finest'], indent=2))


if __name__ == '__main__':
    main()
