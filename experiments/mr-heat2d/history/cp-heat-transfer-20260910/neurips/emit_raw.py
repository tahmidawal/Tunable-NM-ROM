"""Convert h3opt/h2opt run summaries into canonical results/raw/*.json entries.

One file per (cell, variant).  Nothing is quoted from a log; the summary JSON
written by the measurement job is the only input.
"""
import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path

import numpy as np

# Co-residency OBSERVED from the Slurm logs of each job, transcribed here with
# the log path so it stays traceable.  Never asserted, never defaulted to True.
ISOLATION = {
    '1918201': dict(
        node='pax008', other_running_jobs_on_node=[], isolated=True,
        source='/cluster/home/tawal01/nmrom/slurm-logs/hoptall-1918201.out'
               ' line "co-resident on pax008: 1918201"'),
    '1921064': dict(
        node='pax009',
        other_running_jobs_on_node=['1828314', '1891886', '1884591', '1818358'],
        isolated=False,
        source='/cluster/home/tawal01/nmrom/slurm-logs/hopt4-1921064.out'),
    '1921153': dict(
        node='pax009',
        other_running_jobs_on_node=['1828314', '1891886', '1884591', '1818358'],
        isolated=False,
        source='/cluster/home/tawal01/nmrom/slurm-logs/hopt5-1921153.out'),
    '1921388': dict(
        node='pax009',
        other_running_jobs_on_node=['1828314', '1891886', '1884591', '1818358'],
        isolated=False,
        source='/cluster/home/tawal01/nmrom/slurm-logs/hopt6-1921388.out'),
}

VARIANT_NOTES = {
    'base:eager': (
        'CONTROL. Submission-era ROM structure exactly as re-timed on '
        '2026-07-28: the latent rollout is compiled with kappa as a runtime '
        'argument, the ViT encoder is not compiled and runs one XLA op at a '
        'time from Python.'),
    'base': (
        'Encoder compiled. Same residual, same Jacobian, same Gauss-Newton '
        'iteration, same stopping rule as the control; the only change is that '
        'the encoder is inside the compiled graph instead of being dispatched '
        'op by op from Python.'),
    'v5': (
        'Encoder compiled; the data-dependent Gauss-Newton while_loop replaced '
        'by a counted loop that freezes the update once the stopping test is '
        'met. Nothing else changed. Verified BIT-IDENTICAL to the control on 30 '
        'random configurations (same Gauss-Newton counts, same iterates) -- see '
        'test_equiv.py.'),
    'v5w': (
        'Encoder compiled; the (k+1)x(k+1) regularised normal-equation solve '
        'done by Cholesky instead of LU, which is valid because the matrix is '
        'SPD by construction. Loop construct and all residual/Jacobian algebra '
        'byte-for-byte as the control. Differs only in the last bits of the '
        'small solve.'),
    'v5c': ('v5 plus the Cholesky solve.'),
    'v4': (
        'Encoder compiled; counted Gauss-Newton loop; Cholesky solve; and the '
        'residual/Jacobian re-expressed (batched line search, regrouped '
        'gradient norm). Same formulas, but float32 re-association differs from '
        'the control at ~1e-7, which can flip a Gauss-Newton count on a '
        'trajectory sitting on the tolerance.'),
    'v4w': (
        'As v4 but keeping the data-dependent while_loop, so the cost of paying '
        'the iteration cap is not incurred on cells whose Gauss-Newton '
        'converges well before it. Differs from the control at float32 '
        're-association level (~1e-7).'),
    'v3w': (
        'v4w plus contracting the backward-Euler operator into the CP basis '
        'once per kappa and taking the Jacobian through the decoder MLP only. '
        'Measured SLOWER: the re-association changes Gauss-Newton convergence '
        'and the solve takes about twice as many iterations.'),
    'v3w_nofold': ('v4w with the line search batched. Neutral.'),
    'v3w_nolin': ('v3w without the shared-primal Jacobian. Neutral vs v3w.'),
    'v3': (
        'v4 plus algebraic restructuring: the backward-Euler operator is '
        'contracted into the CP basis once per kappa, the Jacobian is taken '
        'through the decoder MLP only and lifted by one matrix product, and '
        'the four line-search trial points are evaluated as one batch. '
        'Mathematically equivalent; differs from the control at f32 '
        're-association level.'),
    'base_enc': (
        'CONTROL for the 2D cells, corrected. The submission-era 2D timed '
        'region starts after the encoder has run, so the published 2D ROM '
        'times exclude it. This arm puts the encoder back in, uncompiled, '
        'which is what the 3D cells were already doing.'),
    'enc_jit': (
        'Encoder included in the measured cost and compiled.'),
    'v4_lu': ('v4 with the LU solve retained.'),
    'v3w_nofold': ('v4w with the line search batched. Neutral.'),
    # --- rounds 8-12: reduced-Gram assembly -------------------------------
    'red': (
        'REDUCED-GRAM assembly. The decoder is affine in a rank-256 (2D) / '
        'rank-512 (3D) feature vector and the heat operator is linear with '
        'affine kappa dependence, so J^T J, J^T r and ||r||^2 are polynomials '
        'in DT*kappa whose coefficients are kappa-free rank x rank contractions '
        'built once offline (1.58 MB in 2D, 6.31 MB in 3D). The online solve '
        'never forms a grid-sized array. In 3D the empirical-quadrature support '
        'and weights are used EXACTLY as released, so the objective is '
        'unchanged and the Gauss-Newton counts are identical on 4 of 5 cells. '
        'In 2D the counts FALL from ~19 to ~4 because the released solver was '
        'stepping on an f32 gradient that test_red2d.py measures as 51% wrong; '
        'capping the dense solver at 4 instead costs 2-8x in rel-L2 (round 10), '
        'so the reduced iterations are not the same iterations. Reduced Grams '
        'are f64: in f32 J^T J is off by 6.8%.'),
    'red_m': ('red plus the counted Gauss-Newton loop. Free where the cap '
              'equals the realised count (h3d_n32 cap 3, h3d_n64_fast cap 2), '
              'ruinous elsewhere.'),
    'red64': ('red with the ITERATE also carried in f64. Within 0.6% of red '
              'everywhere, so carrying the iterate in f64 buys nothing.'),
    'red_jf': ('red with dh/dz from jax.jacfwd instead of the closed-form MLP '
               'Jacobian. Within 1.2% of red: the MLP Jacobian was never the '
               'bottleneck.'),
    'red_lu': ('red with the LU solve retained instead of Cholesky.'),
    'v4w64': (
        'CONTROL for round 10. Dense (N^2 x k+1) assembly with the f64 '
        'ACCUMULATION but the decoder still evaluated in f32. Runs the SAME '
        'Gauss-Newton count as f32 (16.6 vs 16.6 at N=128 acc), which refutes '
        'the hypothesis that precision of the accumulation is what cuts the '
        'count. The f32 error is in the residual entries, not in how they are '
        'summed.'),
    'v4w64f': (
        'CONTROL for round 11. Dense assembly with the DECODER in f64 as well, '
        'so the grid residual itself is f64-accurate. The only dense arm that '
        'can produce an f64-accurate gradient, and the expensive way to do it: '
        'f64 on grid-sized arrays costs ~4.1x the f32 per-iteration cost.'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    files = []
    for pat in args.summaries:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit('no summary files matched')

    for f in files:
        s = json.load(open(f))
        cell = s['cell']
        cfg = s['config']
        N = cfg['N']
        pde = 'heat3d' if cell.startswith('h3d') else 'heat2d'
        arm = cell.split('_')[-1]
        stamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        fom_t = s['fom_times']
        # runs of the same cell are distinguished by a prefix on the summary
        # filename (r4_, r5_graph_, r5_nograph_); round 3 has no prefix.
        bn = os.path.basename(f)
        run = ''
        # longest first, so r8Aa100_redgraph_ is not shadowed by r8Aa100_red_
        for pfx in ('r8Aa100_redgraph_', 'r8Aa100_x64off_', 'r8Aa100_red_',
                    'r8A_redgraph_', 'r8A_x64off_', 'r8A_red_',
                    'r8B_nograph_', 'r8B_graph_',
                    'r11_2d_nograph_', 'r11_2d_graph_', 'r11_attrib_',
                    'r12_nograph_', 'r12_graph_',
                    'r9_redgraph_', 'r9_red_', 'r10_',
                    'r5_nograph_', 'r5_graph_', 'r4_', 'r5_', 'r6_', 'r7_'):
            if bn.startswith(pfx):
                run = '_' + pfx.rstrip('_')
                break

        for vname, v in s['variants'].items():
            tag = (vname.replace(':', '_').replace('@', '_at_')
                   .replace(',', '_').replace('.', 'p'))
            rom_t = v['rom_times']
            ratios = v['per_traj_speedup']
            reps = len(v['rom_all'][0]) if v.get('rom_all') else None
            cap = v.get('knobs', {}).get('max_iters',
                                         cfg.get('max_iters',
                                                 cfg.get('gn_max_iters')))
            gn_mean = float(np.mean([np.mean(x) for x in v['gn_iters']]))
            at_cap = cap is not None and gn_mean >= 0.98 * cap
            iso = (s.get('isolation_evidence')
                   or ISOLATION.get(str(s.get('slurm_job_id')), {}))
            isolated = bool(s.get('exclusive_node',
                                  iso.get('isolated', False)))
            payload = {
                'pde': pde,
                'N': int(N),
                'method': 'nm_rom',
                'variant': f'recompiled_{arm}_{tag}{run}',
                'rel_l2': float(v['rel_l2_mean']),
                'solve_time_s': float(np.median(rom_t)),
                'fom_time_s': float(np.median(fom_t)),
                'speedup': float(v['speedup_median']),
                'config': dict(cfg, **v.get('knobs', {})),
                'seed': 0,
                'timestamp': stamp,
                'notes': (
                    'Heat block cost study during the discussion period. '
                    'EVAL ONLY: submission-era checkpoint, submission-era '
                    'validation trajectories, cached empirical-quadrature '
                    'support, so accuracy is fixed by construction. '
                    'FOM FROZEN: same operator, CG tolerance 1e-6, '
                    '1000-iteration cap, 50 backward-Euler steps, compiled '
                    'with kappa as a runtime argument, warmed and '
                    'median-aggregated exactly like the ROM, and measured '
                    'INTERLEAVED with it. ARM: ' + VARIANT_NOTES.get(
                        vname.split('@')[0], 'see ROUNDS.md')),
                'protocol': {
                    'timing_protocol': 'rebuttal-v3',
                    'warmup_discarded': 2,
                    'repeats': reps,
                    'aggregation': 'median',
                    'device_sync': True,
                    'backend': s.get('backend'),
                    'exclusive_node': isolated,
                    'exclusive_node_source': 'observed',
                    'interleaved': True,
                    # bench.py refuses a ratio without a convergence flag: a
                    # solve that stopped at its iteration cap reports the cap,
                    # not a cost. Recorded per arm rather than assumed.
                    'solver_converged': (not at_cap),
                    'gn_iters_mean': gn_mean,
                    'gn_iter_cap': cap,
                    'gn_at_cap': at_cap,
                },
                'extra': {
                    'speedup_aggregation': 'median_of_per_traj_ratios',
                    'speedup_median_of_ratios': float(np.median(ratios)),
                    'speedup_mean_of_ratios': float(np.mean(ratios)),
                    'speedup_range': v['speedup_range'],
                    'per_traj_kappa': s['kappas'],
                    'per_traj_rel_l2': v['per_traj_rel_l2'],
                    'per_traj_speedup': ratios,
                    'per_traj_rom_s': rom_t,
                    'per_traj_fom_s': fom_t,
                    'rom_repeat_times_s': v['rom_all'],
                    'fom_repeat_times_s': s['fom_all'],
                    'fom_jit_vs_reference_rel_l2': s['fom_fidelity'],
                    'gn_iters_per_step': v['gn_iters'],
                    'node': s.get('node'),
                    'slurm_job_id': s.get('slurm_job_id'),
                    'n_eq': s.get('n_eq'),
                    'source_summary': os.path.basename(f),
                    # XLA graph capture, when used, is set for the FOM as well
                    # as the ROM. The FOM gains ~0% from it; the ROM ~8%.
                    'xla_flags': (s.get('xla_flags')
                                  or ('--xla_gpu_graph_level=3'
                                      if run == '_r5_graph' else '')),
                    'isolation_evidence': iso,
                    'x64': s.get('x64'),
                    'reduced_offline_bytes': s.get('reduced_offline_bytes'),
                    'reduced_build_seconds': s.get('reduced_build_seconds'),
                    'equiv_vs': v.get('equiv_vs'),
                    'equiv_traj_identical_gn': v.get('equiv_traj_identical_gn'),
                    'equiv_total_gn_diff': v.get('equiv_total_gn_diff'),
                    'equiv_rel_l2_delta': v.get('equiv_rel_l2_delta'),
                    'caveat': ('Gauss-Newton stopped at its iteration cap '
                               f'({gn_mean:.1f} of {cap}), so this arm\'s error '
                               'is set by where the cap falls and its time is '
                               'the cap cost, not a converged cost.'
                               if at_cap else None),
                    'control_variant': 'base:eager' if pde == 'heat3d' else 'base',
                    'batched': s.get('batched', {}).get(
                        'variants', {}).get(vname),
                },
            }
            name = (f'{pde}_n{N}_nm_rom_recompiled_{arm}_{tag}{run}'
                    f'_{stamp}.json')
            print(f'{name}  rel_l2={payload["rel_l2"]:.6e}  '
                  f'rom={payload["solve_time_s"]*1e3:.2f}ms  '
                  f'spd={payload["speedup"]:.3f}x')
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
                with open(outdir / name, 'w') as fh:
                    json.dump(payload, fh, indent=2)


if __name__ == '__main__':
    main()
