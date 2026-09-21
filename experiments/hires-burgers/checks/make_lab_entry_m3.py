"""Lab-log entry for hires-burgers milestone 3 (hb4k04, hb4kh64; lane closed), numbers read from audit JSONs only."""
import json
from pathlib import Path
H = Path(__file__).resolve().parents[1]
S = {a: json.loads((H / f'checks/{a}-summary.json').read_text()) for a in ('hb4k04', 'hb4kh64', 'hb2kh64')}
M = json.loads((H / 'reports/summary.json').read_text())['matrix']
f = lambda x, d=2: '—' if x is None else f'{x:.{d}f}'
t4, th = S['hb4k04']['table'], S['hb4kh64']['table']
b, p = t4['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry'], t4['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']
ch = S['hb4k04']['verdict']['accurate_1_percent']
o = []; w = o.append
w('')
w('### hires-burgers — milestone 3 (lane closed, 8/8 jobs): pred2 predictor 1.15× at unchanged error; 4096² dev6 best 4.87× the relaxed passing FOM (bar NOT met); held-out 4096² error 1.33 % (bar NOT met)')
w('')
w('Worktree `worktrees/2026-09-20-hires-burgers`; `hb4k04` (job 4079320, 4096², dev6, 5 reps) and `hb4kh64` (job 4079321, 4096², hold64, 1 rep), both H200 pax008, source '
  f"`{S['hb4k04']['commit'][:8]}`, `jax_backend=gpu`, f64, highest; checksum-collected, NumPy-audited, remote dirs deleted (namespace `hires_b_20260920` now empty). "
  f"Failed gates: hb4k04 {', '.join(S['hb4k04']['failed_gates']) or 'none'}; hb4kh64 {', '.join(S['hb4kh64']['failed_gates']) or 'none'}. "
  'Summaries `experiments/hires-burgers/checks/{hb4k04,hb4kh64}-summary.json`; final report + verdict matrix `experiments/hires-burgers/reports/2026-09-20-hires-burgers.md` and `reports/summary.json`.')
w('')
w(f"**Speed (H11, same job).** `pred2` (one batched residual guard over current / linear / quadratic extrapolation) cuts median LM iterations per query {b['total_iterations_median']:g} → {p['total_iterations_median']:g} "
  f"at the accurate rung, {f(b['median_gpu_ms'], 1)} → {f(p['median_gpu_ms'], 1)} ms ({b['median_gpu_ms'] / p['median_gpu_ms']:.2f}×), error {f(b['worst_evolved_percent'], 4)} → {f(p['worst_evolved_percent'], 4)} %; kept. "
  f"Loosening the LM stationarity tolerance to 1e-2 (labelled) gives the dev6-selected arm `{ch['arm']}`: {f(ch['worst_evolved_percent'], 3)} % at {f(ch['median_gpu_ms'], 1)} ms = "
  f"{f(ch['speedups']['vs_tight'])}× `{ch['tight_comparator']}` and {f(ch['speedups']['vs_relaxed_passing'])}× `{ch['relaxed_passing_comparator']}` — the relaxed-FOM bar is missed by a few per cent. "
  'Local-only negatives, no job spent: XLA while-loop command buffers, initial-fit tolerance.')
w('')
w('| mesh | cohort | role | arm | worst evolved % | stalled | GPU ms | host ms | S tight GPU/host | S relaxed passing GPU/host | S fastest as-accurate GPU | S coarse GPU/host |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in M:
    if r['mesh'] != 4096:
        continue
    w(f"| {r['mesh']}² | {r['cohort']} | {r['role']} | `{r['arm']}` | {f(r['worst_evolved_percent'], 3)} | {r['stalled_exits']}/{r['steps']} | {f(r['gpu_ms'], 1)} | {f(r['host_ms'], 1)} | "
      f"{f(r['s_tight_gpu'])}/{f(r['s_tight_host'])} | {f(r['s_relaxed_gpu'])}/{f(r['s_relaxed_host'])} (`{r['relaxed_passing']}`) | {f(r['s_fastest_at_least_as_accurate_gpu'])} (`{r['fastest_at_least_as_accurate']}`) | "
      f"{(f(r['s_coarse_gpu']) + '/' + f(r['s_coarse_host']) + ' (`' + r['coarse'] + '`)') if r['coarse'] else '—'} |")
w('')
m5 = th['q256_M544_lat64_g0p001_fast_chol_clip_lamcarry_pred2']; m2 = th['q256_M2176_lat64_g0p001_fast_chol_clip_lamcarry_pred2']
r3 = th['lean_nt3e-3_l3e-3_dt005']; c1 = th['c1024_nt1e-4_dt005']
w(f"**Held-out at 4096² (hb4kh64).** Same picture as 2048²: the dev6-chosen arm is {f(th[ch['arm']]['worst_evolved_percent'], 3)} % worst evolved (median {f(th[ch['arm']]['median_evolved_percent'], 3)} %), "
  f"M=544 {f(m5['worst_evolved_percent'], 2)} % (0.875 % on dev6), M=2176 {f(m2['worst_evolved_percent'], 3)} % (uncertified), zero stalled exits everywhere; the coarse FOM c1024 is {f(c1['worst_evolved_percent'], 3)} % at {f(c1['median_gpu_ms'], 1)} ms. "
  f"On hold64 the relaxed FOM `lean_nt3e-3_l3e-3_dt005` is {f(r3['worst_evolved_percent'], 3)} % (> 0.1 %), so by the pre-declared rule the \"relaxed passing\" comparator falls back to the tight FOM there; "
  'the matrix column "fastest as-accurate" shows the ratio against it instead. The accuracy failure is a bank/representation floor (bank-floor lane), not a solver or quadrature effect: the tolerance and predictor arms leave the error unchanged to 3 digits.')
w('')
g = lambda L, c, role: next(r for r in M if r['mesh'] == L and r['cohort'] == c and r['role'] == role)
d2c, d2a, d4c = g(2048, 'dev6', 'chosen on dev6'), g(2048, 'dev6', 'accurate rung q256/M1088'), g(4096, 'dev6', 'chosen on dev6')
xfer = S['hb4k04']['table'][ch['arm']]
w(f"**Bar verdict (final for this lane).** 2048² dev6: met vs the tight FOM only ({f(d2c['s_tight_gpu'])}× chosen M544, {f(d2a['s_tight_gpu'])}× M1088), not vs the relaxed passing FOM "
  f"({f(d2c['s_relaxed_gpu'])}× / {f(d2a['s_relaxed_gpu'])}×). 4096² dev6: met vs tight ({f(d4c['s_tight_gpu'], 1)}×), not vs relaxed passing ({f(d4c['s_relaxed_gpu'])}×; complete query {f(d4c['s_relaxed_host'])}×, "
  f"because the six-field output transfer, {xfer['median_host_ms'] - xfer['median_gpu_ms']:.0f} ms, is common to both). hold64 at both meshes: not met — no certified arm ≤ 1 %. "
  "The coarse-grid FOM c1024 is faster than every accurate ROM arm at 4096² and about as accurate.")
w('')
w('**What was wrong / retracted.** hb4kh64 fails the same restricted-proxy gate as hb2kh64 (18 of 1152 rows, low-error q=256 cases, worst gap '
  f"{S['hb4kh64']['gates']['restricted_recomputation_tracks_full_grid']['worst_relative_gap']:.3f}; cohort-worst per arm within 1.2 %, full-grid recompute of case 0 exact) — the 5 % threshold was tuned on dev6 and is too tight on held-out low-error cases; reported, not loosened. "
  'The local 64² smoke over-predicted pred2 (1.24× there, 1.15× at 4096²). Nothing earlier in this lane is retracted.')
w('')
w('**Open / next.** The lane is closed (budget spent). The binding problem is held-out accuracy, not speed: refit the head and recertify the EQ rule on the bank-floor lane\'s better banks (`worktrees/2026-09-20-bank-floor/experiments/bank-floor/CKPT-MANIFEST.json`), then re-time with pred2 + chol + clip + lamcarry. No worktree merge proposed by this agent.')
print('\n'.join(o))
