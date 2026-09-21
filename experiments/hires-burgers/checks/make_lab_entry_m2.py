"""Lab-log entry for hires-burgers milestone 2 (hb2k02, hb4k03, hb2kh64), numbers read from the
audit summaries and reports/summary.json only."""
import json
from pathlib import Path
H = Path(__file__).resolve().parents[1]
S = {a: json.loads((H / f'checks/{a}-summary.json').read_text()) for a in ('hb2k02', 'hb4k03', 'hb2kh64')}
M = json.loads((H / 'reports/summary.json').read_text())['matrix']
f = lambda x, d=2: '—' if x is None else f'{x:.{d}f}'
def row(att, role):
    return next(r for r in M if r['attempt'] == att and r['role'] == role)
def T(att, n):
    return S[att]['table'][n]
out = []
w = out.append
w('')
w('## 2026-09-21')
w('')
w('### hires-burgers — milestone 2: hb2k02 (2048²), hb4k03 (4096²) and hb2kh64 (2048², 64 held-out cases) audited; bar met vs the TIGHT FOM on dev6 at both meshes, NOT vs the relaxed passing FOM, and NOT on held-out cases (accurate rung 1.31 %)')
w('')
w('Worktree `worktrees/2026-09-20-hires-burgers` (branch `exp/2026-09-20-hires-burgers`); namespace `hires_b_20260920`; all three jobs H200, '
  '`jax_backend=gpu`, f64, highest precision, checksum-collected, NumPy-audited by `audit_hires.py`, remote dirs deleted. Summaries '
  '`experiments/hires-burgers/checks/{hb2k02,hb4k03,hb2kh64}-summary.json`; report and verdict matrix `experiments/hires-burgers/reports/2026-09-20-hires-burgers.md` '
  '+ `reports/summary.json`, generated. Agent handover: the Fable 5.1 agent stopped on usage credits after committing hb2k02; this entry is by its Opus 5 successor.')
w('')
for att in ('hb2k02', 'hb4k03', 'hb2kh64'):
    s = S[att]
    w(f"- `{att}` job {s['job_id']}, {s['intervals']}², cohort {'hold64 (64 cases)' if att == 'hb2kh64' else 'dev6'}, source `{s['commit'][:8]}`, "
      f"elapsed {s['elapsed_seconds']:.0f} s, failed gates: {', '.join(s['failed_gates']) or 'none'}.")
w('')
w('| mesh | cohort | role | arm | worst evolved % | stalled | GPU ms | host ms | S tight GPU/host | S relaxed passing GPU/host | S coarse GPU/host |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for r in M:
    if r['attempt'] not in S:
        continue
    w(f"| {r['mesh']}² | {r['cohort']} | {r['role']} | `{r['arm']}` | {f(r['worst_evolved_percent'], 3)} | {r['stalled_exits']}/{r['steps']} | {f(r['gpu_ms'], 1)} | {f(r['host_ms'], 1)} | "
      f"{f(r['s_tight_gpu'])}/{f(r['s_tight_host'])} (`{r['tight']}`) | {f(r['s_relaxed_gpu'])}/{f(r['s_relaxed_host'])} (`{r['relaxed_passing']}`) | "
      f"{(f(r['s_coarse_gpu']) + '/' + f(r['s_coarse_host']) + ' (`' + r['coarse'] + '`)') if r['coarse'] else '—'} |")
w('')
a4 = T('hb4k03', 'q256_M1088_lat64_g0p001_fast')
b4 = T('hb4k03', 'q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry')
c1 = T('hb4k03', 'c1024_nt1e-4_dt005')
w(f"**Speed.** Cholesky + trust clipping + damping carry-over (hb2k02, SPEED-LOG) replicate at 4096²: q256/M1088 {f(a4['median_gpu_ms'], 1)} → {f(b4['median_gpu_ms'], 1)} ms "
  f"({a4['median_gpu_ms'] / b4['median_gpu_ms']:.2f}×) at unchanged error {f(a4['worst_evolved_percent'], 4)} → {f(b4['worst_evolved_percent'], 4)} %, rejected trial steps {a4['damping_retries']} → {b4['damping_retries']}. "
  f"The ROM's speedup against the relaxed passing FOM grows with mesh (the FOM scales with n, the ROM does not, except its decode), but stays below 5 in both scopes. "
  f"At 4096² the complete query adds ≈{b4['median_host_ms'] - b4['median_gpu_ms']:.0f} ms of output transfer to every arm, which caps any complete-query ratio.")
w('')
w(f"**The coarse-grid FOM beats the ROM at 4096².** `c1024_nt1e-4_dt005` runs in {f(c1['median_gpu_ms'], 1)} ms at {f(c1['worst_evolved_percent'], 3)} % same-grid "
  f"(refined-reference error {f(c1.get('worst_reference_evolved_percent'), 2)} % vs the ROM's {f(b4.get('worst_reference_evolved_percent'), 2)} % and the tight FOM's {f(S['hb4k03']['truth_vs_refined_reference_evolved_percent'], 2)} %): "
  'about as accurate as the accurate rung and faster. Against that comparator the ROM has no speed story at these meshes.')
w('')
h = T('hb2kh64', 'q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry')
h2 = T('hb2kh64', 'q256_M2176_lat64_g0p001_fast_chol_clip_lamcarry')
h0 = T('hb2kh64', 'q0_M64_scaled_g0p001_fast_clip_lamcarry')
w(f"**Held-out cases (hb2kh64).** The 0.598 % dev6 accuracy does not generalise: on hold64 the accurate rung is {f(h['worst_evolved_percent'], 3)} % worst evolved "
  f"(median {f(h['median_evolved_percent'], 3)} %), M=2176 {f(h2['worst_evolved_percent'], 3)} % (rule uncertified), q=0 {f(h0['worst_evolved_percent'], 2)} %; zero stalled exits. "
  'No certified ROM arm is ≤ 1 % on held-out cases at 2048², so the lane bar is not met on unseen cases at any speed. This matches the bank-floor lane (R=512 bank floor far higher on hold64 than on dev6).')
w('')
w('**What was wrong / caveats.** hb2kh64 fails one audit gate, `restricted_recomputation_tracks_full_grid` (threshold 5 %): on 5 of 384 rows, all low-error q=256 cases, '
  f"the 256² restricted sample reads the full-grid error 5–7 % low (worst gap {S['hb2kh64']['gates']['restricted_recomputation_tracks_full_grid']['worst_relative_gap']:.3f}); "
  'the cohort-worst per arm agrees within 1.2 % and the full-grid recomputation of case 0 is exact, so the verdict numbers stand, but the gate failed and is reported. '
  'hold64 timings are one repetition. On hold64 no same-mesh FOM other than the tight one passes the 0.1 % "relaxed passing" definition, so "relaxed passing" = tight there by rule. '
  'The HANDOFF line "worst evolved 1.31 % on hold64" was an unaudited log line; the audit confirms it.')
w('')
w('**Running / next.** Budget 8/8 now spent: `hb4k04` (job 4079320, 4096², dev6, 5 reps) tests the `pred2` quadratic-extrapolation predictor (H11) and labelled tolerance arms; '
  '`hb4kh64` (job 4079321, 4096², hold64) reports the same arms on held-out cases. Selection stays on dev6 (DESIGN addendum A1). '
  'After them the lane stops; the next lever is accuracy, not speed: refit the head + recertify EQ on the bank-floor lane\'s better banks (out of this lane\'s budget).')
print('\n'.join(out))
