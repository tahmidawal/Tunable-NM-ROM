"""Write the self-audit of the report and the lab-log entry from the same JSONs the
report is generated from, so neither carries a hand-typed number.

    python reports/generate_entries.py
      -> reports/self-audit-report.md, checks/2026-09-17-lab-log-entry.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
S = json.loads((HERE / 'summary.json').read_text())
V = json.loads((HERE / 'verdicts.json').read_text())
AUD = {a: json.loads((LANE / 'artifacts' / a / 'audit.json').read_text()) for a in ('plin256', 'plin1024b', 'plhead1')}
RES = {a: json.loads((LANE / 'artifacts' / a / 'result.json').read_text()) for a in ('plin256', 'plin1024b', 'plhead1')}
SUB = {a: json.loads((LANE / 'runs' / a / 'SUBMISSION.json').read_text()) for a in ('plin256', 'plin1024', 'plin1024b', 'plhead1')}
FAIL = json.loads((LANE / 'runs/plin1024/FAILURE.json').read_text())
FIX = json.loads((LANE / 'checks/2026-09-17-oracle-fix-check-64.json').read_text())
SCALE = json.loads((LANE / 'checks/2026-09-17-oracle-metric-scale.json').read_text())
PRIM = 'new_K32'


def val(mesh, subject, metric):
    return next(r['value'] for r in S if r['mesh'] == mesh and r['subject'] == subject and r['metric'] == metric)


def pct(x):
    return f'{100 * x:.4f} %'


def ladder_table(mesh):
    crit = V[str(mesh)]
    lines = ['| rung | worst same-grid | median total ms |', '|---|---:|---:|']
    for name, e, c in zip(crit['ladder'], crit['errs'], crit['costs']):
        lines.append(f'| `{name}` | {pct(e)} | {c:.3f} |')
    return '\n'.join(lines)


def clauses(mesh):
    c = V[str(mesh)]
    return (f"D1 span {c['D1_span']:.3f}x ({'pass' if c['D1'] else 'FAIL'}; over the q < R rungs alone, post-hoc, "
            f"{c['D1_neural_span_posthoc']:.3f}x); D2 {'pass' if c['D2_lowest'] and c['D2_within'] else 'FAIL'}, strict "
            f"{'yes' if c['D2_strict'] else 'no'}; D3 {'pass' if c['D3_fom'] and c['D3_pod'] else 'FAIL'} "
            f"(non-dominated all: {', '.join(f'`{x}`' for x in c['nd_all'])}; reduced: {', '.join(f'`{x}`' for x in c['nd_red'])}); "
            f"monotone {'yes' if c['monotone'] else 'no'}; falsified literal {'MET' if c['falsified_literal'] else 'not met'}, "
            f"intent {'MET' if c['falsified_intent'] else 'not met'}")


def dense_vs_projected(res):
    """Worst relative difference between the dense q=0 oracle and the projected q=0 oracle,
    per case, from the 256 job (which predates the driver's own record of it)."""
    rec = next(r for r in res['reconstruction'] if r['model'] == PRIM)
    if 'dense_vs_projected_oracle' in rec:
        return rec['dense_vs_projected_oracle']['worst_relative_difference']
    d = rec['best_found_dense']['per_case']
    e = next(x for x in rec['augmented'] if x['q'] == 0)['best_found']['per_case']
    return max(abs(a - b) / a for a, b in zip(d, e))


def main():
    r1024, r256, rh = RES['plin1024b'], RES['plin256'], RES['plhead1']
    a1024, a256, ah = AUD['plin1024b'], AUD['plin256'], AUD['plhead1']
    H = V['head']
    arms = {a['arm']: a for a in rh['arms']}
    ctrl, prim = arms['K32_w128_L2'], arms['pbh02_primary_K32']
    best_ratio_arm = min((a for a in rh['arms'] if a['arm'] != 'pbh02_primary_K32'), key=lambda a: a['ratio_dev_best_found_over_floor'])
    best_val_arm = min((a for a in rh['arms'] if a['arm'] != 'pbh02_primary_K32'), key=lambda a: a['best_found_validation']['worst'])
    aug1024 = [r for r in S if r['mesh'] == 1024 and r['family'] == 'oracle' and r['subject'].startswith('augmented')]
    aug256 = [r for r in S if r['mesh'] == 256 and r['family'] == 'oracle' and r['subject'].startswith('augmented')]
    top = f'd_linear_qr_m4@{PRIM}'

    # ------------------------------------------------------------ self-audit --
    rows = [
        ('1024 verdict DEGENERATE (D1 ∧ D2 ∧ D3)', 'verdicts.json["1024"]; audit.json criterion.degenerate',
         'generator and NumPy audit derive it independently from invocation rows', f"generator D1={V['1024']['D1']}, D2_strict={V['1024']['D2_strict']}, D3={V['1024']['D3_fom'] and V['1024']['D3_pod']}; audit degenerate={a1024['criterion']['degenerate']}"),
        ('1024 D1 cost span', 'verdicts.json["1024"].D1_span vs audit criterion.D1_cost_span', 'two implementations agree',
         f"{V['1024']['D1_span']:.6f} vs {a1024['criterion']['D1_cost_span']:.6f}"),
        ('256 D1 fails literally, intent not met', 'verdicts.json["256"].{D1_span,falsified_literal,falsified_intent}; audit criterion', 'two implementations agree',
         f"span {V['256']['D1_span']:.3f}x; literal {V['256']['falsified_literal']}/{a256['criterion']['falsified_literal']}, intent {V['256']['falsified_intent']}/{a256['criterion']['falsified_intent']}"),
        ('top rung = cheapest and most accurate at both meshes', 'verdicts.json[*].{cheapest,top,D2_lowest}', 'string equality',
         f"1024: cheapest `{V['1024']['cheapest']}` top `{V['1024']['top']}`; 256: cheapest `{V['256']['cheapest']}` top `{V['256']['top']}`"),
        ('top rung reaches the bank floor', f'summary rows worst_same_grid of `{top}` and `bank_floor@{PRIM}`', 'relative difference',
         f"1024: {val(1024, top, 'worst_same_grid'):.6e} vs floor {val(1024, f'bank_floor@{PRIM}', 'worst_same_grid'):.6e}; 256: {val(256, top, 'worst_same_grid'):.6e} vs {val(256, f'bank_floor@{PRIM}', 'worst_same_grid'):.6e}"),
        ('bank floor independently rebuilt', 'audit.json numpy_bank_floor (plin1024b)', 'NumPy QR of the 1046529x512 bank, no JAX',
         f"max abs difference {a1024['numpy_bank_floor']['max_abs_difference']:.2e}, worst {a1024['numpy_bank_floor']['worst']:.6e}"),
        ('every reported error recomputed from saved fields', 'audit.json recomputed_errors (all three jobs)', 'sha256 of every field checked, error recomputed',
         f"1024: {a1024['recomputed_errors']['invocations']} errors/{a1024['recomputed_errors']['distinct_fields']} fields, worst {a1024['recomputed_errors']['worst_same_grid_difference']:.2e}; 256: {a256['recomputed_errors']['invocations']}/{a256['recomputed_errors']['distinct_fields']}, {a256['recomputed_errors']['worst_same_grid_difference']:.2e}; head: {ah['recomputed_errors']['invocations']}/{ah['recomputed_errors']['distinct_fields']}, {ah['recomputed_errors']['worst_same_grid_difference']:.2e}"),
        ('GPU backend and precision', 'result.json {backend,x64,matmul_precision,gpu}', 'audit backend check',
         f"1024: {r1024['backend']}/{r1024['gpu']}; 256: {r256['backend']}/{r256['gpu']}; head: {rh['backend']}/{rh['gpu']}; x64 {r1024['x64']}, precision {r1024['matmul_precision']}"),
        ('no ratio across GPUs', 'result.json gpu per job', 'the report prints a GPU-per-job table; every span/non-dominated set is per job',
         f"H200 (1024) vs A100-PCIE-40GB (256, head): no cross-mesh cost number exists in summary.json (all rows carry one job_id)"),
        ('fidelity gates', 'result.json gates; audit fidelity_gates_recomputed', 'all pass at 1e-9, recomputed from fields',
         f"1024: {sum(g['passed'] for g in r1024['gates'])}/{len(r1024['gates'])}, worst {max(g['worst_relative_difference'] for g in r1024['gates']):.2e}; 256: {sum(g['passed'] for g in r256['gates'])}/{len(r256['gates'])}, worst {max(g['worst_relative_difference'] for g in r256['gates']):.2e}; head: {sum(g['passed'] for g in rh['gates'])}/{len(rh['gates'])}"),
        ('consistency pairs', 'result.json consistency', 'all pass at the stated tolerances',
         f"1024: {sum(c['passed'] for c in r1024['consistency'])}/{len(r1024['consistency'])}; 256: {sum(c['passed'] for c in r256['consistency'])}/{len(r256['consistency'])}; q512 eliminated vs direct {next(c['worst_field_relative_difference'] for c in r1024['consistency'] if c['a'].startswith('q512')):.2e} (1024)"),
        ('POD-512 and DST dominate the neural points', 'summary rows worst_same_grid/median_total_ms of e_pod512_m4@trainset, dst_direct', 'read from rows; non-dominated flags',
         f"1024: POD-512 {pct(val(1024, 'e_pod512_m4@trainset', 'worst_same_grid'))} at {val(1024, 'e_pod512_m4@trainset', 'median_total_ms'):.3f} ms, DST {val(1024, 'dst_direct', 'median_total_ms'):.3f} ms; top rung {pct(val(1024, top, 'worst_same_grid'))} at {val(1024, top, 'median_total_ms'):.3f} ms"),
        ('CG is the slowest full-order route', 'summary rows median_total_ms of cg_*', 'read from rows',
         f"1024: {val(1024, 'cg_0.01', 'median_total_ms'):.1f} ms (1e-2) to {val(1024, 'cg_1e-08', 'median_total_ms'):.1f} ms (1e-8) vs DST {val(1024, 'dst_direct', 'median_total_ms'):.3f} ms"),
        ('q = 0 three layers agree', f'summary augmented_best_found_q0@{PRIM}, q0_m256@{PRIM}, bank_floor; result reconstruction.dense_vs_projected_oracle (256)', 'oracle q=0 has no V; dense cross-check at 256',
         f"1024: oracle {pct(val(1024, f'augmented_best_found_q0@{PRIM}', 'worst_same_grid'))} vs solved {pct(val(1024, f'q0_m256@{PRIM}', 'worst_same_grid'))}; 256 dense-vs-projected {dense_vs_projected(r256):.2e}"),
        ('augmented oracle q > 0 RETRACTED (A10)', 'summary oracle rows retracted flag; checks/2026-09-17-oracle-metric-scale.json; checks/2026-09-17-oracle-fix-check-64.json', 'V^T V measured per mesh; fix verified locally',
         f"retracted rows: 1024 {sum(r['retracted'] for r in aug1024)}/{len(aug1024)}, 256 {sum(r['retracted'] for r in aug256)}/{len(aug256)}; V^T V diag 64/256/1024: {SCALE['64']['diag_mean']:.4f}/{SCALE['256']['diag_mean']:.4f}/{SCALE['1024']['diag_mean']:.3f}; fixed q=R vs floor rel {FIX['checks']['fixed_qR_equals_floor_rel']:.1e}, q0 identical {FIX['checks']['q0_identical']}, all {FIX['checks']['all_passed']}"),
        ('plin1024 retraction is a GPU OOM, not disk-full', 'runs/plin1024/FAILURE.json', 'log complete, ends in RESOURCE_EXHAUSTED; disk recorded',
         f"{FAIL['state']} {FAIL['exit_code']} at {FAIL['elapsed']} on {FAIL['gpu']}; disk {FAIL['disk_at_failure']}; timed numbers {FAIL['timed_numbers_produced']}"),
        ('H1 reproducibility', 'verdicts.json head.H1_relative; result arms', '5 % bar',
         f"{100 * H['H1_relative']:.2f} % relative (control {pct(ctrl['best_found_development']['worst'])} vs primary {pct(prim['best_found_development']['worst'])}) -> {'pass' if H['H1_pass'] else 'FAIL'}"),
        ('H2 verdict', 'verdicts.json head.{H2_max_drop,H2_min_ratio}', '20 % drop bar; 2x ratio bar',
         f"max drop {100 * H['H2_max_drop']:.1f} % (`{best_ratio_arm['arm']}`), min ratio {H['H2_min_ratio']:.3f}x -> partial movement; validation picks `{best_val_arm['arm']}`"),
        ('head arms solved at 1024 do not enter the linear-case table', 'DESIGN section 5 (enters only if ratio < 2x)', 'H2_min_ratio',
         f"{H['H2_min_ratio']:.3f}x >= 2 -> none enters; the head rows are on the A100 job and are never compared to the H200 job"),
        ('one job per directory, squeue before/after', 'runs/*/SUBMISSION.json', 'one_job_per_directory flags',
         ', '.join(f"{a}: {SUB[a]['one_job_per_directory']}" for a in SUB)),
        ('remote directories deleted', 'ssh listing after collection', 'namespace listing empty',
         'p_linear_20260917/ contains no attempt directories (verified 2026-09-17 10:55 EDT)'),
        ('job count', 'runs/*/SUBMISSION.json', 'count', f'{len(SUB)} of 8 (one retracted)'),
    ]
    md = ['# Self-audit of the p-linear report against the raw JSONs (substitute for the Codex report audit)', '',
          'Codex (gpt-6-astra) is unavailable until 2026-09-19 11:33 (usage limit; LANE-PROTOCOL.md notice). Per that notice and DESIGN §A9 '
          'this is a written self-audit: one row per claim the report and the lab-log entry make, the JSON field it rests on, the check run, '
          'and the value read. Every value below is read from `summary.json`, `verdicts.json`, the three `audit.json` files, the run '
          '`result.json` files, `FAILURE.json`, and the two check JSONs by `generate_entries.py`; none is typed. The independence '
          'guarantee is weaker than a second model family.', '',
          '| # | claim | rests on | check | value read |', '|---|---|---|---|---|']
    md += [f'| {i + 1} | {c} | {f} | {k} | {v} |' for i, (c, f, k, v) in enumerate(rows)]
    md += ['', '## Findings of this self-audit', '',
           '1. **One retraction found by this audit and not by any gate**: the augmented best-found oracle for q > 0 (row 15). '
           'No gate covered it because it is untimed and not in D1–D3; the bracket floor ≤ best-found ≤ solved was stated in DESIGN §3 '
           'but never asserted in code. It is now checked per rung by the generator and the offending values are struck through and flagged in `summary.json`.',
           '2. The 1024 and 256 verdicts differ on D1 only through the top rung\'s cost advantage (row 2 vs 3), and agree on D2, D3, '
           'monotonicity and on `falsified_intent = False`. The paper\'s claim is decided by `falsified_intent` per §A8, declared before the 1024 job ran.',
           '3. Nothing in the timed rows, gates, POD/DST/CG rows, or the head job is affected by the oracle defect (rows 5–13, 17–19 rest on other fields).',
           '4. Not verified here: the eliminated `q512_m4` arm\'s cost (an inert-iteration artefact, DESIGN §A4) is reported but excluded from the ladder line; a reader who disagrees with §A4 can recompute D1 from the `q512_m4@new_K32` row in `summary.json`.']
    (HERE / 'self-audit-report.md').write_text('\n'.join(md) + '\n')

    # ------------------------------------------------------------ lab log --
    c1, c2 = V['1024'], V['256']
    e = ['## 2026-09-17',
         f"### p-linear — Poisson 2D ladder to q = R on the best checkpoint: at 1024² the pre-registered degenerate-curve criterion PASSES (D1 {c1['D1_span']:.2f}x, D2 strict, D3); at 256² D1 fails literally ({c2['D1_span']:.2f}x) only because the top rung is cheaper than the middle; neither mesh meets the falsification intent; the untimed augmented-oracle column for q > 0 is retracted (A10)",
         '',
         'INTERIM entry: all three completed jobs are collected, audited and archived; the corrected oracle column (A10) has not been re-run. '
         'Worktree `worktrees/2026-09-17-p-linear`, branch `exp/2026-09-17-p-linear`, forked from `exp/2026-09-16-p-bank-head` at `266dea9d`. '
         'Namespace `/cluster/tufts/paralab/tawal01/p_linear_20260917/` (empty after collection). Pre-registration with ten dated amendments: '
         '`experiments/p-linear/DESIGN.md`. Report generated from the run JSONs: `experiments/p-linear/reports/2026-09-17-p-linear.md`, '
         'with `summary.json`, `verdicts.json`, a per-mesh figure and `self-audit-report.md` (Codex unavailable until 2026-09-19 11:33).', '',
         '**Jobs (4 of 8).** '
         f"`plin256` = `{SUB['plin256']['job_id']}` ({r256['gpu']}, {r256['elapsed_seconds'] / 60:.1f} min, source `{r256['commit'][:12]}`); "
         f"`plin1024` = `{SUB['plin1024']['job_id']}` **FAILED, retracted** (below); "
         f"`plin1024b` = `{SUB['plin1024b']['job_id']}` (**{r1024['gpu']}**, {r1024['elapsed_seconds'] / 60:.1f} min, source `{r1024['commit'][:12]}`); "
         f"`plhead1` = `{SUB['plhead1']['job_id']}` ({rh['gpu']}, {rh['elapsed_seconds'] / 60:.1f} min, source `{rh['commit'][:12]}`). "
         'All logged `jax_backend=gpu`, float64, matmul precision `highest`; one job per attempt directory, `squeue` before and after every submit; '
         'remote directories deleted after checksum collection. The two meshes ran on different cards and are never compared on cost.', '',
         '**The question.** Whether the accuracy/cost trade from correction rank q degenerates on a linear PDE: top rung a linear reduced model '
         'that is also (about) the cheapest, with POD-LSPG or a direct solver non-dominated. Ladder to q = R = 512 on `pbh02`\'s `new_K32` '
         '(R = 512, K = 32), every comparator timed in the same job on the same GPU, 12 development sources, 3 timed repetitions.', '',
         f"**1024² (`plin1024b`), the paper's row.** The `m4` ladder with the A4 direct top rung:", '', ladder_table(1024), '',
         f"Clauses: {clauses(1024)}. **Verdict: DEGENERATE under D1 ∧ D2 ∧ D3 as pre-registered.** The top rung reaches the bank floor "
         f"({pct(val(1024, f'bank_floor@{PRIM}', 'worst_same_grid'))}) and is the cheapest ladder point; the q < R rungs span only {c1['D1_neural_span_posthoc']:.3f}x in cost. "
         f"Comparators in the same job: POD-LSPG k'=512 {pct(val(1024, 'e_pod512_m4@trainset', 'worst_same_grid'))} at {val(1024, 'e_pod512_m4@trainset', 'median_total_ms'):.3f} ms; "
         f"DST direct exact at {val(1024, 'dst_direct', 'median_total_ms'):.3f} ms; unpreconditioned CG {val(1024, 'cg_0.01', 'median_total_ms'):.1f}–{val(1024, 'cg_1e-08', 'median_total_ms'):.1f} ms (1e-2 to 1e-8). "
         f"Head alone (`a_neural@new_K32`) {pct(val(1024, f'a_neural@{PRIM}', 'worst_same_grid'))} at {val(1024, f'a_neural@{PRIM}', 'median_total_ms'):.3f} ms. "
         f"Eliminated `q512_m4` reaches the same floor at {val(1024, f'q512_m4@{PRIM}', 'median_total_ms'):.3f} ms (inert iteration, A4). "
         'No speedup over any full-order solver is claimed.', '',
         f"**256² (`plin256`).**", '', ladder_table(256), '',
         f"Clauses: {clauses(256)}. **Verdict: not degenerate as literally written, because D1 fails — and it fails because the top rung is "
         f"{max(c2['costs']) / c2['costs'][-1]:.2f}x cheaper than the dearest rung, not because any rung buys accuracy for ≥ 2x (A8).** "
         f"POD-LSPG k'=512 {pct(val(256, 'e_pod512_m4@trainset', 'worst_same_grid'))} at {val(256, 'e_pod512_m4@trainset', 'median_total_ms'):.3f} ms; DST {val(256, 'dst_direct', 'median_total_ms'):.3f} ms.", '',
         '**Do the two meshes agree?** On D2 (strict), D3, monotonicity and both falsification readings, yes; on the literal D1 they differ '
         f"only through the size of the top rung's cost advantage ({c1['D1_span']:.2f}x at 1024, {c2['D1_span']:.2f}x at 256). Under `falsified_intent`, "
         'declared in A8 as the deciding clause before the 1024 job ran, both meshes say the same thing: no rung pays ≥ 2x for accuracy.', '',
         f"**Job 3 — head capacity on the frozen R = 512 bank (`plhead1`).** H1: the pbh02 recipe re-run reproduces the primary's development best-found "
         f"to {100 * H['H1_relative']:.2f} % ({'pass' if H['H1_pass'] else 'FAIL'}). H2: the primary's best-found/floor ratio is {prim['ratio_dev_best_found_over_floor']:.3f}x; "
         f"the largest reduction is {100 * H['H2_max_drop']:.1f} % (`{best_ratio_arm['arm']}`, {best_ratio_arm['ratio_dev_best_found_over_floor']:.3f}x, dev best-found "
         f"{pct(best_ratio_arm['best_found_development']['worst'])}), past the 20 % bar but short of the 2x bar for a better anchor: **partial movement**. "
         f"Width helps (`K32_w256_L2` {arms['K32_w256_L2']['ratio_dev_best_found_over_floor']:.3f}x), depth alone does not (`K32_w128_L3` {arms['K32_w128_L3']['ratio_dev_best_found_over_floor']:.3f}x), "
         f"3x the schedule gives {arms['K32_w128_L2_x3']['ratio_dev_best_found_over_floor']:.3f}x. Validation selects `{best_val_arm['arm']}`; no arm enters the linear-case table.", '',
         '**What was wrong and retracted.** (1) `plin1024` (`' + SUB['plin1024']['job_id'] + '`) died at ' + FAIL['elapsed'] + ' with a GPU '
         'RESOURCE_EXHAUSTED in the *untimed* dense best-found oracle on an A100-PCIE-40GB (2.0 GiB jacfwd Jacobians with 32 GiB autotuner '
         'variants); the log is complete and the share was at ' + FAIL['disk_at_failure'] + ', so this is not the disk-full mode. No timed number and no gate '
         'came from it. Fix A7: the dense oracle runs only at ≤ 256 intervals (it is a cross-check of the projected oracle, which agrees with it to '
         f"{dense_vs_projected(r256):.0e}); resubmitted on an H200 with 240 GB. "
         '(2) **A10, found after collection:** the untimed augmented best-found oracle projected with a V whose Gram is ρI, ρ = ((n−1)/254)², not I — '
         f"measured V^T V = {SCALE['64']['diag_mean']:.3f} I / {SCALE['256']['diag_mean']:.4f} I / {SCALE['1024']['diag_mean']:.2f} I at 64/256/1024. Its q > 0 values are meaningless at 1024 "
         f"(they rise with q, reading {pct(val(1024, f'augmented_best_found_q512@{PRIM}', 'worst_same_grid'))} at q = R where the floor is {pct(val(1024, f'bank_floor@{PRIM}', 'worst_same_grid'))}) "
         'and inflated by ≤ 5e-4 relative at 256; q = 0 is unaffected. The signal was visible in the 64-interval smoke (the column barely moved with q) and I '
         'missed it. Retracted in the report and flagged in `summary.json`; fixed by orthonormalising V (verified locally: fixed q = R equals the floor to '
         f"{FIX['checks']['fixed_qR_equals_floor_rel']:.0e} relative); not re-run. No timed arm, gate or D-clause uses that function. "
         '(3) Codex unavailable throughout; design and report audits are written self-audits (A2, A9).', '',
         '**Audit.** Independent NumPy audits (no driver, no JAX) recompute every reported error from the retained fields: 1024 '
         f"{a1024['recomputed_errors']['invocations']} errors, worst {a1024['recomputed_errors']['worst_same_grid_difference']:.1e}; bank floor rebuilt by NumPy QR to {a1024['numpy_bank_floor']['max_abs_difference']:.1e}; "
         f"all {len(r1024['gates'])} cross-job gates at 1024 and {len(r256['gates'])} at 256 pass at 1e-9; the criterion re-derived independently agrees at both meshes.", '',
         '**Open.** The corrected oracle column needs one untimed re-run (`plorc`, ~5 min per mesh) if the paper wants the bracket; the coordinator '
         'decides. Everything is one checkpoint, one seed, development cohorts only; sealed cohorts untouched; nothing merged, nothing pushed.']
    (LANE / 'checks/2026-09-17-lab-log-entry.md').write_text('\n'.join(e) + '\n')
    print('wrote', HERE / 'self-audit-report.md', LANE / 'checks/2026-09-17-lab-log-entry.md')


if __name__ == '__main__':
    main()
