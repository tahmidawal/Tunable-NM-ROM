"""Generate the corrected untimed-oracle supplement from audited retained records."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit', type=Path, help='Independent local audit; defaults to the immutable cluster audit.')
    a = ap.parse_args()
    raw_path = a.archive / 'output/result.json'
    audit_path = a.audit or a.archive / 'output/audit.json'
    d = json.loads(raw_path.read_text())
    audit = json.loads(audit_path.read_text())
    history = json.loads((a.archive / 'code/oracle-history.json').read_text())
    assert d['complete'] and not d['smoke'] and audit['passed'] and audit['full_bank_audit']
    assert audit['result_sha256'] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    original_checks = []
    lane = Path(__file__).resolve().parents[1]
    for intervals, attempt in ((256, 'plin256'), (1024, 'plin1024b')):
        original = lane / 'artifacts' / attempt / 'result.json'
        expected = history[str(intervals)]['result_sha256']
        actual = hashlib.sha256(original.read_bytes()).hexdigest()
        assert actual == expected, f'Original historical result changed: {original}'
        original_checks.append(dict(attempt=attempt, result_sha256=actual, unchanged=True))
    rows = []
    for p in d['panels']:
        n, mid = p['intervals'], p['model']
        old = next(x for x in history[str(n)]['reconstruction'] if x['model'] == mid)
        for rung in p['rungs']:
            q = rung['q']
            oldq = next(x for x in old['augmented'] if x['q'] == q)
            stationary = sum(q == p['R'] or x['reasons'][x['selected']] == 4 for x in rung['cases'])
            row = dict(intervals=n, model=mid, R=p['R'], K=p['K'], q=q,
                       original_retracted=q > 0, old_worst=oldq['best_found']['worst'],
                       original_starts_worst=max(x['original_starts_error'] for x in rung['cases']),
                       corrected_worst=max(x['error'] for x in rung['cases']),
                       corrected_median=float(__import__('numpy').median([x['error'] for x in rung['cases']])),
                       bank_floor_worst=max(p['bank_floor']), stationary=stationary,
                       cases=len(rung['cases']), per_case=[dict(
                           case=x['case'], error=x['error'],
                           original_starts_error=x['original_starts_error'],
                           direct_endpoint=q == p['R'],
                           selected_start=None if q == p['R'] else x['start_labels'][x['selected']],
                           stationary=q == p['R'] or x['reasons'][x['selected']] == 4,
                       ) for x in rung['cases']],
                       original_result_sha256=p['original_result_sha256'],
                       source_commit=d['source_commit'], job_id=d['job_id'])
            rows.append(row)
    lines = ['# Corrected Poisson augmented representation fits', '',
             'This supplement confirms the query-metric projector repair on the complete original development cohort. '
             'The corrected untimed fits have passed an independent NumPy field and gradient audit; historical timed results remain unchanged.', '',
             f"Source `{d['source_commit']}`; job `{d['job_id']}`; GPU `{d['gpu']}`. "
             f"Audit checked {audit['cases']} case/rank combinations. "
             f"Maximum independent decoded-field error discrepancy: {audit['maximum_defects']['full_field']:.6e}.", '',
             'Retained evidence: [complete raw-result archive](../artifacts/plorc01/README.md), '
             '[independent local audit](../artifacts/plorc01/audit.json), '
             '[source audit](../artifacts/plorc01/provenance-audit.json), '
             '[archive restore proof](../artifacts/plorc01/restore-audit.json), and '
             '[result/audit hashes and machine-readable tables](oracle-confirmation.json).', '',
             'The historical nonzero-rank oracle values remain retracted under DESIGN A10. '
             'The corrected values below come from a new run and have their own provenance. '
             'The original nearest-code starts and the additional safeguards are reported separately. '
             'Additional starts use previously retained online solutions at the same rank and the preceding rank’s best fit. '
             'These are truth-informed representation diagnostics, with no inference-time cost or generalization claim.', '',
             'For a query-mesh QR factor $R_G$ and an orthonormal basis $Q_q$ for '
             '$\\operatorname{span}(R_G C_q)$, the fitted residual is', '',
             '$$r_q(z) = (I-Q_qQ_q^\\top)(R_Gh_\\theta(z)-T).$$', '',
             'The finite best-found value is an upper bound on the unknown global minimum representation error. '
             'The free-bank floor is a lower bound. At $q=R$, a direct free-coefficient projection returns that floor, '
             'with the redundant nonlinear head removed; the original-starts column is therefore marked direct at that endpoint.', '']
    for mid in ['new_K32', 'incumbent']:
        lines.extend([f'## Frozen checkpoint `{mid}`', '',
                      '| Intervals | Correction rank | Original starts, worst (%) | Safeguarded best found, worst (%) | Median (%) | Bank floor, worst (%) | Stationary / cases |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in [x for x in rows if x['model'] == mid]:
            original_display = 'Direct endpoint' if r['q'] == r['R'] else f"{100*r['original_starts_worst']:.6f}"
            lines.append(f"| {r['intervals']} | {r['q']} | {original_display} | "
                         f"{100*r['corrected_worst']:.6f} | {100*r['corrected_median']:.6f} | "
                         f"{100*r['bank_floor_worst']:.6f} | {r['stationary']} / {r['cases']} |")
        lines.append('')
    lines.extend(['## Retraction history', '',
                  '| Intervals | Checkpoint | Rank | Historical value (%) | Historical status | Corrected value (%) |',
                  '| ---: | --- | ---: | ---: | --- | ---: |'])
    for r in rows:
        if r['q'] > 0:
            lines.append(f"| {r['intervals']} | `{r['model']}` | {r['q']} | {100*r['old_worst']:.6f} | "
                         f"Retracted: unnormalized query projector | {100*r['corrected_worst']:.6f} |")
    lines.extend(['', '## Glossary', '',
                  '- **Intervals**: subdivisions along each side of the square spatial mesh.',
                  '- **Checkpoint**: frozen neural bank and latent-to-coefficient head parameters.',
                  '- **Rank $R$**: number of learned bank functions; **latent dimension $K$**: number of head inputs.',
                  '- **Correction rank $q$**: number of additional nested linear bank directions allowed during fitting.',
                  '- **Worst / median**: largest / middle relative Euclidean field error over the original development sources, normalized by the same-grid exact finite-difference solution.',
                  '- **Original starts**: the unchanged nearest-training-code multistart prescription, with the corrected projector.',
                  '- **Safeguarded best found**: smallest achieved objective after adding feasible starts from retained solves and the previous correction rank; a finite optimizer result, not a proven global minimum.',
                  '- **Bank floor**: the smallest field error obtainable with unrestricted coefficients in the learned linear bank.',
                  '- **Stationary / cases**: cases meeting the recorded normalized-gradient stopping criterion, out of all evaluated sources; the direct full-bank endpoint is stationary by construction.',
                  '- **Query metric / QR factor**: the Euclidean field geometry induced by evaluating bank functions at the query mesh, represented by a triangular matrix.',
                  '- **Projector**: an operation that removes the component along the permitted linear correction directions.',
                  '- **Development cohort**: already opened source cases; these data do not form a new sealed final test.',
                  '- **Independent audit**: separate NumPy/SciPy equations checking saved fields, objectives, derivatives, summaries and numerical provenance.', ''])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / '2026-09-20-poisson-oracle-confirmation.md').write_text('\n'.join(lines))
    (a.out / 'oracle-confirmation.json').write_text(json.dumps(dict(
        source_commit=d['source_commit'], job_id=d['job_id'],
        result_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(), rows=rows,
        original_results=original_checks,
        old_timed_results_unchanged=all(x['unchanged'] for x in original_checks),
        global_minimum_proven=False), indent=2) + '\n')
    print(f'Generated {len(rows)} corrected rows.')


if __name__ == '__main__':
    main()
