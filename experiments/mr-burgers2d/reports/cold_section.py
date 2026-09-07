"""Generate the independent cold-only diagnostic section from raw records."""
import itertools,json
from pathlib import Path
import numpy as np


def generate(path,pilot):
    path=Path(path);d=json.loads(path.read_text());audit=json.loads((path.parent.parent/'AUDIT.json').read_text())
    assert d['complete'] and d['checkpoint_sha256']==pilot['checkpoint_sha256'] and d['physical_cases']==pilot['physical_cases']
    meshes=list(map(int,d['config']['meshes'].split(',')));rules=['edge_gram','edge_qr','fixed_midpoint','fixed_gauss','full_qr']
    expected=set(itertools.product(meshes,rules,[60,180],[1,4],range(d['config']['cases'])))
    keyed={(r['intervals'],r['rule'],r['budget'],r['starts'],r['case']):r for r in d['rows']}
    assert len(d['rows'])==len(keyed)==len(expected) and set(keyed)==expected
    def group(L,rule,budget=180,starts=4):return [keyed[L,rule,budget,starts,c] for c in range(d['config']['cases'])]
    def worst(L,rule,metric='relative_full_grid_error',budget=180,starts=4):return max(r[metric] for r in group(L,rule,budget,starts))
    obs=min(meshes);budget=max(r['budget'] for r in d['rows']);starts=max(r['starts'] for r in d['rows'])
    nq=d['mesh_setup'][0]['quadrature_axis_points'];L=max(meshes)
    lines=['','## Frozen-bank initial-state fitting diagnostic','',
        f"A separate cold-only run, job `{d['job_id']}` at source `{d['commit']}`, retained the same checkpoint and all {d['config']['cases']} physical cases. It fitted initial states only. The complete-query table above still contains the original initializer, and no cost or error below is substituted into that table.",'',
        f"The original rule samples {nq} equally weighted positions per axis between the first and last interior grid nodes. Those physical coordinates move toward the walls with mesh refinement. The fixed midpoint and weighted Gauss–Legendre rules keep their physical coordinates fixed and charge bilinear interpolation from the supplied field. These rules fit field values, never a pointwise PDE residual. The edge QR control changes only the numerical factorization; full-grid QR is a dense diagnostic within the original learned bank, with no POD replacement.",'',
        f"The next table uses budget {budget} and {starts} starts for every rule. Entries are worst-case initial-field errors on each complete requested grid, normalized by that grid's initial-field norm. The unrestricted bank floor allows arbitrary linear feature coefficients and is less restrictive than the nonlinear head. Returned local fits are not stationary or globally optimal oracles.",'',
        '| Intervals | Edge Gram | Edge QR | Fixed midpoint | Fixed Gauss | Full-grid QR | Bank projection floor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mesh in meshes:
        vals=[worst(mesh,r) for r in rules]+[worst(mesh,'full_qr','unrestricted_bank_floor')]
        lines.append('| '+str(mesh)+' | '+' | '.join(f'{v:.8g}' for v in vals)+' |')
    gram_delta=max(abs(keyed[m,'edge_gram',b,s,c]['relative_full_grid_error']-keyed[m,'edge_qr',b,s,c]['relative_full_grid_error']) for m,b,s,c in itertools.product(meshes,[60,180],[1,4],range(d['config']['cases'])))
    gauss_qr=max(abs(keyed[L,'fixed_gauss',budget,starts,c]['relative_full_grid_error']-keyed[L,'full_qr',budget,starts,c]['relative_full_grid_error']) for c in range(d['config']['cases']))
    lines+=['',f"Across every control, changing Gram to QR on the same edge samples changes the full-grid error by at most {gram_delta:.6g}. At {L} intervals, the largest casewise difference between fixed Gauss and full-grid QR is {gauss_qr:.6g}. The sampled objective's coordinate/weight drift therefore contributes materially; Cholesky regularization does not explain the transfer failure. The growing unrestricted bank floor also shows genuine representation loss under the full-grid norm. The continuous hard-wall bank and clipped Gaussian initial data make additional near-wall nodes a plausible contributor, but this diagnostic does not isolate the spatial location or separate all nonlinear-head and local-optimization limitations.",'',
        f"The query benchmark observes the fixed {obs}-interval grid. It can miss errors at additional fine near-wall nodes. This is visible when the same returned fit is scored under both norms:",'',
        '| Intervals | Rule | Full-grid worst | Common-grid worst | Full-grid median |', '|---|---|---:|---:|---:|']
    for mesh,rule in itertools.product(meshes,['edge_gram','fixed_gauss','full_qr']):
        lines.append(f"| {mesh} | `{rule}` | {worst(mesh,rule):.8g} | {worst(mesh,rule,'relative_common_grid_error'):.8g} | {np.median([r['relative_full_grid_error'] for r in group(mesh,rule)]):.8g} |")
    lines+=['',f'Budget/start controls at the largest grid, {L} intervals, retain every case:', '',
        '| Rule | Iteration budget | Starts | Full-grid worst | Budget exits | Improvement stops | Failed stops |', '|---|---:|---:|---:|---:|---:|---:|']
    for rule,b,s in itertools.product(['edge_gram','fixed_gauss','full_qr'],[60,180],[1,4]):
        rows=group(L,rule,b,s);counts=[sum(r['stop_reason']==reason for r in rows) for reason in [0,2,3]]
        lines.append(f"| `{rule}` | {b} | {s} | {worst(L,rule,budget=b,starts=s):.8g} | "+' | '.join(map(str,counts))+' |')
    selected=[r for r in d['rows'] if r['budget']==budget and r['starts']==starts]
    lines+=['',f"The largest normalized gradient diagnostic among the selected fits is {max(r['relative_gradient'] for r in selected):.6g}; small-step/improvement stopping is not a stationarity certificate. Across all {len(d['rows'])} returned fits, {sum(r['stop_reason']==0 for r in d['rows'])} exhausted their budget and {sum(r['stop_reason']==3 for r in d['rows'])} failed. Longer budgets and more starts do not remove the worst-case objective bias or bank floor. Timings in the raw cold records contain one warmed call per case and are diagnostic only; no repeated cold-fit cost frontier is claimed.",'',
        f"The independent audit checked all {audit['declared_fits_verified']} declared fits, source/checkpoint hashes, unchanged cases and collection checksums. It recomputed each retained common-grid error with maximum discrepancy {audit['common_grid_error_recompute_max_difference']:.6g}; the incumbent cold fields agree with the original query run within {audit['original_pilot_cold_field_max_difference']:.6g}. Full-grid errors at finer resolutions and bank projection floors are in-job scalar diagnostics: the archive retains restricted fields, so those full norms are not independently reconstructed. The corrected initializer's evolution accuracy and complete-query latency remain unmeasured."]
    return lines
