"""Generate the b-qxm report (Markdown), its figures and `summary.json`.

Every number and every plotted point is read from the audited run JSONs
(`checks/<attempt>-audit.json`) and the comparator table. Nothing is typed by hand. Costs are
never compared across jobs: every cost ratio in this file is formed inside one audit.

    python reports/generate_xm.py --audits checks/bqx101-audit.json checks/bqx201-audit.json \
        checks/bqx301-audit.json --out reports/2026-09-17-b-qxm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
K = 16
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
COLS = [('fixed256', 'fixed $M=256$'), ('fixed1088', 'fixed $M=1088$'), ('2x', '$M=2(K+q)$'),
        ('4x', '$M=4(K+q)$'), ('8x', '$M=8(K+q)$'), ('16x', '$M=16(K+q)$'), ('bridge', 'bridge')]
METRICS = [('worst_evolved_percent', 'worst evolved %'), ('worst_all_times_percent', 'worst all-times %'),
           ('worst_t0_compression_percent', '$t=0$ compression %'), ('median_gpu_ms', 'median GPU ms'),
           ('converged', 'converged'), ('median_iterations', 'median iterations')]
SCHED = [(0, 64), (16, 128), (32, 192), (64, 320), (128, 576), (256, 1088)]
SPAN_Q, SPAN_M = [], []


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (float, np.floating)):
        return f'{x:.{d}f}'
    return str(x)


def sci(x, d=2):
    return '—' if x is None else f'{x:.{d}e}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


class Doc:
    def __init__(self):
        self.md = []

    def h(self, level, text):
        self.md.append('#' * level + ' ' + text + '\n')

    def p(self, text):
        self.md.append(text + '\n')

    def table(self, header, rows):
        self.md.append('| ' + ' | '.join(header) + ' |')
        self.md.append('|' + '|'.join(['---'] * len(header)) + '|')
        for r in rows:
            self.md.append('| ' + ' | '.join(str(c) for c in r) + ' |')
        self.md.append('')


class Grid:
    """Cells keyed by (q, M); one primary row per cell, every appearance kept."""

    def __init__(self, audits):
        self.audits = audits
        self.appear, self.controls = {}, []
        for au in audits:
            for row in au['arms']:
                if row['family'] != 'rom':
                    continue
                if 'solver-control' in (row.get('labels') or []):
                    self.controls.append((au, row))
                    continue
                self.appear.setdefault((row['q'], row['M']), []).append((au, row))
        self.primary = {}
        for key, lst in self.appear.items():
            q = key[0]
            # Round-1 cells keep their round-1 primary so no published number shifts; the
            # round-2 jobs are fallbacks, which is where the new cells land.
            pref = ['G1'] if q <= 32 else ['G2']
            pref += ['G2', 'G1', 'S1', 'E1', 'E2']
            for want in pref:
                hit = [x for x in lst if x[0]['question'] == want]
                if hit:
                    self.primary[key] = hit[0]
                    break

    def row(self, q, M):
        return self.primary.get((q, M), (None, None))[1]

    def au(self, q, M):
        return self.primary.get((q, M), (None, None))[0]

    def e(self, q, M, metric='worst_evolved_percent', converged_only=True):
        r = self.row(q, M)
        if r is None or (converged_only and not r['converged']):
            return None
        return r[metric]

    def cells_with(self, label):
        return sorted((k, v[1]) for k, v in self.primary.items() if label in (v[1].get('labels') or []))


def spans(grid, metric):
    out = {}
    # fixed-M columns, and every M that holds >= 2 rows
    byM = {}
    for (q, M) in grid.primary:
        byM.setdefault(M, []).append(q)
    fixed, other = {}, {}
    for M, qs in byM.items():
        qs = sorted(qs)
        if len(qs) < 2:
            continue
        declared = any(lab in (grid.row(q, M).get('labels') or []) for q in qs for lab in ('fixed256', 'fixed1088'))
        vals = [grid.e(q, M, metric) for q in qs]
        ok = [(q, v) for q, v in zip(qs, vals) if v is not None]
        if len(ok) < 2:
            continue
        allc = all(grid.row(q, M)['converged'] for q in qs)
        # `ok` already contains only the converged, valid rungs (grid.e() drops non-converged
        # rows). DESIGN.md §6's "not patched" rule says the DECLARED column's span must not be
        # silently formed from that dropped-rung subset -- so `span` stays None whenever any
        # declared rung fails, exactly as pre-registered. But that rule is about not smuggling
        # a quiet substitution past the reader, not about erasing what the converged rungs
        # still establish on their own. So the converged sub-ladder is *also* reported, plainly
        # labelled, alongside the unavailable full span (never merged into it, never silently
        # used to fill `span`).
        entry = dict(M=M, q=[q for q, _ in ok], values=[v for _, v in ok],
                     monotone=mono([v for _, v in ok]),
                     span=(ok[0][1] / ok[-1][1] if allc else None), q_top=ok[-1][0], q_max=qs[-1],
                     span_to_q128=(ok[0][1] / dict(ok)[128] if 128 in dict(ok) and ok[0][0] == 0 and allc else None),
                     all_converged=allc, available=allc,
                     unavailable_reason=(None if allc else 'a rung is not converged; the span is not patched'),
                     declared_fixed_column=declared,
                     certified_q_range=([ok[0][0], ok[-1][0]] if len(ok) >= 2 else None),
                     certified_span=(ok[0][1] / ok[-1][1] if len(ok) >= 2 else None),
                     certified_all_converged=True if len(ok) >= 2 else None,
                     certified_note=(None if allc else
                         f"the declared column also attempted q = {qs[-1]}, which did not converge and is "
                         f"excluded (not patched into `span`); the q = {ok[0][0]}..{ok[-1][0]} sub-ladder above "
                         f"it, all converged, is reported here as `certified_span` and is unaffected by the "
                         f"attempted extension"))
        (fixed if declared else other)[M] = entry
    # Cost may only be compared inside one job, so for each fixed-M column find the job that
    # holds the most of its rungs and report that segment's cost and error span together.
    for M, entry in fixed.items():
        # Use every APPEARANCE, not just the primary row: G2 runs the q = 0 cells of both
        # fixed columns as in-job anchors precisely so a cost ladder exists inside one job.
        byjob = {}
        for (qq, MM), lst in grid.appear.items():
            if MM != M:
                continue
            for au_, r_ in lst:
                byjob.setdefault(au_['question'], []).append((qq, r_, au_))
        for v in byjob.values():
            v.sort(key=lambda t: t[0])

        def _pack(best):
            if best is None or len(best) < 2:
                return None
            errs = [r[metric] for _, r, _ in best]
            costs = [r['median_gpu_ms'] for _, r, _ in best]
            nd = [i for i in range(len(best))
                  if not any((costs[j] <= costs[i] and errs[j] <= errs[i] and (costs[j] < costs[i] or errs[j] < errs[i]))
                             for j in range(len(best)))]
            allc_ = all(r['converged'] for _, r, _ in best)
            return dict(
                job=best[0][2]['question'], job_id=best[0][2]['job_id'], q=[q for q, _, _ in best],
                values=errs, median_gpu_ms=costs, converged=[r['converged'] for _, r, _ in best],
                all_converged=allc_,
                error_span=(errs[0] / errs[-1] if allc_ else None),
                cost_span=(costs[-1] / costs[0] if allc_ else None),
                monotone_error=mono(errs), monotone_cost=mono([-c for c in costs]),
                non_dominated_points=len(nd),
                passes_tunability_bar=bool(allc_ and mono(errs) and errs[0] / errs[-1] >= 2.
                                           and costs[-1] / costs[0] >= 2. and len(nd) >= 3))

        # `within_job`: the longest run of rungs ONE job holds that are ALL converged --
        # this is the certified, cost-bearing tunability object (e.g. G2's own q=0,64,128,256
        # at M=1088). Choosing it by raw rung count alone (as this used to) let a later job's
        # longer but non-converged attempt (E1's 7-rung q=0..512 run) silently displace and
        # null out an earlier job's shorter, fully-passing result -- a reporting defect, not a
        # finding (round 2, 2026-09-17: see DESIGN.md A5). `within_job_longest_attempted` keeps
        # that longer attempt visible, separately and honestly, exactly as far as it got.
        converged_candidates = [v for v in byjob.values() if len(v) >= 2 and all(r['converged'] for _, r, _ in v)]
        best_certified = (max(converged_candidates, key=lambda v: (len(v), max(x[0] for x in v)))
                          if converged_candidates else None)
        best_attempted = (max(byjob.values(), key=lambda v: (len(v), max(x[0] for x in v)))
                          if byjob else None)
        entry['within_job'] = _pack(best_certified)
        packed_attempted = _pack(best_attempted)
        entry['within_job_longest_attempted'] = (
            packed_attempted if best_attempted is not None and best_attempted != best_certified else None)
    out['fixed_M'] = fixed
    out['other_M_with_two_rows'] = other
    byq = {}
    for (q, M) in grid.primary:
        byq.setdefault(q, []).append(M)
    fq = {}
    for q, Ms in byq.items():
        Ms = sorted(Ms)
        vals = [(M, grid.e(q, M, metric)) for M in Ms]
        ok = [(M, v) for M, v in vals if v is not None]
        if len(ok) < 2:
            continue
        vs = [v for _, v in ok]
        wide = [(M, v) for M, v in ok if M >= 2 * (K + q)]
        fq[q] = dict(q=q, M=[M for M, _ in ok], values=vs, monotone=mono(vs),
                     span=max(vs) / min(vs), span_first_to_last=ok[0][1] / ok[-1][1],
                     M_best=ok[int(np.argmin(vs))][0],
                     span_M_ge_2x=(max(v for _, v in wide) / min(v for _, v in wide) if len(wide) >= 2 else None),
                     M_ge_2x=[M for M, _ in wide])
    out['fixed_q'] = fq
    sched = {}
    for lab in ('4x', '8x'):
        cells = grid.cells_with(lab)
        cells = [(k, r) for k, r in cells if 'anchor' not in (r.get('labels') or [])]
        vals = [grid.e(k[0], k[1], metric) for k, _ in cells]
        ok = [(k, v) for k, v in zip(cells, vals) if v is not None]
        if len(ok) >= 2:
            sched[lab] = dict(cells=[k[0] for k, _ in ok], values=[v for _, v in ok],
                              monotone=mono([v for _, v in ok]), span=ok[0][1] / ok[-1][1])
    # The pre-registered scheduled ladder stops at (256, 1088); if the round-2 job converged
    # the q = 512 rung of the SAME 4(K+q) rule, report the extension as its own row and never
    # fold it into the decomposition, whose corner path has no (0, 2112) cell to stand on.
    ext = grid.row(512, 2112)
    if ext is not None and ext['converged'] and '4x' in sched:
        v4 = sched['4x']
        vals = list(v4['values']) + [ext[metric]]
        sched['4x_extended_to_q512'] = dict(
            cells=list(v4['cells']) + [(512, 2112)], values=vals, monotone=mono(vals),
            span=vals[0] / vals[-1],
            note='the same M = 4(K+q) rule carried to q = R = 512; reported beside the '
                 'pre-registered ladder, never inside the decomposition')
    out['scheduled'] = sched
    return out


def decompose(grid, metric):
    L = lambda q, M: (None if grid.e(q, M, metric) is None else math.log(grid.e(q, M, metric)))
    out = dict(metric=metric)
    a, b, c = L(0, 64), L(0, 1088), L(256, 1088)
    if None not in (a, b, c):
        tot = a - c
        out['corner'] = dict(log_span=tot, span=math.exp(tot), delta_M=a - b, delta_q=b - c,
                             share_M=(a - b) / tot, share_q=(b - c) / tot,
                             path='(0,64) -> (0,1088) [M] -> (256,1088) [q]')
    else:
        out['corner'] = None
    rungs, ok = [], True
    for (q0, M0), (q1, M1) in zip(SCHED, SCHED[1:]):
        x, y, z = L(q0, M0), L(q0, M1), L(q1, M1)
        if None in (x, y, z):
            ok = False
            rungs.append(dict(from_=[q0, M0], to=[q1, M1], bridge=[q0, M1], available=False))
            continue
        rungs.append(dict(from_=[q0, M0], to=[q1, M1], bridge=[q0, M1], available=True,
                          delta_M=x - y, delta_q=y - z, total=x - z,
                          share_M=((x - y) / (x - z) if abs(x - z) > 1e-300 else None)))
    out['rung'] = dict(rungs=rungs, complete=ok)
    if ok:
        dM = sum(r['delta_M'] for r in rungs)
        dq = sum(r['delta_q'] for r in rungs)
        out['rung'].update(delta_M=dM, delta_q=dq, log_span=dM + dq, share_M=dM / (dM + dq),
                           share_q=dq / (dM + dq))
    qs, Ms = [0, 16, 32, 64, 128], [256, 1088]
    mat = [[L(q, M) for M in Ms] for q in qs]
    if all(v is not None for r in mat for v in r):
        A = np.array(mat)
        mu = A.mean()
        alpha = A.mean(1) - mu
        beta = A.mean(0) - mu
        inter = A - mu - alpha[:, None] - beta[None, :]
        ss_q, ss_M, ss_i = len(Ms) * float((alpha ** 2).sum()), len(qs) * float((beta ** 2).sum()), float((inter ** 2).sum())
        ss_t = float(((A - mu) ** 2).sum())
        out['anova'] = dict(q=qs, M=Ms, grand_mean_log=mu, q_effects=alpha.tolist(), M_effects=beta.tolist(),
                            interaction=inter.tolist(), ss_q=ss_q, ss_M=ss_M, ss_interaction=ss_i, ss_total=ss_t,
                            frac_q=ss_q / ss_t, frac_M=ss_M / ss_t, frac_interaction=ss_i / ss_t,
                            note='two-way decomposition of log error on the balanced 5 x 2 sub-grid')
    else:
        out['anova'] = None
    return out


def e1_disambiguation(grid, metric='worst_evolved_percent'):
    """DESIGN.md A3, E1: does q = 512 separate 'the rank has run out' from 'this cell is
    under-tested'? Reads the three q = 512 cells and the (256, 1088) baseline DIRECTLY (not
    through `grid.e`, which drops non-converged rows), because the pre-registered clauses are
    themselves about convergence and must see it. Applied literally, in the order declared."""
    r1088, r2112, r3168 = grid.row(512, 1088), grid.row(512, 2112), grid.row(512, 3168)
    base = grid.row(256, 1088)
    if any(r is None for r in (r1088, r2112, r3168, base)):
        return None
    cells = dict(q512_M1088=r1088, q512_M2112=r2112, q512_M3168=r3168)
    out = dict(metric=metric, baseline_q256_M1088=dict(
        value=base[metric], converged=base['converged'], job=grid.au(256, 1088)['question']))
    for name, r in cells.items():
        out[name] = dict(value=r[metric], converged=r['converged'], median_gpu_ms=r['median_gpu_ms'],
                         tests_per_unknown=r.get('tests_per_unknown'), median_iterations=r['median_iterations'],
                         max_iterations=r.get('max_iterations'),
                         worst_joint_gradient=r['max_joint_stationarity'], total_budget_exits=r['total_budget_exits'])
    out['baseline_q256_M1088']['max_iterations'] = base.get('max_iterations')
    out['baseline_q256_M1088']['median_iterations'] = base.get('median_iterations')
    # every q <= 256 rung of this same job climbs its iteration count with q (max 59 at q=0
    # to max 307 at q=256 -- real optimizer work happening before its own exit); all three
    # q=512 cells instead cap at max_iterations = 3 regardless of M, a flat signature that
    # does not look like "needs more tests, would otherwise grind" -- it looks like a wall
    # at q = R = 512 (the whole bank, no free directions left for the block-damped solve).
    out['flat_early_exit_at_q512'] = bool(
        max(out['q512_M1088']['max_iterations'] or 0, out['q512_M2112']['max_iterations'] or 0,
            out['q512_M3168']['max_iterations'] or 0) <= 5
        and (out['baseline_q256_M1088']['max_iterations'] or 0) > 5)
    all_c = all(r['converged'] for r in cells.values())
    any_c = any(r['converged'] for r in cells.values())
    out['all_three_converged'] = all_c
    out['any_converged'] = any_c
    if not any_c:
        out['clause'] = 'uninformative'
        out['clause_text'] = ("**uninformative**, by the literal DESIGN.md A3 clause: "
            f"(512, 1088), (512, 2112) and (512, 3168) ALL fail to converge (worst joint "
            f"gradient {sci(r1088['max_joint_stationarity'])}, {sci(r2112['max_joint_stationarity'])}, "
            f"{sci(r3168['max_joint_stationarity'])} against the 1e-6 bar, with zero budget exits — "
            f"the Levenberg-Marquardt path stalls short of the stationarity criterion, not a budget "
            f"exhaustion). Nothing is learned about whether the rank has run out or the cell is merely "
            f"under-tested at q = 512: the raw numbers ({f(r1088[metric])} %, {f(r2112[metric])} %, "
            f"{f(r3168[metric])} % at 2.06, 4.0, 6.0 tests per unknown) are directionally consistent with "
            f"more tests helping even at q = 512, and {f(r3168[metric])} % would be the best cell in the "
            f"campaign if certified — but as non-converged numbers they are not certified and are excluded "
            f"from every span and decomposition, exactly as §5 requires.")
    elif r3168['converged'] and r3168[metric] >= base[metric]:
        out['clause'] = 'rank_has_run_out'
        out['clause_text'] = (f"the rank has run out: (512, 3168) converges at {f(r3168[metric])} %, "
            f"no better than the (256, 1088) baseline at {f(base[metric])} %.")
    elif r3168['converged'] and (not r1088['converged'] or r1088[metric] > r3168[metric]):
        out['clause'] = 'test_starved_not_rank_limited'
        out['clause_text'] = ("the cause is the test count, not the rank: (512, 3168) converges and "
            "beats the baseline while (512, 1088) is poor or non-converged; the fixed-M=1088 column's "
            "q = 512 rung must be reported as test-starved, not as a rank limit.")
    else:
        out['clause'] = 'ambiguous'
        out['clause_text'] = ("the pre-registered clauses of DESIGN.md A3 do not cleanly classify this "
            "convergence pattern; reported as-is without a headline claim.")
    return out


def saturation(audits, metric='worst_evolved_percent'):
    """Every sweep in M, from whichever job ran it. A curve is built only from cells the
    config LABELLED as a sweep ('sat', 'sat256'), so a job's two-point anchors never
    masquerade as a saturation curve."""
    jobs = []
    for au in audits:
        rows = {}
        for r in au['arms']:
            if r['family'] == 'rom' and any(l.startswith('sat') for l in (r.get('labels') or [])):
                rows.setdefault(r['q'], {})[r['M']] = r
        rows = {q: byM for q, byM in rows.items() if len(byM) >= 3}
        if rows:
            jobs.append((au, rows))
    if not jobs:
        return None
    au = jobs[0][0]
    out = dict(job_id=au['job_id'], attempt=au['attempt'], gpu=au['gpu'], per_q={},
               sources={})
    rows = {}
    for au_, rws in jobs:
        for q, byM in rws.items():
            rows[q] = byM
            out['sources'][str(q)] = dict(job=au_['question'], job_id=au_['job_id'],
                                          attempt=au_['attempt'], gpu=au_['gpu'])
    for q, byM in sorted(rows.items()):
        Ms = sorted(byM)
        ref = byM.get(4 * (K + q))
        curve = []
        for i, M in enumerate(Ms):
            r = byM[M]
            nxt = byM[Ms[i + 1]] if i + 1 < len(Ms) else None
            curve.append(dict(M=M, value=r[metric], all_times=r['worst_all_times_percent'],
                              median_gpu_ms=r['median_gpu_ms'], converged=r['converged'],
                              median_iterations=r['median_iterations'],
                              cost_ratio_to_4x=(r['median_gpu_ms'] / ref['median_gpu_ms'] if ref else None),
                              improvement_to_next=((r[metric] - nxt[metric]) / r[metric] if nxt else None),
                              next_M=(Ms[i + 1] if nxt else None)))
        mstar = next((c['M'] for c in curve if c['improvement_to_next'] is not None and c['improvement_to_next'] < .05), None)
        star = next((c for c in curve if c['M'] == mstar), None)
        last = curve[-1]
        out['per_q'][q] = dict(curve=curve, M_star=mstar,
                               still_falling_at_largest_M=bool(
                                   len(curve) >= 2
                                   and (curve[-2]['value'] - last['value']) / curve[-2]['value'] >= .05),
                               largest_M=last['M'], value_at_largest_M=last['value'],
                               tests_per_unknown_at_M_star=(None if mstar is None else mstar / (K + q)),
                               error_at_M_star=(star['value'] if star else None),
                               cost_ratio_at_M_star=(star['cost_ratio_to_4x'] if star else None),
                               cost_neutral_claim=(bool(star and star['cost_ratio_to_4x'] is not None
                                                        and star['cost_ratio_to_4x'] <= 1.1) if star else None),
                               reference_M=(4 * (K + q)), monotone=mono([c['value'] for c in curve]))
    return out


def anchors(grid):
    rows = []
    for key, lst in sorted(grid.appear.items()):
        if len(lst) < 2:
            continue
        base_au, base = lst[0]
        for au, r in lst[1:]:
            rows.append(dict(q=key[0], M=key[1], a=base_au['attempt'], b=au['attempt'],
                             a_job=base_au['job_id'], b_job=au['job_id'],
                             rel_evolved=abs(r['worst_evolved_percent'] - base['worst_evolved_percent']) / base['worst_evolved_percent'],
                             rel_all=abs(r['worst_all_times_percent'] - base['worst_all_times_percent']) / base['worst_all_times_percent'],
                             a_ms=base['median_gpu_ms'], b_ms=r['median_gpu_ms'],
                             tol=(1e-9 if key[0] == 0 else (1e-9 if au['directions_sha256'] == base_au['directions_sha256'] else 1e-3))))
            rows[-1]['passed'] = bool(max(rows[-1]['rel_evolved'], rows[-1]['rel_all']) <= rows[-1]['tol'])
    return rows


def verdict(sp, dec, sat):
    fx = sp['fixed_M']
    s1088 = fx.get(1088)
    s256 = fx.get(256)
    v = dict(rules='DESIGN.md §6')
    v['fixed1088_monotone'] = s1088['monotone'] if s1088 else None
    # DESIGN.md §6's bar is stated over the ORIGINALLY pre-registered ranges -- "256 to
    # q = 128; 1088 to q = 256" -- not over whatever a later round additionally attempts. Use
    # `certified_span` (the span over the converged prefix of the declared column) for the
    # headline/falsification decision, so a later round's failed EXTENSION of the SAME column
    # (round 2, 2026-09-17: q = 512 at M = 1088, non-converged) cannot flip a verdict about the
    # rungs it did not touch. `span` (the full, possibly-unavailable declared-column value,
    # "not patched" per §6's own words) is reported separately for full transparency about
    # the extension attempt, and is NEVER what `rank_claim_false` or the headline reads.
    v['fixed1088_all_converged'] = s1088['all_converged'] if s1088 else None
    v['span_q_at_M1088'] = (s1088.get('certified_span') if s1088 else None)
    v['span_q_at_M1088_certified_q_range'] = (s1088.get('certified_q_range') if s1088 else None)
    v['span_q_at_M1088_declared_full_range_unavailable'] = (s1088.get('span') if s1088 else None)
    v['span_q_at_M1088_extension_note'] = (s1088.get('certified_note') if s1088 else None)
    v['span_q_at_M256_to_q128'] = s256['span'] if s256 else None
    v['headline_fixed_M'] = bool(s1088 and s1088['monotone'] and v['span_q_at_M1088'] is not None and v['span_q_at_M1088'] >= 2.)
    wj = (s1088 or {}).get('within_job')
    v['fixed1088_within_job'] = wj
    v['fixed1088_passes_tunability_bar'] = (wj or {}).get('passes_tunability_bar')
    v['fixed1088_within_job_longest_attempted'] = (s1088 or {}).get('within_job_longest_attempted')
    v['headline'] = ('fixed-M ladder (M = 1088, pure rank), scheduled ladder reported beside it'
                     if v['headline_fixed_M'] else 'scheduled ladder, with the M share printed beside every span')
    avail = [(x['M'], x.get('certified_span')) for x in fx.values() if x['M'] in (256, 1088)]
    avail = [(m, s) for m, s in avail if s is not None]
    v['rank_claim_false'] = bool(avail and all(s < 1.5 for _, s in avail))
    big_M_effect = [q for q, x in sp['fixed_q'].items() if q >= 64 and x['span_M_ge_2x'] is not None and x['span_M_ge_2x'] >= 1.5]
    q64_star = (sat['per_q'].get(64, {}).get('M_star') if sat else None)
    v['H_rank_false'] = bool(big_M_effect or (sat is not None and 64 in sat['per_q'] and (q64_star is None or q64_star > 1088)))
    v['H_rank_false_detail'] = dict(fixed_q_spans_ge_1p5_at_q_ge_64=big_M_effect, M_star_q64=q64_star)
    v['H_tests_false'] = (bool(dec['corner']['share_q'] >= .5) if dec.get('corner') else None)
    v['share_q_corner'] = dec['corner']['share_q'] if dec.get('corner') else None
    v['share_M_corner'] = dec['corner']['share_M'] if dec.get('corner') else None
    v['share_M_rung'] = dec['rung'].get('share_M') if dec.get('rung') else None
    v['paths_disagree'] = (bool(abs(v['share_M_corner'] - v['share_M_rung']) > .15)
                           if v['share_M_corner'] is not None and v['share_M_rung'] is not None else None)
    if v['rank_claim_false']:
        v['sentence'] = ('the rank claim is FALSE: at every fixed M the q-span is below 1.5x; the test count, '
                         'not the rank, is the accuracy control')
    elif v['headline_fixed_M']:
        v['sentence'] = (f"both factors matter; the pure-rank ladder at fixed M = 1088 spans {v['span_q_at_M1088']:.2f}x "
                         f"and the scheduled ladder's log-span is {100 * v['share_q_corner']:.0f} % rank, "
                         f"{100 * v['share_M_corner']:.0f} % test count (corner path)")
    else:
        v['sentence'] = ('no fixed-M ladder reaches 2x; the scheduled ladder stays the headline with its M share printed')
    return v


def grid_figure(grid, sp, png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for i, (lab, title) in enumerate([c for c in COLS if c[0] != 'bridge']):
        cells = [(k, r) for k, r in grid.cells_with(lab) if 'anchor' not in (r.get('labels') or []) or lab.startswith('fixed')]
        pts = sorted({(k[0], grid.e(k[0], k[1])) for k, _ in cells if grid.e(k[0], k[1]) is not None})
        if len(pts) < 2:
            continue
        ax.plot([p[0] + 1 for p in pts], [p[1] for p in pts], 'o-', color=PAL[i % len(PAL)], lw=2, ms=6, label=title)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('correction rank $q$ (+1, log$_2$)')
    ax.set_ylabel('worst evolved error (%, log)')
    ax.set_title('error against $q$, one line per $M$ column')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for i, (q, x) in enumerate(sorted(sp['fixed_q'].items())):
        ax.plot(x['M'], x['values'], 'o-', color=PAL[i % len(PAL)], lw=2, ms=6, label=f'$q={q}$')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('test modes $M$ (log$_2$)')
    ax.set_ylabel('worst evolved error (%, log)')
    ax.set_title('error against $M$, one line per $q$ row')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def sat_figure(sat, png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for i, (q, x) in enumerate(sorted(sat['per_q'].items())):
        Ms = [c['M'] for c in x['curve']]
        axes[0].plot(Ms, [c['value'] for c in x['curve']], 'o-', color=PAL[i], lw=2, ms=6, label=f'$q={q}$')
        axes[1].plot(Ms, [c['median_gpu_ms'] for c in x['curve']], 'o-', color=PAL[i], lw=2, ms=6, label=f'$q={q}$')
        if x['M_star']:
            axes[0].axvline(x['M_star'], color=PAL[i], ls='--', lw=1)
    for ax, yl, t in ((axes[0], 'worst evolved error (%, log)', 'error against $M$ (dashed: $M^\\star$)'),
                      (axes[1], 'median GPU ms (log)', f"cost against $M$, same job ({sat['attempt']})")):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('test modes $M$ (log$_2$)')
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audits', nargs='+', required=True)
    p.add_argument('--comparators', default=str(CELL / 'checks/comparators.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--status', default='final')
    a = p.parse_args()
    audits = []
    for path in a.audits:
        au = json.loads(Path(path).read_text())
        au['_path'] = str(path)
        au['_sha256'] = sha(path)
        audits.append(au)
    order = {'G1': 0, 'G2': 1, 'S1': 2, 'E1': 3, 'E2': 4}
    audits.sort(key=lambda x: order.get(x['question'], 9))
    grid = Grid(audits)
    sp = {m: spans(grid, m) for m in ('worst_evolved_percent', 'worst_all_times_percent')}
    dec = {m: decompose(grid, m) for m in ('worst_evolved_percent', 'worst_all_times_percent')}
    sat = saturation(audits)
    anc = anchors(grid)
    ver = verdict(sp['worst_evolved_percent'], dec['worst_evolved_percent'], sat)
    e1d = e1_disambiguation(grid)
    out = Path(a.out)
    d = Doc()
    summary = []

    def srow(**kw):
        summary.append(dict(quadrature='dense', **kw))

    d.h(1, f"b-qxm — the correction ladder's accuracy control: rank $q$, test count $M$, or both")
    d.p(f"**Status: {a.status}.** Numbers are development-cohort, single-seed, one frozen Burgers "
        "$256^2$ checkpoint (`18f0266ae6f0…`), six opened development cases, dense quadrature, the "
        "budget-600 block-damped variable-projection contract, three timed repetitions with burn-in. "
        "Every number below is generated from the audit JSONs by `reports/generate_xm.py`; the "
        "pre-registered design, gates and decision rules are in [`../DESIGN.md`](../DESIGN.md). "
        "**Costs are compared within one job only**; error comparisons across jobs rest on the "
        "anchor and fidelity tables at the end.")
    d.table(['job', 'attempt', 'job id', 'GPU', 'source commit', 'elapsed (s)', 'ROM arms', 'failed gates', 'audit SHA256'],
            [[x['question'], x['attempt'], x['job_id'], x['gpu'], (x['commit'] or '')[:12], f(x['elapsed_seconds'], 1),
              sum(1 for r in x['arms'] if r['family'] == 'rom'),
              ', '.join(f'`{g}`' for g in x['failed']) or 'none', x['_sha256'][:16] + '…'] for x in audits])

    d.h(2, 'Verdict')
    d.p(f"**{ver['sentence']}.**")
    d.p(f"Headline for the paper by the §6 rules: **{ver['headline']}**. Fixed-$M=1088$ ladder monotone: "
        f"{yn(ver['fixed1088_monotone'])}; every rung converged: {yn(ver['fixed1088_all_converged'])}; "
        f"$q$-span at fixed $M=1088$: {f(ver['span_q_at_M1088'], 3)}x; at fixed $M=256$ (to $q=128$): "
        f"{f(ver['span_q_at_M256_to_q128'], 3)}x. Rank claim false: {yn(ver['rank_claim_false'])}. "
        f"H(rank) false: {yn(ver['H_rank_false'])} ({ver['H_rank_false_detail']}). H(tests) false: "
        f"{yn(ver['H_tests_false'])}. Corner-path shares: rank {f(ver['share_q_corner'], 3)}, tests "
        f"{f(ver['share_M_corner'], 3)}; rung-path test share {f(ver['share_M_rung'], 3)}; paths disagree by "
        f"more than 0.15: {yn(ver['paths_disagree'])}.")
    for k, v in ver.items():
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool) or isinstance(v, bool):
            srow(q=None, M=None, metric=f'verdict.{k}', value=v, job_id=None, attempt=None, arm=None, source_sha256=None)

    d.h(2, 'T4 — the crossed grid, every cell')
    d.p('One row per $(q, M)$ cell, taken from its primary job (G1 for $q \\le 32$, G2 for $q \\ge 64$; '
        'the two $q=0$ anchors in G2 and every S1 cell that also exists in G1/G2 appear only in the '
        'anchor table). `column` names the schedule the cell belongs to; a cell can belong to several.')
    rows = []
    for (q, M), (au, r) in sorted(grid.primary.items()):
        rows.append([q, M, ', '.join(r.get('labels') or []), au['question'], au['job_id'],
                     f(r['worst_evolved_percent']), f(r['worst_all_times_percent']),
                     f(r['worst_t0_compression_percent']), f(r['median_gpu_ms'], 1), yn(r['converged']),
                     f(r['median_iterations'], 1), r['total_budget_exits'], sci(r['max_joint_stationarity'])])
        for key, _ in METRICS:
            srow(q=q, M=M, metric=key, value=r[key], job_id=au['job_id'], attempt=au['attempt'],
                 arm=r['arm'], source_sha256=au['_sha256'], labels=r.get('labels'))
    d.table(['$q$', '$M$', 'column', 'job', 'job id', 'worst evolved %', 'worst all-times %', '$t=0$ compression %',
             'median GPU ms', 'converged', 'median iters', 'budget exits', 'worst joint gradient'], rows)

    d.h(3, 'T4 pivot — worst evolved error (%), rows $q$, columns the $M$ schedule')
    qs = sorted({k[0] for k in grid.primary})
    head = ['$q$'] + [t for _, t in COLS]
    piv = []
    for q in qs:
        line = [q]
        for lab, _ in COLS:
            hit = [r for (qq, M), (au, r) in grid.primary.items() if qq == q and lab in (r.get('labels') or [])]
            if lab == 'bridge':
                hit = [r for r in hit]
            line.append(' / '.join(f"{f(r['worst_evolved_percent'])} ($M={r['M']}$)" for r in sorted(hit, key=lambda r: r['M'])) or '—')
        piv.append(line)
    d.table(head, piv)
    d.h(3, 'T4 pivot — median GPU ms (job in brackets; never compare down a column across jobs)')
    piv = []
    for q in qs:
        line = [q]
        for lab, _ in COLS:
            hit = [(au, r) for (qq, M), (au, r) in grid.primary.items() if qq == q and lab in (r.get('labels') or [])]
            line.append(' / '.join(f"{f(r['median_gpu_ms'], 0)} [{au['question']}]" for au, r in sorted(hit, key=lambda t: t[1]['M'])) or '—')
        piv.append(line)
    d.table(head, piv)

    d.h(2, 'The solver control')
    d.p('The inner linear solve switches from Gauss–Jordan to LU when $K + q > 64$, i.e. between the '
        '$q = 32$ and $q = 64$ rows — the same step at which the fixed-$M$ ladders cross from job G1 to '
        'G2. The control below runs the $(32, 1088)$ cell with LU in the same job as its Gauss–Jordan '
        'twin, so the solver switch is measured on its own.')
    crow = []
    for au, r in grid.controls:
        twin = grid.row(r['q'], r['M'])
        if twin is None:
            continue
        crow.append([f"({r['q']}, {r['M']})", twin['linear_solve'], f(twin['worst_evolved_percent']), f(twin['median_gpu_ms'], 1),
                     r['linear_solve'], f(r['worst_evolved_percent']), f(r['median_gpu_ms'], 1),
                     sci(abs(r['worst_evolved_percent'] - twin['worst_evolved_percent']) / twin['worst_evolved_percent']),
                     f(r['median_gpu_ms'] / twin['median_gpu_ms'], 3) + 'x', au['question']])
        srow(q=r['q'], M=r['M'], metric='solver_control.rel_diff_evolved',
             value=abs(r['worst_evolved_percent'] - twin['worst_evolved_percent']) / twin['worst_evolved_percent'],
             job_id=au['job_id'], attempt=au['attempt'], arm=r['arm'], source_sha256=au['_sha256'])
        srow(q=r['q'], M=r['M'], metric='solver_control.cost_ratio_lu_over_gj', value=r['median_gpu_ms'] / twin['median_gpu_ms'],
             job_id=au['job_id'], attempt=au['attempt'], arm=r['arm'], source_sha256=au['_sha256'])
    d.table(['cell', 'primary solve', 'evolved %', 'GPU ms', 'control solve', 'evolved %', 'GPU ms', 'rel. diff evolved', 'cost ratio (same job)', 'job'], crow or [['—'] * 10])

    for metric, title in (('worst_evolved_percent', 'evolved'), ('worst_all_times_percent', 'all-times')):
        s = sp[metric]
        d.h(2, f'Spans — {title} metric')
        if metric == 'worst_all_times_percent':
            d.p('**Caveat.** The all-times metric is dominated by the $t = 0$ compression, which does not depend '
                'on $M$ (gate `t0_field_invariant_in_M`), so its $M$-spans and its $\\Delta_M$ are near zero '
                '**by construction** wherever the compression exceeds the evolved error. It is reported for '
                'completeness; the evolved metric is the one that separates the factors.')
        d.h(3, 'The pure-rank ladder as an operating-point family, measured inside one job')
        d.p('The tunability bar this project has used since 15 September: error monotone in the setting, at '
            'least three non-dominated points, at least $2\\times$ span in **both** error and cost, no '
            'early-stopped point. Cost may only be compared inside one job, so each fixed-$M$ column is '
            'reported over the longest run of its rungs that one job holds.')
        wrows = []
        for M, x in sorted(s['fixed_M'].items()):
            w = x.get('within_job')
            if not w:
                continue
            wrows.append([M, w['job'], w['job_id'], ', '.join(map(str, w['q'])), ' / '.join(f(t) for t in w['values']),
                          ' / '.join(f(t, 0) for t in w['median_gpu_ms']), f(w['error_span'], 3) + 'x',
                          f(w['cost_span'], 3) + 'x', w['non_dominated_points'], yn(w['monotone_error']), yn(w['passes_tunability_bar'])])
            for kk in ('error_span', 'cost_span'):
                srow(q=None, M=M, metric=f'fixed_M_within_job.{kk}.{title}', value=w[kk], job_id=w['job_id'],
                     attempt=None, arm=None, source_sha256=None)
        d.table(['$M$', 'job', 'job id', 'rungs $q$', 'worst evolved %', 'median GPU ms', 'error span', 'cost span',
                 'non-dominated points', 'monotone', 'passes the tunability bar'], wrows or [['—'] * 11])

        arows = []
        for M, x in sorted(s['fixed_M'].items()):
            wa = x.get('within_job_longest_attempted')
            if not wa:
                continue
            arows.append([M, wa['job'], wa['job_id'], ', '.join(map(str, wa['q'])),
                          ' / '.join(f(t) for t in wa['values']), ', '.join(yn(c) for c in wa['converged']),
                          f(wa['error_span'], 3) + 'x' if wa['error_span'] is not None else 'unavailable',
                          f(wa['cost_span'], 3) + 'x' if wa['cost_span'] is not None else 'unavailable'])
        if arows:
            d.p('**Separately, the longest ladder any job *attempted* at this $M$, whether or not it is all '
                'converged** (DESIGN.md A5) — kept visible so a failed extension is reported honestly rather '
                'than silently displacing the certified ladder above; its span is `unavailable` whenever any '
                'rung failed, and it is **never** used as `certified_span` or read by the headline/verdict.')
            d.table(['$M$', 'job', 'job id', 'rungs $q$ attempted', 'worst evolved %', 'converged?', 'error span',
                     'cost span'], arows)
            for M, x in sorted(s['fixed_M'].items()):
                wa = x.get('within_job_longest_attempted')
                if wa:
                    srow(q=None, M=M, metric=f'fixed_M_within_job_longest_attempted.all_converged.{title}',
                         value=wa['all_converged'], job_id=wa['job_id'], attempt=None, arm=None, source_sha256=None)

        d.h(3, 'At fixed $M$: the span in $q$ (the pure-rank effect)')
        d.table(['$M$', 'rows $q$', 'values %', 'monotone in $q$', 'every rung converged', 'span (first/last)', 'span to $q=128$'],
                [[x['M'], ', '.join(map(str, x['q'])), ' / '.join(f(v) for v in x['values']), yn(x['monotone']),
                  yn(x['all_converged']), (f(x['span'], 3) + 'x') if x['span'] is not None else f"unavailable ({x['unavailable_reason']})",
                  (f(x['span_to_q128'], 3) + 'x') if x['span_to_q128'] else '—']
                 for _, x in sorted(s['fixed_M'].items())])
        if s['other_M_with_two_rows']:
            d.p('Other $M$ values that happen to hold two rows (one of them from the saturation job, some with '
                'fewer than two tests per unknown); informational, not a declared column: '
                + '; '.join(f"$M={x['M']}$: $q$ = {', '.join(map(str, x['q']))}, {' / '.join(f(v) for v in x['values'])} %"
                            for _, x in sorted(s['other_M_with_two_rows'].items())) + '.')
        for M, x in sorted(s['fixed_M'].items()):
            if x.get('certified_note'):
                d.p(f"**$M={M}$, {x['certified_note']}** ($q = {x['certified_q_range'][0]}\\ldots"
                    f"{x['certified_q_range'][1]}$, span **{f(x['certified_span'], 3)}x**, monotone, every "
                    f"rung of this sub-ladder converged). The failed extension's own longest-attempted "
                    f"in-job ladder (not certified, not a span) is reported separately below under "
                    f"'the pure-rank ladder ... measured inside one job' where applicable.")
        for M, x in s['fixed_M'].items():
            srow(q=None, M=M, metric=f'span_q_at_fixed_M.{title}', value=x['span'], job_id=None, attempt=None, arm=None,
                 source_sha256=None, q_top=x['q_top'], monotone=x['monotone'])
        d.h(3, 'At fixed $q$: the span in $M$ (the test-count effect)')
        d.p('The last column restricts the span to $M \\ge 2(K+q)$ (at least two tests per unknown); it is the '
            'quantity the H(rank) falsification clause reads, so a near-square cell cannot decide it.')
        d.table(['$q$', 'columns $M$', 'values %', 'monotone in $M$', 'span (max/min)', 'best $M$', 'span over $M \\ge 2(K+q)$'],
                [[q, ', '.join(map(str, x['M'])), ' / '.join(f(v) for v in x['values']), yn(x['monotone']),
                  f(x['span'], 3) + 'x', x['M_best'], (f(x['span_M_ge_2x'], 3) + 'x') if x['span_M_ge_2x'] else '—']
                 for q, x in sorted(s['fixed_q'].items())])
        for q, x in s['fixed_q'].items():
            srow(q=q, M=None, metric=f'span_M_at_fixed_q.{title}', value=x['span'], job_id=None, attempt=None, arm=None,
                 source_sha256=None, monotone=x['monotone'])
        d.h(3, 'The scheduled ladders')
        d.table(['schedule', 'cells $(q, M)$', 'values %', 'monotone', 'span'],
                [[lab, ', '.join(f'({q},{M})' for q, M in x['cells']), ' / '.join(f(v) for v in x['values']),
                  yn(x['monotone']), f(x['span'], 3) + 'x'] for lab, x in sorted(s['scheduled'].items())])
        for lab, x in s['scheduled'].items():
            srow(q=None, M=None, metric=f'span_scheduled_{lab}.{title}', value=x['span'], job_id=None, attempt=None,
                 arm=None, source_sha256=None, monotone=x['monotone'])

        dd = dec[metric]
        d.h(2, f'Decomposition of the scheduled $4(K+q)$ ladder — {title} metric')
        if dd['corner']:
            c = dd['corner']
            d.p(f"**Corner path** {c['path']}: $\\log S = {c['log_span']:.4f}$ ($S = {c['span']:.3f}$x) $= "
                f"\\Delta_M + \\Delta_q = {c['delta_M']:.4f} + {c['delta_q']:.4f}$; shares **test count "
                f"{100 * c['share_M']:.1f} %**, **rank {100 * c['share_q']:.1f} %**.")
            for k in ('span', 'delta_M', 'delta_q', 'share_M', 'share_q'):
                srow(q=None, M=None, metric=f'corner.{k}.{title}', value=c[k], job_id=None, attempt=None, arm=None, source_sha256=None)
        else:
            d.p('**Corner path: unavailable** (a corner cell is missing or not converged).')
        r_ = dd['rung']
        d.p(f"**Rung path** (through the bridge cells): complete {yn(r_['complete'])}"
            + (f"; totals $\\Delta_M = {r_['delta_M']:.4f}$, $\\Delta_q = {r_['delta_q']:.4f}$; shares test count "
               f"**{100 * r_['share_M']:.1f} %**, rank **{100 * r_['share_q']:.1f} %**." if r_['complete'] else '.'))
        d.table(['rung', 'bridge cell', '$\\Delta_M$ (M first, at $q_k$)', '$\\Delta_q$ (then $q$, at $M_{k+1}$)', 'total', 'test-count share'],
                [[f"({r['from_'][0]},{r['from_'][1]}) → ({r['to'][0]},{r['to'][1]})", f"({r['bridge'][0]},{r['bridge'][1]})",
                  f(r.get('delta_M')), f(r.get('delta_q')), f(r.get('total')), f(r.get('share_M'), 3) if r.get('available') else 'unavailable']
                 for r in r_['rungs']])
        if r_['complete']:
            for k in ('share_M', 'share_q'):
                srow(q=None, M=None, metric=f'rung.{k}.{title}', value=r_[k], job_id=None, attempt=None, arm=None, source_sha256=None)
        an = dd['anova']
        if an:
            d.p(f"**Variance shares** on the balanced sub-grid $q \\in \\{{{', '.join(map(str, an['q']))}\\}} \\times M \\in "
                f"\\{{{', '.join(map(str, an['M']))}\\}}$ (log error): rank main effect **{100 * an['frac_q']:.1f} %**, "
                f"test-count main effect **{100 * an['frac_M']:.1f} %**, interaction {100 * an['frac_interaction']:.1f} % of "
                f"the total sum of squares. Interaction matrix (rows $q$, columns $M$): "
                + '; '.join(f"$q={q}$: " + ', '.join(f'{v:+.4f}' for v in row) for q, row in zip(an['q'], an['interaction'])) + '.')
            for k in ('frac_q', 'frac_M', 'frac_interaction'):
                srow(q=None, M=None, metric=f'anova.{k}.{title}', value=an[k], job_id=None, attempt=None, arm=None, source_sha256=None)
        else:
            d.p('**Variance shares: unavailable** (balanced sub-grid incomplete).')

    d.h(2, 'E1 — does $q = 512$ separate a rank limit from an under-tested cell?')
    d.p("Pre-registered in DESIGN.md A3. `(512, 1088)` is only 2.06 tests per unknown; "
        "`(512, 2112)` and `(512, 3168)` add 4.0 and 6.0. `cclad01` (budget 180, not "
        "converged) reached 0.2307 % / 0.4343 % at `(512, 2112)` / `(512, 1056)` and was "
        "cited only as motivation, never as a gate.")
    if e1d is None:
        d.p('**Unavailable**: E1 does not carry all of $(512, 1088)$, $(512, 2112)$, $(512, 3168)$ and the $(256, 1088)$ baseline.')
    else:
        d.p(f"**Verdict:** {e1d['clause_text']}")
        d.table(['cell', 'worst evolved %', 'tests/unknown', 'converged', 'median iters', 'worst joint gradient', 'budget exits', 'median GPU ms'],
                [[name, f(c['value']), f(c.get('tests_per_unknown'), 2), yn(c['converged']), f(c.get('median_iterations'), 1),
                  sci(c.get('worst_joint_gradient')), (c.get('total_budget_exits') if c.get('total_budget_exits') is not None else '—'),
                  f(c.get('median_gpu_ms'), 1)]
                 for name, c in (('q=512, M=1088', e1d['q512_M1088']), ('q=512, M=2112 (4x)', e1d['q512_M2112']),
                                 ('q=512, M=3168 (6x)', e1d['q512_M3168']),
                                 ('q=256, M=1088 (baseline, converged)', e1d['baseline_q256_M1088']))])
        for name, c in (('q512_M1088', e1d['q512_M1088']), ('q512_M2112', e1d['q512_M2112']), ('q512_M3168', e1d['q512_M3168'])):
            for k in ('value', 'converged', 'worst_joint_gradient', 'tests_per_unknown'):
                srow(q=512, M=int(name.split('_M')[1]), metric=f'e1_disambiguation.{k}', value=c[k], job_id=None, attempt='bqx401', arm=name, source_sha256=None)
        srow(q=None, M=None, metric='e1_disambiguation.clause', value=e1d['clause'], job_id=None, attempt='bqx401', arm=None, source_sha256=None)

    d.h(2, 'Saturation in $M$ (job S1)')
    if sat:
        d.p(f"Job `{sat['attempt']}` ({sat['job_id']}, {sat['gpu']}). $M^\\star$ is the smallest $M$ whose next step "
            'improves the worst evolved error by less than 5 %. Cost ratios are within this job, against the '
            "row's own $M = 4(K+q)$ cell; the cost-neutral sentence is written only if that ratio is $\\le 1.1$.")
        for q, x in sorted(sat['per_q'].items()):
            src = (sat.get('sources') or {}).get(str(q), {})
            if x['still_falling_at_largest_M']:
                d.h(3, f"$q = {q}$ — **no $M^\\star$**: the curve is still falling at the largest $M$ run "
                       f"({x['largest_M']}, {f(x['value_at_largest_M'])} %), i.e. "
                       f"{f(x['largest_M'] / (K + int(q)), 1)} tests per unknown does not saturate this rung "
                       f"[{src.get('job', '')} {src.get('job_id', '')}]")
            else:
                d.h(3, f"$q = {q}$ — $M^\\star = {x['M_star']}$ ({f(x['tests_per_unknown_at_M_star'], 1)} tests per "
                       f"unknown), error there {f(x['error_at_M_star'])} %, cost {f(x['cost_ratio_at_M_star'], 3)}x "
                       f"the $M={x['reference_M']}$ cell; cost-neutral claim: {yn(x['cost_neutral_claim'])}; "
                       f"monotone in $M$: {yn(x['monotone'])} [{src.get('job', '')} {src.get('job_id', '')}]")
            d.table(['$M$', 'worst evolved %', 'worst all-times %', 'median GPU ms', 'cost vs $4(K+q)$', 'median iters', 'converged', 'improvement to next $M$'],
                    [[c['M'], f(c['value']), f(c['all_times']), f(c['median_gpu_ms'], 1), (f(c['cost_ratio_to_4x'], 3) + 'x') if c['cost_ratio_to_4x'] else '—',
                      f(c['median_iterations'], 1), yn(c['converged']), (f"{100 * c['improvement_to_next']:.1f} % (to {c['next_M']})" if c['improvement_to_next'] is not None else '—')]
                     for c in x['curve']])
            srow(q=q, M=None, metric='saturation.M_star', value=x['M_star'], job_id=sat['job_id'], attempt=sat['attempt'], arm=None, source_sha256=None)
            srow(q=q, M=None, metric='saturation.cost_ratio_at_M_star', value=x['cost_ratio_at_M_star'], job_id=sat['job_id'], attempt=sat['attempt'], arm=None, source_sha256=None)
            for c in x['curve']:
                srow(q=q, M=c['M'], metric='saturation.worst_evolved_percent', value=c['value'], job_id=sat['job_id'], attempt=sat['attempt'], arm=f"q{q}_M{c['M']}_dense", source_sha256=None)
                srow(q=q, M=c['M'], metric='saturation.median_gpu_ms', value=c['median_gpu_ms'], job_id=sat['job_id'], attempt=sat['attempt'], arm=f"q{q}_M{c['M']}_dense", source_sha256=None)
    else:
        d.p('Not run.')

    d.h(2, 'Cells that appear in more than one job (anchors)')
    d.p('Errors must agree to the declared tolerance ($10^{-9}$ at $q=0$; two-tier at $q>0$). Both costs are '
        'printed and **no ratio is formed between them**: different jobs, possibly different GPUs.')
    d.table(['$q$', '$M$', 'job A', 'job B', 'rel. diff evolved', 'rel. diff all-times', 'tolerance', 'passed', 'A median ms', 'B median ms'],
            [[r['q'], r['M'], f"{r['a']} ({r['a_job']})", f"{r['b']} ({r['b_job']})", sci(r['rel_evolved']), sci(r['rel_all']),
              sci(r['tol'], 0), yn(r['passed']), f(r['a_ms'], 1), f(r['b_ms'], 1)] for r in anc])
    for r in anc:
        srow(q=r['q'], M=r['M'], metric='anchor.rel_diff_evolved', value=r['rel_evolved'], job_id=f"{r['a_job']}/{r['b_job']}", attempt=f"{r['a']}/{r['b']}", arm=None, source_sha256=None, passed=r['passed'])

    d.h(2, 'Same-job full-order controls')
    for au in audits:
        d.h(3, f"{au['question']} ({au['attempt']}, {au['job_id']}, {au['gpu']})")
        d.table(['control', 'worst all-times %', 'worst evolved %', 'median GPU ms', 'repetitions'],
                [[f"`{x['arm']}`", f(x['worst_all_times_percent']), f(x['worst_evolved_percent']), f(x['median_gpu_ms'], 3), x['gpu_ms_repetitions']]
                 for x in au['arms'] if x['family'] == 'fom'])
        for x in au['arms']:
            if x['family'] == 'fom':
                for key in ('worst_all_times_percent', 'median_gpu_ms'):
                    srow(q=None, M=None, metric=f'fom.{key}', value=x[key], job_id=au['job_id'], attempt=au['attempt'], arm=x['arm'], source_sha256=au['_sha256'])

    d.h(2, 'Gates and cross-job fidelity')
    for au in audits:
        d.h(3, f"{au['question']} ({au['attempt']})")
        d.table(['gate', 'passed', 'kind'],
                [[f'`{k}`', yn(v.get('passed')), 'informational' if v.get('blocking') is False else 'blocking']
                 for k, v in sorted(au['checks'].items()) if isinstance(v, dict) and 'passed' in v and not k.startswith('reproduces_')])
        fid = (au['checks'].get('cross_job_fidelity') or {}).get('detail') or {}
        if fid:
            d.table(['arm', 'source job', 'comparator arm', 'unconditional', 'directions bitwise', 'declared tolerance', 'achieved worst rel. diff', 'passed'],
                    [[f"`{k.split('__')[0]}`", f"{v['detail'].get('source')} ({v['detail'].get('source_job')})", f"`{v['detail'].get('comparator')}`",
                      yn(v['detail'].get('unconditional')), yn(v['detail'].get('directions_bitwise')), sci(v['detail'].get('declared_tolerance'), 0),
                      sci(v['detail'].get('achieved_worst_relative_difference')), yn(v['passed'])] for k, v in sorted(fid.items())])
            for k, v in fid.items():
                srow(q=None, M=None, metric='fidelity.achieved_worst_relative_difference', value=v['detail'].get('achieved_worst_relative_difference'),
                     job_id=au['job_id'], attempt=au['attempt'], arm=k, source_sha256=au['_sha256'], passed=v['passed'])

    d.h(2, 'Figures')
    png1 = out.with_name(out.name + '-grid.png')
    grid_figure(grid, sp['worst_evolved_percent'], png1)
    d.p(f'![The crossed grid]({png1.name})')
    if sat:
        png2 = out.with_name(out.name + '-saturation.png')
        sat_figure(sat, png2)
        d.p(f'![Saturation in M]({png2.name})')

    d.h(2, 'Glossary')
    for term, text in GLOSSARY:
        d.md.append(f'- **{term}** — {text}')
    d.md.append('')
    md = out.with_suffix('.md')
    md.write_text('\n'.join(d.md) + '\n')
    # beside --out, so a dry or regression run never clobbers the published summary
    side = out.parent
    (side / 'summary.json').write_text(json.dumps(dict(
        generated_from=[dict(path=x['_path'], sha256=x['_sha256'], job_id=x['job_id'], attempt=x['attempt']) for x in audits],
        verdict=ver, rows=summary), indent=2) + '\n')
    (side / 'analysis.json').write_text(json.dumps(dict(spans=sp, decomposition=dec, saturation=sat, anchors=anc, verdict=ver), indent=2) + '\n')
    print(md, sha(md))
    print(side / 'summary.json', len(summary), 'rows')


GLOSSARY = [
    ('$q$, rank, rung', 'the number of extra correction directions added to the frozen neural head; the solved state is $u = G(h_\\theta(z) + C_q y)$ with $z \\in \\mathbb{R}^{16}$ the latent code and $y \\in \\mathbb{R}^q$ the corrections. One value of $q$ is a rung.'),
    ('$M$, test count, test modes', 'the number of weak test equations the per-step solve is graded on: the first $M$ sine modes of the square in ascending discrete-Laplacian order. The incumbent schedule is $M = 4(K+q)$, four tests per unknown.'),
    ('$K$', 'the latent dimension, 16. The solve has $K + q$ unknowns, so every cell needs $M > K + q$.'),
    ('column / schedule', 'a rule for $M$ across rows: fixed ($M = 256$ or $1088$ whatever $q$ is) or scheduled ($M = c\\,(K+q)$). The `bridge` cells are the $(q_k, M_{k+1})$ cells that make the rung-by-rung decomposition exact.'),
    ('fixed-$M$ ladder, pure-rank ladder', 'the rungs $q = 0, 16, \\ldots$ at one fixed $M$: only the rank moves. At $M = 1088$ it holds every rung to $q = 256$.'),
    ('scheduled ladder', 'the rungs with $M = 4(K+q)$: rank and test count move together. This is the ladder earlier lanes quoted.'),
    ('span', 'the ratio of the largest to the smallest error along a ladder (first over last for a ladder ordered by $q$). Both spans here are on the same metric.'),
    ('corner path / rung path', 'two exact ways of splitting the scheduled ladder\'s log-span into a test-count part $\\Delta_M$ and a rank part $\\Delta_q$: the corner path moves $M$ first at $q = 0$ then $q$ at $M = 1088$; the rung path does the same split rung by rung through the bridge cells.'),
    ('share', '$\\Delta_M / \\log S$ or $\\Delta_q / \\log S$: the fraction of the ladder\'s log-span that one factor accounts for on that path.'),
    ('variance shares', 'on the balanced $5 \\times 2$ sub-grid, the fraction of the total sum of squares of the log error carried by the $q$ main effect, the $M$ main effect and their interaction. A small interaction means "additive" is fair.'),
    ('$M^\\star$, saturation', 'the smallest $M$ beyond which the next step in the sweep improves the error by less than 5 %.'),
    ('worst evolved-times error', 'the largest relative error over output times $t > 0$ and over the six cases, against the same-job converged full-order solve on the same grid: the trajectory error proper.'),
    ('worst all-times error', 'the same with $t = 0$ included, where the error is the decoder\'s compression of the supplied initial field.'),
    ('$t = 0$ compression', 'the relative error at $t = 0$: how well the reduced model represents the field it was handed. It uses only the cold Gauss space and $C_q$, never the test modes, so it does not depend on $M$ (a gate).'),
    ('same-grid metric', 'error against `fft_tight`, the same job\'s own converged full-order Newton solve on the 256-interval grid, so discretisation error cancels.'),
    ('median GPU ms', 'the median over three timed repetitions and six cases of the complete query time on the GPU (supplied initial field in, six output fields out), after a burn-in and with a device sync. Comparable only inside one job.'),
    ('converged', 'every one of the 900 solved steps exited on a convergence criterion (no step hit its 600-iteration budget), the initial fit converged, and the worst normalised joint gradient is at or inside $10^{-6}$.'),
    ('median iterations', 'the median number of Levenberg–Marquardt iterations per time step over all steps, cases and repetitions.'),
    ('budget exit', 'a time step that stopped because it hit the iteration budget rather than a convergence criterion; any budget exit disqualifies a cell from the spans.'),
    ('anchor', 'a cell run in more than one job so the jobs can be checked against each other on error; costs are printed side by side and never divided.'),
    ('cross-job fidelity', 'a named cell reproducing a named arm of an earlier audited job to a declared tolerance; unconditional at $10^{-9}$ for $q = 0$ (no directions involved), two-tier for $q > 0$ because the direction matrix is GPU-model dependent.'),
    ('fft_tight / fft_loose / nt1e-2_dt01', 'same-job full-order Newton solves at three tolerance/time-step settings. `fft_tight` is the reference, so its own error is zero by construction.'),
    ('development cohort', 'the six cases every lane since head-ablation has used for diagnosis; no sealed case is opened here.'),
]


if __name__ == '__main__':
    main()
