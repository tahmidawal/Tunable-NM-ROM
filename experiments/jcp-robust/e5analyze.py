"""jcp-robust E5 analysis (CPU, no JAX): cross-hardware comparison of dimensionless timing ratios.

DESIGN 4.3/4.4 as amended by A1.11, A1.12, A2.8, A2.9, A3.7, A3.8, A3.9, A4.1 (later amendments govern).

Per job
  * status: ok | missing | failed (incomplete / wrong GPU type / not on GPU / source check) | blocked (control) |
    blocked (identity). C-E5.1 = P1's doubled-work control: per subject, per-case median over drift-ok repetitions of
    t_double / mean(t_single), then the median over cases, must lie in [1.6, 2.4] (identities and iteration counts equal);
    otherwise the job is "blocked (control)". Any identity failure of an E5 subject (P1 `identity_veto`, c2 veto) blocks.
  * H-cost-N(2): P1's `p1stats.h_cost_n` UNCHANGED on the {256^2, 1024^2} columns; label = its `procedure_outcome`.
    H200 jobs with 4096^2 also get the three-mesh registered outcome (H200 only, never cross-type).
  * S per (ROM subject, pinned comparator): fom2d `c2analyze.speedup` UNCHANGED; S2-subset label pass iff lo >= 2,
    fail iff hi < 2, else inconclusive.
Pooled (A2.8, A3.8)
  * C_cost: P-cost cases with >= 2 valid reps for both subjects at 256^2 and 1024^2 in EVERY job of both types; C_cross:
    P-cross cases with >= 2 valid reps for both ROM subjects and every timed comparator in every job; |C| < 5 ->
    unavailable. q1(T,b,L) = median of all valid (job, case, rep) B/A entries on C; q2(T,b) = max_L q1 / min_L q1;
    q3(T,b,f) = median_c m^FOM_c / median_c m^ROM_c, m_c = median over the type's valid (job, rep) ratios on case c.
  * phi_k = q_k(H200) / q_k(A100); paired case bootstrap (10,000, seed 20261019, case ids resampled jointly for both
    types and all subjects, every entry of a resampled case kept, full statistic recomputed); hardware-consistent iff
    the 95 % interval lies within [0.8, 1.25], hardware-dependent iff entirely outside, else inconclusive.
  * per-type q4 (A3.9): H-cost-N(2) from pooled q2 (pass iff hi <= 1.25, fail iff lo > 1.25), S2-subset from pooled q3
    (pass iff lo >= 2, fail iff hi < 2); "inconclusive (jobs disagree)" if the type's two jobs give opposite definitive
    per-job labels. A4.1: any missing / failed / blocked job -> every phi and that type's labels
    "unavailable (job missing)"; per-job values still reported ("per-job only").
  * unit-to-unit baseline: the same phi between the two jobs of one type (reported, no required outcome).
Fixtures (A3.7): from the first A100 job (>= 5 cases x 2 valid reps): q1 (R128-BE 1024^2 entries vs x 1.5), q2 (the
1024^2 entries copied to both mesh columns vs the 1024^2 column x 1.5), q3 (real FOM/ROM vs FOM x 1.5) must be
hardware-dependent; null (baseline vs itself) must be consistent with interval [1, 1]. A failed fixture for q_k blocks
the phi_k verdicts and the dependent q4 agreement (q2 -> H-cost-N(2), q3 -> S2-subset).
Verdict agreement (4.3): verdict-robust iff every type gives the same definitive label; hardware-dependent verdict iff
two types give opposite definitive labels; inconclusive otherwise; unavailable / blocked carried through.

Required job gates (code audit 3 item 30; decided in load_job, before any pooling, through the A4.1 path): P-cross
G-RR, G-FR, A-valid (c2analyze.py:98) and G-E5.1 = P-cost scoring W reproduce P1c (--p1ref) on the complete registered
subject x case set, finite, <= 1e-10 relative; a missing --p1ref blocks every job. G-E5.2 stays descriptive.
Single-job smoke (item 27): --smoke --job A100-80G=<dir> --p1ref <dir> reports C-E5.1, the A3.7 fixtures from that
job, G-E5.1 and the P-cross gates as explicit pass/fail plus ALL, and exits 1 unless ALL.

    python e5analyze.py --job A100-80G=<dir> --job A100-80G=<dir> --job H200=<dir> --job H200=<dir> \
        [--p1ref <P1c output dir>] [--expect A100-80G=2,H200=2] --out checks/e5-analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE / 'e5') not in sys.path:
    sys.path.insert(0, str(HERE / 'e5'))
import p1stats as S1  # noqa: E402  (P1, unchanged copy)
import c2analyze as C2A  # noqa: E402  (fom2d, unchanged copy)

SEED = 20261019
NBOOT = 10000
MIN_CASES = 5
MIN_REPS = 2
BAND_PHI = (0.8, 1.25)
BAR_COST = 1.25
BAR_S = 2.
DBL_BAND = (1.6, 2.4)
CROSS_MESHES = (256, 1024)
FAMS = ('R128|fib4181|frozen|BE', 'R384|gauss128|frozen|CNR')
FIX_FAM = 'R128|fib4181|frozen|BE'
TYPES = ('A100-80G', 'H200')          # phi = q(TYPES[1]) / q(TYPES[0])
DEFINITIVE = ('pass', 'fail')
INJECT = 1.5
PCROSS_GATES = ('G-RR', 'G-FR', 'A-valid')
REPRO_TOL = 1e-10
NAMED_CASES = {'dev6': [f'dev6|{i}' for i in range(6)], 'dev2': ['dev6|0', 'dev6|1'],
               'sub16': [f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(10)]}


# ------------------------------------------------------------------ job loading ----

def fam_of(label):
    m, rest = label.split('|', 1)
    return int(m), rest


def load_job(gtype, d, p1ref=None):
    """Load one E5 job and decide its status BEFORE any pooling (code audit 3 item 30): P-cross G-RR / G-FR / A-valid
    (as c2analyze.py:98) and G-E5.1 (complete registered reproduction of P1c, needs p1ref) are required."""
    d = Path(d) if d else None
    job = dict(type=gtype, dir=str(d) if d else None, status='ok', reasons=[])
    if d is None or not (d / 'result.json').exists():
        job.update(status='missing', reasons=['no result.json'])
        return job
    top = json.loads((d / 'result.json').read_text())
    job['top_meta'] = {k: top.get(k) for k in ('attempt', 'job_id', 'gpu_type', 'gpu_name', 'gpu_type_ok', 'backend',
                                               'device_kind', 'jax_version', 'config_sha256', 'commit', 'elapsed_seconds')}
    job['records'] = top.get('records', [])
    fail = []
    if top.get('complete') is not True or not (d / 'COMPLETE').exists():
        fail.append('incomplete')
    if top.get('gpu_type') != gtype or top.get('gpu_type_ok') is not True:
        fail.append(f"wrong GPU type ({top.get('gpu_type')}, {top.get('gpu_name')})")
    if top.get('backend') != 'gpu':
        fail.append(f"backend {top.get('backend')}")
    if not all(r.get('ok') for r in top.get('source_check') or [dict(ok=False)]):
        fail.append('source check')
    pc, xc = top.get('pcost') or {}, top.get('pcross') or {}
    if not (pc.get('complete') and xc.get('complete')):
        fail.append('panel incomplete')
    if fail:
        job.update(status='failed (' + '; '.join(fail) + ')', reasons=fail)
        return job
    job.update(extract_cost(pc), cross=extract_cross(xc), dir=str(d))
    job['c_e5_1'] = c_e5_1(pc.get('doubled', []))
    veto = S1.identity_veto(pc.get('invocations', []), pc.get('fused', []), pc.get('doubled', []))
    job['identity_veto_pcost'] = {k: v for k, v in veto.items() if v}
    job['identity_veto_pcross'] = list(xc.get('veto', []))
    job['equivalence_ok'] = bool(pc.get('equivalence')) and all(x.get('passed') for x in pc['equivalence'])
    xg = xc.get('gates') or {}
    job['pcross_gates'] = {k: xg.get(k) is True for k in PCROSS_GATES}
    job['g_e5_1'] = g_e5_1(job, p1ref, pc)
    blocked = []
    if not job['c_e5_1']['fires']:
        blocked.append('blocked (control C-E5.1)')
    if not job['equivalence_ok']:
        blocked.append('blocked (control: split/fused equivalence)')
    if job['identity_veto_pcost'] or job['identity_veto_pcross']:
        blocked.append('blocked (identity)')
    if not all(job['pcross_gates'].values()):
        blocked.append('blocked (gate P-cross ' + ', '.join(k for k, v in job['pcross_gates'].items() if not v) + ')')
    if not job['g_e5_1']['passed']:
        blocked.append('blocked (gate G-E5.1)')
    if blocked:
        job.update(status=blocked[0], reasons=blocked)
    return job


def extract_cost(pc):
    """P1 invocations -> per (family, mesh): {case: array of valid ratios} and the full (case, rep) arrays."""
    subs = pc['subjects']
    cases = list(next(iter(subs.values()))['case_ids'])
    reps = int(pc['config']['reps'])
    meshes = sorted({fam_of(x['B'])[0] for x in pc['invocations']})
    fams = sorted({fam_of(x['B'])[1] for x in pc['invocations']})
    full = {f: (np.full((len(cases), len(meshes), reps), np.nan), np.zeros((len(cases), len(meshes), reps), bool))
            for f in fams}
    vals = {}
    for x in pc['invocations']:
        L, f = fam_of(x['B'])
        r, v = full[f]
        i, j = cases.index(x['case']), meshes.index(L)
        r[i, j, int(x['rep'])] = x['ratio']
        v[i, j, int(x['rep'])] = bool(x['valid'])
        if x['valid']:
            vals.setdefault((f, L), {}).setdefault(x['case'], []).append(float(x['ratio']))
    cost = {k: {c: np.asarray(v_) for c, v_ in d.items()} for k, d in vals.items()}
    tev = {}
    for x in pc['invocations']:
        if x['valid']:
            tev.setdefault(x['B'], []).append(x['t_evolve'])
    return dict(cost=cost, cost_cases=cases, cost_meshes=meshes, cost_full=full,
                seconds_evolve={k: float(np.median(v)) for k, v in tev.items()})


def extract_cross(xc):
    vals = {}
    for x in xc['stage2']:
        if x['valid']:
            vals.setdefault(x['B'], {}).setdefault(x['case'], []).append(float(x['ratio']))
    tb = {}
    for x in xc['stage2']:
        if x['valid']:
            tb.setdefault(x['B'], []).append(x['tB'])
    return dict(vals={k: {c: np.asarray(v) for c, v in d.items()} for k, d in vals.items()},
                stage2=xc['stage2'], pinned=xc['pinned'], cases=[f'{c}|{j}' for c, j in xc['config']['cases']],
                roms=[r['label'] for r in xc['config']['roms']],
                foms=sorted({k for v in xc['pinned'].values() for k in v.values()}),
                seconds_B={k: float(np.median(v)) for k, v in tb.items()}, gates=xc.get('gates'))


def c_e5_1(rows):
    """A1.12: P1's doubled-work control must fire in every job (case-median of doubled/single in [1.6, 2.4])."""
    out = dict(subjects={}, fires=False)
    if not rows:
        out['reason'] = 'no doubled-work rows'
        return out
    subs = sorted({x['subject'] for x in rows})
    ok_all = True
    for s in subs:
        xs = [x for x in rows if x['subject'] == s]
        per = {}
        for x in xs:
            if x.get('drift_ok'):
                per.setdefault(x['case'], []).append(x['ratio'])
        cm = {c: float(np.median(v)) for c, v in per.items()}
        med = float(np.median(list(cm.values()))) if cm else float('nan')
        integ = all(x.get('identity_ok') and x.get('same_iterations') for x in xs)
        ok = bool(cm and np.isfinite(med) and DBL_BAND[0] <= med <= DBL_BAND[1] and integ)
        out['subjects'][s] = dict(case_medians=cm, median=med, identity_and_iterations_ok=integ, fires=ok, n_cases=len(cm))
        ok_all = ok_all and ok
    out['fires'] = bool(ok_all)
    return out


# ------------------------------------------------------------------ labels ----

def phi_label(lo, hi):
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return 'unavailable'
    if BAND_PHI[0] <= lo and hi <= BAND_PHI[1]:
        return 'hardware-consistent'
    if hi < BAND_PHI[0] or lo > BAND_PHI[1]:
        return 'hardware-dependent'
    return 'inconclusive'


def cost_label(lo, hi):
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return 'unavailable'
    return 'pass' if hi <= BAR_COST else ('fail' if lo > BAR_COST else 'inconclusive')


def s2_label(lo, hi):
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return 'unavailable'
    return 'pass' if lo >= BAR_S else ('fail' if hi < BAR_S else 'inconclusive')


def interval(reps):
    reps = np.asarray(reps, float)
    if reps.size == 0 or not np.isfinite(reps).all():
        return None, None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def type_label(pooled_label, job_labels):
    """A3.9: opposite definitive per-job labels override the pooled label."""
    d = {l_ for l_ in job_labels if l_ in DEFINITIVE}
    if len(d) > 1:
        return 'inconclusive (jobs disagree)'
    return pooled_label


def agreement(labels):
    """4.3 verdict agreement over the per-type labels {type: label}."""
    if not labels:
        return 'unavailable'
    if any(str(l_).startswith(('unavailable', 'blocked')) for l_ in labels.values()):
        return 'unavailable'
    v = list(labels.values())
    d = [l_ for l_ in v if l_ in DEFINITIVE]
    if len(set(d)) > 1:
        return 'hardware-dependent verdict'
    if len(d) == len(v) and len(set(d)) == 1:
        return 'verdict-robust'
    return 'inconclusive'


# ------------------------------------------------------------------ pooled estimators ----

def case_set(groups, need, requested, key):
    """Cases (from `requested`, in order) with >= MIN_REPS valid entries for every cell in `need` in every job of every
    group. groups: {name: [job dict]}, key: 'cost' or 'cross' (job[key][cell][case] -> array)."""
    out = []
    for c in requested:
        ok = True
        for jobs in groups.values():
            for j in jobs:
                for cell in need:
                    if len(j[key].get(cell, {}).get(c, ())) < MIN_REPS:
                        ok = False
        if ok:
            out.append(c)
    return out


def boot_rows(n, seed=SEED, nboot=NBOOT):
    return np.random.default_rng(seed).integers(0, n, size=(nboot, n))


def cost_panel(groups, fams, meshes, C, rows, num, den):
    """q1, q2 per group (pooled over its jobs) and phi1, phi2 = num / den, recomputed on every bootstrap row."""
    n = len(C)
    pooled = {g: {(f, L): [np.concatenate([j['cost'][(f, L)][c] for j in jobs]) for c in C] for f in fams for L in meshes}
              for g, jobs in groups.items()}

    def q1(g, cell, sel):
        return float(np.median(np.concatenate([pooled[g][cell][i] for i in sel])))
    full = np.arange(n)
    Q1 = {g: {cell: (q1(g, cell, full), np.array([q1(g, cell, r) for r in rows])) for cell in pooled[g]} for g in groups}
    Q2 = {}
    for g in groups:
        Q2[g] = {}
        for f in fams:
            pt = [Q1[g][(f, L)][0] for L in meshes]
            bs = np.stack([Q1[g][(f, L)][1] for L in meshes])
            Q2[g][f] = (max(pt) / min(pt), bs.max(0) / bs.min(0))
    phi1, phi2 = {}, {}
    for f in fams:
        for L in meshes:
            a, b = Q1[num][(f, L)], Q1[den][(f, L)]
            lo, hi = interval(a[1] / b[1])
            phi1[f'{f}|{L}'] = dict(phi=a[0] / b[0], lo=lo, hi=hi, label=phi_label(lo, hi),
                                    q_num=a[0], q_den=b[0])
        a, b = Q2[num][f], Q2[den][f]
        lo, hi = interval(a[1] / b[1])
        phi2[f] = dict(phi=a[0] / b[0], lo=lo, hi=hi, label=phi_label(lo, hi), q_num=a[0], q_den=b[0])
    return Q1, Q2, phi1, phi2


def cross_panel(groups, pairs, C, rows, num, den):
    """q3 per group and pair (rom, fom); phi3 = num / den."""
    subj = sorted({s for p in pairs for s in p})
    m = {g: {s: np.array([float(np.median(np.concatenate([j['cross'][s][c] for j in jobs]))) for c in C]) for s in subj}
         for g, jobs in groups.items()}
    Q3 = {}
    for g in groups:
        Q3[g] = {}
        for r, f in pairs:
            pt = float(np.median(m[g][f]) / np.median(m[g][r]))
            bs = np.median(m[g][f][rows], axis=1) / np.median(m[g][r][rows], axis=1)
            Q3[g][(r, f)] = (pt, bs)
    phi3 = {}
    for p in pairs:
        a, b = Q3[num][p], Q3[den][p]
        lo, hi = interval(a[1] / b[1])
        phi3[f'{p[0]}|{p[1]}'] = dict(phi=a[0] / b[0], lo=lo, hi=hi, label=phi_label(lo, hi), q_num=a[0], q_den=b[0])
    return Q3, phi3


# ------------------------------------------------------------------ fixtures (A3.7) ----

def fixtures(job1):
    """Estimator-level must-fail and null fixtures from one real A100 job. Returns {k: dict(ok, ...)} for q1, q2, q3."""
    out = {}
    base_cells = job1['cost'].get((FIX_FAM, 1024), {})
    Cf = [c for c in job1['cost_cases'] if len(base_cells.get(c, ())) >= MIN_REPS]
    if len(Cf) < MIN_CASES:
        for k in ('q1', 'q2', 'q3'):
            out[k] = dict(ok=False, reason=f'fixture case set too small ({len(Cf)})')
        return out
    rows = boot_rows(len(Cf))
    base = {c: base_cells[c] for c in Cf}
    inj = {c: base_cells[c] * INJECT for c in Cf}
    # q1
    g = {'den': [dict(cost={(FIX_FAM, 1024): base})], 'num': [dict(cost={(FIX_FAM, 1024): inj})]}
    _, _, p1, _ = cost_panel(g, [FIX_FAM], [1024], Cf, rows, 'num', 'den')
    g0 = {'den': [dict(cost={(FIX_FAM, 1024): base})], 'num': [dict(cost={(FIX_FAM, 1024): base})]}
    _, _, p1n, _ = cost_panel(g0, [FIX_FAM], [1024], Cf, rows, 'num', 'den')
    i1, n1 = p1[f'{FIX_FAM}|1024'], p1n[f'{FIX_FAM}|1024']
    out['q1'] = dict(injected=i1, null=n1, ok=bool(i1['label'] == 'hardware-dependent' and n1['label'] == 'hardware-consistent'
                                                   and n1['lo'] == 1. and n1['hi'] == 1.), cases=Cf)
    # q2
    gb = dict(cost={(FIX_FAM, 256): base, (FIX_FAM, 1024): base})
    gi = dict(cost={(FIX_FAM, 256): base, (FIX_FAM, 1024): inj})
    _, _, _, p2 = cost_panel({'den': [gb], 'num': [gi]}, [FIX_FAM], [256, 1024], Cf, rows, 'num', 'den')
    _, _, _, p2n = cost_panel({'den': [gb], 'num': [gb]}, [FIX_FAM], [256, 1024], Cf, rows, 'num', 'den')
    out['q2'] = dict(injected=p2[FIX_FAM], null=p2n[FIX_FAM],
                     ok=bool(p2[FIX_FAM]['label'] == 'hardware-dependent' and p2n[FIX_FAM]['label'] == 'hardware-consistent'
                             and p2n[FIX_FAM]['lo'] == 1. and p2n[FIX_FAM]['hi'] == 1.), cases=Cf)
    # q3
    x = job1['cross']
    rom = x['roms'][0]
    fom = x['pinned'][rom]['|H|union|joint']
    Cx = [c for c in x['cases'] if len(x['vals'].get(rom, {}).get(c, ())) >= MIN_REPS
          and len(x['vals'].get(fom, {}).get(c, ())) >= MIN_REPS]
    if len(Cx) < MIN_CASES:
        out['q3'] = dict(ok=False, reason=f'fixture case set too small ({len(Cx)})')
        return out
    rx = boot_rows(len(Cx))
    R = {c: x['vals'][rom][c] for c in Cx}
    F = {c: x['vals'][fom][c] for c in Cx}
    Fi = {c: F[c] * INJECT for c in Cx}
    _, p3 = cross_panel({'den': [dict(cross={rom: R, fom: F})], 'num': [dict(cross={rom: R, fom: Fi})]}, [(rom, fom)], Cx, rx,
                        'num', 'den')
    _, p3n = cross_panel({'den': [dict(cross={rom: R, fom: F})], 'num': [dict(cross={rom: R, fom: F})]}, [(rom, fom)], Cx, rx,
                         'num', 'den')
    k = f'{rom}|{fom}'
    out['q3'] = dict(injected=p3[k], null=p3n[k], ok=bool(p3[k]['label'] == 'hardware-dependent'
                                                          and p3n[k]['label'] == 'hardware-consistent'
                                                          and p3n[k]['lo'] == 1. and p3n[k]['hi'] == 1.), cases=Cx)
    return out


# ------------------------------------------------------------------ per job ----

def per_job(job):
    """Per-job values: H-cost-N(2) (P1 h_cost_n unchanged), three-mesh on H200, S (c2analyze.speedup unchanged), q1 / q2."""
    out = dict(status=job['status'])
    if 'cost' not in job:
        return out
    ms = job['cost_meshes']
    hc = {}
    for f in FAMS:
        if f not in job['cost_full'] or any(L not in ms for L in CROSS_MESHES):
            hc[f] = dict(label='unavailable (missing subject/mesh)')
            continue
        r, v = job['cost_full'][f]
        ix = [ms.index(L) for L in CROSS_MESHES]
        o = S1.h_cost_n(r[:, ix], v[:, ix], list(CROSS_MESHES))
        lab = o.get('procedure_outcome') or ('inconclusive' if str(o.get('outcome', '')).startswith('inconclusive')
                                             else 'not reached')
        ent = dict(two_mesh=o, label=lab)
        if 4096 in ms:
            ent['three_mesh_h200_only'] = S1.h_cost_n(r, v, ms)
        hc[f] = ent
    out['h_cost_n2'] = hc
    q1 = {}
    for (f, L), d in job['cost'].items():
        allv = np.concatenate(list(d.values())) if d else np.array([])
        q1[f'{f}|{L}'] = float(np.median(allv)) if allv.size else None
    out['q1'] = q1
    out['q2'] = {f: (max(q1[f'{f}|{L}'] for L in CROSS_MESHES) / min(q1[f'{f}|{L}'] for L in CROSS_MESHES))
                 if all(q1.get(f'{f}|{L}') for L in CROSS_MESHES) else None for f in FAMS}
    x = job['cross']
    S = {}
    for rom, d in x['pinned'].items():
        for suf, fom in d.items():
            sp = C2A.speedup(C2A.per_case(x['stage2'], rom), C2A.per_case(x['stage2'], fom))
            sp['label'] = s2_label(sp.get('lo'), sp.get('hi')) if sp.get('status') == 'ok' else 'unavailable (invalid timing)'
            S[f'{rom}|{fom}'] = dict(sp, suffix=suf)
    out['S'] = S
    out['seconds_evolve'] = job['seconds_evolve']
    out['seconds_B_cross'] = x['seconds_B']
    out['c_e5_1'] = job['c_e5_1']
    out['cross_gates'] = x.get('gates')
    return out


# ------------------------------------------------------------------ positive gates G-E5.1 / G-E5.2 ----

def p1_label(sp):
    """p1time's subject label."""
    return f"{sp['mesh']}|R{sp['Rp']}|{sp['rule']}|{sp.get('policy', 'frozen')}|{sp['scheme']}"


def registered_cases(which):
    if isinstance(which, str):
        return list(NAMED_CASES[which])
    return [f'{c}|{int(i)}' for c, i in which]


def g_e5_1(job, p1ref, pc):
    """G-E5.1 (required): for the COMPLETE registered set (every P-cost subject of the job config incl. A x every
    registered case) the scoring-run coefficients W exist in the job and in P1c, are finite, and agree to <= 1e-10
    relative. Anything missing, partial or non-finite fails the gate; no p1ref fails it."""
    out = dict(passed=False, rows={}, tol=REPRO_TOL)
    if not p1ref:
        out['reason'] = 'no --p1ref given (G-E5.1 is required)'
        return out
    cfg = pc.get('config') or {}
    try:
        labels = sorted({p1_label(sp) for sp in [cfg['A']] + list(cfg['subjects'])})
        cases = registered_cases(cfg['cases'])
    except (KeyError, TypeError) as e:
        out['reason'] = f'job config lacks the registered set ({e!r})'
        return out
    out.update(subjects=labels, cases=cases)
    if not labels or not cases:
        out['reason'] = 'empty registered set'
        return out
    ok = True
    for lb in labels:
        name = f"S_{lb.replace('|', '_')}.npz"
        pa, pb = Path(job['dir']) / 'pcost' / name, Path(p1ref) / name
        row = dict(file=name)
        if not pa.exists() or not pb.exists():
            row.update(status='missing ' + ('job file' if not pa.exists() else 'P1 reference file'), passed=False)
        else:
            a, b = np.load(pa), np.load(pb)
            ia = {str(c): i for i, c in enumerate(a['case_ids'])}
            ib = {str(c): i for i, c in enumerate(b['case_ids'])}
            miss = [c for c in cases if c not in ia or c not in ib]
            if miss:
                row.update(status='missing cases', missing=miss, passed=False)
            else:
                Wa = np.stack([a['W'][ia[c]] for c in cases])
                Wb = np.stack([b['W'][ib[c]] for c in cases])
                fin = bool(np.isfinite(Wa).all() and np.isfinite(Wb).all())
                den = float(np.max(np.abs(Wb))) if fin else float('nan')
                rel = float(np.max(np.abs(Wa - Wb)) / den) if fin and den > 0 else float('inf')
                row.update(status='ok' if fin else 'non-finite W', cases=len(cases), max_rel=rel,
                           bitwise=bool(fin and np.array_equal(Wa, Wb)), passed=bool(fin and den > 0 and rel <= REPRO_TOL))
        ok = ok and row['passed']
        out['rows'][lb] = row
    out['passed'] = bool(ok and len(out['rows']) == len(labels))
    return out


def p1_reference_q1(p1ref):
    """G-E5.2 reference: P1c's median B/A per (family, mesh) and its case-bootstrap 95 % interval (same estimator)."""
    r = json.loads((Path(p1ref) / 'result.json').read_text())
    ex = extract_cost(r)
    out = {}
    for (f, L), d in ex['cost'].items():
        if f not in FAMS:
            continue
        C = [c for c in ex['cost_cases'] if c in d]
        if len(C) < MIN_CASES:
            continue
        rows = boot_rows(len(C))
        Q1, _, _, _ = cost_panel({'p1': [dict(cost={(f, L): d})]}, [f], [L], C, rows, 'p1', 'p1')
        pt, bs = Q1['p1'][(f, L)]
        lo, hi = interval(bs)
        out[f'{f}|{L}'] = dict(q1=pt, lo=lo, hi=hi)
    return dict(job_id=r.get('job_id'), gpu=r.get('gpu'), cells=out)


# ------------------------------------------------------------------ main analysis ----

def analyze(jobs, expect, p1ref=None):
    """jobs: list of load_job() dicts; expect: {type: count}."""
    res = dict(label='conditional on the observed devices (two GPUs per type cannot represent a hardware population); '
                     'frozen H200-selected comparators; two-mesh P-cost panel (256^2, 1024^2) for every cross-type quantity',
               seed=SEED, nboot=NBOOT, jobs=[], per_job={}, fixtures=None, case_sets={}, pooled={}, phi={}, per_type={},
               agreement={}, unit_to_unit={}, gates={})
    by = {t: [j for j in jobs if j['type'] == t] for t in expect}
    for t, n in expect.items():
        while len(by[t]) < n:
            by[t].append(dict(type=t, dir=None, status='missing', reasons=['not provided']))
    for t in expect:
        for i, j in enumerate(by[t]):
            name = f'{t}#{i + 1}'
            j['name'] = name
            res['jobs'].append(dict(name=name, dir=j.get('dir'), status=j['status'], reasons=j.get('reasons'),
                                    meta=j.get('top_meta'), records=j.get('records')))
            pj = per_job(j)
            if j['status'] != 'ok' and 'cost' in j:
                pj['scope'] = 'per-job only'
            res['per_job'][name] = pj
    complete = {t: len(by[t]) >= expect[t] and all(j['status'] == 'ok' for j in by[t][:expect[t]]) for t in expect}
    res['types_complete'] = complete
    # fixtures from the first A100 job
    j1 = by[TYPES[0]][0] if by.get(TYPES[0]) else None
    fx = fixtures(j1) if j1 is not None and j1['status'] == 'ok' else \
        {k: dict(ok=False, reason='fixture job (A100 job 1) not available') for k in ('q1', 'q2', 'q3')}
    res['fixtures'] = fx
    fok = {k: bool(v.get('ok')) for k, v in fx.items()}
    cross_types = all(t in expect for t in TYPES)
    all_ok = all(complete.values()) and cross_types
    ok_jobs = {t: by[t][:expect[t]] for t in expect if complete[t]}

    def panel_sets(groups):
        req_c = groups[next(iter(groups))][0]['cost_cases']
        need_c = [(f, L) for f in FAMS for L in CROSS_MESHES]
        Cc = case_set(groups, need_c, req_c, 'cost')
        x0 = groups[next(iter(groups))][0]['cross']
        need_x = x0['roms'] + x0['foms']
        Cx = case_set({g: [dict(cross=j['cross']['vals']) for j in js] for g, js in groups.items()}, need_x, x0['cases'],
                      'cross')
        pairs = [(r, f) for r, d in x0['pinned'].items() for f in sorted(set(d.values()))]
        return dict(requested_cost=req_c, C_cost=Cc, requested_cross=x0['cases'], C_cross=Cx, pairs=pairs)

    def run_panels(groups, num, den, sets):
        out = {}
        xg = {g: [dict(cross=j['cross']['vals']) for j in js] for g, js in groups.items()}
        if len(sets['C_cost']) >= MIN_CASES:
            Q1, Q2, p1, p2 = cost_panel(groups, list(FAMS), list(CROSS_MESHES), sets['C_cost'], boot_rows(len(sets['C_cost'])),
                                        num, den)
            out.update(Q1=Q1, Q2=Q2, phi1=p1, phi2=p2)
        if len(sets['C_cross']) >= MIN_CASES:
            Q3, p3 = cross_panel(xg, sets['pairs'], sets['C_cross'], boot_rows(len(sets['C_cross'])), num, den)
            out.update(Q3=Q3, phi3=p3)
        return out

    def type_labels(t, out, sets):
        lab = {}
        for f in FAMS:
            jl = [res['per_job'][j['name']]['h_cost_n2'][f]['label'] for j in by[t]]
            if 'Q2' in out:
                lo, hi = interval(out['Q2'][t][f][1])
                pooled = cost_label(lo, hi)
                lab[f'H-cost-N(2)|{f}'] = dict(label=type_label(pooled, jl), pooled=pooled, q2=out['Q2'][t][f][0], lo=lo,
                                               hi=hi, per_job=jl)
            else:
                lab[f'H-cost-N(2)|{f}'] = dict(label=f"unavailable (|C_cost| = {len(sets['C_cost'])} < {MIN_CASES})",
                                               per_job=jl)
        for r, f in sets['pairs']:
            jl = [res['per_job'][j['name']]['S'][f'{r}|{f}']['label'] for j in by[t]]
            if 'Q3' in out:
                lo, hi = interval(out['Q3'][t][(r, f)][1])
                pooled = s2_label(lo, hi)
                lab[f'S2|{r}|{f}'] = dict(label=type_label(pooled, jl), pooled=pooled, q3=out['Q3'][t][(r, f)][0], lo=lo,
                                          hi=hi, per_job=jl)
            else:
                lab[f'S2|{r}|{f}'] = dict(label=f"unavailable (|C_cross| = {len(sets['C_cross'])} < {MIN_CASES})", per_job=jl)
        return lab

    def blocked(k):
        return f'blocked (fixture {k} did not fire)'

    if all_ok:
        sets = panel_sets(ok_jobs)
        res['case_sets'] = sets
        out = run_panels(ok_jobs, TYPES[1], TYPES[0], sets)
        for k, kk, nm in (('phi1', 'q1', 'C_cost'), ('phi2', 'q2', 'C_cost'), ('phi3', 'q3', 'C_cross')):
            if k in out:
                res['phi'][k] = {c: dict(v, label=v['label'] if fok[kk] else blocked(kk)) for c, v in out[k].items()}
            else:
                res['phi'][k] = f'unavailable (|{nm}| = {len(sets[nm])} < {MIN_CASES})'
        res['pooled'] = {t: dict(q1={f'{f}|{L}': out['Q1'][t][(f, L)][0] for f in FAMS for L in CROSS_MESHES} if 'Q1' in out else None,
                                 q2={f: out['Q2'][t][f][0] for f in FAMS} if 'Q2' in out else None,
                                 q3={f'{r}|{f}': out['Q3'][t][(r, f)][0] for r, f in sets['pairs']} if 'Q3' in out else None)
                         for t in expect}
        for t in expect:
            res['per_type'][t] = type_labels(t, out, sets)
        for t in expect:                                    # unit-to-unit baseline (no required outcome)
            if len(by[t]) >= 2:
                g = {'job1': [by[t][0]], 'job2': [by[t][1]]}
                o = run_panels(g, 'job2', 'job1', sets)
                res['unit_to_unit'][t] = {k: o.get(k) for k in ('phi1', 'phi2', 'phi3')}
    else:
        miss = 'unavailable (job missing)' if cross_types else 'unavailable (cross-type pair not requested)'
        res['phi'] = {k: miss for k in ('phi1', 'phi2', 'phi3')}
        for t in expect:
            if not complete[t]:
                res['per_type'][t] = 'unavailable (job missing)'
                continue
            sets = panel_sets({t: ok_jobs[t]})
            out = run_panels({t: ok_jobs[t]}, t, t, sets)
            lab = type_labels(t, out, sets)
            res['per_type'][t] = {k: dict(v, scope='per-type only (case set over this type\'s jobs; cross-type set '
                                                   'unavailable, A4.1)') for k, v in lab.items()}
            res['case_sets'][t] = sets
    # verdict agreement (4.3) with fixture blocks
    items = sorted({k for v in res['per_type'].values() if isinstance(v, dict) for k in v})
    if not items:
        items = [f'H-cost-N(2)|{f}' for f in FAMS]
    for it in items:
        kk = 'q2' if it.startswith('H-cost-N') else 'q3'
        if not all_ok:
            res['agreement'][it] = 'unavailable (job missing)' if cross_types else 'unavailable (single type)'
        elif not fok[kk]:
            res['agreement'][it] = blocked(kk)
        else:
            res['agreement'][it] = agreement({t: res['per_type'][t][it]['label'] for t in expect})
    # positive gates: G-E5.1 already decided each job's status in load_job (required); G-E5.2 descriptive
    ref = None
    if p1ref:
        try:
            ref = p1_reference_q1(p1ref)
        except Exception as e:  # noqa: BLE001
            ref = dict(error=repr(e), cells={})
        res['gates']['p1_reference'] = ref
    for t in expect:
        for j in by[t]:
            g2 = {}
            if ref and t == 'H200':
                for c, v in res['per_job'][j['name']].get('q1', {}).items():
                    rv = ref['cells'].get(c)
                    if rv and v is not None and rv['lo'] is not None:
                        g2[c] = dict(job=v, p1=rv['q1'], p1_lo=rv['lo'], p1_hi=rv['hi'], within=bool(rv['lo'] <= v <= rv['hi']))
            res['gates'][j['name']] = {'G-E5.1': j.get('g_e5_1', dict(passed=False, reason='job not loaded')),
                                       'P-cross gates': j.get('pcross_gates'), 'G-E5.2 (descriptive, H200)': g2 or None}
    return res


def smoke(job):
    """Single-job smoke acceptance (code audit 3 item 27): no cross-type pooling; every required control / gate is an
    explicit pass/fail and ALL is their conjunction (a failed or blocked job makes ALL false)."""
    C = {}

    def chk(name, ok, detail=None):
        C[name] = dict(ok=bool(ok), detail=detail)
    chk('job status ok (complete, GPU type, backend, sources, identities, equivalence, gates)', job['status'] == 'ok',
        dict(status=job['status'], reasons=job.get('reasons')))
    loaded = 'cost' in job
    ce = job.get('c_e5_1') or dict(fires=False)
    chk('C-E5.1 doubled/single case-median in [1.6, 2.4] for every doubled subject', loaded and ce.get('fires'), ce)
    fx = fixtures(job) if loaded else {k: dict(ok=False, reason='job not loaded') for k in ('q1', 'q2', 'q3')}
    for k in ('q1', 'q2', 'q3'):
        f = fx[k]
        chk(f'fixture {k}: injected x1.5 hardware-dependent, null consistent [1, 1]', f.get('ok'),
            {kk: f.get(kk) for kk in ('injected', 'null', 'reason')})
    g1 = job.get('g_e5_1') or dict(passed=False, reason='job not loaded')
    chk('G-E5.1 reproduction of P1c (complete registered set, <= 1e-10)', g1.get('passed'), g1)
    pg = job.get('pcross_gates') or {}
    for k in PCROSS_GATES:
        chk(f'P-cross gate {k}', pg.get(k) is True, pg.get(k))
    pj = per_job(job) if loaded else {}
    return dict(mode='single-job smoke (no cross-type pooling)', job=job.get('dir'), type=job['type'], checks=C,
                ALL=bool(C) and all(v['ok'] for v in C.values()), per_job=pj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', action='append', default=[], help='TYPE=DIR (DIR holds result.json, COMPLETE, pcost/)')
    ap.add_argument('--expect', default='A100-80G=2,H200=2')
    ap.add_argument('--p1ref', default=None, help="P1c p1time output dir (result.json, S_*.npz)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true', help='single-job smoke acceptance; exits 1 unless ALL')
    a = ap.parse_args()
    jobs = [load_job(*x.split('=', 1), p1ref=a.p1ref) for x in a.job]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        if len(jobs) != 1:
            raise SystemExit('--smoke takes exactly one --job')
        r = smoke(jobs[0])
        Path(a.out).write_text(json.dumps(clean(r), indent=1) + '\n')
        for k, v in r['checks'].items():
            print('OK  ' if v['ok'] else 'FAIL', k)
        print('ALL', r['ALL'])
        raise SystemExit(0 if r['ALL'] else 1)
    expect = {k: int(v) for k, v in (x.split('=') for x in a.expect.split(','))}
    res = analyze(jobs, expect, a.p1ref)
    Path(a.out).write_text(json.dumps(clean(res), indent=1) + '\n')
    print('E5 types complete', res['types_complete'], 'fixtures', {k: v.get('ok') for k, v in res['fixtures'].items()})
    for k, v in res['agreement'].items():
        print('AGREEMENT', k, v)


def clean(x):
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else '|'.join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist()) if x.size <= 64 else dict(shape=list(x.shape), median=float(np.median(x)))
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == '__main__':
    main()
