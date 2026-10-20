"""Final results report of the viscous Burgers 2D/3D JCP campaign (AGENT-PROMPT-burgers.md section 8).

Reads every number from the lane outputs (runs/EXPORT.json, checks/*.json, runs/GPU-LEDGER.json, git refs) and
writes report.md, figs/*.png, FINAL-EXPORT.json and PROVENANCE.json next to this file. No measured number is typed
by hand: every value printed in report.md is registered in PROVENANCE.json with its source file, JSON key path and the
file's sha256 (or, for derived values, every source and the derivation). Registered design constants (bars such as
1.25, 2x, 2.5e-4) are quoted as constants of the plan, not as results.

jcp-robust and jcp-regimes (priority 4) are read from their runs/EXPORT.json when present; otherwise their sections
are an explicit "pending" block with the lane's status at compile time.

Run: JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python make_final.py
"""
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
WT = ROOT / 'worktrees'
FIGS = HERE / 'figs'

LANES = [  # (lane, worktree dir, plan items, priority)
    ('jcp-speed3d', '2026-10-14-jcp-speed3d', '3D speed, f1 FOM ladder, m1 mesh panel, H2, K2 (3D)', 'speed3d'),
    ('jcp-p1-scaling', '2026-10-17-jcp-p1-scaling', 'P1a-P1g, Q3: H-cost-N/r, H-acc-N/r, P1g', '1'),
    ('jcp-qualify', '2026-10-18-jcp-qualify', 'Q1-Q3, B1, B4, B5, FRONT-2D, 3D Q2 subset', '2'),
    ('jcp-fom2d', '2026-10-18-jcp-fom2d', 'C1 ladder, C2 crossover, H-comp 2D, K2 (2D)', '2'),
    ('jcp-hr-baselines', '2026-10-06-jcp-hr-baselines', 'C3 minimal, H-base', '2'),
    ('jcp-heads', '2026-10-18-jcp-heads', 'C5 K sweep, linear span vs head', '3'),
    ('jcp-dims', '2026-10-18-jcp-dims', 'A3, A4, B3, B6, 3D front, H2q', '3'),
    ('jcp-budget', '2026-10-18-jcp-budget', 'B2, B7, B8', '3'),
    ('jcp-robust', '2026-10-19-jcp-robust', 'E1, E2, E5', '4'),
    ('jcp-regimes', '2026-10-19-jcp-regimes', 'D1', '4'),
]
LANE_DIR = {l: WT / w / 'experiments' / l for l, w, _, _ in LANES}
LANE_WT = {l: WT / w for l, w, _, _ in LANES}
PENDING_LANES = ('jcp-robust', 'jcp-regimes')

# ----------------------------------------------------------------------------------------------------------------
# provenance registry


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class Reg:
    def __init__(self):
        self.prov = {}
        self._json = {}
        self._sha = {}

    def load(self, path):
        path = Path(path)
        if path not in self._json:
            self._json[path] = json.load(open(path))
            self._sha[path] = sha256(path)
        return self._json[path]

    def src(self, path, keypath, filt=None):
        """keypath: list of keys ('*' = every key at that level), or a '/'-separated string of such keys."""
        path = Path(path)
        self.load(path)
        kp = keypath.split('/') if isinstance(keypath, str) else [str(k) for k in keypath]
        d = dict(file=rel(path), key_path=kp, sha256=self._sha[path])
        if filt:
            d['filter'] = filt
        return d

    def get(self, nid, path, *keys):
        o = self.load(path)
        for k in keys:
            o = o[k]
        self._put(nid, dict(value=o, **self.src(path, [str(k) for k in keys])))
        return o

    def derived(self, nid, value, sources, how):
        self._put(nid, dict(value=value, derivation=how, sources=sources))
        return value

    def git(self, nid, wt, ref):
        v = subprocess.check_output(['git', '-C', str(wt), 'rev-parse', '--short=9', ref], text=True).strip()
        self._put(nid, dict(value=v, file=f'git:{rel(wt)}', key_path=f'rev-parse {ref}', sha256=None))
        return v

    def _put(self, nid, rec):
        if nid in self.prov and self.prov[nid].get('value') != rec.get('value'):
            raise SystemExit(f'provenance id {nid} registered twice with different values')
        self.prov[nid] = rec


R = Reg()


def E(lane):
    return LANE_DIR[lane] / 'runs' / 'EXPORT.json'


# ----------------------------------------------------------------------------------------------------------------
# formatting


def g3(x):
    if x is None:
        return '–'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f'{x:,}'
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    a = abs(x)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f'{x:.3g}'
    return f'{x:.3g}'


def pct(x):
    return '–' if x is None else f'{100 * x:.3g} %'


def cell(c):
    return str(c).replace('|', '\\|')


def table(head, rows):
    out = ['| ' + ' | '.join(cell(h) for h in head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        out.append('| ' + ' | '.join(cell(c) for c in r) + ' |')
    return '\n'.join(out)


def latexify(s):
    """Render plain exponents such as 127^3 from lane text as LaTeX."""
    import re
    return re.sub(r'(?<![$\w])(\d+)\^(\d+)', r'$\1^{\2}$', s)


L = []


def w(s=''):
    L.append(s)


# ----------------------------------------------------------------------------------------------------------------
# figures


def copy_fig(lane, name, newname):
    src = LANE_DIR[lane] / 'plots' / name
    FIGS.mkdir(exist_ok=True)
    dst = FIGS / newname
    shutil.copyfile(src, dst)
    R._put(f'fig:{newname}', dict(value=f'figs/{newname}', file=rel(src), key_path='(copied PNG)', sha256=sha256(src)))
    return f'figs/{newname}'


def fig(path, caption):
    w(f'![{caption}]({path})')
    w(f'*{caption}*')
    w()


INK = '#1c2629'; MUTE = '#5b6a6c'; GRID = '#d6dedb'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=8)


# ================================================================================================================
def main():
    now = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M UTC')
    FIGS.mkdir(exist_ok=True)
    P1, S3, QL, F2, HB, HD, DM, BG = (E(l) for l in ('jcp-p1-scaling', 'jcp-speed3d', 'jcp-qualify', 'jcp-fom2d',
                                                      'jcp-hr-baselines', 'jcp-heads', 'jcp-dims', 'jcp-budget'))
    QLA = LANE_DIR['jcp-qualify'] / 'checks' / 'analysis.json'
    BG3 = LANE_DIR['jcp-budget'] / 'checks' / 'b3-b3d.json'
    p1, s3, ql, f2, hb, hd, dm, bg = (R.load(x) for x in (P1, S3, QL, F2, HB, HD, DM, BG))
    qla = R.load(QLA)
    verdicts = []  # rows for the verdict table and FINAL-EXPORT

    def V(hyp, reading, scope, verdict, key, figure, lane):
        scope, key = latexify(scope), latexify(key)
        verdicts.append(dict(hypothesis=hyp, reading=reading, scope=scope, verdict=verdict, key_number=key,
                             figure=figure, lane=lane))

    # ---------------- H-cost-N --------------------------------------------------------------------------------
    hcn = p1['verdicts']['h_cost_n']
    Rs = {k: R.get(f'p1.hcn.R.{k}', P1, 'detail', 'h_cost_n', k, 'R') for k in hcn}
    ubs = {k: R.get(f'p1.hcn.ci_hi.{k}', P1, 'detail', 'h_cost_n', k, 'ci')[1] for k in hcn}
    n_pass_cn = R.derived('p1.hcn.n_pass', sum(v == 'pass' for v in hcn.values()),
                          [R.src(P1, 'verdicts/h_cost_n/*')], 'count of "pass"')
    hcn_txt = (f'{n_pass_cn}/{len(hcn)} settings pass; R = {g3(min(Rs.values()))}–{g3(max(Rs.values()))}, '
               f'largest 95 % upper end {g3(max(ubs.values()))} (bar 1.25)')
    V('H-cost-N', 'r (bank width)', '2D, 256²–4096², R128–R384, BE/CN-R, frozen + nominal mode lists; evolution time '
      'with internal-state recording (not the full query)',
      'pass' if n_pass_cn == len(hcn) else 'fail', hcn_txt, 'figs/summary_cost_vs_mesh.png', 'jcp-p1-scaling')
    h41 = s3['m1']['H41']
    mm41 = {k: R.get(f's3.H41.maxmin.{k}', S3, 'm1', 'H41', k, 'max_over_min') for k in h41}
    n41 = R.derived('s3.H41.n_pass', sum(v['verdict'] == 'pass' for v in h41.values()), [R.src(S3, 'm1/H41/*/verdict')],
                    'count of "pass"')
    # quadrature qualification of the four m1 ROMs (dims h2q comparators, qualify 3D Q2 for the winners)
    s3h2 = s3['H2']
    q2_m1 = {}
    for comp in ('C_old', 'C_W768', 'C_W1024'):
        q2_m1[comp] = R.get(f'dm.h2q.{comp}.comparator_q2', DM, 'h2q', comp, 'comparator_q2')
        # the comparator dims qualified must be the configuration speed3d timed under that name
        assert dm['h2q'][comp]['comparator'] == s3['r1']['roms'][f'{comp}|mesh']['key'], comp
    win_old = R.get('s3.H2.C_old.winner', S3, 'H2', 'C_old', 'winner')
    win_768 = R.get('s3.H2.C_W768.winner', S3, 'H2', 'C_W768', 'winner')
    q2_win_old = R.get('ql.3d.win_old.label', QLA, 'threeD', 'tested', win_old, 'label')
    q2_win_768 = R.get('ql.3d.win_768.label', QLA, 'threeD', 'tested', win_768, 'label')
    V('H-cost-N', 'r (bank width)', '3D, 64³–384³ (H4.1), mesh-free contract, deployment mode policy, 4 ROMs',
      'pass' if n41 == len(h41) else 'fail',
      f'{n41}/{len(h41)} ROMs flat, max/min {g3(min(mm41.values()))}–{g3(max(mm41.values()))} (bar 1.25); '
      f'quadrature check (Q2) of the four ROMs: C_old {q2_m1["C_old"]}, C_W1024 {q2_m1["C_W1024"]}, '
      f'W_C_old {q2_win_old}, W_C_W768 {q2_win_768} — not all are qualified configurations',
      'figs/summary_cost_vs_mesh.png', 'jcp-speed3d')
    hcnK = hd['h_cost_n']
    RK16 = R.get('hd.hcn.16.R', HD, 'h_cost_n', '16', 'R')
    RKinc = R.get('hd.hcn.inc.R', HD, 'h_cost_n', 'inc', 'R')
    notK = [k for k, v in hcnK.items() if v['outcome'] != 'pass']
    R.derived('hd.hcn.not_reached', notK, [R.src(HD, 'h_cost_n/*/outcome')], 'keys whose outcome is not "pass"')
    V('H-cost-N', 'K (head latent)', '2D, 256²–4096², seed 0 heads, BE, evolution only',
      'pass (K = 16, incumbent); not reached for K = ' + ', '.join(notK),
      f'K = 16: {g3(RK16)}; incumbent K = 16: {g3(RKinc)} (bar 1.25)', 'figs/heads_cost_vs_K.png', 'jcp-heads')

    # ---------------- H-cost-r ---------------------------------------------------------------------------------
    for dim in ('2d', '3d'):
        cr = p1['verdicts'][f'h_cost_r_{dim}']
        fits = {k: R.get(f'p1.hcr.{dim}.{k}.p_fit', P1, 'detail', f'h_cost_r_{dim}', k, 'p_fit') for k in cr}
        decl = {k: R.get(f'p1.hcr.{dim}.{k}.p_decl', P1, 'detail', f'h_cost_r_{dim}', k, 'p_decl') for k in cr}
        npass = R.derived(f'p1.hcr.{dim}.n_pass', sum(v == 'pass' for v in cr.values()),
                          [R.src(P1, f'verdicts/h_cost_r_{dim}/*')], 'count of "pass"')
        V('H-cost-r', 'r (bank width)', f'{dim.upper()} component kernels at fixed (M, m)',
          'fail' if npass < len(cr) else 'pass',
          f'{npass}/{len(cr)} components follow the declared exponent; fitted p: ' +
          ', '.join(f'{k} {g3(fits[k])} (declared {g3(decl[k])})' for k in cr),
          'figs/p1_kernel_cost_vs_r_2D.png' if dim == '2d' else 'figs/p1_kernel_cost_vs_r_3D.png', 'jcp-p1-scaling')
    beta = R.get('dm.elasticity.beta', DM, 'elasticity_2d', 'beta')
    V('H-cost-r', 'r (bank width)', '2D full query, t2 timing panel (descriptive elasticities, no verdict)',
      'descriptive', f'elasticity of query time in r {g3(beta[1])}, in M {g3(beta[2])}, in m {g3(beta[3])}',
      '–', 'jcp-dims')
    idK = {m: R.get(f'hd.costK.{m}.identified', HD, 'cost_vs_K', m, 'fit', 'identified') for m in ('256', '1024', '4096')}
    pK = R.get('hd.costK.4096.p', HD, 'cost_vs_K', '4096', 'fit', 'point', 'p')
    V('H-cost-r', 'K (head latent)', '2D, cost against K = 4–32 (descriptive)', 'inconclusive',
      f'exponent identified at {sum(idK.values())} of 3 meshes (4096²: p = {g3(pK)}, seed 0 only)',
      'figs/heads_cost_vs_K.png', 'jcp-heads')

    # ---------------- H-acc-N ----------------------------------------------------------------------------------
    han = p1['verdicts']['h_acc_n']
    n_an = R.derived('p1.han.n_pass', sum(v == 'pass' for v in han.values()), [R.src(P1, 'verdicts/h_acc_n/*')],
                     'count of "pass"')
    trend_p = R.get('p1.han.R128CNR.trend_p', P1, 'detail', 'h_acc_n', 'R128|CNR|frozen', 'trend', 'p')
    V('H-acc-N', 'r (bank width)', '2D, 256² vs 1024² (38 cases) and 4096² vs 1024² (16 cases), P1 settings',
      'pass' if n_an == len(han) else 'fail',
      f'{n_an}/{len(han)} pass (every case within margin); R128 CN-R has a detectable small trend '
      f'(Wilcoxon p = {g3(trend_p)}): equivalence within margin, not zero dependence', 'figs/p1_error_vs_r_and_mesh_CNR.png',
      'jcp-p1-scaling')
    qhan = ql['h_acc_n']
    cnt = {}
    for v in qhan.values():
        cnt[v] = cnt.get(v, 0) + 1
    R.derived('ql.han.counts', cnt, [R.src(QL, 'h_acc_n/*')], 'count of outcomes')
    V('H-acc-N', 'r (bank width)', f'2D, extended scheme / step set (B1, {len(qhan)} tests)',
      f'pass {cnt.get("pass", 0)}/{len(qhan)}, inconclusive {len(qhan) - cnt.get("pass", 0)}, no fail'
      if 'fail' not in cnt else 'fail in some',
      '; '.join(f'{k} {n}' for k, n in sorted(cnt.items())), 'figs/qualify_error_vs_N.png', 'jcp-qualify')

    # ---------------- H-acc-r ----------------------------------------------------------------------------------
    har = p1['verdicts']['h_acc_r']
    r384 = {k: R.get(f'p1.har.{k}.range_128_384', P1, 'verdicts', 'h_acc_r', k, 'range_128_384') for k in har}
    r512 = {k: R.get(f'p1.har.{k}.range_128_512', P1, 'verdicts', 'h_acc_r', k, 'range_128_512')
            for k in har if 'range_128_512' in har[k]}
    V('H-acc-r', 'r (bank width)', '2D, r = 128 → 256 → 384, P1 cells (256², 1024², 4096² × BE, CN-R)',
      'pass' if all(v == 'monotone' for v in r384.values()) else 'fail',
      f'{sum(v == "monotone" for v in r384.values())}/{len(r384)} cells monotone', 'figs/p1_error_vs_r_and_mesh_CNR.png',
      'jcp-p1-scaling')
    V('H-acc-r', 'r (bank width)', '2D, registered range r = 128–512 (1024² only)',
      'fail' if any(v != 'monotone' for v in r512.values()) else 'pass',
      f'{sum(v == "not monotone" for v in r512.values())}/{len(r512)} cells not monotone (384 → 512 does not improve)',
      'figs/p1_error_vs_r_and_mesh_CNR.png', 'jcp-p1-scaling')
    qhar = ql['h_acc_r']
    c384, c512 = {}, {}
    for v in qhar.values():
        c384[v['range_128_384']] = c384.get(v['range_128_384'], 0) + 1
        c512[v['range_128_512']] = c512.get(v['range_128_512'], 0) + 1
    R.derived('ql.har.c384', c384, [R.src(QL, 'h_acc_r/*/range_128_384')], 'count of outcomes')
    R.derived('ql.har.c512', c512, [R.src(QL, 'h_acc_r/*/range_128_512')], 'count of outcomes')
    V('H-acc-r', 'r (bank width)', f'2D, extended scheme / step set (B1, {len(qhar)} cells)', 'mixed',
      'range 128–384: ' + ', '.join(f'{k} {n}' for k, n in sorted(c384.items())) + '; range 128–512: ' +
      ', '.join(f'{k} {n}' for k, n in sorted(c512.items())), 'figs/qualify_error_vs_r.png', 'jcp-qualify')
    a3 = R.get('dm.A3_2d.fixedMm.range', DM, 'A3_2d', 'fixed (M=2048, gauss256)', 'range')
    V('H-acc-r', 'r (bank width)', '2D, fixed (M, m) = (2048, Gauss 256²) (A3)', 'fail' if a3 != 'monotone' else 'pass',
      f'range {a3}', '–', 'jcp-dims')
    hk = R.get('hd.hacck.BE.range', HD, 'h_acc_k', 'BE', 'range')
    V('H-acc-r', 'K (head latent)', '2D 1024², K = 4 → 8 → 16 → 32 (BE)', hk,
      'too few numerically qualified seeds per K for a pooled verdict', 'figs/heads_error_vs_K.png', 'jcp-heads')

    # ---------------- H-comp + K2 ------------------------------------------------------------------------------
    hc = f2['c2']['hcomp']
    comp_counts = {}
    for k, v in hc.items():
        if not v['scope'].startswith('primary'):
            continue
        mesh, fam = k.split('|')[0], v['family']
        comp_counts.setdefault(mesh, {}).setdefault(fam, {})
        comp_counts[mesh][fam][v['cls']] = comp_counts[mesh][fam].get(v['cls'], 0) + 1
    R.derived('f2.hcomp.primary.counts', comp_counts, [R.src(F2, ['c2', 'hcomp', '*', f]) for f in ('scope', 'family', 'cls')],
              'count of cls per (mesh, family) over primary-scope (LSPG) cells')
    fam_name = {'UB': 'upwind + BE', 'CC': 'central + CN-R', 'PS': 'sine pseudo-spectral', 'union': 'union of families'}
    for mesh in sorted(comp_counts, key=int):
        for fam in ('UB', 'CC', 'PS', 'union'):
            c = comp_counts[mesh].get(fam, {})
            ver = ('fail' if set(c) == {'ROM < 2x'} else 'pass' if set(c) == {'ROM >= 2x'} else
                   'mixed (pass only against the screening-selected comparator)' if 'ROM >= 2x' in c else
                   'fail or unresolved')
            V('H-comp', 'r (bank width)', f'2D {mesh}², vs {fam_name[fam]}, {sum(c.values())} qualified LSPG configurations',
              ver, ', '.join(f'{k} ×{n}' for k, n in sorted(c.items())), 'figs/summary_union_speedup.png', 'jcp-fom2d')
    k2m = f2['c2']['k2']['members']
    S2 = {k: R.get(f'f2.k2.S.{i}', F2, 'c2', 'k2', 'members', k, 'S') for i, k in enumerate(k2m)}
    S2hi = {k: R.get(f'f2.k2.hi.{i}', F2, 'c2', 'k2', 'members', k, 'hi') for i, k in enumerate(k2m)}
    k2v2 = R.get('f2.k2.verdict', F2, 'c2', 'k2', 'verdict')
    V('K2', '–', '2D 4096², registered LSPG schemes vs best matched FOM (union)', 'fires',
      f'{k2v2}; union speed-up {g3(min(S2.values()))}–{g3(max(S2.values()))} over {len(S2)} configurations '
      f'(largest interval end {g3(max(S2hi.values()))})', 'figs/summary_union_speedup.png', 'jcp-fom2d')
    hcl = s3['m1']['Hcomp_largest_mesh']
    Lmax = R.get('s3.hcomp.L', S3, 'm1', 'Hcomp_largest_mesh', 'L')
    c3 = {}
    for rom, cells in hcl['cells'].items():
        for fam, c in cells.items():
            st = ('no ladder match' if c.get('state') == 'no ladder match' else
                  ('ROM >= 2x' if c.get('reaches_2x') else 'ROM < 2x'))
            c3.setdefault(fam, {})
            c3[fam][st] = c3[fam].get(st, 0) + 1
    R.derived('s3.hcomp.counts', c3, [R.src(S3, ['m1', 'Hcomp_largest_mesh', 'cells', '*', '*', f]) for f in ('state', 'reaches_2x')],
              'per family cell: "no ladder match" if state == "no ladder match" (union cells have no state field and '
              'count as matched); otherwise "ROM >= 2x" if reaches_2x is true, else "ROM < 2x" (reaches_2x is absent '
              'only in unmatched cells); then count per family')
    fam3 = {'UB': 'upwind + BE', 'UC': 'upwind + CN-R', 'CC': 'central + CN-R', 'PS': 'sine pseudo-spectral',
            'union': 'union of families'}
    for fam in ('UB', 'UC', 'CC', 'PS', 'union'):
        c = c3[fam]
        ver = ('inconclusive (no ladder member as accurate)' if set(c) == {'no ladder match'} else
               'fail' if set(c) == {'ROM < 2x'} else 'mixed')
        V('H-comp', 'r (bank width)', f'3D L = {Lmax}, vs {fam3[fam]}, {len(hcl["cells"])} selected ROMs (not all quadrature-qualified)',
          ver, ', '.join(f'{k} ×{n}' for k, n in sorted(c.items())), 'figs/speed3d_mesh-time.png', 'jcp-speed3d')
    S3u = {r: R.get(f's3.hcomp.union.{r}', S3, 'm1', 'Hcomp_largest_mesh', 'cells', r, 'union', 'speedup')
           for r in hcl['cells']}
    k2v3 = R.get('s3.K2', S3, 'm1', 'K2')
    V('K2', '–', f'3D L = {Lmax}, selected ROMs vs best matched FOM (union)', k2v3,
      f'union speed-up {g3(min(S3u.values()))}–{g3(max(S3u.values()))} ({len(S3u)} ROMs)', 'figs/summary_union_speedup.png',
      'jcp-speed3d')

    # ---------------- H-base ---------------------------------------------------------------------------------
    hbc = hb['cells']
    hbase = {}
    for cell, filt, label in (('c2d256', None, '2D 256²'), ('c2d1024', None, '2D 1024²'),
                              ('c3d64', 'batch 2', '3D 65³, qualified subset (batch 2)'),
                              ('c3d64', 'batch 1', '3D 65³, diagnostic (batch 1)')):
        X = {k: v for k, v in hbc[cell]['X'].items() if filt is None or filt in v['cell_line']}
        dom = sum('baseline dominates X' in v['cell_line'] for v in X.values())
        withheld = sum('withheld' in v['cell_line'] for v in X.values())
        fdesc = f'X whose cell_line contains "{filt}"' if filt else None
        R.derived(f'hb.{cell}.{filt}.dominated', dom, [R.src(HB, f'cells/{cell}/X/*/cell_line', fdesc)],
                  'count of X whose cell_line contains "baseline dominates X"' + (f' among "{filt}"' if filt else ''))
        R.derived(f'hb.{cell}.{filt}.n', len(X), [R.src(HB, f'cells/{cell}/X/*/cell_line', fdesc)],
                  'count of X' + (f' among "{filt}"' if filt else ''))
        R.derived(f'hb.{cell}.{filt}.withheld', withheld, [R.src(HB, f'cells/{cell}/X/*/cell_line', fdesc)],
                  'count of X whose cell_line contains "withheld"')
        restr = R.get(f'hb.{cell}.restrictions', HB, 'cells', cell, 'flags', 'restrictions')
        bkey = 'budget_2D_V2' if cell.startswith('c2d') else ('budget_3D_V3_batch2' if filt == 'batch 2' else 'budget_3D_V3_batch1')
        bshort = R.get(f'hb.{bkey}.short', HB, 'cells', bkey, 'short')
        hbase[label] = (dom, len(X), withheld, restr)
        ver = 'fail' if dom == len(X) else ('fail (diagnostic only)' if filt == 'batch 1' else
                                             'fail where assessed; neutral conclusions withheld' if dom else 'withheld')
        V('H-base', 'r (bank width)', label, ver,
          f'{dom}/{len(X)} of our configurations dominated (faster and lower worst error) by a registered POD pick'
          + (f'; {withheld} neutral conclusions withheld' if withheld else '') + f'; restrictions: {"; ".join(restr)}'
          + f'; budget: {bshort}; developmental, in-sample, unadjusted',
          {'2D 256²': 'figs/hrb_front-c2d256.png', '2D 1024²': 'figs/hrb_front-c2d1024.png'}.get(label, 'figs/hrb_front-c3d64.png'),
          'jcp-hr-baselines')

    # ---------------- H2 (3D ROM vs ROM) -----------------------------------------------------------------------
    for comp in ('C_old', 'C_W768', 'C_W1024'):
        Sq = R.get(f'dm.h2q.{comp}.S', DM, 'h2q', comp, 'S')
        vq = R.get(f'dm.h2q.{comp}.verdict', DM, 'h2q', comp, 'verdict')
        Sc = R.get(f's3.H2.{comp}.S_confirmed', S3, 'H2', comp, 'S_confirmed')
        scq = R.get(f'dm.h2q.{comp}.scope', DM, 'h2q', comp, 'scope')
        lab = R.get(f'dm.h2q.{comp}.comparator_label', DM, 'h2q', comp, 'comparator_label')
        V('H2q', '–', f'3D, qualified winner vs our earlier configuration {comp} (ROM vs ROM; bar 1/0.6); {scq}; {lab}', vq,
          f'S = {g3(Sq)} (comparator Q2: {q2_m1[comp]}); the unqualified speed3d winners had {g3(Sc)}',
          'figs/dims_h2q_3d.png', 'jcp-dims')

    # ================================================================================================================
    # figures (copies)
    F = {}
    for lane, name, new in [
        ('jcp-p1-scaling', 'solve_time_vs_mesh.png', 'p1_solve_time_vs_mesh.png'),
        ('jcp-p1-scaling', 'kernel_cost_vs_r_2D.png', 'p1_kernel_cost_vs_r_2D.png'),
        ('jcp-p1-scaling', 'kernel_cost_vs_r_3D.png', 'p1_kernel_cost_vs_r_3D.png'),
        ('jcp-p1-scaling', 'error_vs_r_and_mesh_CNR.png', 'p1_error_vs_r_and_mesh_CNR.png'),
        ('jcp-p1-scaling', 'error_vs_r_and_mesh_BE.png', 'p1_error_vs_r_and_mesh_BE.png'),
        ('jcp-p1-scaling', 'p1g_frozen_vs_nominal.png', 'p1_p1g_frozen_vs_nominal.png'),
        ('jcp-speed3d', 'mesh-time.png', 'speed3d_mesh-time.png'),
        ('jcp-qualify', 'front2d_median.png', 'qualify_front2d_median.png'),
        ('jcp-qualify', 'front2d_worst.png', 'qualify_front2d_worst.png'),
        ('jcp-qualify', 'error_vs_r.png', 'qualify_error_vs_r.png'),
        ('jcp-qualify', 'error_vs_N.png', 'qualify_error_vs_N.png'),
        ('jcp-qualify', 'order_b4.png', 'qualify_order_b4.png'),
        ('jcp-fom2d', 'speedup_vs_mesh.png', 'fom2d_speedup_vs_mesh.png'),
        ('jcp-fom2d', 'fom_pareto.png', 'fom2d_fom_pareto.png'),
        ('jcp-hr-baselines', 'front-c2d256.png', 'hrb_front-c2d256.png'),
        ('jcp-hr-baselines', 'front-c2d1024.png', 'hrb_front-c2d1024.png'),
        ('jcp-hr-baselines', 'front-c3d64.png', 'hrb_front-c3d64.png'),
        ('jcp-heads', 'error_cost_1024.png', 'heads_error_cost_1024.png'),
        ('jcp-heads', 'cost_vs_K.png', 'heads_cost_vs_K.png'),
        ('jcp-heads', 'error_vs_K.png', 'heads_error_vs_K.png'),
        ('jcp-dims', 'h2q_3d.png', 'dims_h2q_3d.png'),
        ('jcp-dims', 'front_3d.png', 'dims_front_3d.png'),
        ('jcp-dims', 'family_err_R384M1536.png', 'dims_family_err_R384M1536.png'),
        ('jcp-dims', 'width_mstar_3d.png', 'dims_width_mstar_3d.png'),
        ('jcp-dims', 'b6_2d_kappa.png', 'dims_b6_2d_kappa.png'),
        ('jcp-budget', 'b2_main_1024.png', 'budget_b2_main_1024.png'),
        ('jcp-budget', 'b2_main_3d.png', 'budget_b2_main_3d.png'),
        ('jcp-budget', 'b7_error_time_1024.png', 'budget_b7_error_time_1024.png'),
        ('jcp-budget', 'b8_sampling_1024.png', 'budget_b8_sampling_1024.png'),
    ]:
        F[new] = copy_fig(lane, name, new)

    # ---- new summary figure 1: cost vs mesh, 2D and 3D (two panels, each indexed to its own base) --------------
    fig1, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = [C1, C2, C3, C4, '#e87ba4', '#008300', '#eda100', '#e34948']
    for i, k in enumerate(hcn):
        med = R.get(f'p1.hcn.medians.{k}', P1, 'detail', 'h_cost_n', k, 'medians')
        ms = sorted(med, key=int)
        base = med['1024']
        a1.plot([int(m) for m in ms], [med[m] / base for m in ms], '-o', color=cols[i % 8], lw=2, ms=5,
                label=k.replace('|', ' '))
    a1.set_xscale('log', base=2); a1.set_xticks([256, 1024, 4096]); a1.set_xticklabels(['256²', '1024²', '4096²'])
    a1.axhspan(0.8, 1.25, color=GRID, alpha=0.5, lw=0)
    a1.set_ylim(0.7, 1.35); a1.set_ylabel('evolve time / its value at 1024²', color=INK, fontsize=9)
    a1.set_title('2D (P1): 8 settings, band = 0.8–1.25', fontsize=9, color=INK); style(a1)
    a1.legend(fontsize=6, frameon=False, ncol=2, loc='lower left')
    for i, (k, v) in enumerate(h41.items()):
        sec = R.get(f's3.H41.seconds.{k}', S3, 'm1', 'H41', k, 'seconds')
        ms = sorted(sec, key=int)
        a2.plot([int(m) for m in ms], [sec[m] / sec[ms[0]] for m in ms], '-o', color=cols[i], lw=2, ms=5, label=k)
    a2.set_xscale('log', base=2); a2.set_xticks([64, 128, 256, 384]); a2.set_xticklabels(['64³', '128³', '256³', '384³'])
    a2.axhspan(0.8, 1.25, color=GRID, alpha=0.5, lw=0)
    a2.set_ylim(0.7, 1.35); a2.set_ylabel('query time / its value at 64³', color=INK, fontsize=9)
    a2.set_title('3D (speed3d m1): 4 ROMs, mesh-free contract', fontsize=9, color=INK); style(a2)
    a2.legend(fontsize=6, frameon=False, loc='lower left')
    fig1.tight_layout(); fig1.savefig(FIGS / 'summary_cost_vs_mesh.png', dpi=150); plt.close(fig1)

    # ---- new summary figure 2: union speed-up per configuration (2D 1024², 4096²; 3D L = 384) -------------------
    rows = []
    for k, v in hc.items():
        if v['scope'].startswith('primary') and v['family'] == 'union':
            rows.append((k.split('|')[0], v['target'], v['S'], v['lo'], v['hi']))
    R.derived('f2.union.rows', [[r[0], r[2], r[3], r[4]] for r in rows], [R.src(F2, ['c2', 'hcomp', '*', f], 'primary-scope cells with family == union') for f in ('S', 'lo', 'hi')],
              'primary-scope union cells (mesh, S, lo, hi)')
    fig2, ax = plt.subplots(figsize=(7.5, 3.4))
    for j, (mesh, colr, yoff) in enumerate((('1024', C1, -0.15), ('4096', C2, 0.15))):
        sub = sorted([r for r in rows if r[0] == mesh], key=lambda r: r[2])
        for i, r in enumerate(sub):
            ax.plot([r[3], r[4]], [i + yoff] * 2, color=colr, lw=1.5)
            ax.plot([r[2]], [i + yoff], 'o', color=colr, ms=4, label=f'2D {mesh}² ({len(sub)} LSPG configurations)' if i == 0 else None)
    base = len([r for r in rows if r[0] == '4096']) + 1
    for i, (rom, s) in enumerate(sorted(S3u.items(), key=lambda t: t[1])):
        ax.plot([s], [base + i], 's', color=C3, ms=5, label=f'3D {Lmax}³ ({len(S3u)} ROMs)' if i == 0 else None)
    ax.axvline(1, color=MUTE, lw=1, ls='--'); ax.axvline(2, color=INK, lw=1)
    ax.text(1, -1.6, ' 1× ', fontsize=7, color=MUTE, ha='right'); ax.text(2, -1.6, ' 2× bar', fontsize=7, color=INK)
    ax.set_xscale('log'); ax.set_yticks([]); ax.set_ylim(-2, base + len(S3u))
    ax.set_xlabel('ROM speed-up over the best matched full-order solver (union of families; < 1 = FOM faster)',
                  fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc='lower right'); style(ax)
    fig2.tight_layout(); fig2.savefig(FIGS / 'summary_union_speedup.png', dpi=150); plt.close(fig2)

    # ---- new summary figure 3: per-family speed-up, small multiples (2D primary LSPG cells) ----------------------
    famrows = {}
    for k, v in hc.items():
        if v['scope'].startswith('primary') and v.get('S') is not None:
            famrows.setdefault(v['family'], []).append((k.split('|')[0], v['target'].split('@')[0], v['S'], v['lo'], v['hi']))
    R.derived('f2.family.rows', {f: [[r[0], r[2], r[3], r[4]] for r in rs] for f, rs in famrows.items()},
              [R.src(F2, ['c2', 'hcomp', '*', f], 'primary-scope cells with an S value') for f in ('family', 'S', 'lo', 'hi')],
              'per family: (mesh, S, lo, hi)')
    fig4, axs = plt.subplots(1, 4, figsize=(11, 3.6), sharey=False)
    for ax, fam in zip(axs, ('UB', 'CC', 'PS', 'union')):
        rs = famrows.get(fam, [])
        order = sorted({r[1] for r in rs}, key=lambda t: min(x[2] for x in rs if x[1] == t))
        for mesh, colr, yoff in (('1024', C1, -0.15), ('4096', C2, 0.15)):
            for r in rs:
                if r[0] != mesh:
                    continue
                y = order.index(r[1]) + yoff
                ax.plot([r[3], r[4]], [y, y], color=colr, lw=1.3)
                ax.plot([r[2]], [y], 'o', color=colr, ms=3.5,
                        label=f'{mesh}²' if r is next(x for x in rs if x[0] == mesh) else None)
        ax.axvline(1, color=MUTE, lw=1, ls='--'); ax.axvline(2, color=INK, lw=1)
        ax.set_xscale('log'); ax.set_yticks([]); style(ax)
        ax.set_title(f'{fam_name[fam]} ({len({r[1] for r in rs})} configs × meshes)', fontsize=8.5, color=INK)
        ax.set_xlabel('speed-up (2× bar solid)', fontsize=7.5, color=INK)
    axs[0].legend(fontsize=7, frameon=False, loc='lower right')
    fig4.tight_layout(); fig4.savefig(FIGS / 'summary_family_speedup.png', dpi=150); plt.close(fig4)

    # ---- ledgers ------------------------------------------------------------------------------------------------
    ledg = {}
    for lane, _, _, _ in LANES:
        p = LANE_DIR[lane] / 'runs' / 'GPU-LEDGER.json'
        sp = R.get(f'ledger.{lane}.spent', p, 'spent_gpu_hours')
        cap = R.get(f'ledger.{lane}.cap', p, 'cap_gpu_hours')
        nj = R.derived(f'ledger.{lane}.n_jobs', len(R.load(p)['jobs']), [R.src(p, 'jobs')], 'number of ledger entries')
        ledg[lane] = (sp, cap, nj)
    done_total = R.derived('ledger.total_finished', sum(v[0] for k, v in ledg.items() if k not in PENDING_LANES),
                           [R.src(LANE_DIR[k] / 'runs' / 'GPU-LEDGER.json', 'spent_gpu_hours') for k in ledg
                            if k not in PENDING_LANES], 'sum over the eight finished lanes')
    all_total = R.derived('ledger.total_all', sum(v[0] for v in ledg.values()),
                          [R.src(LANE_DIR[k] / 'runs' / 'GPU-LEDGER.json', 'spent_gpu_hours') for k in ledg],
                          'sum over all ten lanes at compile time')
    fig3, ax = plt.subplots(figsize=(7, 3.4))
    names = [l for l, *_ in LANES]
    ax.barh(names, [ledg[n][1] for n in names], color=GRID, height=0.6, label='cap')
    ax.barh(names, [ledg[n][0] for n in names], color=C1, height=0.36, label='spent (sacct)')
    for i, n in enumerate(names):
        ax.text(ledg[n][0], i, f' {ledg[n][0]:.1f}', va='center', fontsize=7, color=INK)
    ax.invert_yaxis(); ax.set_xlabel('GPU-hours', fontsize=8, color=INK); style(ax)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    fig3.tight_layout(); fig3.savefig(FIGS / 'summary_gpu_ledger.png', dpi=150); plt.close(fig3)

    # ================================================================================================================
    # REPORT
    w('# Viscous Burgers 2D/3D: final results of the JCP experiment campaign (priorities 1–4)')
    w()
    pend_now = [l for l in PENDING_LANES if not E(l).exists()]
    w(f'**Status (compiled {now} by `make_final.py`).** Final for the development and validation cohorts '
      '(2D dev6 + val32, 3D 923801); the historical test cohorts were used for replication only, and the sealed and '
      'reserved out-of-distribution cohorts were **not opened**. Accuracy numbers are scored against the E0 '
      'second-order reference $R_{ST}$, whose per-case uncertainty $\\varepsilon_{\\rm ref}$ is a **conditional '
      'Richardson estimate**, not a bound, so they are conditional E0 estimates. Two labelled exceptions: the '
      'long-horizon windows beyond $t = 0.25$ (section 3.9) use jcp-budget\'s own second-order references (also '
      'conditional Richardson estimates), and the P1g spread columns marked "first-order" (section 4) are against the '
      'earlier first-order references, as in the historical claims they recheck. Post-hoc labels '
      'carry forward unchanged: E0 A10 (the reference cross-check rule was changed after data), combined A10 and A11 '
      '(post-hoc rule selections) and the retracted combined A15. '
      + (f'**Pending:** {", ".join(pend_now)} (priority 4) had not exported results at compile time; their sections '
         'are marked pending and contain no numbers.' if pend_now else 'All ten lanes have exported results.'))
    w()
    w('Every number in this report is read from a lane output (or from git, for commits) and registered in '
      '`PROVENANCE.json` (number id → file, JSON key path, sha256; derived values list every source and the '
      'derivation). `FINAL-EXPORT.json` holds the verdicts and key numbers. Bars such as 1.25, 2×, '
      '$2.5\\times10^{-4}$ and 1/0.6, cohort sizes (2D dev6 + val32 = 38 cases; 3D 64 cases), meshes, time steps and '
      'design grids (r, K, seeds) are registered design constants of the plan, not results.')
    w()
    w('```mermaid')
    w('flowchart LR')
    w('  E0[E0 references, done before]:::prior --> P1[priority 1: p1-scaling]:::lane')
    w('  S3[speed3d: 3D speed, K2 3D]:::lane --> Q[priority 2: qualify]:::lane')
    w('  P1 --> Q --> F2[fom2d: C1 C2, K2 2D]:::lane')
    w('  P1 --> HB[hr-baselines: C3, H-base]:::lane')
    w('  Q --> DM[priority 3: dims]:::lane')
    w('  Q --> BG[budget]:::lane')
    w('  Q --> HD[heads]:::lane')
    w('  DM --> RB[priority 4: robust]:::p4')
    w('  Q --> RG[regimes]:::p4')
    w('  F2 & HB & DM & BG & HD & RB & RG --> FN[this report]:::doc')
    w('  classDef prior fill:#e5e7eb,stroke:#374151; classDef lane fill:#dbeafe,stroke:#1d4ed8;')
    w('  classDef p4 fill:#fef3c7,stroke:#92400e; classDef doc fill:#dcfce7,stroke:#166534;')
    w('```')
    w()

    # ---- 1. headline
    w('## 1. In one paragraph')
    w()
    w(f'Solve cost does not depend on the mesh: in 2D the evolution time of every registered setting is flat from 256² '
      f'to 4096² (H-cost-N {n_pass_cn}/{len(hcn)}; the full-field decode lies outside this verdict and grows with the '
      f'mesh), in 3D every timed ROM is flat from 64³ to 384³ ({n41}/{len(h41)}), and the '
      f'nonlinear-head reading of "k" is flat as well (K = 16, ratio {g3(RK16)}). Accuracy does not depend on the mesh '
      f'within the registered margin (H-acc-N {n_an}/{len(han)}) and improves with r up to 384 but not from 384 to 512. '
      'Kernel cost does not follow the declared operation-count exponents in r (H-cost-r fails as registered: the '
      'normal-matrix and Cholesky kernels grow more slowly than declared, the vector kernels slightly faster). Against full '
      f'solvers, the plan\'s kill criterion **K2 fires in both dimensions**: a coarse sine pseudo-spectral solver reaches '
      f'the ROM\'s accuracy faster (2D 4096² union speed-up {g3(min(S2.values()))}–{g3(max(S2.values()))}; 3D '
      f'{g3(min(S3u.values()))}–{g3(max(S3u.values()))}). Against standard reduced models, **H-base fails**: a tuned '
      f'POD model with ECSW or Q-DEIM hyper-reduction dominates all {hbase["2D 256²"][1]} tested qualified 2D '
      f'configurations at 256² and all {hbase["2D 1024²"][1]} at 1024² (developmental, in-sample, unadjusted; 2D '
      f'tuning budgets {R.prov["hb.budget_2D_V2.short"]["value"]}). The paper therefore cannot claim speed or accuracy advantages over standard '
      'alternatives for viscous Burgers on the box; what remains is mesh-independent cost, mesh-free evaluation and '
      'memory.')
    w()

    # ---- 2. verdicts
    w('## 2. Hypothesis verdicts')
    w()
    w('"k" has two readings in this project: **r** (the width of the ordered linear bank, the number of coefficients '
      'solved for) and **K** (the latent size of the nonlinear head). Both are reported. "Inconclusive" means the '
      'registered rule could not decide (usually because too few configurations were numerically qualified), not '
      'a negative result.')
    w()
    w(table(['hypothesis', 'k reading', 'scope', 'verdict', 'key numbers', 'figure', 'lane'],
            [[v['hypothesis'], v['reading'], v['scope'], f'**{v["verdict"]}**', v['key_number'],
              v['figure'] if v['figure'] == '–' else f'[{Path(v["figure"]).name}]({v["figure"]})', v['lane']]
             for v in verdicts]))
    w()
    w('Provisional labels that travel with these verdicts: all errors are conditional E0 estimates; every matched '
      'FOM speed-up is an **upper bound** on the ROM speed-up (finite FOM ladder, screening-selected comparator; 2D '
      'timing intervals describe the dev6 workload, 3D matches use speed3d\'s paired match panels); H-base comparisons are **developmental, in-sample, unadjusted**; heads results are '
      '**exploratory validation evidence** with no family-wise claim.')
    w()

    # ---- 3. per-priority
    w('## 3. Results by question')
    w()
    w('### 3.1 Scaling against k')
    w()
    w('**Bank width r.** At fixed $(M, m)$ the component kernels (tested advection $N$, Jacobian $J$, normal matrix '
      '$J^\\top J$, Cholesky) were timed over r. Fitted exponents of $t = a + b\\,r^p$ against the declared '
      'operation-count exponent:')
    w()
    rows = []
    for dim in ('2d', '3d'):
        for k in p1['verdicts'][f'h_cost_r_{dim}']:
            d = p1['detail'][f'h_cost_r_{dim}'][k]
            ci = R.get(f'p1.hcr.{dim}.{k}.ci', P1, 'detail', f'h_cost_r_{dim}', k, 'ci')
            sec = R.get(f'p1.hcr.{dim}.{k}.secondary', P1, 'detail', f'h_cost_r_{dim}', k, 'secondary')
            rows.append([dim.upper(), k, g3(R.prov[f'p1.hcr.{dim}.{k}.p_decl']['value']),
                         g3(R.prov[f'p1.hcr.{dim}.{k}.p_fit']['value']), f'{g3(ci[0])}–{g3(ci[1])}',
                         p1['verdicts'][f'h_cost_r_{dim}'][k] + (f' ({sec})' if sec else '')])
    w(table(['dim', 'kernel', 'declared p', 'fitted p', '95 % interval', 'outcome'], rows))
    kabove, kbelow, kwithin = [], [], []
    for dim in ('2d', '3d'):
        for k in p1['verdicts'][f'h_cost_r_{dim}']:
            lo_, hi_ = R.prov[f'p1.hcr.{dim}.{k}.ci']['value']
            pd_ = R.prov[f'p1.hcr.{dim}.{k}.p_decl']['value']
            (kabove if lo_ > pd_ else kbelow if hi_ < pd_ else kwithin).append(f'{dim.upper()} {k}')
    w()
    w('Against the declared exponent, the fitted exponent is ' + '; '.join(
        f'**{lab}** for ' + (', '.join(ks) if ks else 'none') for lab, ks in (
            ('above (interval entirely above)', kabove), ('below (interval entirely below)', kbelow),
            ('consistent (interval contains it)', kwithin))) + '. The registered rule sub-labels each fail '
      '"hardware scaling" or "operation-model"; these are classifications, not causal explanations.')
    w()
    fig(F['p1_kernel_cost_vs_r_2D.png'], '2D kernel cost against r at fixed (M, m) (jcp-p1-scaling)')
    fig(F['p1_kernel_cost_vs_r_3D.png'], '3D kernel cost against r at fixed (M, m) (jcp-p1-scaling)')
    nheads = R.derived('hd.n_heads', sum(k.startswith('mid_k') for k in hd['training']), [R.src(HD, 'training')],
                       'count of trained arms mid_k*')
    w(f'**Head latent K.** {nheads} matched heads (K = 4, 8, 16, 32 × seeds 0–2, the registered design) were trained '
      'with the incumbent recipe. '
      f'Numerically qualified BE seeds at 1024²: ' + '; '.join(
          f'K = {K}: {R.get(f"hd.S_K.{K}", HD, "S_K", "BE", K) or "none"}' for K in ('4', '8', '16', '32')) +
      f'. The cost-against-K exponent is identified only at 4096² (p = {g3(pK)}, seed 0 only; descriptive) and not at '
      '256² or 1024², and the error-against-K range is '
      f'**{hk}**.')
    w()
    fig(F['heads_cost_vs_K.png'], 'Head online cost against K (jcp-heads; descriptive)')

    w('### 3.2 Scaling against the mesh')
    w()
    w('2D evolve time (P1, A–B–A normalised, median over dev6) and 3D warm query time (speed3d m1) against the '
      'mesh, each indexed to its own base so both fit one scale. The grey band is the registered [0.8, 1.25].')
    w()
    fig('figs/summary_cost_vs_mesh.png', 'Cost against mesh: 2D (left) and 3D (right); flat lines = mesh-independent cost')
    rows = [[k.replace('|', ' / '), g3(Rs[k]), g3(ubs[k]), hcn[k]] for k in hcn]
    w(table(['2D setting', 'R = max/min over meshes', '95 % upper end', 'outcome'], rows))
    w()
    rows = [[k, *[g3(1000 * R.prov[f's3.H41.seconds.{k}']['value'][m]) for m in ('64', '128', '256', '384')],
             g3(mm41[k]), h41[k]['verdict']] for k in h41]
    w(table(['3D ROM', 'L=64 (ms)', 'L=128 (ms)', 'L=256 (ms)', 'L=384 (ms)', 'max/min', 'H4.1'], rows))
    w()
    w('The 3D flat-cost result is a statement about the deployment mode policy (mode set varies with mesh) under the '
      'mesh-free / scoring-lattice contract; no fixed-mode 3D cost panel was run. The 2D result holds for both the '
      'frozen mode list and the nominal per-mesh lists.')
    w()
    fig(F['p1_solve_time_vs_mesh.png'], '2D solve time against mesh for each bank width (jcp-p1-scaling)')

    w('### 3.3 Accuracy and the accuracy–cost front')
    w()
    w('2D error at 1024² (P1 settings, 38 cases; conditional E0 estimates):')
    w()
    rows = []
    for k in p1['key_numbers']['configurations']:
        mesh, cfg, sch = k.split('|')
        if mesh != '1024' or not cfg.startswith('R'):
            continue
        med = R.get(f'p1.kn.{k}.median', P1, 'key_numbers', 'configurations', k, 'e0_median')
        wst = R.get(f'p1.kn.{k}.worst', P1, 'key_numbers', 'configurations', k, 'e0_worst')
        rows.append([cfg, sch, pct(med), pct(wst)])
    w(table(['r', 'scheme', 'median e', 'worst e'], rows))
    w()
    nq = R.derived('ql.n_qualified', sum(v['numerically_qualified'] for v in ql['configurations'].values()),
                   [R.src(QL, 'configurations/*/numerically_qualified')], 'count of true')
    nc = R.derived('ql.n_configs', len(ql['configurations']), [R.src(QL, 'configurations')], 'count')
    w(f'**Numerical qualification (jcp-qualify).** {nq} of {nc} 2D configurations (r = 128–512 × LSPG-BE / LSPG-CN-R '
      '/ GAL-CN × step × mesh) pass Q1 (solver) and Q2 (quadrature). The qualified error–cost front at each mesh '
      'is dominated by GAL-CN, a registered diagnostic scheme; the registered deployment schemes LSPG-BE and '
      'LSPG-CN-R share the worst-error front.')
    w()
    for mesh in ('256', '1024', '4096'):
        fr = R.get(f'ql.front.{mesh}.median', QL, 'fronts', mesh, 'fronts', 'median|c')
        w(f'- {mesh}² median-error front ({len(fr)} members): ' + ', '.join(f'`{x.split("|", 1)[1]}`' for x in fr))
    w()
    fig(F['qualify_front2d_median.png'], '2D qualified accuracy–cost front, median error (jcp-qualify)')
    b4 = ql['B4']
    rows = [[k.replace('|', ' / '), R.get(f'ql.B4.{k}.label', QL, 'B4', k, 'label'),
             g3(R.get(f'ql.B4.{k}.p_median', QL, 'B4', k, 'p_median'))] for k in b4 if k != 'TH06_control']
    w('Observed temporal order (B4, over $\\Delta t_0 \\dots \\Delta t_0/4$):')
    w()
    w(table(['configuration', 'order label', 'median observed order'], rows))
    w()
    cov = dm['coverage']
    w('**3D front (jcp-dims).** Numerically qualified 3D configurations per bank width (a qualified subset of the '
      'tested candidates; never a completeness claim): ' +
      ', '.join(f'{k} {R.get(f"dm.coverage.{k}", DM, "coverage", k)}' for k in cov) +
      f'; {R.get("dm.open_candidates.count", DM, "open_candidates_3d", "count")} W1024 candidates remain untested.')
    w()
    fig(F['dims_front_3d.png'], '3D qualified front (jcp-dims; qualified subset of tested candidates)')
    w('**3D ROM-vs-ROM speed-up (H2q).** After quadrature qualification the cheapest qualified configuration is '
      f'`{R.get("dm.h2q.C_old.winner", DM, "h2q", "C_old", "winner")}` for all three comparators:')
    w()
    rows = []
    for comp in ('C_old', 'C_W768', 'C_W1024'):
        rows.append([comp, g3(R.prov[f's3.H2.{comp}.S_confirmed']['value']), g3(R.prov[f'dm.h2q.{comp}.S']['value']),
                     R.prov[f'dm.h2q.{comp}.verdict']['value'], q2_m1[comp]])
    w(table(['comparator', 'speed3d S (winners not Q2-qualified)', 'H2q S (qualified winner)', 'H2q verdict (bar 1/0.6)',
             'comparator Q2'], rows))
    w()
    w('These are ROM-vs-ROM speed-ups over our own earlier 3D configurations, not comparisons with the best FOM.')
    w()
    fig(F['dims_h2q_3d.png'], '3D: qualified speed-up over earlier configurations (jcp-dims)')

    w('### 3.4 Speed-up against each full-order family')
    w()
    nfc = R.get('f2.c1.configurations', F2, 'c1', 'configurations')
    nfa = R.get('f2.c1.admissible', F2, 'c1', 'admissible')
    w(f'2D (jcp-fom2d): {nfc} FOM configurations ({nfa} admissible) in three families (sign-upwind + BE, central + CN-R, sine '
      'pseudo-spectral ETDRK4), each validation-matched to every qualified ROM configuration on both median and '
      'worst error. Counts over the qualified LSPG configurations ("ROM ≥ 2×" against upwind is against the '
      'screening-selected comparator only):')
    w()
    rows = []
    for mesh in sorted(comp_counts, key=int):
        for fam in ('UB', 'CC', 'PS', 'union'):
            c = comp_counts[mesh].get(fam, {})
            rows.append([f'{mesh}²', fam_name[fam], ', '.join(f'{k} ×{n}' for k, n in sorted(c.items()))])
    w(table(['ROM mesh', 'FOM family', 'cells'], rows))
    w()
    fig('figs/summary_union_speedup.png', 'ROM speed-up over the best matched full-order solver (union), with 95 % '
        'intervals (2D) — every point lies left of the 2× bar, most left of 1×')
    fig('figs/summary_family_speedup.png', '2D: ROM speed-up over the matched FOM of each family, per qualified LSPG '
        'configuration, with 95 % intervals (one panel per family; cells without a matched FOM are omitted and '
        'counted in the table above)')
    fig(F['fom2d_speedup_vs_mesh.png'], '2D: ROM speed-up over the union-matched FOM against mesh (jcp-fom2d)')
    w(f'3D (jcp-speed3d, L = {Lmax}): ' + '; '.join(
        f'{fam3[f]}: ' + ', '.join(f'{k} ×{n}' for k, n in sorted(c3[f].items())) for f in ('UB', 'UC', 'CC', 'PS', 'union'))
      + '. "No ladder match" means no FOM in the ladder was as accurate as the ROM; nothing is claimed for those '
      f'cells. 3D cells reaching 2×: {sum(c.get("ROM >= 2x", 0) for c in c3.values())} (against '
      + ', '.join(fam3[f] for f, c in c3.items() if c.get('ROM >= 2x')) + ').')
    w()

    w('### 3.5 POD baselines (H-base)')
    w()
    w('Tuned POD-LSPG baselines (no hyper-reduction, ECSW, Q-DEIM) against our configurations, timed in the same job '
      'on the same GPU. "Dominated" = a baseline chosen by the registered role rule has lower full-query time and '
      'lower worst error, with non-overlapping intervals.')
    w()
    rows = [[lab, f'{d}/{n}', wh, '; '.join(rs)] for lab, (d, n, wh, rs) in hbase.items()]
    w(table(['cell', 'ours dominated', 'neutral conclusions withheld', 'restrictions'], rows))
    w()
    rows = []
    for cell in ('c2d256', 'c2d1024', 'c3d64'):
        for xk, x in hbc[cell]['X'].items():
            if 'batch 1' in x['cell_line']:
                continue
            ba = x['roles'].get('B_acc') or {}
            Wx = R.get(f'hb.{cell}.{xk}.W_X', HB, 'cells', cell, 'X', xk, 'W_X')
            Wb = R.get(f'hb.{cell}.{xk}.B_acc.W', HB, 'cells', cell, 'X', xk, 'roles', 'B_acc', 'W') if ba else None
            Tb = R.get(f'hb.{cell}.{xk}.B_acc.T', HB, 'cells', cell, 'X', xk, 'roles', 'B_acc', 'T') if ba else None
            rows.append([cell, f'`{xk}`', pct(Wx), f'`{ba.get("id", "–")}`', pct(Wb), g3(Tb),
                         'dominated' if 'baseline dominates X' in x['cell_line'] else 'withheld'])
    w('Per configuration, against the accuracy-role baseline pick (worst error; time = baseline / ours):')
    w()
    w(table(['cell', 'ours', 'ours worst e', 'baseline pick (B_acc)', 'pick worst e', 'time ratio', 'outcome'], rows))
    w()
    b2v = R.get('hb.budget2D.statement', HB, 'cells', 'budget_2D_V2', 'statement')
    b3v = R.get('hb.budget3D.statement', HB, 'cells', 'budget_3D_V3_batch2', 'statement')
    w(f'Tuning budgets (validation-informed GPU-hours, baseline / ours): 2D {b2v}; 3D {b3v}. Some dominations pair '
      'our first-order BE with a second-order baseline pick; the like-for-like CN-R pairs are dominated too. '
      'Median-error reversals (ours better on the median) exist in every cell: ' +
      ', '.join(f'{c} {len(R.get(f"hb.{c}.median_reversals", HB, "cells", c, "median_reversals"))}'
                for c in ('c2d256', 'c2d1024', 'c3d64')) + '.')
    w()
    fig(F['hrb_front-c2d1024.png'], '2D 1024²: POD baselines and ours, accuracy against cost (jcp-hr-baselines)')
    fig(F['hrb_front-c3d64.png'], '3D 65³: POD baselines and ours (jcp-hr-baselines)')

    w('### 3.6 Quadrature rule families')
    w()
    rows = []
    for rm in dm['families_2d']:
        for fam in dm['families_2d'][rm]['mstar']:
            rows.append([f'2D {rm}', fam, R.get(f'dm.fam2d.{rm}.{fam}', DM, 'families_2d', rm, 'mstar', fam, 'family', 'outcome')])
    for fam in dm['families_3d']['mstar']:
        rows.append(['3D W512', fam, R.get(f'dm.fam3d.{fam}', DM, 'families_3d', 'mstar', fam, 'family', 'outcome')])
    w('Smallest qualifying point count $m^*$ per family (Q2 successor agreement; for randomised families the same '
      'point count must qualify for every draw; "none" = no candidate size qualifies simultaneously for every required '
      'draw, even where individual draws qualify at some size; separate from the cross-family gate):')
    w()
    w(table(['panel', 'family', 'm* outcome'], rows))
    w()
    p1e2 = R.get('p1.p1e.2d.fail_succ.dist', P1, 'detail', 'p1e', 'two_d', 'R256', 'fail|fail_succ', 'worst_distance')
    p1e3 = R.get('p1.p1e.3d.lat8192.dist', P1, 'detail', 'p1e', 'three_d', 'lat8192|lat32768', 'worst_distance')
    w(f'P1e slice: the 2D rule registered as failing (Gauss 64²) **passes** against its successor (distance '
      f'{g3(p1e2)}); the 3D lattice 8192 rule fails (distance {g3(p1e3)}; bar $2.5\\times10^{{-4}}$).')
    w()
    fig(F['dims_family_err_R384M1536.png'], '2D rule families at (r, M) = (384, 1536) (jcp-dims)')

    w('### 3.7 Test space')
    w()
    ctl = R.get('dm.B6_2d.R128.control', DM, 'B6_2d', 'R128|LSPG-CNR', 'control')
    lmfe = R.get('dm.LMF.gauss.efficiency', DM, 'LMF', 'gauss', 'efficiency')
    lmfa = R.get('dm.LMF.accuracy', DM, 'LMF', 'accuracy_gauss256', 'label')
    w(f'B6 (test-space size $M/r$): the registered weak control $M = r$ is **{ctl}** — no square configuration '
      'passes solver qualification in 2D or 3D — so every test-space transition verdict is blocked by the registered '
      f'rule. The low-mixed-frequency (hyperbolic-cross) test set is stable; Gauss: {lmfe}; accuracy {lmfa}.')
    w()
    fig(F['dims_b6_2d_kappa.png'], '2D: error against test-space ratio M/r (jcp-dims; transitions blocked)')

    w('### 3.8 Error budget')
    w()
    w('Which ingredient dominates the error (Q quadrature, T time stepping, F/S/I/M other packages, $e_P$ the bank '
      'projection floor), per 2D member and mesh (jcp-budget B2):')
    w()
    members = sorted({k for m in bg['b2'] for k in bg['b2'][m]})
    MC = LANE_DIR['jcp-budget'] / 'checks' / 'meshcmp.json'
    mc = R.load(MC)['members']
    rows = []
    for m in members:
        full = [R.get(f'bg.b2.{ms}.{m}.dominant', BG, 'b2', ms, m, 'dominant') if m in bg['b2'][ms] else '–'
                for ms in ('256', '1024', '4096')]
        sub = [R.get(f'bg.meshcmp.{m}.{ms}', MC, 'members', m, 'dominant', ms) if m in mc else '–'
               for ms in ('256', '1024', '4096')]
        lab = R.get(f'bg.meshcmp.{m}.label', MC, 'members', m, 'label') if m in mc else '–'
        rows.append([m.replace('|', ' / '), *full, *sub, lab])
    w('The first three columns use each mesh\'s own case population; the cross-mesh verdict compares the shared '
      '16-case subset, whose dominant labels are shown next to it.')
    w()
    w(table(['member', '256² (own cases)', '1024² (own cases)', '4096² (own cases)', '256² (shared 16)',
             '1024² (shared 16)', '4096² (shared 16)', 'same across meshes (shared 16)'], rows))
    w()
    w('3D (65³): ' + ', '.join(f'`{k}` {R.get(f"bg.b3.{k}", BG, "b3", k)}' for k in bg['b3']) + '.')
    w()
    fig(F['budget_b2_main_1024.png'], 'Error budget at 1024² (jcp-budget)')
    adeq = {m: R.get(f'bg.lane_statement.{m}.ranking', BG, 'lane_statement', m, 'ranking') for m in bg['lane_statement']}
    w('B8 output sampling: the 257² scoring lattice is ' + ', '.join(f'{v} at {m}²' for m, v in adeq.items()) +
      ' for the LSPG members (relative to a 1025² route); the R384 GAL-CN member is mixed or inadequate, so the '
      'statement is scoped to LSPG.')
    w()

    w('### 3.9 Long horizons')
    w()
    rows = []
    for m in bg['b7']:
        mem = bg['b7'][m]
        nq_ = sum(v['qualified_at_T'] for v in mem.values())
        nst = sum(v['stable'] for v in mem.values())
        ng = sum(v['growth']['W2/W1'] == 'no growth' for v in mem.values())
        R.derived(f'bg.b7.{m}.counts', [nq_, nst, ng, len(mem)], [R.src(BG, ['b7', m, '*', 'qualified_at_T']), R.src(BG, ['b7', m, '*', 'stable']), R.src(BG, ['b7', m, '*', 'growth', 'W2/W1'])],
                  'counts of qualified_at_T, stable, growth W2/W1 == "no growth", members')
        fo = R.get(f'bg.front_order.{m}', BG, 'front_order_T05', m)
        rows.append([f'2D {m}², T = 0.5', f'{nq_}/{len(mem)}', f'{nst}/{len(mem)}', f'{ng}/{len(mem)}',
                     f'LSPG {fo["lspg"]}, all {fo["all"]}', '–'])
    b73 = R.load(BG3)['b7']
    n3q = sum(v['qualified_at_T'] for v in b73.values())
    n3s = sum(v['stability']['pass'] for v in b73.values())
    n3g = sum(v['growth']['W2/W1']['label'] == 'no growth' for v in b73.values())
    R.derived('bg.b7.3d.counts', [n3q, n3s, n3g, len(b73)],
              [R.src(BG3, ['b7', '*', 'qualified_at_T']), R.src(BG3, ['b7', '*', 'stability', 'pass']), R.src(BG3, ['b7', '*', 'growth', 'W2/W1', 'label'])], 'counts')
    fo3 = R.get('bg.b7.3d.front_order', BG3, 'front_order', 'label')
    rows.append(['3D 65³, T = 0.5', f'{n3q}/{len(b73)}', f'{n3s}/{len(b73)}', f'{n3g}/{len(b73)}', fo3, '–'])
    t1 = bg['b7_T1']['members']
    nt1 = sum(v['Q1'] and v['Q2'] == 'pass' for v in t1.values())
    nt1s = sum(v['stable'] for v in t1.values())
    ng1 = sum(v['growth']['W2/W1'] == 'no growth' for v in t1.values())
    w32 = {}
    for v in t1.values():
        w32[v['growth']['W3/W2']] = w32.get(v['growth']['W3/W2'], 0) + 1
    R.derived('bg.b7T1.counts', [nt1, nt1s, ng1, len(t1)],
              [R.src(BG, ['b7_T1', 'members', '*', f]) for f in ('Q1', 'Q2', 'stable')] +
              [R.src(BG, ['b7_T1', 'members', '*', 'growth', 'W2/W1'])], 'counts')
    R.derived('bg.b7T1.W3W2', w32, [R.src(BG, ['b7_T1', 'members', '*', 'growth', 'W3/W2'])], 'count of labels')
    rows.append(['2D 1024², T = 1.0', f'{nt1}/{len(t1)}', f'{nt1s}/{len(t1)}', f'{ng1}/{len(t1)}',
                 R.get('bg.b7T1.front_W1W2', BG, 'b7_T1', 'front_order_W1W2')['lspg'] + ' (LSPG)',
                 '; '.join(f'{k} ×{n}' for k, n in w32.items())])
    w(table(['panel', 'qualified at T', 'stable', 'no error growth (W2/W1)', 'front order W1 → W2',
             'W3/W2 growth (T = 1 only)'], rows))
    w()
    w('The T = 1 window W3 is **provisional**: its pseudo-spectral cross-check tier is unvalidated, so the lane gives '
      'values without a verdict (a validation reference failed its order check; jcp-budget report). '
      'In 3D no member is quadrature-qualified at T = 0.5, though all are stable.')
    w()
    fig(F['budget_b7_error_time_1024.png'], 'Error against time out to T = 1 at 1024² (jcp-budget)')

    w('### 3.10 Linear span against the nonlinear head')
    w()
    r16 = hd['lh']['BE']['per_K']['16']
    mh = R.get('hd.lh.16.median_H', HD, 'lh', 'BE', 'per_K', '16', 'median_H')
    m16 = R.get('hd.lh.16.r16.median_L', HD, 'lh', 'BE', 'per_K', '16', 'rungs', '16', 'median_L')
    c16 = R.get('hd.lh.16.r16.cost', HD, 'lh', 'BE', 'per_K', '16', 'rungs', '16', 'cost', 'median')
    m128 = R.get('hd.lh.16.r128.median_L', HD, 'lh', 'BE', 'per_K', '16', 'rungs', '128', 'median_L')
    c128 = R.get('hd.lh.16.r128.cost', HD, 'lh', 'BE', 'per_K', '16', 'rungs', '128', 'cost', 'median')
    amc = R.get('hd.lh.16.at_most_cost', HD, 'lh', 'BE', 'per_K', '16', 'lh_cost', 'at_most_cost')
    w(f'At matched dimension the K = 16 head (median {pct(mh)}) is more accurate than the linear span at r = 16 '
      f'(median {pct(m16)}, at {g3(c16)}× the head\'s cost): a trade-off. At no more online cost the linear span '
      f'wins: r = 128 reaches {pct(m128)} at {g3(c128)}× the head\'s cost ({amc}). Exploratory, BE, 1024², '
      'qualified seeds only. Under either reading of "k" the cost does not depend on the mesh; under the head reading '
      'the accuracy-per-cost comparison favours the linear span.')
    w()
    fig(F['heads_error_cost_1024.png'], 'Head (K) against linear span (r): error and cost at 1024² (jcp-heads)')

    # ---- priority 4
    for lane, title, items in (('jcp-robust', '3.11 Robustness (E1 randomised rules, E2 retrain seeds, E5 A100 vs H200)', 'E1, E2, E5'),
                               ('jcp-regimes', '3.12 Regimes (D1: ν extrapolation, IC-family change, stiffness)', 'D1')):
        w(f'### {title}')
        w()
        ex = E(lane)
        if not ex.exists():
            hbp = LANE_DIR[lane] / 'HEARTBEAT.json'
            hbj = {}
            if hbp.exists():
                hbj = dict(step=R.get(f'{lane}.heartbeat.step', hbp, 'step'), utc=R.get(f'{lane}.heartbeat.utc', hbp, 'utc'))
            w(f'> **PENDING — no results.** `{lane}` ({items}) had not written `runs/EXPORT.json` when this report was '
              f'compiled. Lane status at compile time (from its heartbeat, not a result): step "{hbj.get("step", "unknown")}", '
              f'heartbeat {hbj.get("utc", "unknown")}. GPU-hours spent so far are in the ledger section. This section '
              'will be regenerated from the lane\'s EXPORT when the lane hands back.')
            w()
            continue
        render_generic(lane, ex)

    # ---- 4. P1g
    w('## 4. P1g recheck: how much earlier "flat accuracy" moved under frozen mode lists')
    w()
    w('The 2D test-mode set is built by sorting mesh-dependent discrete Laplacian eigenvalues. P1g reran the earlier '
      'cross-mesh comparisons with one frozen list (taken at 1024²) and compared with the nominal per-mesh lists.')
    w()
    rows = []
    for arm in p1['detail']['p1g']:
        g = p1['detail']['p1g'][arm]
        d256 = R.get(f'p1g.{arm}.256.dist', P1, 'detail', 'p1g', arm, 'meshes', '256', 'dist_max')
        e256 = R.get(f'p1g.{arm}.256.de0', P1, 'detail', 'p1g', arm, 'meshes', '256', 'de0_max_abs_pp')
        d4k = R.get(f'p1g.{arm}.4096.dist', P1, 'detail', 'p1g', arm, 'meshes', '4096', 'dist_max')
        sf = R.get(f'p1g.{arm}.spread3_frozen', P1, 'detail', 'p1g', arm, 'spread3_frozen', 'first_order_max_pp')
        sn = R.get(f'p1g.{arm}.spread3_nominal', P1, 'detail', 'p1g', arm, 'spread3_nominal', 'first_order_max_pp')
        sfe = R.get(f'p1g.{arm}.spread3_frozen_e0', P1, 'detail', 'p1g', arm, 'spread3_frozen', 'e0_max_pp')
        sne = R.get(f'p1g.{arm}.spread3_nominal_e0', P1, 'detail', 'p1g', arm, 'spread3_nominal', 'e0_max_pp')
        cfg_, sch_ = arm.split('|')
        qn = R.get(f'p1g.{arm}.qualified', P1, 'qualification', f'1024|{cfg_}|{sch_}')
        rows.append([arm.replace('|', ' / '), 'yes' if qn else 'no (descriptive historical arm)', g3(d256),
                     f'{g3(e256)} pp', g3(d4k), f'{g3(sfe)} / {g3(sne)} pp', f'{g3(sf)} / {g3(sn)} pp'])
    w(table(['arm', 'numerically qualified (1024²)', '256²: frozen vs nominal trajectory distance', '256²: largest E0 error change',
             '4096²: distance', '3-mesh spread vs E0, frozen / nominal',
             '3-mesh spread vs first-order reference, frozen / nominal'], rows))
    w()
    w('Reading: the lists differ only at 256² (a few modes swapped at M = 1536); at 4096² the nominal list is a '
      'permutation of the frozen one (distances at round-off). Freezing moves trajectories by more than the Q1 bar '
      'but errors only by hundredths of a percentage point; the nominal runs reproduce the historical jobs to '
      'round-off. **Correction of interpretation, not of numbers:** the earlier "flat accuracy" comparisons at 256² '
      'used a slightly different test space; the conclusion stands.')
    w()
    fig(F['p1_p1g_frozen_vs_nominal.png'], 'P1g: frozen against nominal mode lists (jcp-p1-scaling)')

    # ---- 5. retractions
    w('## 5. Retractions and corrections')
    w()
    w('Consolidated from LAB-LOG.md (entries 2026-10-17 onward) and the lane reports. Numbers are read from lane '
      'outputs; the source of each item is given. Amendment-level detail (every DESIGN amendment) is in the sources '
      'listed in section 5.1.')
    w()
    dold = R.get('ql.3d.win_old.dist', QLA, 'threeD', 'tested', win_old, 'worst_dist_qp')
    d768 = R.get('ql.3d.win_768.dist', QLA, 'threeD', 'tested', win_768, 'worst_dist_qp')
    h5 = hist_h5()
    comb_rows, comb_cases = [], set()
    for mesh in ('256', '1024', '4096'):
        for r_ in ('R128', 'R256'):
            k = f'{mesh}|{r_}|LSPG|CNR|2'
            if k in qla['configurations']:
                comb_rows.append(f'{mesh}² {r_} {R.get(f"ql.Q1dist.{k}", QLA, "configurations", k, "worst_dist_tight"):.3g}')
                comb_cases.add(R.get(f'ql.Q1case.{k}', QLA, 'configurations', k, 'worst_dist_tight_case'))
    items = [
        ('speed3d H2 winners are not quadrature-qualified',
         f'The H2 speed-ups ({g3(R.prov["s3.H2.C_old.S_confirmed"]["value"])}×, '
         f'{g3(R.prov["s3.H2.C_W768.S_confirmed"]["value"])}×, {g3(R.prov["s3.H2.C_W1024.S_confirmed"]["value"])}×) '
         f'were measured correctly, but the winners fail Q2 (successor-rule distance {g3(dold)} and {g3(d768)}; bar '
         f'$2.5\\times10^{{-4}}$). After qualification (jcp-dims H2q) the speed-up over C_old is '
         f'{g3(R.prov["dm.h2q.C_old.S"]["value"])}× and **fails** the 1/0.6 bar; over C_W768 it is '
         f'{g3(R.prov["dm.h2q.C_W768.S"]["value"])}×. K2 is unaffected.', 'jcp-qualify report "Corrections"; jcp-dims H2q'),
        ('Combined H5 speed-up is a narrow-scope result',
         f'Combined H5 ({h5["sel"]} selected LSPG-CN-R, {h5["best"]} best arm, vs a second-order FOM) holds only '
         'against a same-mesh sign-upwind FOM matched on worst error. Against the full FOM ladder every qualified '
         'LSPG configuration is slower than its comparator and K2 fires. The paper must not quote it as a speed-up '
         'over the best FOM.', f'jcp-fom2d report; {h5["src"]}'),
        ('Combined finalist labels "qualified" withdrawn for LSPG-CN-R at 2Δt₀',
         'R128 and R256 at 2Δt₀ fail the every-case Q1 contract (worst distance to the tight solve: ' +
         ', '.join(comb_rows) + f'; bar $2.5\\times10^{{-4}}$; worst case: {", ".join(sorted(comb_cases))}). Their errors and timings are not '
         'retracted; their qualification label is.', 'jcp-qualify report'),
        ('"1024² reused from combined" was false for R256',
         'The archived R256 arm used Gauss 96², the frozen P1 rule is Gauss 128²; all 1024² P1 runs were redone '
         'fresh with in-job certification.', 'LAB-LOG 2026-10-17/18; P1-RESTATEMENT'),
        ("Coordinator's \"mode lists differ at 4096²\"",
         'Wrong: at 4096² the nominal list is a permutation of the frozen one (trajectory distance ' +
         g3(R.prov['p1g.R384|BE.4096.dist']['value']) + ' for R384 BE). The lists differ only at 256².',
         'jcp-p1-scaling P1g'),
        ('Registered failing 2D rule does not fail',
         f'Gauss 64² was registered as the failing 2D rule in P1e; it passes (distance {g3(p1e2)}). Reported, not '
         're-labelled.', 'jcp-p1-scaling P1e'),
        ('"The head is necessary at matched latent dimension" (2026-09-14) is qualified',
         'It holds at matched dimension only; at no more online cost the linear span r = 128 is more accurate than '
         'the K = 16 head.', 'jcp-heads §6d'),
        ('Early "flat accuracy" at 256² used a slightly different test space', 'See section 4 (P1g).',
         'jcp-p1-scaling'),
        ('Post-data tolerance and rule changes (labels carried)',
         'E0 A10 (reference cross-check rule changed after data); jcp-hr-baselines A41 (reproduction tolerance '
         'loosened after data; outcomes unchanged) and A44.3 (role picks chosen on the Stage-B data); jcp-dims A6.1 / '
         'A9.1 (certificate bar relaxed after the smoke; first discrepancy figures were wrong and replaced); '
         'jcp-qualify A1.3 / A2.5 (GAL stationarity replaced by a residual criterion); jcp-p1-scaling A2.2 '
         '(component-sum control demoted to a diagnostic).', 'lane DESIGN amendments'),
        ('Lane-internal errors found by audits and fixed before results were accepted',
         'speed3d A9 G2 attribution (corrected in A10) and m1 calibration population; jcp-hr-baselines budget '
         'under-count (results audit 1, fixed in A44.2) and the ill-posed ECSW tolerance knob (replaced by a support '
         'target); jcp-budget A9/A9.1 eligibility bug, A12.x caching bugs (no value changed), A14.1 corrected by '
         'A14.2, A14.3 tie rule; jcp-dims A7, A10.2 underpriced certificates, A12.1 omitted configurations, A14 '
         'timing controls missing before the first t2 configuration.', 'lane reports, LAB-LOG'),
        ('Plan corrections', 'Sealed-cohort counts in the draft plan were case-kinds, not physical cases; the plan\'s '
         'GPU-hour total was mis-summed; the proposed ≤ 5 % flatness bar was replaced by the registered [0.8, 1.25] '
         'band; 3D S references are never matched (3D step 0.01 vs S at 0.005). From the plan audits: the P1 '
         'accuracy run count was mis-stated; the 2D query samples the initial condition at the quadrature points '
         'rather than projecting the full field; a finite FOM ladder bounds our speed-up from above (it does not bound '
         'FOM efficiency); the degenerate three-mesh sign test was replaced by per-case slopes with a Wilcoxon test plus '
         'TOST; the P1a fixture could not supply the widest bank from truncated narrower states and was replaced; '
         'the Q2 bar originally mixed two quantities; several must-fail controls were not guaranteed to fire and '
         'were replaced; "P1 uses no randomised rules" was false (the Fibonacci lattices carry a seeded shift).',
         'LAB-LOG 2026-10-17 entries'),
        ('jcp-p1-scaling smoke acceptance', 'The first smoke acceptance failed on defects of the checks themselves '
         '(DESIGN A6); a mislabelled smoke-acceptance file was replaced by a reconstruction.', 'LAB-LOG 2026-10-18'),
        ('jcp-speed3d wording', 'The screening wording "all replay labels mixed" was wrong (not every replay label was "mixed"); '
         'the m1 FOM calibration population at L ≥ 256 was corrected before submission (A9).',
         'jcp-speed3d report'),
        ('jcp-hr-baselines construction defects', 'Phase I acceptance failed on real defects, then fixed: the ECSW '
         'tolerance knob was ill-posed (replaced by a support target); the NNLS solver cycled in 3D (replaced by an '
         'different exact solver, later reworked again after a performance bug, A38–A40); out-of-memory paths were '
         'made memory-lean; cancelled attempts t3d64a, s2d1024a and s3d64a were caused by the NNLS column deletion, '
         'repeated whole-file JSON rewrites, and an inexact step identifier (A42, A44.1); a resource-model correction '
         '(A43.1).', 'LAB-LOG 2026-10-18; jcp-hr-baselines report "Cancelled attempts"'),
        ('jcp-budget corrections', 'A registered budget-gate bar was restated after the smoke (A6); the offline reference '
         'identity compares case descriptors because of last-bit differences across NumPy versions (A6); the first '
         'resource certificates under-priced the retry path (A7); at T = 1 the first eligibility code took W2 '
         'eligibility from the wrong horizon (A11.1); the 3D shared-IC requirement was restated level-wise after a '
         'last-bit inconsistency inside E0 (A13); case durations quoted in A14/A14.1 were cumulative log timestamps '
         '(A14.3).', 'jcp-budget report "What was wrong or corrected"'),
        ('jcp-dims corrections', 'Missing in-job gate evidence for 2D family draws was first labelled "offline (b) '
         'disagrees" and relabelled "missing" (A12.4); protected jobs were missing from the ledger reserve (A11); the first memory-coverage argument for the t2 '
         'timing panel was wrong and was replaced by production evidence (A14.5).',
         'jcp-dims report; LAB-LOG 2026-10-20'),
        ('jcp-heads', 'Under the successor-agreement quadrature policy the incumbent head needs Gauss 96² at 1024², '
         'where the quadrature-study lane had picked Gauss 32² post hoc (a policy outcome, not a universal requirement). '
         'A tighter initial-fit tolerance (A9) was set after smk1, before production; the seed-0 K = 16 '
         'retrain equals the incumbent to round-off (a same-seed replay, not a replication).', 'LAB-LOG 2026-10-18'),
        ('jcp-regimes (design stage; the lane is still running)', 'Withdrawn during audit: a "clean one-factor '
         'attribution" (now distributional contrasts), a wrong reference acceptance rule (now E0\'s predicates), and a '
         'cost model missing the refinement time at low viscosity.', 'LAB-LOG 2026-10-19'),
    ]
    for t, body, src in items:
        w(f'- **{t}.** {body} *(Source: {src}.)*')
    w()
    for lane in PENDING_LANES:
        if not E(lane).exists():
            w(f'- `{lane}`: its retractions will be added when it hands back (pending).')
    w()

    pointer_record()

    # ---- 6. deferred
    w('## 6. Deferred items')
    w()
    bdef = R.get('bg.deferred', BG, 'deferred')
    deferred = [
        ('coordinator', 'C3-full: 2D 4096², 3D meshes finer than 64 intervals, GNAT, POD + exact tensor, best-practice '
         'EQ; per-mesh setup cost for break-even counts', 'C3-minimal already falsified H-base (LAB-LOG 2026-10-19)'),
        ('coordinator', 'C4', 'excluded from the freeze roster (plan V3-6)'),
        ('coordinator / jcp-regimes', '3D regimes and the regimes FOM comparison', 'regimes brief defer order'),
        ('jcp-budget', latexify('; '.join(d['item'] for d in bdef)), '; '.join(d['reason'] for d in bdef)),
        ('jcp-speed3d', 'upwind FOM ladder at L ≥ 256 (decides the upwind 3D cells); fixed-mode 3D cost panel',
         'speed3d report, open items'),
        ('jcp-fom2d', 'same-resolution spectral FOM at 1024² and 4096²; upwind + CN-R, Newton–GMRES, IMEX-CG families',
         'deferred by scope (A2.4)'),
        ('jcp-qualify', 'fourth 3D Q2 round', 'routed to jcp-dims (A4)'),
        ('jcp-dims', f'{R.prov["dm.open_candidates.count"]["value"]} untested W1024 candidates; square-cell (M = r) runs '
         'not reproducible across jobs (cause unexplained; they also fail Q1)', 'jcp-dims report'),
        ('jcp-heads', 'head-repair hypotheses (E3c); more seeds or looser solver tolerance for K = 4, 8, 32',
         'jcp-heads §6d'),
        ('jcp-p1-scaling', 'none', 'spending stayed below the cap'),
        ('user', 'merges of the finished lanes; custody of the sealed-cohort key; the 2D floor check before opening; '
         'authorship with Hari', 'LAB-LOG "Where things stand"'),
    ]
    w(table(['owner', 'item', 'reason / source'], deferred))
    w()

    # ---- 7. ledger
    w('## 7. GPU-hour ledger')
    w()
    rows = [[l, g3(ledg[l][0]), g3(ledg[l][1]), ledg[l][2], 'running (priority 4)' if l in PENDING_LANES and not E(l).exists()
             else 'finished'] for l, *_ in LANES]
    w(table(['lane', 'spent (GPU-h, sacct)', 'cap', 'ledger entries', 'state'], rows))
    w()
    w('"Spent" is each lane ledger\'s `spent_gpu_hours`, which counts jobs that have reached a terminal state; '
      'jobs still running are not yet included.')
    w()
    w(f'Total over the eight finished lanes: **{g3(done_total)} GPU-h**; including the two priority-4 lanes at '
      f'compile time: {g3(all_total)} GPU-h (still growing while they run). speed3d\'s ledger includes its jobs from '
      'before the campaign started.')
    w()
    fig('figs/summary_gpu_ledger.png', 'GPU-hours spent against each lane\'s cap')

    # ---- 8. lanes
    w('## 8. Lanes (awaiting the user\'s merge decision)')
    w()
    rows = []
    lane_list = []
    for lane, wtd, what, prio in LANES:
        wt = LANE_WT[lane]
        br = f'exp/{wtd}'
        head = R.git(f'git.{lane}.head', wt, 'HEAD')
        mir = R.git(f'git.{lane}.mirror', wt, f'refs/remotes/origin/codeonly/exp/{wtd}')
        rep = LANE_DIR[lane] / 'report.md'
        rows.append([prio, lane, what, f'`{br}`', head, mir,
                     f'`{rel(rep)}`' if rep.exists() else 'not yet written',
                     'running' if lane in PENDING_LANES and not E(lane).exists() else 'awaiting merge decision'])
        lane_list.append(dict(lane=lane, branch=br, head=head, mirror=mir, report=rel(rep) if rep.exists() else None))
    w(table(['priority', 'lane', 'items', 'branch', 'last commit', 'code-only mirror', 'report', 'state'], rows))
    w()
    w('Last commits and mirror commits are read from git at compile time (the mirror is the local tracking ref '
      '`origin/codeonly/exp/<slug>` updated by each lane\'s `sync_github.sh`). For jcp-qualify and jcp-fom2d the last '
      'commit is the coordinator\'s archive-offload commit made after the lane closed.')
    w()

    # ---- 9. glossary
    w('## 9. Glossary')
    w()
    for t, d in GLOSSARY:
        w(f'- **{t}**: {d}')
    w()
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')

    fx = dict(compiled_utc=now, label='E0 R_ST conditional Richardson estimates; development/validation cohorts only; '
              'sealed and OOD cohorts unopened', pending=pend_now, verdicts=verdicts,
              gpu_hours=dict(per_lane={l: dict(spent=v[0], cap=v[1]) for l, v in ledg.items()},
                             total_finished=done_total, total_all=all_total),
              lanes=lane_list, figures=sorted(str(p.relative_to(HERE)) for p in FIGS.glob('*.png')))
    (HERE / 'FINAL-EXPORT.json').write_text(json.dumps(fx, indent=1) + '\n')
    for p in sorted(FIGS.glob('summary_*.png')):
        R._put(f'fig:{p.name}', dict(value=f'figs/{p.name}', file='generated by make_final.py', key_path='(new plot)',
                                     sha256=sha256(p)))
    (HERE / 'PROVENANCE.json').write_text(json.dumps(R.prov, indent=1, default=str) + '\n')
    print(f'wrote report.md ({len(L)} lines), {len(R.prov)} provenance entries, pending: {pend_now}')


def hist_h5():
    """Combined-H5 matched speed-ups, read exactly as jcp-fom2d's make_report.HIST_H5 does (pinned commit)."""
    commit = 'a3c9e170d8f47a53f34e8cb3672e546148383151'
    jf = 'experiments/jcp-combined/checks/analysis-c2grid-2d_e0.json'
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{jf}'])
    a = json.loads(raw)
    h = a['hypotheses']['H5']['ST']
    sel = max(v['speedup'] for k, v in h['matches'].items() if '|CNR|' in k and v.get('speedup') is not None)
    best = h['best_combined'][0]
    src = dict(file=f'git:{commit[:9]}:{jf}', sha256=hashlib.sha256(raw).hexdigest())
    R._put('comb.H5.selected', dict(value=sel, key_path='hypotheses/H5/ST/matches/*|CNR|*/speedup (max)', **src))
    R._put('comb.H5.best', dict(value=best, key_path='hypotheses/H5/ST/best_combined/0', **src))
    return dict(sel=f'{sel:.3g}×', best=f'{best:.3g}×', src=f'{jf} @ {commit[:9]}, hypotheses.H5.ST')


CORRECTION_SECTIONS = [  # (lane, heading of the lane report's own correction section)
    ('jcp-speed3d', '## What is provisional, and what was wrong'),
    ('jcp-qualify', '## Corrections to earlier reports (stated plainly)'),
    ('jcp-qualify', '## Deviations and retractions'),
    ('jcp-fom2d', '## What is provisional, and what changes earlier claims'),
    ('jcp-hr-baselines', '## Cancelled attempts and coverage'),
    ('jcp-heads', '## 6d. Relation to earlier results, and what is provisional'),
    ('jcp-dims', '## Retractions and corrections during the lane (details in DESIGN.md amendments)'),
    ('jcp-budget', '## What was wrong or corrected'),
]


def pointer_record():
    """Where the full correction records live (pointers only, no quoted text)."""
    w('### 5.1 Where the full records are')
    w()
    w('The list above consolidates the corrections. The full text lives in each lane report\'s own correction section '
      'and in LAB-LOG.md; this report points to them rather than quoting them.')
    w()
    rows = []
    for lane, head in CORRECTION_SECTIONS:
        rp = LANE_DIR[lane] / 'report.md'
        lines = rp.read_text().split('\n')
        i = lines.index(head)
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith('## ')), len(lines))
        R._put(f'pointer:{lane}:{head}', dict(value=f'lines {i + 1}-{j}', file=rel(rp), key_path=f'section "{head}"',
                                              sha256=sha256(rp)))
        rows.append([lane, f'`{rel(rp)}`', head[3:], f'{i + 1}–{j}'])
    log = ROOT / 'LAB-LOG.md'
    ll = log.read_text().split('\n')
    start = ll.index('## 2026-10-17')
    R._put('pointer:LAB-LOG', dict(value=f'lines {start + 1}-{len(ll)}', file=rel(log), key_path='from "## 2026-10-17"',
                                   sha256=sha256(log)))
    rows.append(['coordinator', '`LAB-LOG.md`', 'entries from 2026-10-17 on ("Wrong", "Retracted", "Corrections" bullets)',
                 f'{start + 1}–{len(ll)}'])
    w(table(['lane', 'file', 'section', 'lines at compile time'], rows))
    w()


def render_generic(lane, ex):
    """Auto-render a priority-4 EXPORT that arrived after this script was written: every scalar leaf whose key path
    names a verdict or label, with provenance. To be tailored when the coordinator resumes the compilation."""
    d = R.load(ex)
    w(f'Results read from `{rel(ex)}` (auto-rendered; labels as exported by the lane).')
    w()
    rows = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [k])
        elif isinstance(o, (str, int, float, bool)) and path and any(
                t in str(path[-1]).lower() for t in ('verdict', 'label', 'outcome')):
            nid = f'{lane}.' + '.'.join(str(p) for p in path)
            R.get(nid, ex, *path)
            rows.append(['/'.join(str(p) for p in path), o if isinstance(o, str) else g3(o)])
    walk(d, [])
    w(table(['key', 'value'], rows[:200]) if rows else '_No verdict or label fields found in the export._')
    w()


GLOSSARY = [
    ('r (bank width, $R\'$)', 'number of ordered linear bank coefficients the reduced model solves for; one reading of "k".'),
    ('K (head latent)', 'latent size of the nonlinear head that maps $K$ numbers to bank coefficients; the other reading of "k".'),
    ('mesh, N, L', 'size of the full-order mesh on which the input is sampled and the output decoded. 2D: $N^2$ grid points '
     '($256^2$–$4096^2$). 3D: $L$ intervals per axis ($L = 64$–$384$); "$65^3$" is $L = 64$ (65 nodes, $63^3$ interior unknowns).'),
    ('M, m', '$M$ = number of sine test functions; $m$ = number of quadrature points used to evaluate the nonlinear term.'),
    ('LSPG, GAL, BE, CN-R, CN, BDF2', 'least-squares Petrov–Galerkin (minimises the tested residual); Galerkin (a '
     'registered diagnostic scheme here); backward Euler; Crank–Nicolson with backward-Euler start-up steps; '
     'Crank–Nicolson; second-order backward differentiation.'),
    ('FS, FSP, QN; K1, K2; k1, k4, kinf', '3D solvers: fixed-sweep Levenberg–Marquardt (K1, K2 = one or two full-Jacobian '
     'sweeps per step); fixed-sweep with extra initial-guess candidates; quasi-Newton with the exact gradient and a '
     'Gauss–Newton matrix refreshed every k steps (k1, k4; kinf = only when forced).'),
    ('Jacobian J, normal matrix $J^\\top J$, Cholesky', 'the derivative of the reduced residual with respect to the '
     'coefficients; the small square matrix of the least-squares step; the factorisation used to solve with it. With '
     'the tested advection vector N these are the four timed kernels.'),
    ('$\\Delta t_0$, "2Δt₀", factor', 'the reference step 0.005 (2D); configuration labels give the step as a multiple of it.'),
    ('E0, $R_{ST}$, $R_S$, $\\varepsilon_{\\rm ref}$', 'the second-order reference solutions; $R_{ST}$ (space and time) scores the '
     'errors (exceptions named in the status paragraph); $R_S$ (space only, backward Euler at 0.005) is matched only to 2D BE; $\\varepsilon_{\\rm ref}$ is the per-case '
     'reference uncertainty, a conditional Richardson estimate (not a bound).'),
    ('e (error)', '$e=\\max_{k}\\lVert u(t_k)-R(t_k)\\rVert_2/\\lVert R(t_0)\\rVert_2$ over the five output times, on the '
     '$257^2$ (2D) or $63^3$ (3D) scoring lattice; "median e" and "worst e" are over the cases.'),
    ('dev6, val32, 923801, 16-case subset', 'development and validation cohorts (2D: $6+32=38$ cases; 3D: 64 cases); '
     'selection uses only these. The 16-case subset is dev6 plus the first ten val32 cases.'),
    ('sealed / OOD cohorts', 'held-back test cohorts that stay unopened until the user decides.'),
    ('Q1 (solver qualification)', 'every run valid, at most 1 % of steps not certified (LSPG: stationarity ratio '
     '$\\gamma=\\lVert J^\\top r\\rVert/(\\lVert J\\rVert_F\\lVert r\\rVert)\\le10^{-3}$; GAL: relative Galerkin residual '
     '$\\eta\\le10^{-8}$), distance to a tight solve at most $2.5\\times10^{-4}$ on every case, tight vs tighter at most '
     '$2.5\\times10^{-5}$.'),
    ('Q2 (quadrature qualification)', 'the next finer rule gives the same rollout (distance at most $2.5\\times10^{-4}$) and the '
     'same tested advection ($\\rho\\le0.116$) on the configuration\'s own states, and the finer rule itself agrees '
     'with the next one.'),
    ('numerically qualified', 'passes Q1 and Q2 (and the registered reproduction gate where one applies). Primary 2D verdicts use only such configurations; the 3D H-cost-N, '
     'H-comp and K2 rows include speed3d configurations that are not all qualified, and say so in the row.'),
    ('H-cost-N', 'cost does not depend on the mesh: $R=\\max/\\min$ over meshes of the median cost ratio; pass if the upper '
     'end of its 95 % case-bootstrap interval is at most 1.25 (3D H4.1: pass if $\\max/\\min\\le1.25$).'),
    ('H-cost-r, p, declared p', 'cost grows with r as the operation count predicts; $p$ is the exponent of $t=a+b\\,r^p$ '
     'fitted to kernel times at fixed $(M,m)$, compared with the operation-count ("declared") exponent. "Hardware '
     'scaling" / "operation-model" are the registered sub-labels of a fail.'),
    ('elasticity', 'descriptive slope of log query time against log r, log M and log m in the 2D timing panel; not a kernel exponent.'),
    ('H-acc-N', 'error does not depend on the mesh: two one-sided tests (TOST) that per-case error differences lie within '
     '$\\pm\\max(2\\varepsilon_{\\rm ref},\\,0.1\\,e)$, plus a trend test.'),
    ('H-acc-r, monotone', 'error decreases as r grows: at each step a majority of resolved cases improve and the median '
     'change is negative; a range is monotone if every step improves.'),
    ('H-comp', 'the ROM is at least 2× faster than a full-order solver matched to its accuracy.'),
    ('K2', 'the plan\'s pre-registered kill criterion: no selected ROM reaches 2× against the best matched full-order '
     'solver at the largest mesh.'),
    ('H-base', 'our model is not dominated by a tuned standard POD reduced model.'),
    ('H2, H2q', 'speed-up of the cheapest 3D configuration over our own earlier 3D configurations (ROM vs ROM; full-field '
     '$65^3$ mesh-path query on one H200); H2q restricts the winner to quadrature-qualified configurations; the '
     'comparator is qualified for C_old and C_W768 but not for C_W1024.'),
    ('C_old, C_W768, C_W1024; W_C_old, W_C_W768', 'the earlier deployed 3D BE configurations of the M2_512, W768 and W1024 '
     'banks; the speed3d winners selected against them.'),
    ('M2_512, W256 … W1024', '3D banks: the narrower M2 bank of width 512, and prefixes of width 256–1024 of the ordered W1024 bank.'),
    ('mesh path / mesh-free', '3D query contracts: input given on the mesh and outputs decoded on it, or input and outputs at '
     'the method\'s own points and the scoring lattice.'),
    ('union speed-up, matched', 'FOM time divided by ROM time for the cheapest full-order solver, over all families, that ties '
     'or beats the ROM on both median and worst error; below 1 the FOM is faster. An upper bound, because the FOM ladder is finite.'),
    ('UB, UC, CC, PS, ETDRK4', 'full-order families: sign-upwind + BE, upwind + CN-R, central differences + CN-R, sine '
     'pseudo-spectral; ETDRK4 is the fourth-order exponential time integrator used by the 2D spectral solver.'),
    ('POD, ECSW, Q-DEIM', 'proper orthogonal decomposition basis; energy-conserving sampling and weighting and Q-DEIM '
     'interpolation, two standard hyper-reduction methods.'),
    ('B_acc, B_cost', 'role picks among the baselines: the fastest baseline whose worst error is at most ours; the most '
     'accurate baseline whose median time is at most ours.'),
    ('dominated, time ratio', 'a baseline is both faster and has lower worst error, with non-overlapping intervals; time '
     'ratio = baseline query time / our query time.'),
    ('developmental, in-sample, unadjusted', 'both methods were tuned on the same cases; no held-out confirmation, no '
     'multiplicity correction.'),
    ('front', 'the configurations not beaten on both error and cost by another qualified configuration.'),
    ('frozen / nominal mode list', 'one test-mode list reused at every mesh / each mesh\'s own list (deployment policy).'),
    ('P1g, pp', 'the recheck of earlier cross-mesh accuracy claims under a frozen mode list; pp = percentage points of relative error.'),
    ('H64, H96, H640, HF6765', 'historical 2D quadrature rules at $r=384$: Gauss $64^2$, $96^2$, $640^2$ and Fibonacci 6765 points.'),
    ('rule families, $m^*$', 'gauss / gl: tensor Gauss–Legendre; fib: Fibonacci lattice; lattice / lat: rank-1 lattice; '
     'sobol, halton / hal: quasi-random sequences; smolyak: sparse grid; mc: Monte Carlo. $m^*$ is the smallest point '
     'count in a family that passes the quadrature check.'),
    ('A–B–A', 'timing design: configuration B timed between two runs of the reference A on the same GPU; cost = B / mean(A).'),
    ('error-budget ingredients', 'Q quadrature, T time stepping, F reduced form with its stopping rule, S solver tolerance, '
     'I initial condition, M test space; $e_P$ = error of the best fit by the bank (projection floor).'),
    ('W1, W2, W3', 'output-time windows $(0,0.25]$, $(0.25,0.5]$, $(0.5,1]$.'),
    ('stable (long horizons)', 'every run valid and, at every recorded output time, the reduced kinetic energy no more '
     'than $10^{-3}$ (relative) above its initial value; nothing is checked between output times.'),
    ('no error growth', 'per case $g=e_{W_2}/e_{W_1}$ (or $e_{W_3}/e_{W_2}$) over cases whose errors exceed twice their '
     'reference uncertainty in both windows; "no growth" if the upper end of the bootstrap 95 % interval of the median '
     '$g$ is at most 1; "inconclusive (population)" below the minimum number of cases.'),
    ('dominant, no single dominant', 'per case, the largest resolved error ingredient is the winner (ties within '
     'a registered threshold); an ingredient is dominant if it wins on more than half of the complete cases, otherwise '
     '"no single dominant"; "inconclusive" if too few cases are complete.'),
    ('front order preserved', 'no resolved pair of members reverses its ranking by median window error between the '
     'two windows, over their common eligible cases (not necessarily identical raw rankings).'),
    ('TH06', '$\\theta$-method with $\\theta=0.6$, a first-order control scheme in the temporal-order check.'),
    ('C-A, C-E', 'jcp-budget\'s registered must-fire controls for the output-sampling check: an injected sampling alias '
     'and an injected energy drift.'),
    ('EQ', 'empirical quadrature: hyper-reduction weights fitted to training data (here a deferred baseline variant).'),
    ('GPU-h', 'GPU-hours from Slurm accounting (elapsed time × GPUs).'),
    ('Wilcoxon test, TOST, case-bootstrap', 'a rank-based test on paired per-case differences; two one-sided tests '
     'for equivalence within a margin; a confidence interval obtained by resampling whole cases with replacement.'),
    ('hyperbolic-cross test set', 'a set of sine test functions chosen to favour low mixed frequencies (products of '
     'small frequencies in each direction) rather than the lowest Laplacian eigenvalues.'),
    ('NNLS, GNAT, IMEX-CG', 'non-negative least squares (used to fit ECSW weights); Gauss–Newton with approximated '
     'tensors, another hyper-reduced ROM (deferred); an implicit–explicit time stepper with conjugate-gradient solves '
     '(an untested FOM family).'),
]

if __name__ == '__main__':
    main()
