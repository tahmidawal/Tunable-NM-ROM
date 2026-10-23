"""Final compilation of the 2026-10-21 improvement campaign (AGENT-PROMPT.md section 8).

Reads every measured number from the lane outputs (experiments/<lane>/EXPORT.json, runs/GPU-LEDGER.json, a few
report-generating check JSONs, campaign/LANES.json, and git for commits) and writes, next to this file:
report.md, figs/*.png (copied lane plots), FINAL-EXPORT.json and PROVENANCE.json.

No measured number is typed by hand: every value printed in report.md is registered in PROVENANCE.json with its source
file, JSON key path and the file's sha256 (derived values list their sources and the derivation). Registered design
constants of the plan (bars such as 2x, 3x, 0.5, kappa <= 16, the 1 GPU-h smoke allowance) are quoted as constants.

E1 (imp-box) is read from its EXPORT.json only when that file says stage == "final" AND campaign/LANES.json marks the
lane done; otherwise every E1 cell of the report is "pending".

Run: JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python make_final.py
"""
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
WT = ROOT / 'worktrees'
FIGS = HERE / 'figs'

LANES = [  # lane, worktree dir, experiments, problem
    ('imp-box', '2026-10-22-imp-box', 'E1', 1),
    ('imp-geometry', '2026-10-21-imp-geometry', 'E2', 1),
    ('imp-varcoef', '2026-10-21-imp-varcoef', 'E3', 1),
    ('imp-faststep', '2026-10-21-imp-faststep', 'E5', 2),
    ('imp-hrcost', '2026-10-21-imp-hrcost', 'E6', 2),
    ('imp-multimesh', '2026-10-21-imp-multimesh', 'E7', 2),
    ('imp-fronts', '2026-10-21-imp-fronts', 'E8, E9, E10', 3),
]
LD = {l: WT / w / 'experiments' / l for l, w, _, _ in LANES}
LW = {l: WT / w for l, w, _, _ in LANES}
LBRANCH = {l: 'exp/' + w for l, w, _, _ in LANES}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


class Reg:
    def __init__(self):
        self.prov = {}
        self._j = {}

    def load(self, path):
        path = Path(path)
        if path not in self._j:
            self._j[path] = json.loads(path.read_text())
        return self._j[path]

    def get(self, nid, path, *keys):
        o = self.load(path)
        for k in keys:
            o = o[k]
        self.prov[nid] = {'value': o, 'file': rel(path), 'key_path': list(keys), 'sha256': sha256(path)}
        return o

    def derive(self, nid, value, sources, how):
        self.prov[nid] = {'value': value, 'derived_from': sources, 'derivation': how}
        return value

    def git(self, nid, value, cmd):
        self.prov[nid] = {'value': value, 'source': 'git', 'command': cmd}
        return value


R = Reg()


def pct(x, d=2):
    return f'{100 * x:.{d}f} %'


def xr(x, d=2):
    return f'{x:.{d}f}×'


def f3(x):
    return f'{x:.3g}'


def esc(s):
    return str(s).replace('|', '\\|')


def gh(x):
    return f'{x:.2f}'


# ------------------------------------------------------------------------------------------------ lane status
LANESJ = CAMP / 'campaign' / 'LANES.json'
lane_status = {d['lane']: d['status'] for d in R.load(LANESJ)['lanes']}
lane_cap = {d['lane']: d['gpu_hours_cap'] for d in R.load(LANESJ)['lanes']}
e1_path = LD['imp-box'] / 'EXPORT.json'
e1_stage = R.load(e1_path).get('stage') if e1_path.exists() else None
E1_FINAL = bool(e1_stage == 'final' and lane_status.get('imp-box') == 'done')

# ------------------------------------------------------------------------------------------------ git
def run(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()


branch_head = {}
for l in LD:
    out = run(['git', 'log', '-1', '--format=%H|%cI|%s', LBRANCH[l]])
    h, t, s = out.split('|', 2)
    branch_head[l] = R.git(f'git.head.{l}', {'commit': h, 'date': t, 'subject': s},
                           f'git log -1 --format=%H|%cI|%s {LBRANCH[l]}')
ls = run(['git', 'ls-remote', 'origin', 'refs/heads/codeonly/exp/2026-10-2*-imp-*'])
mirror = {}
for line in ls.splitlines():
    h, ref = line.split('\t')
    for l in LD:
        if ref.endswith(LBRANCH[l]):
            mirror[l] = R.git(f'git.mirror.{l}', h,
                              "git ls-remote origin 'refs/heads/codeonly/exp/2026-10-2*-imp-*'")

# ------------------------------------------------------------------------------------------------ figures
FIGS.mkdir(exist_ok=True)
FIGLIST = {
    'imp-faststep': ['plots/worst_error_vs_evolve_time.png'],
    'imp-hrcost': ['plots/error-vs-speedup.png', 'plots/f2-ladder-q2.png'],
    'imp-multimesh': ['plots/cost_vs_queries.png'],
    'imp-varcoef': ['plots/gate.png', 'plots/fom_cost_vs_error.png'],
    'imp-fronts': ['plots/a1_smearing.png', 'plots/a3_truncation.png', 'plots/a4_hybrid_floors.png',
                   'plots/phaseb_medians.png'],
    'imp-geometry': ['plots/S_per_family.png'],
    'imp-box': ['plots/S_vs_W_median.png', 'plots/S_vs_W_worst.png'] if E1_FINAL else [],
}
fig = {}
fig_prov = {}
for l, lst in FIGLIST.items():
    for p in lst:
        src = LD[l] / p
        if src.exists():
            dst = FIGS / f'{l}__{Path(p).name}'
            shutil.copy2(src, dst)
            fig[(l, Path(p).name)] = f'figs/{dst.name}'
            fig_prov[dst.name] = {'source': rel(src), 'sha256': sha256(src)}


def img(l, name, cap):
    k = (l, name)
    return f'![{cap}]({fig[k]})\n\n*Figure ({l}): {cap}*\n' if k in fig else ''


# ================================================================================================ E5 (imp-faststep)
FS = LD['imp-faststep'] / 'EXPORT.json'
FSC = {m: LD['imp-faststep'] / 'checks' / f'f{m}.json' for m in (1024, 4096)}
e5 = {}
for m in (1024, 4096):
    for arm in ('G-CN', 'G-E4a', 'G-E4b', 'G-R1'):
        k = f'f{m}|{arm}'
        e5[(m, arm, 'worst')] = R.get(f'e5.{m}.{arm}.worst384', FS, 'verdicts', k, 'worst', '384')
        e5[(m, arm, 'arm1')] = R.get(f'e5.{m}.{arm}.arm1worst384', FS, 'verdicts', k, 'worst_arm1', '384')
        e5[(m, arm, 'S')] = R.get(f'e5.{m}.{arm}.speedup384', FS, 'verdicts', k, 'speedup_evolve_r384')
        e5[(m, arm, 'fired')] = R.get(f'e5.{m}.{arm}.fired', FS, 'verdicts', k, 'fired')
        e5[(m, arm, 'qual')] = R.get(f'e5.{m}.{arm}.qualified', FS, 'verdicts', k, 'qualified')
        e5[(m, arm, 'better')] = R.get(f'e5.{m}.{arm}.better384', FSC[m], 'stop', arm, 'worst_all38', '384',
                                       'ranking_vs_arm1', 'better')
        e5[(m, arm, 'worse')] = R.get(f'e5.{m}.{arm}.worse384', FSC[m], 'stop', arm, 'worst_all38', '384',
                                      'ranking_vs_arm1', 'worse')
    e5[(m, 'fixed')] = R.get(f'e5.{m}.fixed', FS, 'verdicts', f'f{m}|E5_fixed_on_completed_scope')
e5_all_worst = all(e5[(m, a, 'fired')]['worst_error'] for m in (1024, 4096)
                   for a in ('G-CN', 'G-E4a', 'G-E4b', 'G-R1'))
e5_fixed = any(e5[(m, 'fixed')] for m in (1024, 4096))
e5_any_unstable = any(e5[(m, a, 'fired')]['stability_validity'] for m in (1024, 4096)
                      for a in ('G-CN', 'G-E4a', 'G-E4b', 'G-R1'))
e4S = [e5[(m, a, 'S')] for m in (1024, 4096) for a in ('G-E4a', 'G-E4b')]
e5_e4_min = R.derive('e5.etdrk4.speedup.min', min(e4S), [f'e5.{m}.{a}.speedup384' for m in (1024, 4096)
                                                        for a in ('G-E4a', 'G-E4b')], 'min')
e5_e4_max = R.derive('e5.etdrk4.speedup.max', max(e4S), [f'e5.{m}.{a}.speedup384' for m in (1024, 4096)
                                                        for a in ('G-E4a', 'G-E4b')], 'max')
e5_pod = {a: R.get(f'e5.pod_ratio.{a}', FS, 'verdicts', 'f1024|matches', f'{a}|384', 'direct', 'value')
          for a in ('G-E4a', 'G-E4b', 'L-CNR')}
e5_pod_B = {a: R.get(f'e5.pod_B.{a}', FS, 'verdicts', 'f1024|matches', f'{a}|384', 'B') for a in ('G-E4a', 'G-E4b')}
e5_podlab = {a: R.get(f'e5.pod_Q2label.{a}', FSC[1024], 'pod', e5_pod_B[a], 'Q2', 'label') for a in ('G-E4a', 'G-E4b')}
e5_best4096 = R.get('e5.best_stepper.4096', FS, 'verdicts', 'f4096|best_stepper')
e5_best1024 = R.get('e5.best_stepper.1024', FS, 'verdicts', 'f1024|best_stepper')
e5_gpu = R.get('e5.gpu', LD['imp-faststep'] / 'runs' / 'GPU-LEDGER.json', 'spent_gpu_hours')

# ================================================================================================ E6 (imp-hrcost)
HR = LD['imp-hrcost'] / 'EXPORT.json'
e6_F2 = R.get('e6.stop.F2', HR, 'stop', 'F2')
e6_F3 = R.get('e6.stop.F3', HR, 'stop', 'F3')
e6_mstar = R.get('e6.F2.mstar', HR, 'stop', 'F2_mstar')
e6_R_worst = R.get('e6.R.worst', HR, 'arms', 'R', 'worst')
e6_R_m = R.get('e6.R.m', HR, 'arms', 'R', 'm')
e6_F2_m = R.get('e6.F2.m', HR, 'arms', f'F2_{e6_mstar}', 'm')
e6_F2_worst = R.get('e6.F2.worst', HR, 'arms', f'F2_{e6_mstar}', 'worst')
e6_F2_e2e = R.get('e6.F2.e2e', HR, 'arms', f'F2_{e6_mstar}', 'e2e')
f3arms = [f'F3{v}_k{k}' for v in 'ab' for k in (4, 8, 16)]
e6_f3 = {a: {'e2e': R.get(f'e6.{a}.e2e', HR, 'arms', a, 'e2e'), 'q2': R.get(f'e6.{a}.q2', HR, 'arms', a, 'q2'),
             'worst': R.get(f'e6.{a}.worst', HR, 'arms', a, 'worst')} for a in f3arms}
e6_f3_best = max(f3arms, key=lambda a: e6_f3[a]['e2e'])
e6_f3_best_e2e = R.derive('e6.F3.best_e2e', e6_f3[e6_f3_best]['e2e'], [f'e6.{a}.e2e' for a in f3arms], 'max')
e6_f3_q2pass = R.derive('e6.F3.n_q2_pass', sum(e6_f3[a]['q2'] == 'pass' for a in f3arms),
                        [f'e6.{a}.q2' for a in f3arms], 'count q2 == pass')
e6_ff = R.get('e6.F2F3_k16.e2e', HR, 'arms', 'F2F3_k16', 'e2e')
e6_ff_w = R.get('e6.F2F3_k16.worst', HR, 'arms', 'F2F3_k16', 'worst')
e6_ff_q = R.get('e6.F2F3_k16.qualified', HR, 'arms', 'F2F3_k16', 'qualified')
e6_jd = R.get('e6.jac.dup', HR, 'jac_bench', 'dup_ratio')
e6_jq = R.get('e6.jac.quad', HR, 'jac_bench', 'quad_ratio')
e6_gpu = R.get('e6.gpu', LD['imp-hrcost'] / 'runs' / 'GPU-LEDGER.json', 'total_gpu_hours')
e6_F2_fired = e6_F2.startswith('fired')
e6_F3_fired = e6_F3.startswith('fired')

# ================================================================================================ E7 (imp-multimesh)
MM = LD['imp-multimesh'] / 'EXPORT.json'
e7_out = R.get('e7.outcome', MM, 'outcome_primary')
e7_vc = {c: R.get(f'e7.counts.{c}', MM, 'verdict_counts', f'{c}|A_dep') for c in
         ('worst', 'worst_eps', 'casewise', 'worst_median', 'rollout')}
e7_vc100 = R.get('e7.counts.worst.A_100', MM, 'verdict_counts', 'worst|A_100')
e7_vcnh = R.get('e7.counts.worst.A_nohead', MM, 'verdict_counts', 'worst|A_nohead')
e7_n = R.derive('e7.n_pairs', sum(e7_vc['worst'].values()), ['e7.counts.worst'], 'sum of MATCH+NO-MATCH+UNRESOLVED')
e7_rmin = R.get('e7.ratio.min', MM, 'primary_ratio', 'min')
e7_rmed = R.get('e7.ratio.median', MM, 'primary_ratio', 'median')
e7_rmax = R.get('e7.ratio.max', MM, 'primary_ratio', 'max')
e7_noadm_case = R.get('e7.noadm.casewise', MM, 'nomatch_without_admissible', 'casewise|A_dep')
e7_noadm_roll = R.get('e7.noadm.rollout', MM, 'nomatch_without_admissible', 'rollout|A_dep')
e7_gpu = R.get('e7.gpu', LD['imp-multimesh'] / 'runs' / 'GPU-LEDGER.json', 'spent_gpu_hours')

# ================================================================================================ E3 (imp-varcoef)
VC = LD['imp-varcoef'] / 'EXPORT.json'
VG = LD['imp-varcoef'] / 'checks' / 'gate.json'
e3_gate = R.get('e3.gate', VC, 'gate')
e3_reg = R.get('e3.gate_registered', VC, 'gate_registered_rule', 'outcome')
e3_contrast = R.get('e3.contrast', VC, 'gate_detail', 'contrast')
e3 = {}
for t in ('0.03', '0.1'):
    e3[t] = {'G': R.get(f'e3.{t}.G', VC, f'tau_{t}', 'G'), 'ci': R.get(f'e3.{t}.G_ci95', VC, f'tau_{t}', 'G_ci95'),
             'fom': R.get(f'e3.{t}.fom', VC, f'tau_{t}', 'fom'), 'rom': R.get(f'e3.{t}.rom', VC, f'tau_{t}', 'rom'),
             'romw': R.get(f'e3.{t}.rom_worst', VG, 'targets', f'{t}|C100', 'rom', 'worst'),
             'out': R.get(f'e3.{t}.outcome', VC, 'gate_detail', 'target_outcomes', t)}
e3_test10 = R.get('e3.test.etd10.worst', VC, 'test', 'etd|N48|0.0125|mid@C100', 'worst')
e3_test3 = R.get('e3.test.etd3.worst', VC, 'test', 'etd|N48|0.003125|mid@C100', 'worst')
VT = LD['imp-varcoef'] / 'runs' / 'test' / 'output' / 'vtest_test-run' / 'result.json'
VRT = LD['imp-varcoef'] / 'runs' / 'test' / 'output' / 'vref_ref-test' / 'result.json'
e3_test_nel = R.get('e3.test.etd10.n_eligible', VT, 'keys', 'etd|N48|0.0125|mid@C100', 'n_eligible')
_rr = R.load(VRT)['cases']
e3_test_n = R.derive('e3.test.n_cases', len(_rr), [rel(VRT) + ':cases'], 'count of test reference cases')
e3_test_nfail = R.derive('e3.test.n_ref_not_accepted', sum(c['judge']['outcome'] != 'accepted' for c in _rr.values()),
                         [rel(VRT) + ':cases[*].judge.outcome'], "count outcome != 'accepted'")
e3_t1 = R.get('e3.0.01.status', VC, 'tau_0.01', 'status')
e3_gpu = R.get('e3.gpu', LD['imp-varcoef'] / 'runs' / 'GPU-LEDGER.json', 'total_gpu_hours')

# ================================================================================================ E2 (imp-geometry)
GE = LD['imp-geometry'] / 'EXPORT.json'
e2 = {}
for c in ('L|wide|S1', 'H|wide|S1'):
    sv = R.get(f'e2.{c}.S_val', GE, 'cells', c, 'S_val')
    st = R.get(f'e2.{c}.S_test', GE, 'cells', c, 'S_test')
    pf = R.get(f'e2.{c}.per_family_val', GE, 'cells', c, 'per_family_val')
    al = R.get(f'e2.{c}.allowance', GE, 'cells', c, 'allowance')
    lab = R.get(f'e2.{c}.labels', GE, 'cells', c, 'labels')
    bc = R.get(f'e2.{c}.best_competitor', GE, 'cells', c, 'best_competitor')
    svv = [v for v in sv.values() if v is not None]
    stv = [v for v in st.values() if v is not None]
    semw = [pf[f'{t}|worst']['sem'] for t in ('0.01', '0.03', '0.1')]
    e2[c] = {
        'sv_min': R.derive(f'e2.{c}.S_val.min', min(svv), [f'e2.{c}.S_val'], 'min over targets x statistics'),
        'sv_max': R.derive(f'e2.{c}.S_val.max', max(svv), [f'e2.{c}.S_val'], 'max over targets x statistics'),
        'st_min': R.derive(f'e2.{c}.S_test.min', min(stv), [f'e2.{c}.S_test'], 'min over non-null'),
        'st_max': R.derive(f'e2.{c}.S_test.max', max(stv), [f'e2.{c}.S_test'], 'max over non-null'),
        'st_null': R.derive(f'e2.{c}.S_test.n_null', sum(v is None for v in st.values()), [f'e2.{c}.S_test'],
                            'count null (finalists excluded at test)'),
        'sem_min': R.derive(f'e2.{c}.sem_worst.min', min(semw), [f'e2.{c}.per_family_val'], 'min sem worst S'),
        'sem_max': R.derive(f'e2.{c}.sem_worst.max', max(semw), [f'e2.{c}.per_family_val'], 'max sem worst S'),
        'fd10': pf['0.1|worst']['fdmg'], 'fd3': pf['0.03|worst']['fdmg'],
        'fd_gt1': [k for k in pf if pf[k]['fdmg'] > 1],
        'inel': al['n_ineligible'], 'nref': al['n_ineligible'] + al['n_eligible'],
        'labels': {t: lab[t]['label'] for t in lab}, 'bc': sorted(set(bc.values())),
    }
e2_A10 = R.get('e2.A10', GE, 'A10')
e2_glab = R.get('e2.geometry_labels', GE, 'geometry_labels')
e2_gpu = R.get('e2.gpu', LD['imp-geometry'] / 'runs' / 'GPU-LEDGER.json', 'total_gpu_hours')
e2_smoke = R.get('e2.smoke_gpu', LD['imp-geometry'] / 'runs' / 'GPU-LEDGER.json', 'smoke_gpu_hours')
e2_over = R.derive('e2.smoke_over', e2_smoke - 1.0, ['e2.smoke_gpu'], 'smoke_gpu_hours - 1.0 (registered allowance)')
e2_win = any(lbl == 'win' for c in e2 for lbl in e2[c]['labels'].values())
e2_Smax_all = R.derive('e2.S.max_all', max(e2[c]['sv_max'] for c in e2), [f'e2.{c}.S_val.max' for c in e2], 'max')

# ================================================================================================ E8/E9/E10 (imp-fronts)
FR = LD['imp-fronts'] / 'EXPORT.json'
FL = LD['imp-fronts'] / 'runs' / 'GPU-LEDGER.json'
e8A = {k: R.get(f'e8.{k}.verdict', FR, k, 'verdict' if k != 'A2' else 'decision_arm_c') for k in ('A1', 'A2', 'A3', 'A4')}
e8v = R.get('e8.phaseb.verdicts', FR, 'phase_b', 'verdicts')
b_arm = R.get('e8.b.fc1_median', FR, 'phase_b', 'routes', 'b', 'rom', 'arm_median')
b_ctl = R.get('e8.K1.fc1_median', FR, 'phase_b', 'routes', 'b', 'rom', 'ctrl_median')
b_ratio = R.derive('e8.b.ratio', b_ctl / b_arm, ['e8.K1.fc1_median', 'e8.b.fc1_median'], 'ctrl / arm')
b_q = R.get('e8.b.fc1_qualified', FR, 'phase_b', 'score_b', 'FC1', 'qualified')
b_n = R.get('e8.b.fc1_n', FR, 'phase_b', 'score_b', 'FC1', 'n_eligible')
b_guard = R.get('e8.b.guard', FR, 'phase_b', 'guards', 'b', 'status')
b_gr = R.get('e8.b.guard_ratio', FR, 'phase_b', 'guards', 'b', 'ratio')
c_nodes = R.get('e8.c.nodes', FR, 'phase_b', 'routes', 'c', 'mstar', 'arm_nodes')
c_bar = R.get('e8.c.nodes_bar', FR, 'phase_b', 'routes', 'c', 'mstar', 'bar')
c_rule = R.get('e8.c.rule', FR, 'phase_b', 'routes', 'c', 'mstar', 'rule')
c_ta = R.get('e8.c.tail_arm', FR, 'phase_b', 'routes', 'c', 'tail', 'arm_median')
c_tc = R.get('e8.c.tail_ctrl', FR, 'phase_b', 'routes', 'c', 'tail', 'ctrl_median')
c_tr = R.derive('e8.c.tail_ratio', c_tc / c_ta, ['e8.c.tail_ctrl', 'e8.c.tail_arm'], 'ctrl / arm')
c_fc1 = R.get('e8.c.fc1_median', FR, 'phase_b', 'score_c', 'FC1', 'median_e_diagnostic')
c_guard = R.get('e8.c.guard', FR, 'phase_b', 'guards', 'c', 'status')
c_gr = R.get('e8.c.guard_ratio', FR, 'phase_b', 'guards', 'c', 'ratio')
c_wall = R.get('e8.c.wall_max', FR, 'phase_b', 'armc_wall', 'max_rel_wall_rms')
c_wall_bar = R.get('e8.c.wall_bar', FR, 'phase_b', 'armc_wall', 'registered_bar')
k1_fn0s = R.get('e8.K1.fn0s_q', FR, 'phase_b', 'score_K1', 'FN0s', 'qualified')
e9_tau = R.get('e9.tau_s', FR, 'phase_b', 'e9', 'tau_s')
e9_n = R.get('e9.n', FR, 'phase_b', 'e9', 'n')
e9_flag = R.get('e9.flagged', FR, 'phase_b', 'e9', 'flagged_fraction')
e9_t = {t: {k: R.get(f'e9.{t}.{k}', FR, 'phase_b', 'e9', 'targets', t, k)
            for k in ('auc', 'detection', 'false_alarm', 'n_bad', 'n_good', 'label', 'threshold')} for t in 'AB'}
e9_fi = R.get('e9.FI_alarm', FR, 'phase_b', 'e9', 'flagged_by_cohort', 'FI', 'alarm')
s10 = R.get('e10.s', FR, 'e10_results', 'verdicts', 'E10-s')
q10 = R.get('e10.q', FR, 'e10_results', 'verdicts', 'E10-q')
e10s_q = R.get('e10.s.fc1_q', FR, 'e10_results', 'score_e10s', 'FC1', 'qualified')
fl_jobs = R.load(FL)['jobs']
e8_gpu_tot = R.get('e8.gpu_total', FL, 'cumulative_gpu_hours')
e10_gpu = R.derive('e10.gpu', sum(j['charged_gpu_h'] for j in fl_jobs if j.get('bucket') == 'E10'),
                   [rel(FL) + ':jobs[*].charged_gpu_h'], "sum over jobs with bucket == 'E10'")
e8_gpu = R.derive('e8.gpu', sum(j['charged_gpu_h'] for j in fl_jobs if j.get('bucket') != 'E10'),
                  [rel(FL) + ':jobs[*].charged_gpu_h'], "sum over jobs with bucket != 'E10'")
smc = [j for j in fl_jobs if j['name'] == 'SM-c'][0]
R.prov['e8.smc.note'] = {'value': smc['note'], 'file': rel(FL), 'key_path': ['jobs', 'name=SM-c', 'note'],
                         'sha256': sha256(FL)}
fr_def = R.get('e8.deferred', FL, 'deferred')
e8_fixed = not any(v.startswith('STOP') for v in (e8v['b'], e8v['c']))
e10_fixed = not (s10['outcome'] == 'STOP' and q10['outcome'] == 'STOP')

# ================================================================================================ E1 (imp-box)
BX = LD['imp-box'] / 'EXPORT.json'
BL = LD['imp-box'] / 'runs' / 'GPU-LEDGER.json'
e1_gpu = R.get('e1.gpu', BL, 'total_gpu_hours') if BL.exists() else None
if E1_FINAL:
    e1_sr = R.get('e1.stop_rule', BX, 'stop_rule')
    e1_maxS = R.get('e1.stop_rule.max_S', BX, 'stop_rule', 'max_S')
    e1_scope = R.get('e1.scope', BX, 'scope')
    e1_cells = R.load(BX)['cells']
    e1_test = R.load(BX)['test']
    ARMN = {'podg': 'R-PODG (POD-Galerkin)', 'ecsw': 'R-ECSW (POD + ECSW)', 'trans': 'R-TRANS (translation model)',
            'bank_lspg': 'B-LSPG (our bank, deployed LSPG stepper)', 'bank_e4': 'B-E4 (our bank, Galerkin ETDRK4)'}
    e1_rows, e1_inc = [], []   # S >= 2 with complete FOM search / with incomplete FOM search
    e1_by_arm = {}
    for c, d in e1_cells.items():
        for k, v in d['entries'].items():
            S = v.get('S')
            arm = k.split('|')[0]
            if S is None or v.get('status') != 'finalist':
                continue
            complete = all(x == 'finalist' for x in v['fom_status'].values())
            e1_by_arm.setdefault(arm, []).append((S, c, k, complete))
            if S >= 2:
                R.get(f'e1.S.{c}.{k}', BX, 'cells', c, 'entries', k, 'S')
                t = e1_test.get(c, {}).get('pairs', {}).get(k, {})
                st = R.get(f'e1.S_test.{c}.{k}', BX, 'test', c, 'pairs', k, 'S_test') if t.get('matched') else None
                row = (c, k, S, st, v['competitor_key'], v.get('ladder_minimum') or [], v['fom_status'])
                (e1_rows if complete else e1_inc).append(row)
    e1_lspg = [x[0] for x in e1_by_arm.get('bank_lspg', [])]
    e1_lspg_min = R.derive('e1.bank_lspg.S.min', min(e1_lspg), [rel(BX) + ':cells[*].entries[bank_lspg|*].S (finalist)'], 'min')
    e1_lspg_max = R.derive('e1.bank_lspg.S.max', max(e1_lspg), [rel(BX) + ':cells[*].entries[bank_lspg|*].S (finalist)'], 'max')
    e1_lspg_n = R.derive('e1.bank_lspg.n', len(e1_lspg), [rel(BX) + ':cells[*].entries[bank_lspg|*]'], 'count finalist entries')
    def _wide(c):
        return int(c[1:3]) >= 28
    e1_e4w = [x[0] for x in e1_by_arm.get('bank_e4', []) if _wide(x[1]) and x[3]]
    e1_e4w_max = R.derive('e1.bank_e4.wide.S.max', max(e1_e4w), [rel(BX) + ':cells[W28*,W40*].entries[bank_e4|*].S (finalist, complete FOM search)'], 'max')
    e1_a16 = R.load(BX)['a16_timing']
    e1_a16d = [e1_a16[c]['fom_dense']['ratio'] for c in e1_a16]
    e1_a16_min = R.derive('e1.a16.fom_dense.min', min(e1_a16d), [rel(BX) + ':a16_timing[*].fom_dense.ratio'], 'min')
    e1_a16_max = R.derive('e1.a16.fom_dense.max', max(e1_a16d), [rel(BX) + ':a16_timing[*].fom_dense.ratio'], 'max')
    e1_lm_n = R.derive('e1.S2.complete.n_ladder_min', sum(bool(r[5]) for r in e1_rows), ['e1.S.*'], 'count complete S>=2 entries with a ladder-minimum label')
    e1_arms2 = sorted({r[1].split('|')[0] for r in e1_rows})
else:
    e1_sr = None
    e1_interim_speed = R.get('e1.interim.stop_speed', BX, 'stop_rule', 'speed') if BX.exists() else None
    R.get('e1.stage', BX, 'stage')

# ------------------------------------------------------------------------------------------------ verdicts
def v_e1():
    if not E1_FINAL:
        return 'pending'
    if e1_sr['speed'] != 'NOT FIRED':
        return 'stop (speed)'
    return 'partly (see §2)' if e1_rows else 'no'


e3_fail = e3_gate != 'PASS'
p1 = 'not fixed so far (E2, E3 negative; E4 not run; E1 pending)' if not E1_FINAL else (
    'partly fixed (box only, under conditions; see E1)' if (e1_rows and not e2_win and e3_fail) else
    ('not fixed' if not e1_rows else 'see E1'))
p2 = 'fixed' if (e5_fixed or not e6_F2_fired or not e6_F3_fired) else 'not fixed'
p3 = 'fixed' if (e8_fixed or e10_fixed) else 'not fixed'
if p2 == 'not fixed':
    pass
P_VERDICT = {1: p1, 2: p2, 3: p3}

# ================================================================================================ report
L = []
w = L.append
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
status = ('**provisional** — E1 (box, family width × steepness) is still running and every E1 entry is "pending"'
          if not E1_FINAL else '**final** for every lane')
w('# Improvement campaign (2026-10-21): can the three known weaknesses of the off-mesh bank ROM be fixed?\n')
w(f'**Status (compiled {now} by `make_final.py`):** {status}. E4 was not run. Accuracy numbers are conditional estimates against second-order (E0 ST) or lane spectral references '
  '(conditional Richardson $\\varepsilon_{\\rm ref}$; the E0 A10 and combined A10/A11/A15 labels carry forward). '
  'POD comparisons are developmental and unadjusted: in-sample on dev6 + val32 in E5–E7; in E1 selected on val32 with the '
  'frozen pairs checked once on a fresh test32 cohort; every speed-up against a tuned ladder '
  'is an upper bound "against the strongest tested competitor". Every number below is read from a lane output and '
  'registered in `PROVENANCE.json`; bars such as 2×, 3×, 0.5 are registered constants of the plan.\n')

w('```mermaid\nflowchart LR\n'
  '  P1["Problem 1: coarse spectral FOM is faster"]:::prob --> E1["E1 box width x steepness"]\n'
  '  P1 --> E2["E2 geometry without FFT"]\n  P1 --> E3["E3 variable coefficient"]\n'
  '  P1 --> E4["E4 3D holes (only if E2 wins)"]\n'
  '  P2["Problem 2: tuned POD+ECSW dominates"]:::prob --> E5["E5 faster step"]\n'
  '  P2 --> E6["E6 cheaper nonlinear term"]\n  P2 --> E7["E7 multi-mesh full cost"]\n'
  '  P3["Problem 3: fronts steeper than training"]:::prob --> E8["E8 retrain a better bank"]\n'
  '  P3 --> E9["E9 out-of-regime flag"]\n  P3 --> E10["E10 numerics near fronts"]\n'
  '  classDef prob fill:#e5e7eb,stroke:#374151;\n  classDef stop fill:#fee2e2,stroke:#991b1b;\n'
  '  classDef pend fill:#fef3c7,stroke:#92400e;\n  classDef skip fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray: 4 3;\n'
  '  classDef part fill:#dcfce7,stroke:#166534;\n'
  f'  class E2,E3,E5,E6,E7,E8,E10 stop;\n  class {"E1," if not E1_FINAL else ""}E9 pend;\n  class E4 skip;\n'
  + ('  class E1 part;\n' if E1_FINAL else '') + '```\n')
w('*Red: negative or unsupported outcome under the reported rules (E3: FAIL under the post-hoc A10 rule, UNRESOLVED under the registered rule; E2: H withdrawn, L descriptive only). Green: partly (E1: $S\\ge2$ in a few cells, under conditions). Amber: pending or descriptive only. Dashed: not run.*\n')

w('## 1. In one paragraph\n')
w(f'{"Problem 1 is only partly addressed, and Problems 2 and 3 are not fixed." if E1_FINAL and e1_rows else "None of the three weaknesses is fixed by the experiments that have finished."} **Problem 1** (spectral FOM faster): '
  f'without a fast transform (E2) the bank is still slower than the strongest tested competitor on both geometries '
  f'(largest validation $S$ = {f3(e2_Smax_all)}, bar 2), and on a high-contrast variable coefficient (E3) the fastest '
  f'tuned FOM already costs only {f3(e3["0.03"]["G"])}–{f3(e3["0.1"]["G"])} of our box ROM (bar: ≥ 2× slower), so no '
  f'ROM was trained; E4 was not run; {"E1 is pending" if not E1_FINAL else ("on the box (E1) a reduced model beats the dense spectral FOM by $S\\ge2$ only in a few cells: our bank only with the ETDRK4 step on the narrow, low-steepness family, where POD arms do as well, and on the wide family only the translation model (our deployed LSPG bank stays at " + f3(e1_lspg_min) + "–" + f3(e1_lspg_max) + ")" if e1_rows else "E1 finds no $S\\ge2$")}. **Problem 2** (POD + '
  f'ECSW dominates): a precontracted Galerkin ETDRK4 step makes the $r=384$ evolution {f3(e5_e4_min)}–'
  f'{f3(e5_e4_max)}× faster than LSPG CN-R but its worst error is slightly higher, so E5 fails its rule; neither '
  f'bank tests nor fitted weights make the nonlinear term cheaper (E6, best fitted-weight gain {xr(e6_f3_best_e2e)}, '
  f'none Q2-qualified); and a continuous POD built once matches us on {e7_vc["worst"]["MATCH"]}/{e7_n} multi-mesh '
  f'workload pairs at {f3(e7_rmin)}–{f3(e7_rmax)} of our cost (E7), so the comparative multi-mesh cost claim is '
  f'dropped. **Problem 3** (steep fronts): neither the coverage retrain (FC1 median {xr(b_ratio)} lower, bar 2×) nor '
  f'the boundary-compatible bank nor a finer time step (E10-s) qualifies a single FC1 case; the binding failure is '
  f'time-step certification on steep states, not quadrature (E10-q).\n')

# ---------------------------------------------------------------- Problem 1
w(f'## 2. Problem 1 — a coarse spectral solver is faster at equal accuracy · verdict: **{P_VERDICT[1]}**\n')
w('### E1 — box: family width × steepness (imp-box)\n')
if not E1_FINAL:
    w(f'**Pending.** The lane is still running (LANES.json status `{lane_status.get("imp-box")}`; its EXPORT.json is '
      f'stage `{e1_stage}`). No E1 number is reported here until the lane hands back a final export. '
      f'GPU-hours to date: {gh(e1_gpu)} of {lane_cap["imp-box"]} (interim, ledger read at compile time). '
      'Interim deviations recorded by the coordinator (INTERVENTIONS.md, 2026-10-22 ~16:40 UTC): the lane uses '
      'spectral conditional references (DESIGN A2.2); a dense-matrix FOM of the same discretisation was added as the '
      'fair competitor (A11); H200 FFT runs at $N=12$ returned non-finite values and are excluded (A10).\n')
else:
    w('Burgers 2D on the unit box; family width $W$ (0.08 narrow … 0.40 wide; the stop rule judges the wide family, '
      '$W\\ge0.28$) × steepness level A1–A4 (A4 steepest). Arms: the tuned sine pseudo-spectral FOM (fixed or adaptive '
      'ETDRK4, FFT or dense transform; every $S$ is against the faster, which was the dense transform, A11), our bank '
      'with the deployed LSPG stepper (B-LSPG) and with the Galerkin ETDRK4 step from E5 (B-E4), POD-Galerkin (R-PODG, '
      'the pilot stand-in), POD + ECSW (R-ECSW) and a translation model (R-TRANS). $S$ = FOM time / reduced-model time '
      'at matched median or worst error $\\tau\\in\\{1,3,10\\}\\,\\%$, val32 selection. **Provisional throughout:** '
      'errors are against conditional spectral references (lane deviation A2.2, accepted by the coordinator); every '
      '$S$ is an upper bound against the tested FOM ladder; ladders were not extended below their minima, and '
      'entries sitting at a minimum are labelled (accepted by the coordinator).\n')
    collected = [c for c, st in e1_scope.items() if st == 'collected']
    deferred = [f'{c} ({st})' for c, st in e1_scope.items() if st != 'collected']
    w(f'Cells collected: {", ".join(collected)}. Deferred (coordinator ruling, cap): {", ".join(deferred)}.\n')
    w(f'Stop rules: SPEED-STOP **{e1_sr["speed"]}** (largest matched wide-family $S$ = {f3(e1_maxS)} over '
      f'{e1_sr["n_match"]} matched comparisons), scoped reading {e1_sr["scoped"]}, BANK-STOP {e1_sr["bank"]}.\n')
    w('Entries with $S\\ge2$ and a complete FOM search (validation; test-once $S$ of the same frozen pair):\n')
    w('| cell | arm | target | $S$ val | $S$ test | FOM finalist | labels |')
    w('|---|---|---|---|---|---|---|')
    for c, k, S, st, fk, lab, fs in e1_rows:
        a, t, stt = k.split('|')
        w(f'| {c} | {ARMN[a]} | {pct(float(t), 0)} {stt} | {f3(S)} | {f3(st) if st is not None else "not matched"} | '
          f'`{esc(fk)}` | {"; ".join(lab) or "none"} |')
    w('')
    if e1_inc:
        w('Larger values with an **incomplete** adaptive-FOM search (a faster FOM may exist, so these are weaker upper '
          'bounds): ' + '; '.join(f'{c} {ARMN[k.split("|")[0]]} {esc(k.split("|", 1)[1])}: {f3(S)}'
                                 + (f' (test {f3(st)})' if st is not None else '') for c, k, S, st, fk, lab, fs in e1_inc)
          + '.\n')
    bank2 = [r for r in e1_rows if r[1].startswith('bank')]
    w('Reading: ' + (f'the bank reaches $S\\ge2$ only with the ETDRK4 step and only in '
                     f'{", ".join(sorted({r[0] for r in bank2}))} (narrow family $W=0.08$, low steepness), at one target '
                     f'({", ".join(sorted({esc(r[1].split("|", 1)[1]) for r in bank2}))}), where the POD arms do as well '
                     f'(the same cell and target: ' + ', '.join(f'{ARMN[r[1].split("|")[0]]} {f3(r[2])}'
                     for r in e1_rows if r[0] in {b[0] for b in bank2} and r[1].split("|", 1)[1] in
                     {b[1].split("|", 1)[1] for b in bank2}) + '). ' if bank2 else 'the bank never reaches $S\\ge2$. ')
      + f'On the wide family the bank with ETDRK4 reaches at most $S$ = {f3(e1_e4w_max)}. The bank with its deployed '
      f'LSPG stepper stays at $S$ = {f3(e1_lspg_min)}–{f3(e1_lspg_max)} over its {e1_lspg_n} matched entries. '
      'The only wide-family arm at $S\\ge2$ is ' + (', '.join(sorted({ARMN[r[1].split("|")[0]] for r in e1_rows
                                                          if int(r[0][1:3]) >= 28})) or 'none')
      + f', which is not our method. {e1_lm_n} of the {len(e1_rows)} complete-search $S\\ge2$ entries carry the '
      'reduced-model ladder-minimum label (a smaller, possibly faster reduced configuration was not tried). '
      '"Complete" means both the fixed-step and the adaptive FOM searches returned a finalist; within each '
      'six-candidate screening batch the ordering is recorded only for the attempted candidates, so superiority over '
      'unattempted batch members cannot be independently reconstructed (lane search limitation). Timing caveat (A16): '
      f'some cells were timed with constants hoisted as arguments and others embedded in the compiled program; the '
      f'dense-FOM hoisted/embedded time ratio is {f3(e1_a16_min)}–{f3(e1_a16_max)}, so comparisons of $S$ across cells '
      'timed differently (including the width plots) carry this uncertainty; within-cell comparisons do not. '
      f'GPU-hours {gh(e1_gpu)} of {lane_cap["imp-box"]}.\n')
    w(img('imp-box', 'S_vs_W_median.png', 'S against family width, median-error targets'))
    w(img('imp-box', 'S_vs_W_worst.png', 'S against family width, worst-error targets'))

w('### E2 — geometry without a fast transform (imp-geometry)\n')
w('Burgers 2D on an L-shape (L) and a box with a circular hole (H), wide family, steepness level S1 only. '
  'Competitors: spectral elements (SEM), finite differences with multigrid (FD+MG), FD-RK4, POD-Galerkin and POD + ECSW. '
  'Win rule: $S\\ge2$ against the best competitor on the wide family; otherwise withdraw the geometry claim.\n')
w('| cell (conditional references; upper bounds) | $S$ validation (min–max) | $S$ test (min–max; empty cells) | vs SEM alone, worst targets | vs FD+MG, worst, τ = 3 % / 10 % | best competitor | decision labels (τ = 1 / 3 / 10 %) |')
w('|---|---|---|---|---|---|---|')
for c in e2:
    d = e2[c]
    w(f'| {esc(c)} | {f3(d["sv_min"])}–{f3(d["sv_max"])} | {f3(d["st_min"])}–{f3(d["st_max"])} ({d["st_null"]} empty) | '
      f'{f3(d["sem_min"])}–{f3(d["sem_max"])} | {f3(d["fd3"])} / {f3(d["fd10"])} | {", ".join(d["bc"])} | '
      f'{" / ".join(d["labels"][t] for t in ("0.01", "0.03", "0.1"))} |')
w('')
w(f'Outcome (registered A10 readout): **{e2_A10}**. Geometry H: **{e2_glab["H"]}**. Geometry L: {e2_glab["L"]} — '
  f'{e2["L|wide|S1"]["inel"]} of {e2["L|wide|S1"]["nref"]} validation references fail the Richardson order gate, so '
  'L is descriptive only, although every measured $S$ is below 2. $S$ is an upper bound against the strongest tested '
  'competitor (finite ladders; static condensation and multi-domain Chebyshev were not built). Against FD+MG alone the bank is faster only at the '
  f'looser targets ($S>1$ at: L {", ".join(e2["L|wide|S1"]["fd_gt1"])}; H {", ".join(e2["H|wide|S1"]["fd_gt1"])}; '
  'target|statistic). Competitors are admitted on validity and accuracy only; their Q1 is reported, not gated '
  '(DESIGN A4.2), which can only lower $S$. Narrow family, S2 and the box controls were not run. GPU-hours '
  f'{gh(e2_gpu)} of {lane_cap["imp-geometry"]}.\n')
w(img('imp-geometry', 'S_per_family.png', 'speed ratio S of the bank against each competitor family, per target'))

w('### E3 — variable coefficient (imp-varcoef)\n')
w(f'FOM-only benchmark at contrast $C={e3_contrast:g}$; a ROM is trained only if the fastest tuned FOM is ≥ 2× '
  'slower than our (constant-coefficient) box ROM. $G$ = fastest FOM cost / ROM cost at matched worst error.\n')
w('| worst-error target | ROM (constant κ), worst error (conditional, dev+val) | fastest FOM | $G$ [95 % CI] | outcome |')
w('|---|---|---|---|---|')
w(f'| 1 % | – | – | – | {e3_t1} (no ROM qualified) |')
for t, lab in (('0.03', '3 %'), ('0.1', '10 %')):
    d = e3[t]
    w(f'| {lab} | {pct(d["romw"])} | `{esc(d["fom"])}` | {f3(d["G"])} [{f3(d["ci"][0])}, {f3(d["ci"][1])}] | {d["out"]} |')
w('')
w(f'Gate: **{e3_gate}** under the post-hoc A10 reference cross-check rule; **{e3_reg}** under the registered A5.5 rule '
  '(references not decidable). Both readings mean no ROM is trained, so the conclusion does not depend on the rule. '
  'Provisional: the ROM denominator is the inherited constant-κ configuration (an optimistic comparator; a variable-κ '
  'ROM is unmeasured), and $G$ is against the strongest tested FOM (Krylov exponentials, AMG and POD-ECSW not built), '
  f'so a stronger FOM only lowers it. The gate is a development/validation result; on the {e3_test_nel} eligible of {e3_test_n} test-once cases '
  f'({e3_test_nfail} references not accepted; their registered A8.3 fallback was not run) the frozen 10 % winner reaches {pct(e3_test10, 1)} worst error (above its target) and the 3 % winner {pct(e3_test3, 2)}. '
  'Deferred: $C=10$ timing and validation ladder, CCv at $L=512$; reused factorisations not built. '
  f'GPU-hours {gh(e3_gpu)} of {lane_cap["imp-varcoef"]}.\n')
w(img('imp-varcoef', 'gate.png', 'gate ratio G per target with bootstrap interval'))
w(img('imp-varcoef', 'fom_cost_vs_error.png', 'FOM cost against worst error, variable coefficient'))
w('### E4 — 3D cube with small spherical holes\n')
w('Not run: it was conditional on an E2 win, and E2 did not win. Its reserve was released unspent.\n')

# ---------------------------------------------------------------- Problem 2
w(f'## 3. Problem 2 — tuned POD + ECSW dominates · verdict: **{P_VERDICT[2]}**\n')
w('### E5 — a faster reduced time step (imp-faststep)\n')
w('Stop if any run is unstable or invalid, its worst error is above the incumbent LSPG CN-R (arm 1), or the $r=384$ '
  'evolution is < 3× faster. Errors: worst over dev6 + val32 (38 cases), conditional E0 estimates.\n')
w('| mesh | arm | worst error $r=384$, conditional E0 (arm 1) | cases better / worse than arm 1 | evolve speed-up | conditions fired | qualified |')
w('|---|---|---|---|---|---|---|')
AN = {'G-CN': 'Galerkin CN (precontracted)', 'G-E4a': 'Galerkin ETDRK4, Δt 0.0125', 'G-E4b': 'Galerkin ETDRK4, Δt 0.025',
      'G-R1': 'one-pass linearised Galerkin CN'}
for m in (1024, 4096):
    for a in AN:
        fd = e5[(m, a, 'fired')]
        w(f'| {m}² | {AN[a]} | {pct(e5[(m, a, "worst")])} ({pct(e5[(m, a, "arm1")])}) | {e5[(m, a, "better")]} / '
          f'{e5[(m, a, "worse")]} | {f3(e5[(m, a, "S")])}× | {", ".join(k for k, v in fd.items() if v) or "none"} | '
          f'{"yes" if e5[(m, a, "qual")] else "no"} |')
w('')
w(f'E5 fixed on the completed scope: 1024² {"yes" if e5[(1024, "fixed")] else "no"}, 4096² '
  f'{"yes" if e5[(4096, "fixed")] else "no"}. No run was unstable or invalid ({"none" if not e5_any_unstable else "some"} '
  'fired the stability condition); the Galerkin arms are more accurate on most cases, but on the hardest case the '
  'worst error exceeds arm 1, so the worst-error condition fires for every arm. The ETDRK4 speed-ups are '
  f'evolution-only against the incumbent. Frozen dev6 best-stepper selection: 1024² {e5_best1024}; 4096² '
  f'`{e5_best4096}`, reported as a failed frozen selection because its 38-case worst error exceeds arm 1. Against the '
  f'matched POD + ECSW with the same ETDRK4 step (1024², in-sample, upper bound), the competitor full query takes '
  f'{f3(e5_pod["G-E4a"])} (`{e5_pod_B["G-E4a"]}`) and {f3(e5_pod["G-E4b"])} (`{e5_pod_B["G-E4b"]}`) of our ETDRK4 '
  f'query time (< 1: the competitor is faster; both competitors Q1-qualified but Q2-POD label '
  f'{e5_podlab["G-E4a"]} / {e5_podlab["G-E4b"]}), and {f3(e5_pod["L-CNR"])} of the incumbent LSPG query time. '
  f'Deferred at 4096²: $r=128$ and the $M=1152$ control (cap). GPU-hours {gh(e5_gpu)} of {lane_cap["imp-faststep"]}.\n')
w(img('imp-faststep', 'worst_error_vs_evolve_time.png', 'worst error against median evolution time per arm and rank'))

w('### E6 — a cheaper nonlinear term (imp-hrcost)\n')
w('F2: bank-tested Galerkin with classical rules (stop if $m^*>16384$, accuracy worse, or not faster). F3: sine-tested '
  'Galerkin with fitted ECSW-style weights (stop if no $\\kappa\\le16$ passes Q2 at equal worst error with ≥ 1.5× '
  'gain). Speed-ups are paired, same-job, end-to-end ratios against the lane reference arm R (an internal ablation, '
  'in-sample, selection-unadjusted).\n')
w('| arm | nodes $m$ | worst error (conditional E0) | end-to-end gain vs R (in-sample) | Q2 |')
w('|---|---|---|---|---|')
w(f'| R (sine tests, Gauss 128²) | {e6_R_m} | {pct(e6_R_worst, 3)} | 1 (reference) | pass |')
w(f'| F2, $m^*$: {e6_mstar} | {e6_F2_m} | {pct(e6_F2_worst, 3)} | {xr(e6_F2_e2e)} | pass |')
for a in f3arms:
    w(f'| {a} | {R.get(f"e6.{a}.m", HR, "arms", a, "m")} | {pct(e6_f3[a]["worst"], 3)} | {xr(e6_f3[a]["e2e"])} | '
      f'{e6_f3[a]["q2"]} |')
w(f'| F2F3_k16 (fastest qualified) | {R.get("e6.F2F3_k16.m", HR, "arms", "F2F3_k16", "m")} | {pct(e6_ff_w, 3)} | '
  f'{xr(e6_ff)} | {"pass" if e6_ff_q else "fail"} |')
w('')
w(f'F2: `{e6_F2}`. F3: **{e6_F3}** ({e6_f3_q2pass} of {len(f3arms)} fitted-weight arms pass Q2; best gain '
  f'{xr(e6_f3_best_e2e)}, {e6_f3_best}). Point work scales sub-linearly: the step-Jacobian microbenchmark costs '
  f'{xr(e6_jd)} for 2× the points and {xr(e6_jq)} for 4×, so fewer points buy little. GPU-hours {gh(e6_gpu)} of '
  f'{lane_cap["imp-hrcost"]}.\n')
w(img('imp-hrcost', 'error-vs-speedup.png', 'worst error against end-to-end gain per arm'))
w(img('imp-hrcost', 'f2-ladder-q2.png', 'F2 classical-rule ladder and its Q2 distances'))

w('### E7 — multi-mesh full cost (imp-multimesh)\n')
w('21 workloads (every non-empty subset of {256², 1024², 4096²} × 10/100/1000 queries per mesh) × 6 qualified '
  'configurations of ours, against per-mesh POD + ECSW and the fair control: a continuous POD built once and reused. '
  'Rule: if the control matches us (no more cost at equal or better worst error) on every workload, drop the '
  'comparative multi-mesh cost claim.\n')
w('| accuracy criterion (A_dep accounting) | MATCH | NO-MATCH | of which no admissible competitor |')
w('|---|---|---|---|')
for c in ('worst', 'worst_eps', 'worst_median', 'casewise', 'rollout'):
    na = R.get(f'e7.noadm.{c}', MM, 'nomatch_without_admissible', f'{c}|A_dep')
    w(f'| {c}{" (primary)" if c == "worst" else ""} | {e7_vc[c]["MATCH"]} | {e7_vc[c]["NO-MATCH"]} | {na} |')
w('')
w(f'Primary outcome: **{e7_out}** — the continuous POD matches us on {e7_vc["worst"]["MATCH"]} of {e7_n} '
  f'(workload, configuration) pairs at {f3(e7_rmin)}–{f3(e7_rmax)} of our cost (median {f3(e7_rmed)}); the same '
  f'under the A100 ({e7_vc100["MATCH"]}) and no-head ({e7_vcnh["MATCH"]}) accountings. The casewise and rollout '
  'sensitivities do not fire, but every one of their NO-MATCH cells has no accuracy-admissible competitor at all '
  '(cost advantage undetermined), not a cost win. Costs are the measured items only (bank rotation and rule selection '
  'uncosted, which favours us); our bank-fit time is a phase-wise projection from measured H200 rates, and every cost '
  'verdict is conditional on compilation projections; developmental, in-sample. GPU-hours '
  f'{gh(e7_gpu)} of {lane_cap["imp-multimesh"]}.\n')
w(img('imp-multimesh', 'cost_vs_queries.png', 'full cost against number of queries, ours vs continuous POD'))

# ---------------------------------------------------------------- Problem 3
w(f'## 4. Problem 3 — fails on fronts steeper than training · verdict: **{P_VERDICT[3]}**\n')
w('### E8 — retrain a better bank (imp-fronts)\n')
w(f'Phase A (CPU diagnostics, 0 GPU-h): A1 smearing test **{e8A["A1"]}**; A2 wall curvature → arm (c) '
  f'**{e8A["A2"]}**; A3 truncation screen **{e8A["A3"]}**; A4 hybrid floors **{e8A["A4"]}**.\n')
w('Phase B: single-change retrains against the control retrain K1 of the incumbent recipe; arm (a) data fidelity '
  'was deferred (both retrain arms would exceed the 15 GPU-h cap), arm (b) and then arm (c) were released.\n')
w('| arm | registered bar | measured (diagnostic errors: conditional, qualification ignored) | in-range guard (worst representation floor vs K1, bar ≤ 2×) | verdict |')
w('|---|---|---|---|---|')
w(f'| (a) data fidelity | N0s floor ≥ 3× below K1 | not run | – | {e8v["a"]} |')
w(f'| (b) steep / low-ν coverage | FC1 ROM median ≥ 2× lower | {xr(b_ratio)} lower ({pct(b_arm)} vs K1 {pct(b_ctl)}); '
  f'{b_q}/{b_n} FC1 cases numerically qualified | {b_guard} ({xr(b_gr)}) | {e8v["b"]} |')
w(f'| (c) boundary-compatible bank | stop if $m^*$ is not ≥ 2× smaller **and** the tail is not ≥ 3× smaller (either improvement continues) | $m^*$ = {c_rule} = {c_nodes:.0f} '
  f'nodes (bar ≤ {c_bar:.0f}); tail {xr(c_tr)} smaller; FC1 median {pct(c_fc1)} | {c_guard} ({xr(c_gr)}) | '
  f'{e8v["c"]} |')
w('')
w(f'GPU-hours (E8 bucket) {gh(e8_gpu)} of 15.\n')
w(img('imp-fronts', 'phaseb_medians.png', 'Phase B diagnostic medians per cohort, arms vs control K1'))
w(img('imp-fronts', 'a4_hybrid_floors.png', 'A4 hybrid (sine-enriched) floors'))
w(img('imp-fronts', 'a1_smearing.png', 'A1 smearing test'))
w(img('imp-fronts', 'a3_truncation.png', 'A3 truncation screen'))
w('### E9 — out-of-regime indicator (folded into E8)\n')
w(f'Flag queries with $s=a/\\nu>\\tau_s$, $\\tau_s$ = {f3(e9_tau)} (frozen before test). On {e9_n} test cases it '
  f'flags {pct(e9_flag, 1)} (in-range FI cohort {pct(e9_fi, 1)}). Target A (bad = conditional error above '
  f'{pct(e9_t["A"]["threshold"], 3)}): AUC {f3(e9_t["A"]["auc"])}, detection '
  f'{f3(e9_t["A"]["detection"])}, false alarms {f3(e9_t["A"]["false_alarm"])} ({e9_t["A"]["n_bad"]} bad / '
  f'{e9_t["A"]["n_good"]} good); target B (bad = error above {pct(e9_t["B"]["threshold"], 0)}): AUC {f3(e9_t["B"]["auc"])}, false alarms '
  f'{f3(e9_t["B"]["false_alarm"])}. Label: **{e9_t["A"]["label"]}** — descriptive only. $s$ is a known input on this '
  'family, so the indicator is a safety feature, not a speed one; the FOM-fallback cost is unmeasured.\n')
w('### E10 — numerics near fronts (activated because E8 fell short)\n')
w(f'- **E10-s** (time step refined to 1/8, single change on K1): FC1 median {pct(s10["fc1_median"])} (diagnostic, conditional E0-style error; qualification ignored for the medians) against the bar '
  f'{pct(s10["bar"])} (half of K1\'s {pct(s10["k1_fc1_diagnostic_median"])}); {e10s_q}/{s10["n_fc1"]} FC1 cases '
  f'qualified, {s10["fc1_step_converged"]} step-converged → **{s10["outcome"]}**.\n'
  f'- **E10-q** (state-triggered quadrature escalation): its trigger never fires — $\\max\\rho$ = '
  f'{f3(q10["evidence"]["rho_max_all"])} over {q10["evidence"]["accepted_step_states"]} accepted step states '
  f'({q10["evidence"]["states_over_bar"]} over the bar) → **{q10["outcome"]}** without a job: quadrature is not the '
  'binding failure.\n')
w(f'E10 reused the Phase B test cases (not test-once for E10). GPU-hours (E10 bucket) {gh(e10_gpu)} of 10.\n')

# ---------------------------------------------------------------- table
w('## 5. Every experiment and its stop-rule outcome\n')
w('| exp | problem | lane | registered stop / win rule | outcome | fixed? |')
w('|---|---|---|---|---|---|')
w(f'| E1 | 1 | imp-box | stop if every reduced model has $S<0.5$ at every target on the wide family | '
  f'{"pending (running)" if not E1_FINAL else e1_sr["speed"]} | {v_e1()} |')
w(f'| E2 | 1 | imp-geometry | win: $S\\ge2$ vs best competitor on the wide family, else withdraw | {e2_A10}; H '
  f'{e2_glab["H"]}; L not decidable | no |')
w(f'| E3 | 1 | imp-varcoef | train a ROM only if the fastest FOM is ≥ 2× slower | gate {e3_gate} (A10 rule) / '
  f'{e3_reg} (registered); $G$ = {f3(e3["0.03"]["G"])}, {f3(e3["0.1"]["G"])} | no |')
w('| E4 | 1 | – | only if E2 wins | not run | – |')
w(f'| E5 | 2 | imp-faststep | stop if unstable/invalid, worst error above arm 1, or < 3× faster | '
  f'{"fired (worst error) for every arm" if e5_all_worst else "see E5 section"} | {"yes" if e5_fixed else "no"} |')
w(f'| E6 | 2 | imp-hrcost | F2: $m^*\\le16384$, no worse, faster; F3: κ ≤ 16 Q2-qualified with ≥ 1.5× | F2 '
  f'{e6_F2.split(":")[0]}; F3 {e6_F3} | no |')
w(f'| E7 | 2 | imp-multimesh | drop the claim if continuous POD matches us on every workload | {e7_out} '
  f'({e7_vc["worst"]["MATCH"]}/{e7_n}) | no (claim dropped) |')
w(f'| E8 | 3 | imp-fronts | (a) N0s ≥ 3×; (b) C1 median ≥ 2×; (c) stop if neither $m^*$ ≥ 2× nor tail ≥ 3×; floor guard ≤ 2× | (a) '
  f'{e8v["a"]}; (b) {e8v["b"]}; (c) {e8v["c"]} | no |')
w(f'| E9 | 3 | imp-fronts | report coverage and false alarms (no stop rule) | {e9_t["A"]["label"]} | descriptive |')
w(f'| E10 | 3 | imp-fronts | only if E8 falls short; single changes vs K1 | E10-s {s10["outcome"]}; E10-q '
  f'{q10["outcome"]} | no |')
w('')

# ---------------------------------------------------------------- paper claims
w('## 6. What this means for the paper\'s claims\n')
w('**Withdrawn or not made:**\n')
w('- Geometry without a fast transform (E2): withdrawn for the holed box (measured loss); not supported on the '
  'L-shape (S1 only, references not decidable). The 3D holes case (E4) is not claimed.')
w('- Variable coefficient (E3): no ROM advantage is claimed; the tuned FOM is already about as cheap as the box ROM.')
w('- Comparative multi-mesh cost (E7): dropped. Mesh-independent evolution and mesh-free evaluation remain stated as '
  'properties of the method, not as comparative advantages over a continuous POD.')
w('- "A faster step closes the gap to POD + ECSW" (E5) and "a cheaper nonlinear term" (E6): not claimed; the '
  'deployed stepper remains LSPG CN-R with the frozen quadrature.')
w('- Robustness to fronts steeper than training (E8, E10): not claimed; it stays a stated limitation, with $s=a/\\nu$ '
  'as a descriptive out-of-regime flag (E9) and FOM fallback.')
w('\n**Kept (unchanged by this campaign):** the properties established before it — mesh-independent evolution cost and '
  'mesh-free evaluation (E7 keeps them as properties); the incumbent configuration and its accuracy as previously '
  'reported (E5 arm 1 is the incumbent LSPG CN-R; E6 reproduces E5\'s precontracted Galerkin CN arm as its reference R). The finding that tuned POD + ECSW dominates our solve cost '
  'stands and should be reported, with the E5 Galerkin ETDRK4 evolution speed-up as a mechanism result that does not '
  'change the comparison.')
if not E1_FINAL:
    w('\n**Open:** the box regime map (E1) is pending; the spectral-FOM claim is undecided until it hands back.\n')
else:
    w('\n**Problem 1 on the box (E1), narrowed:** the paper may say that on the box a reduced model beats the tuned '
      'dense spectral FOM by $S\\ge2$ only in the cells and targets listed in §2 (upper bounds, conditional '
      'references, ladder-minimum labels); that our bank does so only with the Galerkin ETDRK4 step, only on the '
      'narrow, low-steepness family, where POD-Galerkin and POD + ECSW do as well; and that the deployed LSPG bank '
      f'stays at $S$ = {f3(e1_lspg_min)}–{f3(e1_lspg_max)}. A general "faster than a spectral solver" claim is not '
      'supported.\n')

# ---------------------------------------------------------------- retractions/deviations
w('## 7. Retractions and deviations\n')
w('**Procedure deviations (disclosed):**\n')
w('- **E6 smoke submitted against a DO-NOT-SUBMIT job audit** (`imp-hrcost/audits/codex-job-smk1-1.md`); the lane '
  'argued the computation was unaffected (`audits/response-job-smk1-1.md`) and fixed the offline-acceptance gaps '
  'before acceptance; accepted by the coordinator as a smoke-only deviation.')
w(f'- **E8 SM-c admission breach** (lane ledger note, verbatim): `{smc["note"]}` The results audit marks it WRONG as a procedure breach.')
w('- **E3 post-hoc A10 cross-check rule:** adopted after the reference data but before any solver result; carried as a '
  f'post-hoc label. Both outcomes are reported ({e3_gate} / {e3_reg}); neither trains a ROM.')
w('- **E2 geometry chosen after inspecting smoke rows:** the first geometry showed no bump–feature interaction and was '
  'redesigned (v4a); the claim that no solver or ROM result existed at selection time was retracted (A3.1): the '
  'redesign was diagnostic-guided problem development, not outcome-blind.')
w(f'- **E2 smoke overrun:** the 1 GPU-h smoke allowance was exceeded by {gh(e2_over)} GPU-h (smoke5 ran to its limit '
  'during a usage-limit pause).')
w('- **E2 scope reduced:** the first L cell timed out in the ladder stage; registered reductions (A9) were applied and '
  'the cell resumed; the narrow family, S2 and box controls were deferred under the 62 GPU-h cap.')
w('- **E7 / E5 shared-scratchpad collision:** imp-multimesh overwrote a shared prompt file, so E5 design audit 3 ran '
  'with a partly foreign prompt; it audited the right design (DESIGN note 2026-10-22) and audits 4–5 re-covered it.')
w('- **E5 pkill incident:** a `pkill -f codex-design-4.md` may have terminated another lane\'s Codex audit '
  '(`imp-faststep/audits/NOTE-pkill-incident.txt`); dead audits were re-run by rule; lanes use private work dirs since.')
if E1_FINAL:
    w('- **E1 spectral conditional references** (lane deviation A2.2 from the E0 second-order recipe): accepted by the '
      'coordinator as a labelled deviation. **No downward ladder extension:** FOM and reduced ladders were not '
      'extended below their minima; entries at a minimum are labelled as truncated (coordinator ruling). A '
      'dense-transform FOM of the same discretisation was added as the fair competitor (A11); H200 FFT runs at $N=12$ '
      'gave non-finite values and were excluded (A10). Query timing changed mid-lane from embedded to hoisted constants (A16); cross-cell $S$ comparisons carry the measured ratio (§2).')
w('- **E8 released early:** Phase B retrains started before the other wave-1 lanes handed back (independent, capped).')
w('- **E10 reused Phase B test cases** (not test-once for E10).')
w('- **Interruptions:** two account usage-limit stops halted every lane agent (2026-10-21 ~19:30 UTC and 2026-10-22 '
  '~01:00 UTC; resumed after reset); GitHub credentials expired 2026-10-21 ~23:40 UTC, so mirrors were not pushed '
  'until the user restored them (2026-10-22 ~19:45 UTC), when every lane was re-synced. Self-matching `pgrep` waiters '
  'and a dead detached waiter were killed or replaced (INTERVENTIONS.md).')
w('\n**Retractions and corrections inside lanes** (most before any verdict; the E8 wall-residual finding and the E7 resampling correction came from results audits):\n')
w('- E6: an unattainable mass/stiffness bar replaced before verdict data (A5); a projected-ρ gate justification '
  'withdrawn and the tested-vector gate restored (A2.1); defective P5 fixtures replaced (A6); a timer control that did '
  'not fire at real size replaced by C5′ (A9).')
w('- E8: the original A2 curvature control was retracted (it fired on only part of the in-range cases where it had to fire); draft A3/A4 debugging '
  'outputs were seen and discarded before Amendment 1; the bank-values check was respecified (Amendments 6–7).')
w(f'- E8 arm (c) identity: on the trained columns the wall residual reaches {f3(c_wall)} of the column RMS against the registered bar {f3(c_wall_bar)} (round-off of $\\sin(\\pi x)$ at $x=1$; the bar was tested on the initial bank; scoring zeroes wall nodes).')
w('- E6: F2 changes both the test space and the diffusion discretisation, so it is not a pure quadrature change.')
w('- E7: the single-rate bank-fit cost model was corrected to a phase-wise projection, and resampling-cost accounting was corrected.')
w('- E5: design audit 2 audited the wrong lane and was voided; F5 secondary criteria rejected arm 4.')
w('- E3: none within the lane; some timed subjects did not reproduce their ladder error (G-FR False) and are excluded '
  'from the gate (none is a decisive winner).')
w('\n**Provisional labels carried in this report:** conditional E0 references (and E1\'s spectral conditional '
  'references); in-sample, unadjusted POD comparisons; upper-bound speed-ups against finite ladders; E7 conditional on '
  'compilation projections; E3 ROM denominator is constant-κ.\n')

# ---------------------------------------------------------------- deferred
w('## 8. Deferred items\n')
if E1_FINAL:
    w('- E1: cells ' + ', '.join(f'{c} ({st})' for c, st in e1_scope.items() if st != 'collected')
      + ' (coordinator ruling: the A4 cells, W16_A2/A3 and W08_A3 deferred under the cap); the second training-data '
      'size; no downward ladder extension below the ladder minima.')
else:
    w('- E1: all of it until the lane hands back (running).')
w('- E2: narrow family, S2 steepness, box attribution controls; static condensation and multi-domain Chebyshev '
  'competitors.')
w('- E3: a variable-κ ROM (never trained: gate failed); Krylov/Leja exponentials, reused factorisations, AMG and POD-ECSW FOM competitors; $C=10$ timing and validation ladder; CCv at $L=512$; the A8.3 test-reference fallback.')
w('- E4: not run (conditional on E2).')
w('- E5: $r=128$ and the $M=1152$ control at 4096² (cap).')
w('- E7: bank rotation and quadrature-rule selection costs (uncosted, favour us).')
for d in fr_def:
    w(f'- E8/E9/E10: {d["item"]} — {d["reason"]}.')
w('- Problem 3 next step (not a single change, so not released): an embedded adaptive integrator or a shift/local-basis '
  'bank — open for the paper or a later campaign.\n')

# ---------------------------------------------------------------- ledger
w('## 9. GPU-hour ledger\n')
w('| lane | experiments | cap (GPU-h) | spent (GPU-h) | status |')
w('|---|---|---|---|---|')
spent = {'imp-faststep': e5_gpu, 'imp-hrcost': e6_gpu, 'imp-multimesh': e7_gpu, 'imp-varcoef': e3_gpu,
         'imp-fronts': e8_gpu_tot, 'imp-geometry': e2_gpu, 'imp-box': e1_gpu}
for l, _, ex, _ in LANES:
    st = 'final' if l != 'imp-box' or E1_FINAL else 'running (interim)'
    extra = f' (E8 {gh(e8_gpu)} of 15, E10 {gh(e10_gpu)} of 10)' if l == 'imp-fronts' else ''
    w(f'| {l} | {ex} | {lane_cap[l]} | {gh(spent[l])}{extra} | {st} |')
done_total = R.derive('ledger.total_done', sum(v for k, v in spent.items() if k != 'imp-box'),
                      ['e5.gpu', 'e6.gpu', 'e7.gpu', 'e3.gpu', 'e8.gpu_total', 'e2.gpu'], 'sum')
all_total = R.derive('ledger.total_all', done_total + (e1_gpu or 0), ['ledger.total_done', 'e1.gpu'], 'sum')
cap_total = R.derive('ledger.cap_total', sum(lane_cap[l] for l in spent), [rel(LANESJ) + ':lanes[*].gpu_hours_cap'],
                     'sum')
w(f'| **total** | | {cap_total} | {gh(all_total)}'
  + (f' ({gh(done_total)} in finished lanes; E1 interim)' if not E1_FINAL else '') + ' | |')
w('\nE4 (cap 80) was not run and its reserve was not spent. Sources: each lane\'s `runs/GPU-LEDGER.json`.\n')

# ---------------------------------------------------------------- lanes
w('## 10. Lanes, branches and mirrors\n')
w('| lane | branch | last local commit | GitHub mirror (`origin/codeonly/<branch>`) | status |')
w('|---|---|---|---|---|')
for l, _, ex, _ in LANES:
    bh = branch_head[l]
    w(f'| {l} ({ex}) | `{LBRANCH[l]}` | `{bh["commit"][:9]}` ({bh["date"][:16]} {bh["date"][19:]}) | '
      f'`{mirror.get(l, "missing")[:9]}` | {lane_status.get(l)} |')
w('\nMirror heads were read with `git ls-remote origin` when this report was compiled (an auditor without network access can corroborate them against the local `refs/remotes/origin/codeonly/...` refs). Mirror commits are code-only filtered copies, so their hashes differ from the local commits by design. '
  '**Branches awaiting the user\'s merge decision:** all seven lane branches above (no branch was merged in this '
  f'campaign){"; imp-box only once it finishes" if not E1_FINAL else ""}.\n')

# ---------------------------------------------------------------- glossary
w('## 11. Glossary\n')
G = [
    ('ROM / FOM', 'reduced-order model (our fast surrogate) / full-order model (the accurate, expensive solver).'),
    ('bank, $r$', 'the ordered set of learned coordinate-network basis functions; $r$ is how many are used (the number of unknowns solved for).'),
    ('off-mesh quadrature, $m$', 'the set of $m$ sample points where the nonlinear term is evaluated, independent of any mesh.'),
    ('POD, POD + ECSW, POD-Galerkin', 'the classical competitor: a basis from snapshots on a mesh (POD), with a small set of weighted mesh points for the nonlinear term (ECSW).'),
    ('continuous POD (CPR/CPU)', 'a POD basis built once and resampled to other meshes; the fair multi-mesh control in E7.'),
    ('LSPG CN-R', 'the deployed time stepper: least-squares residual minimisation with Crank–Nicolson and two backward-Euler start steps.'),
    ('Galerkin, precontracted, ETDRK4', 'projection-based stepping with matrices multiplied together offline; ETDRK4 is a 4th-order exponential integrator.'),
    ('arm', 'one configuration being compared; arm 1 / R / K1 are the reference arms of E5 / E6 / E8.'),
    ('$S$', 'speed-up: competitor time divided by ROM time at matched error; $S>1$ means we are faster. Always an upper bound against a finite tested ladder.'),
    ('$G$', 'E3 gate ratio: fastest FOM cost divided by ROM cost; $G\\ge2$ was needed to train a ROM.'),
    ('worst / median error', 'the largest / median, over the evaluated cohort (dev6 + val32 for E5–E7, the stated cohort elsewhere), of the relative field error.'),
    ('$C$, CCv', 'E3: contrast of the variable diffusion coefficient; CCv = the E0 scheme (CN-R, Newton–BiCGStab) used as one FOM family.'),
    ('$\\tau$ (target)', 'an error level (1 %, 3 %, 10 %) at which speeds are matched.'),
    ('conditional E0 / Richardson $\\varepsilon_{\\rm ref}$', 'errors are measured against reference solutions whose own error is estimated, not bounded; numbers are conditional on that estimate.'),
    ('Q1 / Q2, qualified', 'numerical qualification (every run valid and step-certified) / quadrature qualification (a finer rule changes nothing beyond the bar).'),
    ('in-sample, unadjusted', 'measured on the cases used for selection, without correcting for having picked the best of several.'),
    ('dev6, val32, test-once', 'development, validation and once-only test case cohorts.'),
    ('MATCH / NO-MATCH', 'E7: the control is at least as accurate and no more costly / it is not.'),
    ('FI, FN0s, FC1, FC2', 'E8 test cohorts: in-range; the stiff corner inside the training range (low viscosity, high amplitude); and two steeper-than-training cohorts.'),
    ('$s=a/\\nu$', 'steepness: advection amplitude over viscosity; large $s$ means sharp fronts.'),
    ('AUC, false alarm', 'E9 indicator quality: area under the ROC curve; fraction of good queries flagged.'),
    ('$m^*$, tail', 'E8(c)/E6: smallest certified quadrature size; the high-frequency remainder of decoded fields.'),
    ('guard, representation floor', 'E8: an arm also stops if its in-range worst representation floor (best error the bank can represent, by projection) is more than 2× the control\'s.'),
    ('SEM, FD+MG, FD-RK4', 'spectral elements; finite differences with multigrid; finite differences with explicit Runge–Kutta.'),
    ('S1 / S2, wide / narrow', 'E2 steepness levels and parameter-family widths.'),
    ('A10, A3.1, A9 …', 'numbered design amendments in each lane\'s DESIGN.md.'),
    ('A_dep / A_100 / A_nohead', 'E7 cost accountings: H200-measured offline cost with projected bank fit / historical A100 bank fit and head / head excluded.'),
    ('worst, worst_eps, worst_median, casewise, rollout', 'E7 accuracy criteria: aggregate worst error no worse; the same within $2\\max\\varepsilon_{\\rm ref}$; worst and pooled median; every case within $2\\varepsilon_{\\rm ref}$; worst restricted to competitors whose $\\kappa=4$ and $\\kappa=8$ rollouts agree.'),
    ('CPR-$s$ / CPU-$s$', 'continuous POD from source mesh $s$ with the ECSW rule refitted on each mesh / with the source rule reused on finer nested meshes.'),
    ('$M$', 'number of test functions in LSPG (E5 control $M=1152$).'),
    ('$\\kappa$', 'overloaded: in E6/E7, nodes per unknown in a fitted (ECSW-style) rule, $m=\\kappa r$ (k4/k8/k16 in arm names); in E3, the diffusion coefficient (constant-κ vs variable-κ).'),
    ('F2, F3a / F3b, F2F3', 'E6 arms: classical rules with bank tests; fitted ECSW-style weights with sine tests, non-negative (NNLS) weights fitted on the Gauss 128² points to the projected (F3a) or the full tested (F3b) advection; fitted weights with bank tests.'),
    ('$\\rho$', 'E10-q quadrature indicator: relative change of the nonlinear term under a finer rule; escalation triggers above a registered bar.'),
    ('CI', '95 % bootstrap confidence interval.'),
    ('GPU-h, cap', 'GPU-hours charged by the cluster; the budget a lane may not exceed.'),
    ('$W$, A1–A4 (E1)', 'E1 family width (bump centres drawn in a square of side $W$; 0.08 narrow, 0.40 wide) and steepness levels (viscosity bands a factor 3 apart; A4 steepest); a cell is one (W, level) pair, e.g. W08_A2.'),
    ('B-LSPG / B-E4, R-PODG / R-ECSW / R-TRANS', 'E1 arms: our bank with the deployed LSPG stepper / with the Galerkin ETDRK4 step; POD-Galerkin; POD + ECSW; static translation model (POD of centred bumps decoded at the true centre).'),
    ('dense transform, FOM finalist', 'E1: the spectral FOM implemented with a dense transform matrix instead of an FFT (faster here); the fastest qualified FOM configuration (`spec|N…|dense|nt…`: modes, transform, time steps).'),
    ('ladder minimum, search-incomplete', 'the selected configuration is the smallest one tried, so a cheaper one may exist; the adaptive-FOM search hit its attempt cap before finishing.'),
    ('mirror (codeonly)', 'the filtered GitHub backup branch of a lane, without large archive files.'),
]
for k, v in G:
    w(f'- **{k}**: {v}')
w('')

(HERE / 'report.md').write_text('\n'.join(L) + '\n')
R.prov['_figures'] = fig_prov
(HERE / 'PROVENANCE.json').write_text(json.dumps(R.prov, indent=1, default=str) + '\n')
FX = {'compiled_utc': now, 'E1_final': E1_FINAL, 'problem_verdicts': P_VERDICT,
      'experiments': {'E1': v_e1(), 'E2': e2_A10, 'E3': e3_gate, 'E4': 'not run', 'E5': 'fixed' if e5_fixed else ('stop fired (worst error, every arm)' if e5_all_worst else 'stop fired'),
                      'E6': {'F2': e6_F2, 'F3': e6_F3}, 'E7': e7_out, 'E8': e8v, 'E9': e9_t['A']['label'],
                      'E10': {'E10-s': s10['outcome'], 'E10-q': q10['outcome']}},
      'gpu_hours': {**spent, 'total': all_total}}
(HERE / 'FINAL-EXPORT.json').write_text(json.dumps(FX, indent=1, default=str) + '\n')
print('wrote report.md,', len(R.prov), 'provenance entries,', len(fig), 'figures; E1_FINAL =', E1_FINAL)
