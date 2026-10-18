"""Development manifest (DESIGN 2.1, AM-1 "item 0"), written and hashed BEFORE any lane job: runs/MANIFEST-dev.json
(+ .sha256). CPU only (JAX_PLATFORMS=cpu).

Records: bank checkpoint and rotation sha256; every (mesh, policy, M) ordered mode list with its hash and the set
difference against the frozen list; every rule used (name, m, sha256 of points and weights; Fibonacci with its frozen
seed-0 shift); schemes, dt, solver tolerances (prod / tight / tighter, LSPG and GAL), budget; initialisation contract;
test weighting; output times; E0 store version, metric id and per-case-kind eligibility (dev6, val32); cohort hashes;
config hashes; the code commit.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import p1common as P  # noqa: E402

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
sys.path.insert(0, str(E0))
import refstore as RS  # noqa: E402

RULES = ['fib4181', 'fib6765', 'fib10946', 'fib17711', 'gauss32', 'gauss64', 'gauss128', 'gauss160', 'gauss192', 'gauss256',
         'gauss320', 'gauss384']


def main():
    m = dict(revision=1, revision_note='jcp-qualify development manifest, written before any lane job (copy of P1 manifest.py, rules and output path adapted)',
             commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
             checkpoint_sha256=P.QS.sha_file(P.T2.CKPT), rotation_sha256=P.QS.sha_file(P.Q.ROTATION))
    ml = {}
    for L in (256, 1024, 4096):
        for pol in ('frozen', 'nominal'):
            for M in (128, 256, 512, 1024, 1536, 2048):
                kx, ky = P.mode_list(pol, L, M)
                ml[f'{L}|{pol}|{M}'] = dict(hash=P.list_hash(kx, ky), symdiff_vs_frozen=P.symdiff(L, M) if pol == 'nominal' else None)
    for k, v in ml.items():
        if v['symdiff_vs_frozen']:
            v['symdiff_vs_frozen'] = {a: [list(map(int, t)) for t in b] if isinstance(b, list) else b for a, b in v['symdiff_vs_frozen'].items()}
    m['mode_lists'] = ml
    kx, ky = P.mode_list('frozen', 1024, 2048)                 # A1.6: the explicit frozen list F_2048 (prefix M = 4r)
    mf = json.dumps(dict(source='modes_lean(1024, 2048)', kx=[int(x) for x in kx], ky=[int(y) for y in ky]))
    (HERE / 'inputs').mkdir(exist_ok=True)
    (HERE / 'inputs' / 'modes_F2048.json').write_text(mf)
    m['modes_F2048'] = dict(path='inputs/modes_F2048.json', sha256=hashlib.sha256(mf.encode()).hexdigest(), hash=P.list_hash(kx, ky))
    m['rules'] = {}
    for r in RULES:
        X, w, h = P.rule_points(r)
        m['rules'][r] = dict(m=int(len(X)), sha256=h, weight_sum=float(np.sum(w)))
    m['code'] = dict(tol_of_prod_LSPG=list(P.tol_of('LSPG', 'prod')), tol_of_tight_LSPG=list(P.tol_of('LSPG', 'tight')),
                     tol_of_tighter_LSPG=list(P.tol_of('LSPG', 'tighter')), tol_of_tight_GAL=list(P.tol_of('GAL', 'tight')),
                     tol_of_tighter_GAL=list(P.tol_of('GAL', 'tighter')), budget=600, dt0=P.DT0,
                     schemes={k: list(v) for k, v in P.T2.SCHEMES.items() if k in ('BE', 'CNR', 'CN', 'TH06')}, tol_of_prod_GAL=list(P.tol_of('GAL', 'prod')),
                     p1core_sha256=P.QS.sha_file(HERE / 'p1core.py'), p1common_sha256=P.QS.sha_file(HERE / 'p1common.py'), qacc_sha256=P.QS.sha_file(HERE / 'qacc.py'), qcert_sha256=P.QS.sha_file(HERE / 'qcert.py'),
                     init_contract='full-field input sampled at the cold quadrature points; reduced projection solve; full-field decode',
                     weighting='LSPG D = 1/(a0 + dt b0 nu lam)', output_times=[0, .05, .1, .15, .2, .25])
    # A6.2: rule identity = identity of the rule-generating code (point hashes differ across CPU platforms at round-off)
    C = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/vendor/quad2d')
    gen = [('qcore.py', 'qcore.py'), ('vendor/hari_quadrature.py', 'vendor/hari_quadrature.py')]
    m['code']['rule_generator_sha256'] = {a: P.QS.sha_file(HERE / 'vendor' / 'quad2d' / a) for a, _ in gen}
    m['code']['rule_generator_identical_to_archive'] = all(P.QS.sha_file(HERE / 'vendor' / 'quad2d' / a) == P.QS.sha_file(C / b)
                                                           for a, b in gen)
    m['rules_hash_platform_note'] = 'point sha256 computed on the local CPU; cluster jobs record their own (platform round-off)'
    m['e0_store_version'] = RS.store_version()
    m['e0_metric_id'] = RS.V.METRIC_ID[2]
    m['eligibility'] = {f'{coh}|{c}|{k}': RS.reference_status(2, coh, c, k)['eligible']
                        for coh, n in (('dev6', 6), ('val32', 32)) for c in range(n) for k in ('ST', 'S')}
    m['cohorts'] = P.cohort_hashes()
    m['configs'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / 'configs').glob('*.json'))}
    (HERE / 'runs').mkdir(exist_ok=True)
    b = json.dumps(m, indent=1, default=str)
    (HERE / 'runs' / 'MANIFEST-dev.json').write_text(b)
    (HERE / 'runs' / 'MANIFEST-dev.sha256').write_text(hashlib.sha256(b.encode()).hexdigest() + '  MANIFEST-dev.json\n')
    print('manifest', hashlib.sha256(b.encode()).hexdigest()[:16], 'non-eligible:', [k for k, v in m['eligibility'].items() if not v])


if __name__ == '__main__':
    main()
