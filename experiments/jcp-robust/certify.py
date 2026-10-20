"""jcp-robust execution inventory and resource certificate (DESIGN 7, A3.10, A4.2; common brief 6).

    python certify.py inventory <config.json>                 print the unique-execution inventory of one config
    python certify.py certify <attempt> [--timings checks/smoke-timings.json]
        -> checks/inventory-<attempt>.json  (unique execution keys of every step: bank, mesh, rule, spec / arm, solver
                                             level, case; controls and gates included; reuse marked)
           checks/resources-<attempt>.json  (projection_hours = per GPU group: sum of setup + count x measured seconds of
                                             that execution class, x 1.25; the job's GPU-h = groups run in parallel on
                                             `gpus` GPUs, so the reservation is gpus x max-group hours; bound to the config
                                             sha256s, the inventory sha256 and the GPU count)
Measured seconds come from the smokes (checks/smoke-timings.json: {"<driver>|<class>": seconds per execution,
"<driver>|setup|<mesh or setting>": seconds}); a class with no measurement makes certification fail (never a guess),
except shell (training) steps, whose projection is stated in the job spec with its basis.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARGIN = 1.25


def _cases(spec, n_all=38):
    if spec in (None, 'all38'):
        return n_all
    if spec == 'dev6':
        return 6
    if spec == 'sub16':
        return 16
    return len(spec)


def inv_r2acc(c):
    keys = []
    m = c['mesh']
    bank = (c.get('bank') or {}).get('ckpt_sha256', 'incumbent')
    ids = list(range(_cases(c.get('cases', 'all38'))))
    for g in c['groups']:
        tids = list(range(min(len(ids), _cases(g.get('tighter_cases', 'dev6')))))
        cids = list(range(min(len(ids), _cases(g.get('check_cases', 'dev6')))))
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            if g.get('target'):
                keys += [(bank, m, g['name'], g['target'], ak, 'prod', 'target', i) for i in ids]
            if g.get('check'):
                keys += [(bank, m, g['name'], g['check'], ak, 'prod', 'check', i) for i in cids]
            for r in g['rules']:
                keys += [(bank, m, g['name'], r['rule'], ak, 'prod', 'rule', i) for i in ids]
                if r.get('level', 'full') == 'full':
                    keys += [(bank, m, g['name'], r['rule'], ak, 'tight', 'rule', i) for i in ids]
                    keys += [(bank, m, g['name'], r['rule'], ak, 'tighter', 'rule', i) for i in tids]
                    if r.get('succ'):
                        keys += [(bank, m, g['name'], r['succ'], ak, 'prod', f"succ-of-{r['rule']}", i) for i in ids]
        if g.get('permute'):
            pr = next(r for r in g['rules'] if r['label'] == g['permute']['label'])
            keys += [(bank, m, g['name'], pr['rule'], 'arm' + str(g['permute']['arm']), 'prod', 'permuted-replay', i) for i in ids]
    return keys


def inv_e13d(c):
    keys = []
    ids = c.get('case_subset') or list(range(c['cohort_count']))
    bank_of = {st['name']: st['bank'] for st in c['settings']}     # bank from the settings (job audit 3D-1 #5)
    for cf in c['configs']:
        sp = cf['spec']
        B = bank_of[cf['setting']]
        tcs = [j for j in cf.get('tighter_cases', list(range(8))) if j in ids]
        chk = [j for j in cf.get('check_cases', list(range(8))) if j in ids]
        keys += [(B, cf['setting'], cf['target'], sp, 'prod', 'target', j) for j in ids]
        keys += [(B, cf['setting'], cf['check'], sp, 'prod', 'check', j) for j in chk]
        for s in cf['draws']:
            rl, su = f"lat{cf['n']}s{s}", f"lat{cf['succ_n']}s{s}"
            keys += [(B, cf['setting'], rl, sp, 'prod', 'draw', j) for j in ids]
            keys += [(B, cf['setting'], rl, sp, 'adt-tight', 'draw', j) for j in ids]
            keys += [(B, cf['setting'], rl, sp, 'adt-tighter', 'draw', j) for j in tcs]
            keys += [(B, cf['setting'], su, sp, 'prod', f'succ-of-{rl}', j) for j in ids]
        for rl in cf.get('extra', []) + ([cf['bad']] if cf.get('bad') else []):
            keys += [(B, cf['setting'], rl, sp, 'prod', 'diag', j) for j in ids]
        if cf.get('permute'):
            keys += [(B, cf['setting'], f"lat{cf['n']}s0", sp, 'prod', 'permuted-replay', j) for j in ids]
    return keys


def inv_d3(c):
    keys = []
    ids = c.get('case_subset') or list(range(c['cohort_count']))
    bank = {s['name']: s['bank'] for s in c['settings']}
    for b in c['blocks']:
        if b['kind'] == 'ladder':
            bids = b.get('cases') or ids                     # block-specific case set (code audit 2 #12)
            tcs = [j for j in b.get('tighter_cases', list(range(8))) if j in bids]
            st = b['setting']
            for rl in b['rungs']:
                for sp in b['specs']:
                    keys += [(bank[st], st, rl, sp, 'prod', 'rung', j) for j in bids]
                for sd in sorted({tuple(sp.split('|')[1:3]) for sp in b['specs']}):
                    keys += [(bank[st], st, rl, f'ad|{sd[0]}|{sd[1]}', 'adt-tight', 'rung', j) for j in bids]
                    keys += [(bank[st], st, rl, f'ad|{sd[0]}|{sd[1]}', 'adt-tighter', 'rung', j) for j in tcs]
        elif b['kind'] == 'panel':
            for sp in [b['A']] + b['subjects']:
                for j in range(int(b['cases'])):
                    for r_ in range(int(b['reps'])):
                        keys.append((bank[sp['setting']], sp['setting'], sp['rule'], sp['spec'], 'timed-triplet', b['label'], j, r_))
    return keys


def inv_dtime(c):
    """dtime: per subject and case an untimed scoring run + tight / tighter / successor / check replays, then reps A-B-A
    triplets (+ a fused triplet) per (subject, case, rep), and the doubled-work control."""
    keys = []
    bank = (c.get('bank') or {}).get('ckpt_sha256', 'incumbent')
    n = _cases(c.get('cases', 'dev6'))
    for sp in [c['A']] + c['subjects']:
        lab = f"{sp['Rp']}|{sp['rule']}|{sp['scheme']}|{sp['factor']}"
        for i in range(n):
            keys += [(bank, lab, 'score', i), (bank, lab, 'tight', i), (bank, lab, 'tighter', i), (bank, lab, 'qp', i),
                     (bank, lab, 'qpp', i)]
            if sp is not c['A']:                       # A is the reference: certified, never a timed subject
                keys += [(bank, lab, 'triplet', i, r) for r in range(int(c['reps']))]
                keys += [(bank, lab, 'fused-triplet', i, r) for r in range(int(c['reps']))]
    for sp in c.get('doubled', []):
        keys += [(bank, f"{sp['Rp']}|{sp['rule']}|{sp['scheme']}|{sp['factor']}|doubled", 'doubled', i, r) for i in range(n)
                 for r in range(int(c['reps']))]
    return keys


def code_tree_sha(commit, root=None):
    """sha256 over the (path, blob id) list of every staged lane file at `commit` (runs/, audits/, checks/, plots/,
    report* and HEARTBEAT excluded) -- binds a certificate to the code it was computed for (code audit 3 #12)."""
    import subprocess
    root = root or HERE
    out = subprocess.check_output(['git', 'ls-tree', '-r', commit, '--', '.'], cwd=root, text=True)
    skip = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json')
    rows = sorted(l.split(None, 3)[3] + ' ' + l.split()[2] for l in out.splitlines() if not l.split(None, 3)[3].startswith(skip))
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()


def _inv_e5(c):
    import e5common as EC
    return EC.inv_e5time(c)


INV = {'e5time': _inv_e5, 'r2acc': inv_r2acc, 'e1run3d': inv_e13d, 'd3run': inv_d3, 'e2run3d': inv_d3, 'dtime': inv_dtime}


def klass(driver, k):
    """Execution class used for the measured-seconds lookup."""
    if driver == 'r2acc':
        bank, m, g, rule, arm, lev, kind, _ = k
        big = rule.startswith('gauss') and int(rule[5:]) >= 256
        return f"r2acc|{m}|{g}|{lev}|{'big' if big else 'std'}"
    if driver == 'e5time':
        import e5common as EC
        return EC.klass(k)          # one-argument signature (subagent note)
    if driver == 'dtime':
        return f'dtime|{k[2]}'
    if driver == 'e1run3d':
        return f"e1run3d|{k[1]}|{k[4]}|{'gl' if k[2].startswith('gl') else 'lat'}"
    if k[4] == 'timed-triplet':
        return f'd3|{k[1]}|triplet'
    return f"d3|{k[1]}|{k[4]}|{k[2]}"                 # per rule (A13.4)


def certify(att, timings_path):
    spec = json.loads((HERE / 'jobs' / f'{att}.json').read_text())
    T = json.loads((HERE / timings_path).read_text())
    T = T.get('timings', T)
    inv, group_hours, cfgs, missing = {}, [], {}, set()
    for gi, g in enumerate(spec['groups']):
        h = 0.
        for st in g:
            if 'driver' in st:
                c = json.loads((HERE / st['config']).read_text())
                cfgs[st['config']] = hashlib.sha256((HERE / st['config']).read_bytes()).hexdigest()
                keys = INV[st['driver']](c)
                uniq = sorted(set(map(tuple, keys)), key=str)
                inv[st['config']] = dict(n=len(keys), unique=len(uniq), duplicates=len(keys) - len(uniq),
                                         keys=[list(map(str, k)) for k in uniq])
                setup = T.get(f"{st['driver']}|setup|{c.get('mesh', c.get('mode'))}", T.get(f"{st['driver']}|setup"))
                if st['driver'] == 'e5time':
                    setup = 0.                           # A10.1: the smoke's elapsed already contains setup
                if setup is None:
                    missing.add(f"{st['driver']}|setup|{c.get('mesh', c.get('mode'))}")
                    setup = 0.
                sec = setup
                if st['driver'] == 'e5time':            # A10.1: smoke-measured seconds per execution of the same GPU type
                    gt = c.get('gpu_type')
                    kk = f'e5time|per_execution|{gt}'
                    if kk not in T:
                        missing.add(kk)
                    else:
                        sec += T[kk] * len(uniq)
                    uniq = []
                ov = T.get(f"r2acc|overhead|{c.get('mesh')}", 0.) if st['driver'] == 'r2acc' else 0.   # A10.3
                for k in uniq:
                    kl = klass(st['driver'], k)
                    if kl not in T:
                        missing.add(kl)
                    else:
                        sec += T[kl] + ov
                h += sec * MARGIN / 3600
            else:
                assert 'projection_hours' in st and st.get('projection_basis'), 'a shell step states its projection and basis'
                h += float(st['projection_hours'])
        group_hours.append(h)
    if missing:
        print('UNMEASURED classes (certificate refused):', sorted(missing))
        sys.exit(2)
    (HERE / 'checks').mkdir(exist_ok=True)
    ip = HERE / 'checks' / f'inventory-{att}.json'
    ip.write_text(json.dumps(inv, indent=0) + '\n')
    proj = max(group_hours)
    import math
    import subprocess
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    cert = dict(attempt=att, commit=commit, code_tree_sha256=code_tree_sha(commit), jobspec_sha256=hashlib.sha256((HERE / 'jobs' / f'{att}.json').read_bytes()).hexdigest(),
                time_limit_hours=math.ceil(float(spec['hours']) * 4) / 4, projection_hours=round(proj, 4), group_hours=[round(x, 4) for x in group_hours],
                gpus=int(spec.get('gpus', 1)), projection_gpu_hours=round(proj * int(spec.get('gpus', 1)), 4),
                configs=cfgs, inventory_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(), margin=MARGIN,
                estimates=sorted({'A10.1 E5 per-execution rate from the smoke' for g in spec['groups'] for st in g if st.get('driver') == 'e5time'}
                                 | {'A10.2 dtime classes derived from measured rollout classes' for g in spec['groups'] for st in g
                                    if st.get('driver') == 'dtime'}
                                 | {'A11.3 E1 3D W512 / M2_512 classes = 2 x the measured W256 class (unmeasured widths)'
                                    for g in spec['groups'] for st in g if st.get('driver') == 'e1run3d'}),
                timings=timings_path, timings_sha256=hashlib.sha256((HERE / timings_path).read_bytes()).hexdigest())
    (HERE / 'checks' / f'resources-{att}.json').write_text(json.dumps(cert, indent=1) + '\n')
    print(json.dumps({k: v for k, v in cert.items() if k != 'configs'}))


if __name__ == '__main__':
    if sys.argv[1] == 'inventory':
        c = json.loads(Path(sys.argv[2]).read_text())
        k = INV[c['driver']](c)
        print(len(k), 'executions', len(set(map(tuple, k))), 'unique')
    else:
        tp = sys.argv[sys.argv.index('--timings') + 1] if '--timings' in sys.argv else 'checks/smoke-timings.json'
        certify(sys.argv[2], tp)
