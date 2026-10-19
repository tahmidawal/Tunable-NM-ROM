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
    for cf in c['configs']:
        sp = cf['spec']
        tcs = [j for j in cf.get('tighter_cases', list(range(8))) if j in ids]
        chk = [j for j in cf.get('check_cases', list(range(8))) if j in ids]
        keys += [('W1024', cf['setting'], cf['target'], sp, 'prod', 'target', j) for j in ids]
        keys += [('W1024', cf['setting'], cf['check'], sp, 'prod', 'check', j) for j in chk]
        for s in cf['draws']:
            rl, su = f"lat{cf['n']}s{s}", f"lat{cf['succ_n']}s{s}"
            keys += [('W1024', cf['setting'], rl, sp, 'prod', 'draw', j) for j in ids]
            keys += [('W1024', cf['setting'], rl, sp, 'adt-tight', 'draw', j) for j in ids]
            keys += [('W1024', cf['setting'], rl, sp, 'adt-tighter', 'draw', j) for j in tcs]
            keys += [('W1024', cf['setting'], su, sp, 'prod', f'succ-of-{rl}', j) for j in ids]
        for rl in cf.get('extra', []) + ([cf['bad']] if cf.get('bad') else []):
            keys += [('W1024', cf['setting'], rl, sp, 'prod', 'diag', j) for j in ids]
        if cf.get('permute'):
            keys += [('W1024', cf['setting'], f"lat{cf['n']}s0", sp, 'prod', 'permuted-replay', j) for j in ids]
    return keys


def inv_d3(c):
    keys = []
    ids = c.get('case_subset') or list(range(c['cohort_count']))
    bank = {s['name']: s['bank'] for s in c['settings']}
    for b in c['blocks']:
        if b['kind'] == 'ladder':
            tcs = [j for j in b.get('tighter_cases', list(range(8))) if j in ids]
            st = b['setting']
            for rl in b['rungs']:
                for sp in b['specs']:
                    keys += [(bank[st], st, rl, sp, 'prod', 'rung', j) for j in ids]
                for sd in sorted({tuple(sp.split('|')[1:3]) for sp in b['specs']}):
                    keys += [(bank[st], st, rl, f'ad|{sd[0]}|{sd[1]}', 'adt-tight', 'rung', j) for j in ids]
                    keys += [(bank[st], st, rl, f'ad|{sd[0]}|{sd[1]}', 'adt-tighter', 'rung', j) for j in tcs]
        elif b['kind'] == 'panel':
            for sp in [b['A']] + b['subjects']:
                for j in range(int(b['cases'])):
                    for r_ in range(int(b['reps'])):
                        keys.append((bank[sp['setting']], sp['setting'], sp['rule'], sp['spec'], 'timed-triplet', b['label'], j, r_))
    return keys


INV = {'r2acc': inv_r2acc, 'e1run3d': inv_e13d, 'd3run': inv_d3, 'e2run3d': inv_d3}


def klass(driver, k):
    """Execution class used for the measured-seconds lookup."""
    if driver == 'r2acc':
        bank, m, g, rule, arm, lev, kind, _ = k
        big = rule.startswith('gauss') and int(rule[5:]) >= 256
        return f"r2acc|{m}|{g}|{lev}|{'big' if big else 'std'}"
    if driver == 'e1run3d':
        return f"e1run3d|{k[1]}|{k[4]}|{'gl' if k[2].startswith('gl') else 'lat'}"
    if k[4] == 'timed-triplet':
        return f'd3|{k[1]}|triplet'
    return f"d3|{k[1]}|{k[4]}|{'gl' if k[2].startswith('gl') else 'lat'}"


def certify(att, timings_path):
    spec = json.loads((HERE / 'jobs' / f'{att}.json').read_text())
    T = json.loads((HERE / timings_path).read_text())
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
                setup = T.get(f"{st['driver']}|setup|{c.get('mesh', c.get('mode'))}")
                if setup is None:
                    missing.add(f"{st['driver']}|setup|{c.get('mesh', c.get('mode'))}")
                    setup = 0.
                sec = setup
                for k in uniq:
                    kl = klass(st['driver'], k)
                    if kl not in T:
                        missing.add(kl)
                    else:
                        sec += T[kl]
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
    cert = dict(attempt=att, projection_hours=round(proj, 4), group_hours=[round(x, 4) for x in group_hours],
                gpus=int(spec.get('gpus', 1)), projection_gpu_hours=round(proj * int(spec.get('gpus', 1)), 4),
                configs=cfgs, inventory_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(), margin=MARGIN,
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
