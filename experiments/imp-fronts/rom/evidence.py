"""imp-fronts smoke control evidence (DESIGN Amendment 12: "no production item is admitted unless C-B1 (hash
rejection), C-B6 (gamma fixture) and the smoke dry run have hash-bound passing records"). Pure Python.

SM-r outputs (pulled locally under runs/SM-r/output/):
  fpolicy_smkpolK1/result.json   C-B1 (controls.cb1) and C-B6 (controls.CB6) on K1, FC1 dev j = 0, 1, round 1
  fpolicy_smkdryK1/result.json   the reference-free K1 policy dry run on the 16 fresh dev/val cases
build() evaluates every check and binds it to the sha256 of the file it read; verify() re-reads the files, recomputes the
hashes and the checks, and passes only if both agree with the stored record and every check passes.

    python evidence.py build  <SM-r output dir> [--out runs/SM-r/SMOKE-EVIDENCE.json]
    python evidence.py verify <SMOKE-EVIDENCE.json>
    python evidence.py build-e10 <SM-e output dir> [--out runs/SM-e/E10-EVIDENCE.json];  verify-e10 <file>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schedule as SCH  # noqa: E402
import policy_sm as SM  # noqa: E402

K1_BANK = HERE / 'inputs' / 'banks' / 's1' / 'BANK.json'
FILES = {'controls': 'fpolicy_smkpolK1/result.json', 'dry_run': 'fpolicy_smkdryK1/result.json'}
OPEN_TERMINALS = ('BUDGET-CAP', 'ROUND-CAP')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _common(r, attempt):
    k1 = json.loads(K1_BANK.read_text())
    return {'complete': r.get('complete') is True, 'backend_gpu': r.get('backend') == 'gpu',
            'precision_highest': r.get('precision') == 'highest', 'attempt': r.get('config', {}).get('attempt') == attempt,
            'bank_is_K1': (r.get('bank') or {}).get('ckpt_sha256') == k1['ckpt_sha256']
            and (r.get('bank') or {}).get('rotation_sha256') == k1['rotation_sha256']}


def check_controls(r):
    c = r.get('controls') or {}
    cb1, cb6 = c.get('cb1') or {}, c.get('CB6') or {}
    out = _common(r, 'smkpolK1')
    out.update(cb1_swapped_rejected=bool((cb1.get('swapped_bank_json') or {}).get('fired')),
               cb1_config_hash_rejected=bool((cb1.get('config_hash_mismatch') or {}).get('fired')),
               cb1_matched_passes=bool((cb1.get('matched_positive') or {}).get('passed')),
               cb6_fixture_found=bool(cb6.get('fixture')), cb6_fired=cb6.get('fired') is True,
               cb6_not_blocked=cb6.get('blocked') is False,
               cb6_agree=bool((cb6.get('fixture') or {}).get('agree')),
               cb6_case=cb6.get('run', '').startswith('FC1|0|cert|') and int(cb6.get('step', -1)) == 10)
    return out


CLS = {'prod': 'cert', 'half': 'cert', 'tight': 'tight', 'succ': 'succ', 'tighter': 'tighter'}


def _rkey(cid, kind, rule, f):
    return f'{cid}|{CLS[kind]}|{rule}|{float(f):g}'


def consistency(r):
    """Solve / history consistency: every charged solve row is a run the case histories requested (or a terminal
    tighter), every requested run has a record, rounds = len(history), terminal = the last decision."""
    rows, recs, pol = r.get('rows') or [], r.get('records') or {}, r.get('policy') or []
    solved = {_rkey(x['case'], x['kind'], x['rule'], x['factor']) for x in rows}
    want, ok = set(), True
    for p in pol:
        hist = p.get('history') or []
        ok = ok and int(p.get('rounds', -1)) == len([h for h in hist if h.get('value') != 'BUDGET-CAP'])
        for h in hist:
            for k in (h.get('prod'), h.get('succ'), h.get('half')):
                if k:
                    want.add(k)
            if h.get('prod'):
                want.add(h['prod'].replace('|cert|', '|tight|'))
        last = hist[-1] if hist else {}
        ok = ok and (last.get('decision') == 'terminal' and last.get('value') == p.get('terminal')
                     or p.get('terminal') in ('ROUND-CAP',))
    ran = {k for k, v in recs.items() if not v.get('skipped')}
    tighter = {k for k in ran if '|tighter|' in k}
    return bool(ok and want <= set(recs) and solved == ran and (ran - tighter) <= want)


def mandatory_solves(r):
    """Every evaluated state executed its mandatory solves with timing rows: prod (valid or not) and, if prod is valid,
    its tight, its succ and (f > 0.25) its half; each executed record has at least one charged row."""
    recs, rows = r.get('records') or {}, r.get('rows') or []
    fmin = SM.factor_set((r.get('config') or {}).get('factors'))[0]
    timed = {_rkey(x['case'], x['kind'], x['rule'], x['factor']) for x in rows if x.get('seconds', 0) > 0}
    for p in r.get('policy') or []:
        for h in p.get('history') or []:
            if h.get('value') == 'BUDGET-CAP' or not h.get('prod'):
                continue
            pr = recs.get(h['prod']) or {}
            if pr.get('skipped') or h['prod'] not in timed or pr.get('valid') not in (True, False):
                return False
            if pr.get('valid') is True:
                need = [h['prod'].replace('|cert|', '|tight|'), h.get('succ')] + ([h['half']] if h.get('half') else [])
                if SM.half_factor(float(h['f']), fmin) is not None and not h.get('half'):
                    return False
                for k in need:
                    rk = recs.get(k) or {}
                    if not k or rk.get('skipped') or k not in timed or rk.get('valid') not in (True, False):
                        return False
    return True


def replay(r):
    """Re-derive every case's path with policy_sm from the recorded evidence (records as JSON: non-finite -> null, which
    the state machine treats exactly as non-finite) and require the same histories, terminals and scored runs."""
    recs, cfg = r.get('records') or {}, r.get('config') or {}
    for p in r.get('policy') or []:
        if p.get('terminal') not in SM.TERMINALS:
            return False
        fmin, cap = SM.factor_set(cfg.get('factors'))
        c = SM.Case(p['case'], cfg.get('selected'), fmin)
        while c.terminal is None and c.rounds < len(p.get('history') or []):
            SM.advance(c, recs, int(cfg.get('max_rounds', cap)))
        if c.terminal is None and p.get('terminal') == 'BUDGET-CAP':
            SM.budget_cap(c, recs)
        got = [(h['rule'], h['f'], h['decision'], list(h['value']) if isinstance(h['value'], (list, tuple)) else h['value'])
               for h in c.history]
        want = [(h['rule'], h['f'], h['decision'], list(h['value']) if isinstance(h['value'], (list, tuple)) else h['value'])
                for h in p.get('history') or []]
        if got != want or c.terminal != p.get('terminal') or c.scored != p.get('scored'):
            return False
    return True


def transitions(results):
    """Registered map coverage over the given results: q2_successor (a succ solve), the factor-halving map (a half
    solve) and next_candidate (an escalation decision)."""
    succ = half = esc = False
    for r in results:
        for x in r.get('rows') or []:
            succ = succ or x.get('kind') == 'succ'
            half = half or x.get('kind') == 'half'
        for p in r.get('policy') or []:
            for h in p.get('history') or []:
                v = h.get('value')
                esc = esc or (h.get('decision') == 'next' and isinstance(v, (list, tuple)) and v[0] != h.get('rule'))
    return dict(map_q2_successor=succ, map_half=half, map_next_candidate=esc)


def check_dry(r):
    """The registered dry run: config and policy-result case ids are exactly the 16 fresh dev/val ids (unique), K1 from
    gauss160, the full policy (10 rounds), complete, no case left open (BUDGET-CAP / ROUND-CAP), every case charged at
    least one valid solve, solves consistent with the case histories, and the successor and halving maps exercised
    (the escalation map is checked over SM-r's two runs together, build())."""
    cfg = r.get('config') or {}
    out = _common(r, 'smkdryK1')
    rows = r.get('rows') or []
    per = {}
    for x in rows:
        per.setdefault(x.get('case'), []).append(x)
    pol = r.get('policy') or []
    pids = [p.get('case') for p in pol]
    tr = transitions([r])
    out.update(case_ids=tuple(cfg.get('cases', ())) == SCH.DEVVAL_IDS,
               policy_ids=len(pids) == len(set(pids)) == len(SCH.DEVVAL_IDS) and set(pids) == set(SCH.DEVVAL_IDS),
               start_gauss160=cfg.get('selected') == 'gauss160', full_policy=int(cfg.get('max_rounds', 0)) == 10,
               no_open_cases=all(p.get('terminal') and p.get('terminal') not in OPEN_TERMINALS for p in pol),
               every_case_charged=set(per) == set(SCH.DEVVAL_IDS)
               and all(any(x.get('valid') for x in xs) and all(x.get('seconds', 0) > 0 for x in xs) for xs in per.values()),
               solves_match_history=consistency(r), mandatory_solves=mandatory_solves(r), replay_state_machine=replay(r),
               step_wall_clock=isinstance(r.get('driver_wall_seconds'), (int, float)) and 0 < r['driver_wall_seconds'] < 1e6,
               map_q2_successor=tr['map_q2_successor'], map_half=tr['map_half'])
    return out


CONTROL_RULE = 'gauss256'          # SM-r dry run: FC1|0 escalated gauss160 -> 192 -> 256 at f = 1, then failed at (256, 0.5)


def check_e10_smoke(r):
    """SM-e (E10-DESIGN.md section 9): K1 from gauss160 with the E10-s factor set and round cap 11 on FC1 dev cases incl.
    the must-fail control FC1|0, complete on a GPU with precision 'highest' and K1's checkpoint AND rotation hashes; the
    PINNED control fired = FC1|0's state (gauss256, 0.5) failed its step check (decision (gauss256, 0.5) ->
    (gauss256, 0.25)), the state SM-r's dry run recorded (code audit C-2); some case visited all four factors (reached
    0.125); solves consistent with the histories, mandatory solves executed and the state machine replays the decisions."""
    cfg = r.get('config') or {}
    k1 = json.loads(K1_BANK.read_text())
    fired = four = False
    for p in r.get('policy') or []:
        fs = {float(h['f']) for h in p.get('history') or [] if 'f' in h}
        four = four or {1., .5, .25, .125} <= fs
        if p.get('case') == 'FC1|0':
            fired = any(h.get('decision') == 'next' and h.get('rule') == CONTROL_RULE and float(h.get('f', 0)) == .5
                        and isinstance(h.get('value'), (list, tuple)) and list(h['value']) == [CONTROL_RULE, .25]
                        for h in p.get('history') or [])
    ids = list(cfg.get('cases') or [])
    return dict(complete=r.get('complete') is True, backend_gpu=r.get('backend') == 'gpu',
                precision_highest=r.get('precision') == 'highest',
                bank_is_K1=(r.get('bank') or {}).get('ckpt_sha256') == k1['ckpt_sha256']
                and (r.get('bank') or {}).get('rotation_sha256') == k1['rotation_sha256'],
                factor_set_e10s=[float(x) for x in cfg.get('factors') or []] == [1., .5, .25, .125]
                and int(cfg.get('max_rounds', 0)) == 11,
                start_gauss160=cfg.get('selected') == 'gauss160',
                fc1_dev_cases='FC1|0' in ids and len(ids) >= 2 and all(c.split('|')[0] == 'FC1' and int(c.split('|')[1]) < 4 for c in ids),
                control_fired=fired, all_four_factors=four, solves_match_history=consistency(r),
                mandatory_solves=mandatory_solves(r), replay_state_machine=replay(r),
                step_wall_clock=isinstance(r.get('driver_wall_seconds'), (int, float)) and 0 < r['driver_wall_seconds'] < 1e6)


def build(outdir):
    outdir = Path(outdir)
    ev = dict(schema='imp-fronts-smoke-evidence/2', root=str(outdir), files={}, checks={})
    loaded = {}
    for k, rel in FILES.items():
        p = outdir / rel
        if not p.exists():
            ev['files'][k], ev['checks'][k] = dict(path=rel, sha256=None), dict(present=False)
            continue
        r = json.loads(p.read_text())
        ev['files'][k] = dict(path=rel, sha256=sha(p))
        ev['checks'][k] = (check_controls if k == 'controls' else check_dry)(r)
        loaded[k] = r
    ev['checks']['coverage'] = dict(map_next_candidate=transitions(list(loaded.values()))['map_next_candidate'])
    ev['passed'] = bool(all(c and all(c.values()) for c in ev['checks'].values()))
    return ev


def verify(ev_path):
    """True iff the stored evidence passed AND the files still hash and check to the same record."""
    try:
        ev = json.loads(Path(ev_path).read_text())
        again = build(ev['root'])
    except Exception as e:  # noqa: BLE001
        return False, f'evidence unreadable: {e}'
    if again['files'] != ev['files'] or again['checks'] != ev['checks']:
        return False, 'evidence differs from the files it is bound to'
    if not (ev.get('passed') and again['passed']):
        bad = {k: [n for n, v in c.items() if not v] for k, c in again['checks'].items()}
        return False, f'controls / dry run not passing: {bad}'
    return True, 'ok'


E10_FILE = 'fpolicy_smke10s/result.json'


def build_e10(outdir):
    """Hash-bound SM-e evidence (E10-s): check_e10_smoke on fpolicy_smke10s/result.json, bound to its sha256."""
    outdir = Path(outdir)
    p = outdir / E10_FILE
    if not p.exists():
        return dict(schema='imp-fronts-e10-evidence/1', root=str(outdir), file=dict(path=E10_FILE, sha256=None),
                    checks=dict(present=False), passed=False)
    chk = check_e10_smoke(json.loads(p.read_text()))
    return dict(schema='imp-fronts-e10-evidence/1', root=str(outdir), file=dict(path=E10_FILE, sha256=sha(p)), checks=chk,
                passed=bool(all(chk.values())))


def verify_e10(ev_path):
    try:
        ev = json.loads(Path(ev_path).read_text())
        again = build_e10(ev['root'])
    except Exception as e:  # noqa: BLE001
        return False, f'E10 evidence unreadable: {e}'
    if again['file'] != ev['file'] or again['checks'] != ev['checks']:
        return False, 'E10 evidence differs from the file it is bound to'
    if not (ev.get('passed') and again['passed']):
        return False, f"SM-e control / contract not passing: {[k for k, v in again['checks'].items() if not v]}"
    return True, 'ok'


def main():
    cmd = sys.argv[1]
    if cmd in ('build-e10', 'verify-e10'):
        if cmd == 'build-e10':
            ev = build_e10(sys.argv[2])
            out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else HERE.parent / 'runs' / 'SM-e' / 'E10-EVIDENCE.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(ev, indent=1) + '\n')
            print(out, 'passed' if ev['passed'] else 'FAILED', ev['checks'])
            sys.exit(0 if ev['passed'] else 4)
        ok, msg = verify_e10(sys.argv[2])
        print(ok, msg)
        sys.exit(0 if ok else 4)
    if cmd == 'build':
        ev = build(sys.argv[2])
        out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else HERE.parent / 'runs' / 'SM-r' / 'SMOKE-EVIDENCE.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ev, indent=1) + '\n')
        print(out, 'passed' if ev['passed'] else 'FAILED', ev['checks'])
        sys.exit(0 if ev['passed'] else 4)
    ok, msg = verify(sys.argv[2])
    print(ok, msg)
    sys.exit(0 if ok else 4)


if __name__ == '__main__':
    main()
