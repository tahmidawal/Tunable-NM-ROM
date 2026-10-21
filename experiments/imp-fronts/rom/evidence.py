"""imp-fronts smoke control evidence (DESIGN Amendment 12: "no production item is admitted unless C-B1 (hash
rejection), C-B6 (gamma fixture) and the smoke dry run have hash-bound passing records"). Pure Python.

SM-r outputs (pulled locally under runs/SM-r/output/):
  fpolicy_smkpolK1/result.json   C-B1 (controls.cb1) and C-B6 (controls.CB6) on K1, FC1 dev j = 0, 1, round 1
  fpolicy_smkdryK1/result.json   the reference-free K1 policy dry run on the 16 fresh dev/val cases
build() evaluates every check and binds it to the sha256 of the file it read; verify() re-reads the files, recomputes the
hashes and the checks, and passes only if both agree with the stored record and every check passes.

    python evidence.py build  <SM-r output dir> [--out runs/SM-r/SMOKE-EVIDENCE.json]
    python evidence.py verify <SMOKE-EVIDENCE.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schedule as SCH  # noqa: E402

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


def check_dry(r):
    """The registered dry run: exactly the 16 fresh dev/val ids, K1 from gauss160, the full policy (10 rounds, no smoke
    round limit), complete, no case left open (BUDGET-CAP / ROUND-CAP), every case charged at least one valid solve."""
    cfg = r.get('config') or {}
    out = _common(r, 'smkdryK1')
    rows = r.get('rows') or []
    per = {}
    for x in rows:
        per.setdefault(x.get('case'), []).append(x)
    pol = r.get('policy') or []
    out.update(case_ids=tuple(cfg.get('cases', ())) == SCH.DEVVAL_IDS,
               start_gauss160=cfg.get('selected') == 'gauss160', full_policy=int(cfg.get('max_rounds', 0)) == 10,
               no_open_cases=all(p.get('terminal') not in OPEN_TERMINALS and p.get('terminal') for p in pol)
               and len(pol) == len(SCH.DEVVAL_IDS),
               every_case_charged=set(per) == set(SCH.DEVVAL_IDS)
               and all(any(x.get('valid') for x in xs) and all(x.get('seconds', 0) > 0 for x in xs) for xs in per.values()))
    return out


def build(outdir):
    outdir = Path(outdir)
    ev = dict(schema='imp-fronts-smoke-evidence/1', root=str(outdir), files={}, checks={})
    for k, rel in FILES.items():
        p = outdir / rel
        if not p.exists():
            ev['files'][k], ev['checks'][k] = dict(path=rel, sha256=None), dict(present=False)
            continue
        r = json.loads(p.read_text())
        ev['files'][k] = dict(path=rel, sha256=sha(p))
        ev['checks'][k] = (check_controls if k == 'controls' else check_dry)(r)
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


def main():
    cmd = sys.argv[1]
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
