"""imp-fronts test-once enforcement (DESIGN Amendment 12): every evaluation of test-split cases is registered immutably
BEFORE it runs, keyed by (bank hashes, policy hash, cohort-table hash); a second evaluation of the same key is refused.
Pure Python (hashlib / json).

  key           sha256 of {ckpt_sha256, rotation_sha256, policy_sha256, table_sha256}
  policy hash   sha256 over policy_sm.py, fpolicy.py, schedule.py sources and the canonical config (minus 'note')
  registration  rom/registrations/<key>.json, written once (exclusive create) and committed before staging:
                {key, bank, policy_sha256, table_sha256, config_sha256, config, created}
Driver side (fpolicy.main, test_once configs): the registration for the job's key must exist and name this config, the
output directory must not exist (or be empty), and the job then CLAIMS the key atomically and once
(<IMPF_TESTONCE_DIR>/<key>.claim on the shared cluster namespace, exclusive create, bound to the output directory and
the Slurm job id); any second claim, even to another empty directory, is refused. A killed job is never re-run: its
recovery is fcollect.py --claim <claim file>, which finalises only the bound directory / job from its persisted
per-case outputs (RUN.json carries the key and job id).

    python testonce.py register <config.json>     -> rom/registrations/<key>.json (refuses an existing key)
    python testonce.py show <config.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = HERE / 'registrations'
SOURCES = ('policy_sm.py', 'fpolicy.py', 'schedule.py')


class TestOnceError(RuntimeError):
    pass


def canon(cfg):
    return json.dumps({k: v for k, v in cfg.items() if k != 'note'}, sort_keys=True, separators=(',', ':'))


def policy_sha(cfg):
    h = hashlib.sha256()
    for s in SOURCES:
        h.update((HERE / s).read_bytes())
    h.update(canon(cfg).encode())
    return h.hexdigest()


def key_of(bank, cfg, table_sha):
    d = dict(ckpt_sha256=bank['ckpt_sha256'], rotation_sha256=bank['rotation_sha256'], policy_sha256=policy_sha(cfg),
             table_sha256=table_sha)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def register(bank, cfg, table_sha, reg=REG):
    k, d = key_of(bank, cfg, table_sha)
    reg.mkdir(parents=True, exist_ok=True)
    p = reg / f'{k}.json'
    try:
        with open(p, 'x') as fh:                     # exclusive create: a key is registered exactly once
            json.dump(dict(key=k, **d, config_sha256=hashlib.sha256(canon(cfg).encode()).hexdigest(), config=cfg,
                           evaluation=('RE-USE of already scored test cases (E10)' if cfg.get('reused_test_cases')
                                       else 'test-once'),
                           created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), fh, indent=1)
    except FileExistsError:
        raise TestOnceError(f'test evaluation {k[:12]} already registered: a second evaluation is refused')
    return k, p


def claim(k, out, job_id, claims_dir):
    """Atomically claim key k ONCE (exclusive create of <claims_dir>/<k>.claim on the shared cluster namespace), bound to
    this output directory and job id. A second claim -- for any directory, empty or not -- is refused."""
    if not claims_dir:
        raise TestOnceError('test-once jobs need a shared claims directory (IMPF_TESTONCE_DIR)')
    cd = Path(claims_dir)
    cd.mkdir(parents=True, exist_ok=True)
    rec = dict(key=k, out=str(Path(out).resolve()), job_id=str(job_id), claimed=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        with open(cd / f'{k}.claim', 'x') as fh:
            json.dump(rec, fh, indent=1)
    except FileExistsError:
        prev = json.loads((cd / f'{k}.claim').read_text())
        raise TestOnceError(f'test evaluation {k[:12]} already claimed by job {prev.get("job_id")} at {prev.get("out")}: '
                            'a second evaluation is refused (recovery = fcollect.py on the bound directory)')
    return rec


def require(bank, cfg, table_sha, out, claims_dir=None, job_id=None, reg=REG):
    """In-job check: registered key, registration names this config, output directory absent or empty, then the atomic
    claim. Returns (key, claim record)."""
    k, _ = key_of(bank, cfg, table_sha)
    p = reg / f'{k}.json'
    if not p.exists():
        raise TestOnceError(f'test evaluation {k[:12]} is not registered (python testonce.py register <config>)')
    r = json.loads(p.read_text())
    if r['config_sha256'] != hashlib.sha256(canon(cfg).encode()).hexdigest():
        raise TestOnceError('registration names another config')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise TestOnceError(f'{out} exists: a second test evaluation is refused (recovery of a killed job = fcollect.py)')
    return k, claim(k, out, job_id, claims_dir)


def bound(run_json, claim_file):
    """Collector-side check: a killed test-once run may be finalised only from the directory / job its claim binds."""
    run = json.loads(Path(run_json).read_text())
    c = json.loads(Path(claim_file).read_text())
    if run.get('testonce_key') != c['key'] or str(run.get('job_id')) != str(c['job_id']):
        raise TestOnceError('collector refused: this output is not the directory / job bound by the claim')
    return True


def main():
    cmd, cfgp = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, str(HERE))
    import bankguard as BG
    import fcases as FC
    cfg = json.loads(cfgp.read_text())
    assert cfg.get('test_once'), 'only test-once configs are registered'
    bank = BG.verify(cfg['bank'], HERE, expect_label=cfg.get('bank_label'))
    tsha = FC.load_table(FC.LANE / cfg.get('table', 'phaseb/fresh_cohorts.json'))[0]['sha256']
    if cmd == 'register':
        k, p = register(bank, cfg, tsha)
        print('registered', k, p)
    else:
        print(json.dumps(dict(zip(('key', 'parts'), key_of(bank, cfg, tsha))), indent=1))


if __name__ == '__main__':
    main()
