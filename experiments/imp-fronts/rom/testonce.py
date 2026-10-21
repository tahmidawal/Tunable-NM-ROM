"""imp-fronts test-once enforcement (DESIGN Amendment 12): every evaluation of test-split cases is registered immutably
BEFORE it runs, keyed by (bank hashes, policy hash, cohort-table hash); a second evaluation of the same key is refused.
Pure Python (hashlib / json).

  key           sha256 of {ckpt_sha256, rotation_sha256, policy_sha256, table_sha256}
  policy hash   sha256 over policy_sm.py, fpolicy.py, schedule.py sources and the canonical config (minus 'note')
  registration  rom/registrations/<key>.json, written once (exclusive create) and committed before staging:
                {key, bank, policy_sha256, table_sha256, config_sha256, config, created}
Driver side (fpolicy.main, test_once configs): the registration for the job's key must exist and name this config, and
the output directory must not exist (or be empty). A killed job is never re-run: its recovery is fcollect.py, which
finalises the same registration from the persisted per-case outputs only (RUN.json carries the key).

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
                           created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), fh, indent=1)
    except FileExistsError:
        raise TestOnceError(f'test evaluation {k[:12]} already registered: a second evaluation is refused')
    return k, p


def require(bank, cfg, table_sha, out, reg=REG):
    """In-job check. Returns the key; raises TestOnceError when unregistered or when the output already exists."""
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
    return k


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
