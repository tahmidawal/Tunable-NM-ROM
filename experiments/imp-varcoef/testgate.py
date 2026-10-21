"""Test-split opening gate (DESIGN A2.18, A3.27; code audit 1 item 13).

The test cohort opens only for the test-once step: a config must name the frozen timing selection
(FROZEN-SELECTION-4096.json from the timing job, copied into checks/) and the committed opening record
checks/TEST-OPENED.json, both bound by sha256; the record names the permitted keys (the frozen union and per-family
winners at the decisive targets for C = 100) and is written once (make_configs.py test refuses to overwrite it).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECISIVE = ('0.03', '0.1')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def permitted(selection):
    keys = set()
    for gid, g in selection['selection'].items():
        parts = gid.split('|')
        if len(parts) == 3 and parts[1] in DECISIVE and parts[2] == 'C100' and g.get('selected'):
            keys.add(g['selected'])
    return sorted(keys)


def verify(cfg):
    fs, rec = HERE / cfg['frozen_selection'], HERE / cfg['opening_record']
    assert sha(fs) == cfg['frozen_selection_sha256'], 'frozen selection differs from the config binding'
    assert sha(rec) == cfg['opening_record_sha256'], 'opening record differs from the config binding'
    r = json.loads(rec.read_text())
    assert r['frozen_selection_sha256'] == cfg['frozen_selection_sha256'], 'opening record bound to another selection'
    allowed = permitted(json.loads(fs.read_text()))
    assert sorted(r['permitted_keys']) == allowed, 'opening record keys differ from the frozen winners'
    for k in cfg.get('keys', []):
        assert k in allowed, ('key not a frozen winner', k)
    return allowed
