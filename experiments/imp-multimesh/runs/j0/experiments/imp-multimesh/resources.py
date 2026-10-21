"""Smoke-derived resource certificates (DESIGN A1.7): measured smoke phases x workload counts x 1.5.
    python resources.py <attempt> <config> <smoke attempt>
Measured inputs (smoke s1c, H200): 64-trajectory 1024^2 snapshots, POD of 768 snapshots, 24-trajectory 1024^2 ECSW
states, r=128 ECSW path, 4096^2 states (2 trajectories), blocked contributions (120 states, r=384), NNLS (r=384,
tiled, memory pilot), p1acc replication (R128 BE, 6 dev6 cases, all roles) and the 4096^2 bank pilot."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    att, cfgn, smk = sys.argv[1:4]
    sm = json.loads((HERE / 'runs' / smk / 'pulled' / 'output' / 'smoke' / 'smoke.json').read_text())
    rp = json.loads((HERE / 'runs' / smk / 'pulled' / 'output' / 'repl' / 'result.json').read_text())
    src, pl = sm['src1024'], sm['pilot']
    nnls = max(pl['nnls_seconds'], src['ecsw']['128']['nnls_seconds'])       # per path, r <= 384
    snap1024 = src['snap_seconds'] * 576 / 64
    per_case_p1 = (rp['elapsed_seconds'] - rp['phases']['setup']) / 6          # one case, every role, R128
    cfg = json.loads((HERE / cfgn).read_text())
    if att == 'j0':
        s256 = snap1024 * (255 ** 2 / 1023 ** 2) * 51 / 12
        t = snap1024 + 120 + src['states_seconds'] + 3 * nnls + s256 + 120 + src['states_seconds'] / 4 + 3 * nnls
    elif att == 'j5':
        t = sum(6 * 38 * per_case_p1 * f for f in (0.5, 1.0, 3.0)) + 3 * 120      # 4096^2 decode-heavy x3
    else:
        L = cfg['L']
        st = {256: src['states_seconds'] / 4, 1024: src['states_seconds'], 4096: pl['states_seconds'] * 12}[L]
        q = {256: 0.5, 1024: 1.0, 4096: 4.0}[L] * per_case_p1 * 0.5
        nref = sum(1 for a in cfg['arms'] if a['kind'] == 'refit' and a['source'] != L)
        t = st + nref * 3 * (nnls + (pl['contrib_seconds'] if L == 4096 else 30)) + len(cfg['arms']) * 6 * 38 * q \
            + (8 * 5 if cfg.get('pm_estimate') else 0)
    out = dict(attempt=att, kind='prod', projected_hours=round(1.5 * t / 3600, 3), raw_seconds=t, smoke=smk,
               basis=__doc__.split('\n')[1], config=cfgn,
               config_sha256=hashlib.sha256((HERE / 'runs' / att / 'experiments' / 'imp-multimesh' / cfgn).read_bytes()).hexdigest())
    (HERE / 'checks' / f'resources-{att}.json').write_text(json.dumps(out, indent=1))
    print(att, out['projected_hours'])


if __name__ == '__main__':
    main()
