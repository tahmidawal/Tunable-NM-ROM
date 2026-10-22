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
        # 1024^2 12-step: jcp-hr-baselines t2d1024a measured H200 phases of exactly this recipe (tune.json timers /
        # phases): snapshots 208.4, POD 33.9, certificate 30.8, ECSW states 11.6, ECSW paths r128/256/384 31.5/310.8/
        # 2286.5 s (to kappa 16, an upper bound for kappa 8). 256^2 all-step: t2d256a measured A100 phases (upper bound
        # for an H200): saturation rebuild (all-step snapshots + POD) 539, states 5.5, paths r128/256/384 39/99/241 s.
        # Codes: s1c 1024 codes 1.0 s x 9 (snapshot count).
        t = (208.4 + 33.9 + 30.8 + 11.6 + 31.5 + 310.8 + 2286.5 + 9 * src['codes_seconds']) + (539 + 5.5 + 39 + 99 + 241 + 60)
    elif att == 'j5':
        # jcp-p1-scaling accuracy jobs (p1acc, same code): acc256 0.4389 h, acc1024 0.3694 h (A100-80G), acc4096
        # 0.3381 h (H200), each with more configurations than here but sub16 Q2; here all38 Q2 adds 22 successor runs
        # per configuration-scheme (~+30 %); plus the offline checker (CPU) ~120 s per mesh.
        t = (0.4389 + 0.3694 + 0.3381) * 3600 * 1.3 + 3 * 120
    elif att == 'j6':
        # per mesh: three bank builds (s1c 4096 bank pilot 31.6 s incl. setup) + every timed configuration: 76 timed
        # calls x (0.05 s burn + <= 0.15 s query) + AOT compile <= 40 s; 12 fresh-process compile probes <= 120 s each
        ncomp = sum(sum(len(a['configs']) for a in m['arms']) for m in cfg['meshes'].values())
        nours = 2 * 6 * len(cfg['meshes'])                    # A1 and A2, 38 calls each (+ 38 identity-free)
        # one attempt per mesh: the drift retry is disabled (retry_after_seconds = 0, DESIGN A11)
        t = 3 * 3 * 40 + ncomp * (76 * 0.2 + 40) + nours * (38 * 0.2 + 40) + 12 * 120
    elif att == 'j4':
        # J1 (job 6174908, H200) measured per-configuration 38-case run times 7-20 s at 256^2; 120 s per configuration
        # is allowed at 4096^2. Smoke s1c 4096^2 pilot: 2-trajectory states 22 s (x 12 for 24), blocked contributions
        # 75 s and NNLS 252 s per (source, rank) refit; PM estimate 8 trajectories x 5 s.
        t = pl['states_seconds'] * 12 + 6 * (pl['contrib_seconds'] + pl['nnls_seconds']) + 24 * 120 + 8 * 5
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
