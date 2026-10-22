"""imp-geometry staged ladder search (DESIGN 5.4, A1.7).

Every method family supplies `build(cfg) -> query(d) -> (lattice outputs (6, P), stats)`. The engine runs a
configuration on a cohort (vmapped over cases), scores every visited configuration (worst and median over the cohort),
and applies the step rule: per spatial size, n = 5 * 2^j ascending from the family's first rung until the trajectory
distance delta_n = max_cases max_k ||u_n(t_k) - u_{n/2}(t_k)|| / ||u_ref(0)|| <= 1e-4 (both valid), two consecutive
invalid rungs after a valid one, or n = n_cap. The first rung has no delta. Every visited configuration is a candidate
for both statistics. Results are rows (dict) appended to a JSONL file as they are produced (checkpointable).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import score as SC

import os
N_CAP = int(os.environ.get('IGEO_NCAP', 20480))
DELTA_STOP = 1e-4


def run_cohort(q, D, chunk=8):
    """Run query q on descriptors D (n, 5) in vmapped chunks -> outputs (n, 6, P), stats (n, k), seconds."""
    vq = jax.jit(jax.vmap(q))
    outs, sts = [], []
    t0 = time.perf_counter()
    for s in range(0, len(D), chunk):
        o, st = vq(jnp.asarray(D[s:s + chunk]))
        outs.append(np.asarray(o))
        sts.append(np.asarray(st))
    return np.concatenate(outs), np.concatenate(sts), time.perf_counter() - t0


def time_one(q, d, reps=3):
    """Warm single-query time (screen): compile + warm call excluded; median of reps."""
    dj = jnp.asarray(d)
    o = q(dj)
    jax.block_until_ready(o)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(q(dj))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def stats_valid(cfg, st):
    """Benchmark validity from solver statistics (A6): FD-MG / SEM-pcg: no unconverged solve; reduced Newton/LSPG
    steppers: no non-finite step, no damping saturation, unconverged steps <= 1 % (LSPG) / 0 (Galerkin); ETDRK4: no
    non-finite stage and no growth-bound violation."""
    st = np.asarray(st)
    fam = cfg.get('fam')
    if fam in ('fdmg', 'sem'):
        return bool(np.all(st[:, 2] == 0))
    if fam in ('podg', 'ecsw', 'bank'):
        if cfg.get('st') == 'gal_etdrk4':
            return bool(np.all(st[:, 0] == 0) and np.all(st[:, 3] == 0))
        unc = st[:, 6].sum() / max(st[:, 2].sum(), 1)          # pooled over steps of all cases (DESIGN 7)
        lim = 0.01 if cfg.get('st') == 'lspg_cnr' else 0.0
        return bool(np.all(st[:, 3] == 0) and np.all(st[:, 5] == 0) and unc <= lim)
    return True


class Ladder:
    def __init__(self, out_path, Uref, D, U_allow, tag):
        self.path = Path(out_path)
        self.Uref = Uref                   # (n, 6, P) reference lattice values
        self.D = D
        self.U = U_allow
        self.tag = tag
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.done[r['key']] = r
        self.n0 = np.linalg.norm(Uref[:, 0], axis=-1)

    def run(self, key, cfg, build, prev_out=None, extra=None, keep_out=True, record=True):
        if key in self.done and record:                 # A9 resume: a recorded row is never recomputed
            return self.done[key], None
        t0 = time.perf_counter()
        try:
            q = build(cfg)
            O, st, sec = run_cohort(q, self.D)
            ok_build = True
            msg = ''
        except Exception as ex:                                           # noqa: BLE001  (recorded, not hidden)
            O, st, sec, ok_build, msg = None, None, 0.0, False, repr(ex)[:300]
        if not ok_build:
            row = dict(key=key, cfg=cfg, valid=False, error=msg, tag=self.tag)
            self._write(row)
            return row, None
        e = SC.err(O, self.Uref)
        svalid = stats_valid(cfg, st)
        e = np.where(svalid, e, np.inf)
        valid = np.isfinite(e)
        try:
            screen = time_one(q, self.D[0]) if svalid else None          # warm single-query screen time (A6)
        except Exception:                                                 # noqa: BLE001
            screen = None
        row = dict(key=key, cfg=cfg, tag=self.tag, e=e.tolist(), worst=float(np.max(e)), median=float(np.median(e)),
                   screen=screen, stats_valid=svalid,
                   n_valid=int(valid.sum()), valid=bool(valid.all()), stats=st.tolist(), cohort_seconds=sec,
                   wall=time.perf_counter() - t0)
        if prev_out is not None:
            d = np.max(np.linalg.norm(O[:, 1:] - prev_out[:, 1:], axis=-1), axis=-1) / self.n0
            row['delta_prev'] = float(np.max(d)) if np.all(np.isfinite(d)) else float('inf')
        if extra:
            row.update(extra)
        if record:
            self._write(row)
        return row, O

    def _write(self, row):
        self.done[row['key']] = row
        with self.path.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def step_ladder(self, base_key, cfg, build, n0, n_cap=N_CAP, nkey='nsteps'):
        """Step rule for one spatial size; returns the visited rows. Resumable (A9): a rung already recorded is
        replayed from its row (validity, delta_prev); the previous outputs are recomputed only when the next rung
        has to be computed and needs them for its delta."""
        rows = []
        prev = None
        prev_cfg = None
        n = n0
        bad = 0
        seen_valid = False
        while n <= n_cap:
            c = dict(cfg)
            c[nkey] = n
            key = f'{base_key}|nt{n}'
            if key in self.done:
                row, O = self.done[key], None
            else:
                if prev is None and prev_cfg is not None:          # resume: recompute the previous rung's outputs
                    _, prev = self.run(prev_cfg[0], prev_cfg[1], build, record=False)
                row, O = self.run(key, c, build, prev_out=prev)
            rows.append(row)
            if row.get('valid'):
                seen_valid = True
                bad = 0
                if 'delta_prev' in row and row['delta_prev'] <= DELTA_STOP:
                    break
                prev = O
                prev_cfg = (key, c)
            else:
                prev, prev_cfg = None, None
                if seen_valid:
                    bad += 1
                    if bad >= 2:
                        break
            n *= 2
        return rows
