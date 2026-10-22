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
PRUNE_ERR = 0.3


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


def run_unbatched_cohort(q, D):
    """imp-box A9: run the (single-compiled) query case by case -> outputs, stats, seconds (compile included in the
    first call). The same compiled query then gives the screen time without a second compilation."""
    outs, sts = [], []
    t0 = time.perf_counter()
    for d in D:
        o, st = q(jnp.asarray(d))
        outs.append(np.asarray(o))
        sts.append(np.asarray(st))
    return np.stack(outs), np.stack(sts), time.perf_counter() - t0


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
    if fam in ('spec', 'specad'):                               # imp-box FOMs (spec.fom_valid)
        import spec as _SP
        return _SP.fom_valid(fam, st)
    if fam == 'trans':                                          # imp-box R-TRANS: Galerkin ETDRK4 statistics
        fam = 'podg'
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

    def run(self, key, cfg, build, prev_out=None, extra=None, keep_out=True):
        if key in self.done and not keep_out:
            return self.done[key], None
        t0 = time.perf_counter()
        try:
            q = build(cfg)
            O, st, sec = run_unbatched_cohort(q, self.D)                     # imp-box A9: one compile per row
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
        screen_error = None
        try:
            screen = time_one(q, self.D[0]) if svalid else None          # warm single-query screen time (A6)
        except Exception as ex:                                           # noqa: BLE001 (recorded; imp-box)
            screen, screen_error = None, repr(ex)[:200]
        row = dict(key=key, cfg=cfg, tag=self.tag, e=e.tolist(), worst=float(np.max(e)), median=float(np.median(e)),
                   screen=screen, screen_error=screen_error, stats_valid=svalid,
                   n_valid=int(valid.sum()), valid=bool(valid.all()), stats=st.tolist(), cohort_seconds=sec,
                   wall=time.perf_counter() - t0)
        if prev_out is not None:
            d = np.max(np.linalg.norm(O[:, 1:] - prev_out[:, 1:], axis=-1), axis=-1) / self.n0
            row['delta_prev'] = float(np.max(d)) if np.all(np.isfinite(d)) else float('inf')
        if extra:
            row.update(extra)
        self._write(row)
        return row, O

    def _write(self, row):
        self.done[row['key']] = row
        with self.path.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def step_ladder(self, base_key, cfg, build, n0, n_cap=N_CAP, nkey='nsteps'):
        """Step rule for one spatial size; returns the visited rows. imp-box: the termination reason (converged | pruned |
        invalid-rungs | cap | never-valid) is appended to ladder_terms.jsonl (code audit 1)."""
        rows = []
        prev = None
        n = n0
        bad = 0
        seen_valid = False
        reason = 'cap'
        while n <= n_cap:
            c = dict(cfg)
            c[nkey] = n
            row, O = self.run(f'{base_key}|nt{n}', c, build, prev_out=prev)
            rows.append(row)
            if row.get('valid'):
                seen_valid = True
                bad = 0
                if prev is not None and row.get('delta_prev', np.inf) <= DELTA_STOP:
                    reason = 'converged'
                    break
                # imp-box A5: hopeless-rung prune -- both statistics above 3 x the largest target and the step change
                # below a tenth of the error (the error is not time-dominated), so finer steps cannot reach any target
                if prev is not None and min(row['worst'], row['median']) > PRUNE_ERR and \
                        row.get('delta_prev', np.inf) < 0.1 * row['worst']:
                    reason = 'pruned'
                    break
                prev = O
            else:
                prev = None
                if seen_valid:
                    bad += 1
                    if bad >= 2:
                        reason = 'invalid-rungs'
                        break
            n *= 2
        if not seen_valid:
            reason = 'never-valid'
        with self.path.with_name('ladder_terms.jsonl').open('a') as f:
            f.write(json.dumps(dict(base=base_key, reason=reason, last_n=int(n), n_rows=len(rows),
                                    errors=[r.get('error') for r in rows if r.get('error')][:2])) + '\n')
        return rows
