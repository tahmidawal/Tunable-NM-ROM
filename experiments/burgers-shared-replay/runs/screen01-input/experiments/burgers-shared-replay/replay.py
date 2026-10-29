"""A standalone small trajectory followed by sequential guarded fine replay."""
import jax
import jax.numpy as jnp
import hcore as H
import p1core as T2
import replaycore as C

INFO = ('accepted', 'coarse_history_valid', 'fine_f0', 'fine_f1', 'candidate_finite')


def make_guard(hfine, nl, force_failure=False):
    def seed(k, warm, args):
        hist, nu, data, co, dt = args
        candidate = jnp.pad(data['replay_states'][k + 1], (0, 48))
        valid = data['replay_valid'][k] & (not force_failure)
        rc = T2.lspg_residual(nl, co, dt, nu, data, hist)
        r0, r1 = rc(hfine(warm)), rc(hfine(candidate))
        f0, f1 = jnp.vdot(r0, r0), jnp.vdot(r1, r1)
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(f0) & jnp.isfinite(f1)
        take = valid & finite & (f1 < f0) & (f1 <= (1 - 1e-8) * f0)
        info = jnp.array([take, valid, f0, f1, finite], dtype=jnp.float64)
        return jnp.where(take, candidate, warm), info
    return seed


def make_query(coarse, fine, L=1024, record=False, force_failure_step=None):
    """All work is inside this fused query, including both IC fits and previews.

    The coarse query records every state because replay needs all 50 guesses.
    Fine replay receives only those states and validity flags as extra data;
    its residual histories and carried damping are the original fine ones.
    """
    small = C.make_head_query(coarse.pf, 16, L, coarse.trust, nrec=51, nstep=50,
                              stop_on_invalid=True, force_failure_step=force_failure_step)
    nl = H.Q.tested_value('point', L)
    seed = make_guard(H.head_fn(fine.pf), nl)
    large = C.make_head_query(fine.pf, 64, L, fine.trust, nrec=51 if record else 0,
                              nstep=50, seed=seed)

    def query(u0, nu, data, cold_fine, cold_coarse, sch):
        cv = small['fused'](u0, nu, data, cold_coarse, sch)
        cs = cv['stats']
        ic_ok = ((cs['ic_reason'] == 1) | (cs['ic_reason'] == 4)) & jnp.all(jnp.isfinite(cv['w0']))
        step_ok = (jnp.all(jnp.isfinite(cs['states'][1:]), axis=1) & jnp.isfinite(cs['step_gn']) &
                   jnp.isfinite(cs['step_tolratio']) &
                   ((cs['step_gn'] <= sch['gtol']) | (cs['step_tolratio'] <= 1.)) &
                   (cs['step_reason'] != 0) & (cs['step_reason'] != 3))
        # A failed coarse history cannot become a qualified history later.
        valid = ic_ok & jnp.cumprod(step_ok.astype(jnp.int32)).astype(bool)
        fd = dict(data, replay_states=cs['states'], replay_valid=valid)
        fv = large['fused'](u0, nu, fd, cold_fine, sch)
        stats = dict(fv['stats'], **{'coarse_' + k: v for k, v in cs.items()}, coarse_replay_valid=valid)
        return dict(W=fv['W'], fields=fv['fields'], w0=fv['w0'], stats=stats,
                    coarse_W=cv['W'], coarse_fields=cv['fields'], coarse_w0=cv['w0'])

    return dict(fused=jax.jit(query), small=small, large=large)
