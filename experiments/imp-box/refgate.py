"""imp-box spectral reference gates and the conditional estimator (DESIGN A1.2, A2.2, A3.1)."""
from __future__ import annotations

import numpy as np

BAR_SPACE, BAR_TIME, BAR_PAD = 2e-5, 2e-6, 2e-6
ROUNDOFF = 1e-11
TRAIN_BAR = 1e-4


def pad_to(B, n):
    """(..., m, m) coefficients -> (..., n, n): zero-padded or truncated."""
    m = B.shape[-1]
    if m >= n:
        return B[..., :n, :n]
    pw = [(0, 0)] * (B.ndim - 2) + [(0, n - m), (0, n - m)]
    return np.pad(B, pw)


def coef_diff(B, Bref):
    """Continuous L2 difference per case (exact in coefficient space): max_{k>=1} ||B_k - Bref_k|| / ||Bref_0||.
    B, Bref (cases, T, n, n) (B padded/truncated to Bref's size)."""
    n = max(np.asarray(B).shape[-1], Bref.shape[-1])                 # zero-pad BOTH to the common size (audit 2)
    B, Bref = pad_to(np.asarray(B), n), pad_to(np.asarray(Bref), n)
    num = np.sqrt(np.sum((B - Bref) ** 2, axis=(-1, -2)))
    den = np.sqrt(np.sum(Bref[:, 0] ** 2, axis=(-1, -2)))
    e = np.max(num[:, 1:], axis=1) / den
    return np.where(np.all(np.isfinite(B), axis=(1, 2, 3)), e, np.inf)


def lat_diff(O, Oref):
    num = np.linalg.norm(O - Oref, axis=-1)
    den = np.linalg.norm(Oref[:, 0], axis=-1)
    e = np.max(num[:, 1:], axis=1) / den
    return np.where(np.all(np.isfinite(O), axis=(1, 2)), e, np.inf)


def case_labels(d34, d12, dt, dp):
    """Per-case eligibility and eps (A2.2/A3.1). All arrays (cases,). Returns dict of lists."""
    roundoff = (d12 <= ROUNDOFF) & (d34 <= ROUNDOFF)
    decay = (d34 <= d12 / 4) | roundoff
    ok = (d34 <= BAR_SPACE) & decay & (dt <= BAR_TIME) & (dp <= BAR_PAD) & np.isfinite(d34 + d12 + dt + dp)
    eps = d34 + dt + dp
    reason = []
    for i in range(len(d34)):
        r = []
        if not d34[i] <= BAR_SPACE:
            r.append('space')
        if not decay[i]:
            r.append('decay')
        if not dt[i] <= BAR_TIME:
            r.append('time')
        if not dp[i] <= BAR_PAD:
            r.append('pad')
        reason.append(','.join(r))
    return dict(eligible=ok.tolist(), eps=eps.tolist(), roundoff=roundoff.tolist(), reason=reason)


def allowance(eps, eligible):
    eps = np.asarray(eps)[np.asarray(eligible, bool)]
    return float(2 * np.max(eps)) if len(eps) else float('inf')


def selftest():
    """refgate.selftest (A1.2, A3.1): geometric sequence eligible; non-decaying rejected; round-off branch needs both
    differences negligible; the space bar rejects."""
    a = lambda *v: [np.array([x]) for x in v]
    lab = lambda *v: case_labels(*a(*v))['eligible'][0]
    out = dict(geometric_eligible=lab(1e-6, 1e-4, 1e-8, 1e-8), flat_rejected=not lab(1e-6, 2e-6, 1e-8, 1e-8),
               roundoff_eligible=lab(1e-13, 1e-13, 0.0, 0.0), roundoff_both_required=not lab(1e-9, 1e-12, 0.0, 0.0),
               space_bar_rejects=not lab(3e-5, 1e-3, 0.0, 0.0), time_bar_rejects=not lab(1e-6, 1e-4, 3e-6, 0.0),
               pad_bar_rejects=not lab(1e-6, 1e-4, 0.0, 3e-6))
    out['passed'] = bool(all(out.values()))
    return out
