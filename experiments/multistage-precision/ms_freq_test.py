"""Unit test (F2): residual-frequency estimator + Fourier-feature schedule on
known 2-D sinusoids.  Pure numpy (JAX_PLATFORMS=cpu; nothing trained)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import ms_parametric as mp

n = 64
x = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, x, indexing="ij")
rng = np.random.default_rng(0)
ok = True
for f, kind in [(3, "x"), (7, "x"), (5, "diag"), (12, "diag"), (20, "x")]:
    if kind == "x":
        E = np.sin(2 * np.pi * f * X)          # radial freq f
        expect = f
    else:
        E = np.sin(2 * np.pi * f * X) * np.sin(2 * np.pi * f * Y)   # radius f*sqrt2
        expect = f * np.sqrt(2)
    fd = mp.dominant_radial_freq(E[None], n)
    nf = mp.freq_schedule(fd, 0, n)
    # a feature sin(pi*j*x) with j = 2f reproduces f cycles/unit exactly
    good = abs(fd - expect) <= 1.0 and nf >= 2 * expect
    ok &= good
    print(f"{kind:5s} f={f:2d}: f_d={fd:5.1f} (expect {expect:5.1f})  n_freq={nf} "
          f"(needs >= {2*expect:.0f})  {'OK' if good else 'FAIL'}")
# white noise must NOT return the Nyquist ring (old sum-per-annulus bug did)
W = rng.standard_normal((8, n, n))
fdw = mp.dominant_radial_freq(W, n)
print(f"white noise: f_d={fdw:.1f} (Nyquist ring would be ~{(n-1)/2*np.sqrt(2):.0f}); "
      f"{'OK' if fdw < 0.8*(n-1)/2*np.sqrt(2) else 'FAIL'}")
ok &= fdw < 0.8 * (n - 1) / 2 * np.sqrt(2)
# schedule cap = half-cycle Nyquist n-1
print(f"cap: freq_schedule(1e9) = {mp.freq_schedule(1e9, 0, n)} (expect {n-1})")
ok &= mp.freq_schedule(1e9, 0, n) == n - 1
print("ALL OK" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
