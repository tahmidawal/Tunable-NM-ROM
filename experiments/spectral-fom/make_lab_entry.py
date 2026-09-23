"""Generate the LAB-LOG entry body (numbers read from reports/summary.json only) -> reports/lab-entry.generated.md"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'reports' / 'summary.json').read_text())
sha = hashlib.sha256((HERE / 'reports' / 'summary.json').read_bytes()).hexdigest()


def rng(v):
    v = [x for x in v if x is not None]
    return '—' if not v else (f'{min(v):.2f}' if len(v) == 1 else f'{min(v):.2f}–{max(v):.2f}')


L = []
w = L.append
w(f'Generated from `experiments/spectral-fom/reports/summary.json` (sha256 `{sha[:16]}…`). Ratio = spectral FOM GPU ms / '
  f'NM-ROM GPU ms (<1: spectral faster), same allocation, A–B–A.')
for key, t, u in (('poisson2d', 'Poisson 2D (DST-I, exact)', '²'), ('poisson3d', 'Poisson cube (3D DST-I, exact)', '³')):
    for r in S[key]:
        w(f"- {t} {r['mesh']}{u}: accurate `{r['accurate']['arm']}` {100 * r['accurate']['worst']:.3f}% {r['accurate']['ms']:.3f} ms, "
          f"fast `{r['fast']['arm']}` {100 * r['fast']['worst']:.3f}% {r['fast']['ms']:.3f} ms; DST {r['spectral_ms']:.3f} ms "
          f"(`{r['spectral_fom']}`, err {r['spectral_worst']:.1e}) -> ratio {r['ratio_accurate']:.3f} / {r['ratio_fast']:.3f}; "
          f"job {r['job']}, {r['gpu']}.")
for r in S['burgers2d']:
    a, f = r['arms']['rom_accurate'], r['arms']['rom_fast']
    w(f"- Burgers 2D {r['mesh']}²: accurate {100 * a['worst']:.3f}% {a['ms']:.1f} ms vs `{a['matched']}` "
      f"{100 * a['matched_worst']:.3f}% {a['matched_ms']:.1f} ms -> {a['ratio_matched']:.3f}; fast {100 * f['worst']:.3f}% "
      f"{f['ms']:.1f} ms vs `{f['matched']}` {100 * f['matched_worst']:.3f}% {f['matched_ms']:.1f} ms -> {f['ratio_matched']:.3f}; "
      f"job {r['job']}, {r['gpu']}.")
for key, t, u in (('heat2d', 'Heat 2D', '²'), ('heat3d', 'Heat 3D', '³')):
    for r in S[key]:
        parts = [f"{ro[4:]} `{a['arm']}` {100 * a['worst']:.4f}% {a['ms']:.3f} ms vs `{a['matched']}` {a['matched_ms']:.3f} ms -> {a['ratio_matched']:.3f}"
                 for ro, a in r['arms'].items()]
        w(f"- {t} {r['mesh']}{u}: " + '; '.join(parts) + f"; job {r['job']}, {r['gpu']}.")
for r in S['ns3d']['rows']:
    w(f"- NS 3D {r['mesh']}³ ({r['cohort']}, recorded from the lane): {r['role'][4:]} {r['arm']} {100 * r['worst']:.3f}% "
      f"{r['rom_ms']:.3f} ms vs {r['cnab2_matched']['label']} {r['cnab2_matched']['ms']:.3f} ms -> {r['ratio_matched']:.3f}.")
(HERE / 'reports' / 'lab-entry.generated.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
