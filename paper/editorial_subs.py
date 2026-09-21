"""Editorial label substitutions applied to legacy generated tables (gen_tables.py output) after generation.

Reviewed 2026-09-21: replace internal study names with descriptive terms, use k for the latent dimension,
and bring the problem-specification table's meshes and cohorts up to the rows of Table 1.  Every substitution
is an exact string replacement; check_rewrite.py verifies current == apply(BASE) so no number can change here.
"""
from pathlib import Path

WIDE = (r'Heat 2D, wide bank & $u_t=\kappa\Delta u$, $(0,1)^2$ & $1024^2$--$4096^2$ & Crank--Nicolson; '
        r'batched exact-propagator fit & $k=8$, $R=128$ & exact same-grid propagation (DST) & 12 development cases; '
        r'16 sealed held-out cases, opened once \\' + '\n')
SUBS = {
    'T01_problems': [
        ('$256^2$ (ladder 64--1024)', '$256^2$--$4096^2$'),
        ('$K=16$, $R=512$', '$k=16$, $R=512$'),
        ('6 development cases; 32 held-out (tuning); sealed cohort opened once (job 3804465)',
         '6 development cases; 32 held-out (tuning); 64 held-out at $2048^2$, $4096^2$; sealed cohort opened once'),
        ('$256^2$, $1024^2$ & none (elliptic) & $K=16$, $R=128$ (incumbent); $K=32$, $R=512$',
         '$256^2$--$4096^2$ & none (elliptic) & $k=16$, $R=128$ (original); $k=32$, $R=512$'),
        ('Wave 2D (reflective) & ', WIDE + 'Wave 2D (reflective) & '),
        ('$K=32$, $R=64$', '$k=32$, $R=64$'),
        (r'$256^2$, $512^2$ & none & $K\in\{16,32\}$', r'$256^2$--$2048^2$ & none & $k\in\{16,32\}$'),
    ],
    'T13_sealed': [('incumbent sealed', 'original model sealed'), ('for the incumbent', 'for the original model'),
                   ('b-seeds sealed cohort', 'sealed cohort'), ('4(K+q)', '4(k+q)')],
    'T13b_sealed_verdicts': [('incumbent', 'original'), ('b-seeds sealed cohort', 'sealed cohort')],
    'T09_eq_ladder': [('b-eqtop job', 'EQ-ladder job')],
}


def apply(name: str, text: str) -> str:
    for old, new in SUBS.get(name, []):
        text = text.replace(old, new)
    return text


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    for name in SUBS:
        for p in (here / 'tables' / f'{name}.tex', here / 'tables-md' / f'{name}.md'):
            t = p.read_text(); n = apply(name, t)
            if p.suffix == '.tex':
                for old, new in SUBS[name]:
                    assert old not in n or old in new, (p, old)          # every substitution took effect
            p.write_text(n)
    print('editorial substitutions applied:', ', '.join(SUBS))
