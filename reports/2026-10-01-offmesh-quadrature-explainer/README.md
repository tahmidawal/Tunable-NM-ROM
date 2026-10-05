# Off-mesh quadrature explainer video

An animated (Manim) walk-through of Hari's off-mesh quadrature and what it changed for our
Burgers 2D and 3D reduced models. Status: **final**. It draws on the numbers in
`reports/2026-10-01-burgers3d-offmesh-quadrature.md` and
`reports/2026-10-01-burgers2d-offmesh-quadrature.md`, which are both final.

| file | what it is |
|---|---|
| `offmesh.py` | the scenes (S0 intro … S8 summary) |
| `vidnums.py` | every number shown on screen, each with the verbatim report snippet it comes from; `check()` fails the build if a snippet or value is missing from its report |
| `hari_quadrature.py` | unchanged copy of `nmrom/quadrature.py` from Hari's package (`quadrature-study-2026-09-30.tar.gz`), used to draw the point sets |
| `build.sh` | renders every scene at 1080p30 and joins them into `offmesh-explainer.mp4` |

The bump fields, bank outputs and probe readouts in scenes 1 and 3 are illustrations, not
model output, and are labelled that way on screen. Every plotted or quoted number comes
from `vidnums.py`.

## Rebuilding

Manim's text rendering needs Cairo/Pango headers, which the GB10 system Python lacks.
Build it from conda-forge in a throwaway prefix (this is not a JAX environment):

```bash
curl -fsSL https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj bin/micromamba
MAMBA_ROOT_PREFIX=$PWD/mroot ./bin/micromamba create -y -p $PWD/menv -c conda-forge python=3.12 manim ffmpeg
```

```bash
MANIM_ENV=$PWD/menv sh build.sh
```

LaTeX (`latex` and `dvisvgm`) comes from the system TeX install.

## Glossary

- **ROM / reduced model**: solves for a few hundred coefficients instead of one value per mesh node.
- **bank (G)**: the frozen coordinate network. It maps a point $x$ to $R'$ numbers, and the field is $u(x)=G(x)c$.
- **$R'$**: the number of bank outputs the solve uses (512 or 256 here).
- **tests $\psi_{ab}$**: sine functions $\sin(a\pi x)\sin(b\pi y)$ that the residual is projected onto. There are $M\approx 4R'$ of them.
- **tested advection $N(c)$**: $\int\psi_{ab}\,u\,(u_x+u_y)\,dx$, the nonlinear term every iteration must evaluate.
- **mesh sum / upwind**: the full solver's discrete version of that integral. It is first-order accurate, an $O(h)$ error.
- **sub-lattice, EQ, tensor**: our three earlier ways to evaluate $N(c)$ cheaply. All three reproduce the mesh sum.
- **off-mesh rule**: Hari's approach. A fixed classical quadrature rule (Gauss, Fibonacci/CBC lattice, Sobol or Smolyak) at points chosen without reference to the mesh.
- **$\rho$**: the relative error of a rule's $N(c)$ against a target (the continuum integral or the mesh sum), worst case over reached states.
- **certificate bar 0.116**: the $\rho$ threshold the project uses to accept a quadrature rule.
- **refined reference**: a full-order solve on a much finer grid (513³ in 3D, 8192² in 2D), used as ground truth. It is itself first-order accurate.
- **held-out / test cases**: initial conditions that played no part in choosing settings.
- **pp**: percentage points.
- **speedup (same-mesh rule / matched to refined accuracy)**: full-solver time divided by ROM time. The full solver uses either the setting that matches the ROM's accuracy on the same mesh, or the one that matches it against the refined reference.
