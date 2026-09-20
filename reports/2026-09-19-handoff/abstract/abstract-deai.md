# Title and abstract, de-machined

Editing pass only. Every claim and every number is the original's; nothing added, dropped,
softened, or sharpened.

## Title options (no colon-and-subtitle, each 12 words or fewer)

1. **One trained decoder, many operating points, and where the nonlinear manifold fails** (12 words)
2. **A rank-q knob for nonlinear-manifold ROMs, and when linear POD still wins** (12 words)
3. **Tuning accuracy against cost after training, and the limits of nonlinear-manifold ROMs** (12 words)

---

## Version A (222 words)

A neural operator gives one accuracy-and-speed point per trained model. A different
evaluation grid at deployment changes cost; what the model can represent is unchanged. Our
nonlinear-manifold reduced-order model reaches many operating points from one trained
decoder, a coordinate-network bank times a small head. The solver may switch on a rank-q
correction in the bank's coefficient space, and q is the primary knob; the iteration cap,
tolerance and quadrature count are secondary. The residual is a matrix-free least-squares
Petrov–Galerkin projection whose empirical quadrature we validate on the states the solver
actually reaches, not by its fitting residual. On 2D Burgers the ladder in q is monotone
across three training seeds and a sealed cohort opened once, and meets a bar fixed before
any run. All comparisons are timed in one job. Against a tuned full-order solver, nothing
reduced is on the frontier at 256² or 512². On an L-shaped Poisson domain, where no fast
transform exists, the reduced model is 2.8× and 6.1× cheaper than the cheapest full-order
solve at 256² and 512²; plain POD is cheaper still, for 15% more error. Neural operators
trained on the same data are more accurate. On Poisson, heat and waves the family collapses
to a linear model, and on Navier–Stokes the learned manifold never beats linear POD at
matched dimension. Those failures set the conditions for choosing a nonlinear manifold.

*Changed:* "moves cost but not what the model can represent" → "changes cost; what the model
can represent is unchanged" (kills the antithesis, keeps both halves of the claim);
"exposes a family of operating points" → "reaches many operating points" (verb instead of
abstract noun); the head clause split off as an appositive so the knob sentence carries only
knobs; "with … as secondary knobs" → "are secondary" (drops the second use of "knobs");
"Every comparison is timed in the same job" → "All comparisons are timed in one job"
(shorter, breaks the uniform sentence length); "These negatives state the condition under
which the nonlinear manifold is worth having" → "Those failures set the conditions for
choosing a nonlinear manifold" (loses "worth having" and the summing-up cadence). *Note:*
222 words, not 180–200; every further cut I tried removed a claim, and the claims were
non-negotiable.

---

## Version B (240 words)

A neural operator gives one accuracy-and-speed point per trained model. Evaluating it on a
different grid at deployment changes what a solve costs; what the model can represent is
unchanged. We present a nonlinear-manifold reduced-order model that reaches many operating
points from a single trained decoder, a coordinate-network bank times a small head. The
solver may switch on a rank-q correction in the bank's coefficient space, and q is the
primary knob; the iteration cap, tolerance and quadrature count are secondary. The residual
is a matrix-free least-squares Petrov–Galerkin projection, and we validate its empirical
quadrature on the states the solver actually reaches, not by its fitting residual. On 2D
Burgers the ladder in q is monotone on three training seeds and on a sealed cohort opened
once, and meets a bar fixed before any run. Every comparison is timed in the same job.
Against a tuned full-order solver nothing reduced is on the frontier at 256² or 512². On an
L-shaped Poisson domain, where no fast transform exists, the reduced model is 2.8× and 6.1×
cheaper than the cheapest full-order solve at 256² and 512²; plain POD is cheaper still, for
15% more error. Neural operators trained on the same data are more accurate. On Poisson,
heat and waves the family collapses to a linear model, and on Navier–Stokes the learned
manifold never beats linear POD at matched dimension. Those failures fix the conditions for
choosing a nonlinear manifold.

*Changed:* opening semicolon split into two sentences so the first is short and the second
long; "moves cost but not what the model can represent" → "changes what a solve costs; what
the model can represent is unchanged"; "exposes a family of operating points" → "reaches
many operating points"; the three-clause decoder sentence broken so "coordinate-network bank
times a small head" sits as an appositive and the rank-q clause stands on its own; "with the
iteration cap, tolerance, and quadrature count as secondary knobs" → "the iteration cap,
tolerance and quadrature count are secondary"; the long Poisson clause repunctuated with a
semicolon so "plain POD is cheaper still" is a statement rather than a trailing concession;
final sentence reworded off "negatives … worth having".

---

## Version C (247 words, original is 242)

A neural operator gives one accuracy-and-speed point per trained model. Changing the
evaluation grid at deployment changes what a solve costs; what the model can represent stays
the same. We present a nonlinear-manifold reduced-order model that reaches a family of
operating points from a single trained decoder. The decoder is a coordinate-network bank
times a small head, and the solver may switch on a rank-q correction in the bank's
coefficient space. q is the primary knob, with the iteration cap, tolerance and quadrature
count as secondary knobs. The residual is a matrix-free least-squares Petrov–Galerkin
projection, and we validate its empirical quadrature on the states the solver actually
reaches, not by its fitting residual. On 2D Burgers the ladder in q is monotone on three
training seeds and on a sealed cohort opened once, and meets a bar fixed before any run.
Every comparison is timed in the same job. Against a tuned full-order solver, nothing reduced
is on the frontier at 256² or 512². On an L-shaped Poisson domain, where no fast transform
exists, the reduced model is 2.8× and 6.1× cheaper than the cheapest full-order solve at 256²
and 512², though plain POD is cheaper still for 15% more error. Neural operators trained on
the same data are more accurate. On Poisson, heat and waves the family collapses to a linear
model, and on Navier–Stokes the learned manifold never beats linear POD at matched dimension.
Those failures set the conditions for choosing a nonlinear manifold.

*Changed:* only four places — the grid clause ("moves cost but not what the model can
represent" → "changes what a solve costs; what the model can represent stays the same"),
"exposes a family of operating points" → "reaches a family of operating points", the
40-word decoder-plus-knobs sentence split after "coefficient space" so q starts its own
sentence, "The residual is … with empirical quadrature validated on" → "…, and we validate
its empirical quadrature on" (a person doing the validating), and the closing sentence
("These negatives state the condition under which the nonlinear manifold is worth having" →
"Those failures set the conditions for choosing a nonlinear manifold"). Everything else,
including "though plain POD is cheaper still for 15% more error", is the original wording.
