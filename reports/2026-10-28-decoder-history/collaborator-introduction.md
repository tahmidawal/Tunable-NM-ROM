# How our decoders evolved

## A guide for collaborators

We have been studying how to solve a PDE using a small number of unknowns instead of solving for the field at every mesh point. The **decoder** turns those few unknowns into the full solution field. During a reduced solve, we adjust the unknowns so that the decoded field satisfies the governing equations as closely as possible.

The main question throughout this work has been: **which decoder gives an accurate solution that is also inexpensive to compute?** A network that reconstructs saved solutions well is a promising starting point, but it still has to work inside the physics solver.

### The story in five steps

**We started with grid-based tensor decoders.** These assembled fields from learned spatial factors. They were inexpensive to evaluate, but the factors were tied to a particular grid, and accuracy did not reliably improve when the grid was refined.

**We tried networks that take spatial coordinates as inputs.** These could evaluate a solution at new locations without learning a new output grid. They improved several reconstruction tests, but evaluating and differentiating a large network during every solver iteration was costly. Some also produced unreliable solves despite good reconstruction.

**We separated the spatial functions from the small nonlinear network.** The expensive spatial functions could then be evaluated once and reused. A small network, called the head, produced their coefficients. This became the main learned-bank approach.

**We tested whether the nonlinear head was always needed.** We tried different heads, added correction directions, and also solved directly for coefficients in the learned spatial bank. Direct coefficient solving became an important option. Classical data-derived bases remained strong competitors.

**We improved the numerical solver and tested stronger alternatives.** Better integration and time stepping improved accuracy. However, careful comparisons showed that a tuned conventional solver could often reach the same accuracy faster. The latest architecture proposal adds local detail where needed; it remains a hypothesis to test.

### What we can say confidently

The learned coordinate bank can support reduced computations whose cost does not grow with the output mesh in the qualified settings. It also allows evaluation at selected spatial points. Producing a large full-field output still has a cost.

We have **not established a general speed and accuracy advantage over the best conventional methods** for the tested Burgers problems. There are useful results for particular settings, including moving-frame models for translating flows, but they should keep their stated scope.

The most consistent lesson is simple: **a better reconstruction is not necessarily a better PDE solve.** Representation, solver convergence, physical accuracy and total cost must be checked separately.

### How to read the rest

The following sections describe what each architecture changed, why we tried it, what happened, and its status. Blue links open code or supporting evidence directly on GitHub. The code links point to preserved source files at a fixed revision; they do not depend on access to a local workstation. Running an experiment still requires its original configuration, dependencies and checkpoints.

This edition exports the historical review completed on October 28, 2026. It is not a fresh numerical audit. A short update below distinguishes subsequent campaign developments from that frozen review. Earlier results that were corrected or withdrawn remain clearly marked.

**Repository:** [Tunable NM-ROM on GitHub](https://github.com/tahmidawal/Tunable-NM-ROM). **Full code map:** [decoder code and evidence index](https://github.com/tahmidawal/Tunable-NM-ROM/blob/main/reports/2026-10-28-decoder-history/CODE-INDEX.github.md).
