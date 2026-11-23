# Fresh physical phase/shape figures

These are measured outcomes from seed2026112301, with 32 registered
cases and 32 observed case records. Every registered ID is present
in the export and qualification matrix; missing cases are explicit. Full
trajectory qualification passes 32/32. No failed case is filtered out.

The primary curves show pointwise medians and full case ranges, at evolved
times only, using the tighter solver tolerance1e-09. Ordinary and diagnostic
centroid-aligned field errors are both divided by each case's original physical
initial-field norm. A full-cohort statistic is absent where any registered case
has missing/nonfinite data. Initial errors are plotted separately. Physical phase
is the norm of wrapped energy-centroid displacement in unit-domain fractions.
The concentration panel includes each prediction and the reference; its line at
0.1 is a descriptive ambiguity threshold,
not an accuracy or trajectory gate.

The paired panel divides each common-span case's maximum evolved error by the
same head case's maximum evolved error, separately for ordinary and aligned
errors. Ratios above one favor the head. It does not divide timewise medians or
select a favorable time. Native-span initialization is labeled separately.
The separate centroid-displacement pairing applies the same per-case evolved
maximum rule to physical displacement. Red crosses identify a larger head
displacement; all such counterexamples are retained. This diagnostic does not
equate phase displacement with the ordinary or aligned field error.

The reference is the same-grid discrete CNAB2 solution at dt0.000125,
checked at dt6.25e-05, under the unchanged absolute and
fractional gates. This is not a continuum-error claim. Actual initial fits,
operator checks, all rollout stops, tighter-trajectory Jacobian-action checks,
solver refinement, both reference initial fields and actual common time-zero
states are independently qualified. Loose trajectories are not separately
Jacobian-action checked. The 64 actual phase initializer invocations
retain 24 ordinary-direct objective flags in this export, alongside
their separate qualification checks. These are distinct from the standalone
initializer matrix's flags. Neither set is silently relabeled as a pass under
the original ordinary-arithmetic objective comparison.

The head, bank, four starting-code identities and solver method were frozen
before this cohort. The new initializer changes arithmetic and bounded fitting
work, not training. No timing comparison, speed claim, new classical comparator
qualification, tangent-spectrum claim or manifold-distance claim follows from
these figures. Historical failed confirmations remain preserved.
