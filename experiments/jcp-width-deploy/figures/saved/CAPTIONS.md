# Saved Burgers manuscript panels

These plots are deterministic postprocessing of the accepted development results. They introduce no new PDE solve. All are a linear-span, dense discrete-upwind study; they do not establish sparse speed.

**Derivative error through time.** Median and full range across all six development cases for the baseline, wider fixed-test model and wider enriched-test model. The field denominator is the original initial-field norm, whereas derivative and operator denominators are instantaneous corresponding reference norms. Gradient includes the original padded-grid convention. All derivatives are the implemented discrete operators; no analytic/continuum certificate is implied. The Laplacian panel retains errors above unity.

**Casewise derivative maxima.** Each point is one physical case's maximum over all saved times including initialization. Black bars give medians; timing repetitions and time samples are not independent cases. The near-zero-denominator audit and absolute numerator ranges accompany the plot in plotted-data.json.

**Measured width–test map.** Only the five registered width/test cells with a common independent observer are shown; untested cells are grey. Entries are worst casewise temporal maxima, except the initial projection panel and qualification counts. The reported evolution distance is computed from fields relative to the reference's mass projection, not by subtracting scalar errors.

**Transient regressions.** For each existing case, plot the actual field-error change when active width increases at fixed tests. Positive regions retain adverse transient behavior. The native initial projection differs with width; this comparison does not attribute the entire change to time integration alone.

## Glossary

**R′:** active linear-span width. **M:** actual scalar weak tests. **Weak residual:** equation error integrated against test functions. **Independent observer:** unused test rows assessing the residual. **Projection:** the least-squares representation of the reference in a fixed span. **Qualified:** passed registered numerical checks. **Derivative denominator:** norm of the corresponding reference operator at that time. **Percentage point:** absolute difference between two percentage errors. **Case range:** smallest through largest physical-case value.
