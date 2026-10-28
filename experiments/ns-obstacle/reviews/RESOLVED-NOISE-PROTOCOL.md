# Operational definition of resolved reference differences

This prospectively fixes the previously required resolved-noise assessment before interpreting the pending solver/quadrature controls. It does not change physical accuracy limits, the conditional uncertainty formula, or training eligibility. Independent numerical review proposed the rule; the coordinator adopts it as a conditional empirical convention, not a rigorous floating-point or PDE error bound.

For every baseline state used in a spatial/order/geometry contrast, retain measured same-state solver-tightening change $S_y(s)$, quadrature-change $Q_y(s)$ and independent evaluation/replay change $R_y(s)$. Use the same declared output metric, complete physical comparison rule and adopted P4 normalization for each comparison family. A retained independent replay may supply $R$; missing data must remain missing. Define

$$\eta_y(s)=S_y(s)+Q_y(s)+\max(R_y(s),10^{-12}).$$

The fixed floor is a numerical-resolution convention in the dimensionless field/drag metric or absolute transverse-coefficient metric, not a new accuracy target. For a pair of states, define $N=\eta_y(a)+\eta_y(b)$ and the empirical interval

$$[\max(0,d-N),\ d+N].$$

A nonzero difference is resolved when $d>N$. A decreasing h trend is resolved when the earlier interval's lower endpoint exceeds the finest interval's upper endpoint, with the finest difference itself resolved. Retain both raw differences and intervals. Zero/noise-limited geometry may establish no resolved sensitivity only when its upper interval fits the already applicable adequacy budget; zero alone is insufficient. This does not require geometry sensitivity to be nonzero.

The frozen estimate remains exactly

$$\delta_y=2\max(d_{h,\mathrm{fine}},d_p)+d_{\mathrm{solver}}+d_{\mathrm{quad}}.$$

The empirical intervals assess whether the evidence resolves ordering/adequacy; they do not silently replace the constituents or add another geometry margin. Retain original pairwise diagnostics alongside common-normalization versions. The factor two is shared and conditional, not a rigorous independent allowance for P4 geometry interaction.

Create a finite coverage ledger for every actually used baseline state: P4 h1/h2/h3 and P3 g3/g4 h3. Every train/development/validation reference needs its own direct physical, spatial/order and error-budget evidence. Endpoint/Re20 noise measurements must not silently populate other rows. Reuse existing retained controls; do not repeat a valid measurement merely to change its packaging. Qualifying reference truth at validation parameters is not ROM tuning; method choices still freeze before validation use.

The reference owner must provide an explicit compute/disk plan before releasing the population. Keep the global12GPU-hour ceiling and local reserve; preserve recoverable exact evidence during duplicate-only cold retention. If necessary evidence cannot be completed within the bounded campaign, report the cases unresolved rather than weakening thresholds or claiming admission through an unproven coverage assumption. Domain/outlet adequacy remains a separate required control, with common pressure-drop diagnostics and explicit remeshing disclosure.
