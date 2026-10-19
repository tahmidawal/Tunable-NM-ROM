REJECT

Paths below are relative to `experiments/jcp-dims/`.

1. **Medium — Cost contracts remain incomplete.** `make_report.py:256` copies `h2q_Cq` without contracts. All three secondary results contain `S` and `S_screen` without a `contract` field (`runs/EXPORT.json:17882`). The other requested export sections are now present and match their analysis sources.

2. **High — The front plot misidentifies its plotted population.** `make_report.py:343` includes the unqualified `W1024|lat65536|fs|BE|0.01|K1` as an ordinary coloured point under “qualified configurations.” Its width filter also omits **18 screened configurations**, including κ-panel and fixed-test-list configurations. Filter by qualification and group these configurations by actual bank width; show the unqualified comparator separately and label it.

3. **Medium — Registered plots remain incomplete or misleading.** `make_report.py:277` generates heat maps only at gauss256, whereas `DESIGN.md:397` requires them **per rung**. The cost heat map labels blanks “not run” (`make_report.py:287`), although some represent executed but unusable timings. In `plots/h2q_3d.png`, the pass-bar annotation visibly overlaps the title (`make_report.py:365`). All ten PNGs were viewed.

4. **Medium — Inline provisional labels remain missing.** The A3 headlines (`report.md:44`), LMF accuracy verdict (`report.md:619`), primary/secondary H2q tables (`report.md:743`), and 3D accuracy-dependent transitions (`report.md:760`) lack adjacent E0-provisional qualification. The H2q figure also lacks it.

5. **Medium — Scope is not carried on every 3D speed presentation.** Headlines are fixed, but the H2q figure’s title omits “mesh-path,” and its caption omits the required scope altogether (`make_report.py:369`, `report.md:896`). Include the full scope beside the secondary comparison as well (`report.md:751`).

6. **Medium — Glossary remains incomplete.** `report.md:905` onward does not explain screening versus confirmation, median/worst front membership, open candidates, resolution, `fsp`, or the rule-key encodings (`gl`, `lat`, `fib`, seed suffixes). Reference A is described generically without identifying its actual 2D configuration. Several table fields therefore still require external documentation.

7. **Low — Literal-number and LaTeX requirements are only partly fixed.** Thresholds now use `DSTATS` constants, but prose still hard-codes the H2q threshold (`make_report.py:529`) and numerical experiment settings (`:145`, `:246`). Mathematical expressions remain outside LaTeX, including `65^3` in headlines (`report.md:52`) and `2 ε_ref`, `M = 4r`, and `≥ 1/0.6` in the glossary.

**Verified fixes:** the front table and two open candidates are present; Shapley bars are explicitly unavailable; family plots show draw ranges; κ-form and width-m* plots exist. Generated report sections and exported sections match their sources in memory. Checked results-audit-2 values remain consistent: **275/299 usable, 208 qualified**, cost **1.012615667304**, elasticities **0.653726/0.527682/0.653404**, primary H2q **1.149138/2.952802/9.874135**, and secondary **1.145831/2.929196/9.670805**.

Read-only throughout; no files changed, submissions made, arrays copied, or scratch files created.