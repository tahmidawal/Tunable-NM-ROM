REJECT

The numbers reproduce, but required deliverables and scope annotations are missing. References below are relative to `experiments/jcp-dims/`.

1. **High — EXPORT is incomplete.** `make_report.py:258–261` exports neither 2D A3 verdicts, elasticities and Shapley outcomes nor 3D width-m*, qualification records, front, open candidates and secondary H2q comparisons. `runs/EXPORT.json:3189` also lacks the required `contract` field on costs. These violate `DESIGN.md:396`.

2. **High — The qualified front and unresolved candidates are omitted.** `report.md:707` provides coverage counts and examples, not the registered cost–accuracy front. `checks/analysis-3d.json:21626` contains the front; `:21651` identifies two open W1024 candidates. Neither appears in the report/export, despite `DESIGN.md:307–309` requiring the front and open list.

3. **Medium — Headline speed ratios lack mandatory scope.** `report.md:52–54` quotes H2q without “full-field $65^3$ mesh-path query, H200; not a comparison with the best FOM.” That scope appears later at `:718`, but `DESIGN.md:319` requires it on every 3D speed claim.

4. **Medium — Provisional status appears only in the preamble.** `report.md:3` supplies the conditional-Richardson caveat, but accuracy-dependent verdicts and H2q tables lack adjacent provisional labels. The error plot likewise omits this caveat (`make_report.py:164–165`). Repository rules explicitly require inline qualification.

5. **Medium — Registered plots are missing.** `make_report.py:148–185` produces only two curves. `DESIGN.md:397–399` also promises factorial heatmaps, family error curves with draw ranges, curves by form, and qualified-front/H2q panels. Unavailable Shapley bars should be explicitly marked unavailable. The report contains no plot links.

6. **Medium — Glossary is incomplete.** `report.md:835–885` does not define reference A/cost normalization, identity, usable, censored, $\rho$, q+, fs/qn, BE/CN/BDF2, or configuration-key components. Thus several tables and the cost plot cannot be interpreted independently.

7. **Low — Numerical literals and math formatting violate report rules.** `make_report.py:295` hard-codes the omitted-configuration count; `:349–350` hard-codes qualification thresholds rather than sourcing JSON. `report.md:599` prints elasticity equations outside LaTeX; `:873` similarly uses an unformatted frequency product.

**Verified:** report and EXPORT reproduce byte-for-byte through intercepted, in-memory generation; both PNGs reproduce pixel-for-pixel. All 21 tables have consistent column counts and escaped pipes. Ledger hours sum correctly. Displayed verdicts match the analysis JSONs; B6 remains blocked, C_W1024 is labelled unqualified, and no prohibited historical comparison appears.

COMMON-BRIEF §10 hand-back completeness remains unverified: no coordinator hand-back was supplied. No files were changed or arrays copied.