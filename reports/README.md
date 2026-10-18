# Reports

Written-up results. One report per question, never one per session.

**Naming:** `YYYY-MM-DD-<what-it-contains>.md`, where the date is when the *numbers* were
finalised, not when the prose was edited. The slug says what the report is about, so the
filename alone tells you whether to open it.

**Every report opens with** a level-1 title, then one or two sentences saying what it covers and
what state its numbers are in — final, provisional, or superseded.

**Numbers are generated, never hand-typed.** The scripts here read the run JSONs directly:

```
make_report_figs.py   figures for the illustrated four-PDE report
make_cost_figs.py     cost(k), the frontier tables, the hybrid tables
make_eq_fig.py        error and cost against quadrature-point count
make_talk_figs.py     slide-sized figures with takeaway titles
build_report.py       inlines figures as data URIs -> *.built.md for publishing
```

`*.built.md` files are generated output — edit the source, not the build.

Current state and the full history are in `../LAB-LOG.md`, not here.
