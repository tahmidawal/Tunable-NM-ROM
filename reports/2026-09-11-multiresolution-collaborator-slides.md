# Multiresolution results: collaborator presentation

This LaTeX presentation covers Poisson 2D, heat 2D, Burgers 2D and reflective waves. Its numbers are audited development results; independent final paper confirmation remains open.

- [Compiled presentation](2026-09-11-multiresolution-collaborator-slides.pdf)
- [Editable LaTeX source](2026-09-11-multiresolution-collaborator-slides.tex)
- [Portable source and PDF bundle](2026-09-11-multiresolution-collaborator-slides.zip)
- [Presented numerical records](2026-09-11-multiresolution-collaborator-slides.json)
- [Full underlying results and definitions](2026-09-11-iterative-fom-multiresolution.md)
- [Generator](generate_2026_09_11_collaborator_slides.py)

The deck contains 17 slides, including a compact overview, protocol and decoder context, runtime scaling, individual PDE tables, FOM-tolerance/direct-control comparisons, the reflective runtime breakdown, proposed network/solver tunability studies, provenance and a glossary. Tunability proposals are explicitly separated from measured results. No new numerical experiments were run.

Build the supplied source with its adjacent figure directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error 2026-09-11-multiresolution-collaborator-slides.tex
```

To regenerate every measured number from the repository's accepted JSON records:

```bash
/home/tahmid/Dev/.venv/bin/python reports/generate_2026_09_11_collaborator_slides.py
```

All experiment statistics use the accepted pooled-repetition aggregation. Physical failures, Burgers stalls, wave component failures, exact comparator tolerances and development selection are retained. The portable bundle does not include large raw field archives; their locations and content hashes are recorded in the data extract and full report.

## Glossary

- **LaTeX / Beamer / PDF:** the editable typesetting source / its slide format / the compiled presentation.
- **Generator / JSON / SHA-256:** code rebuilding tables and figures / structured numerical records / a content checksum.
- **ROM / FOM:** reduced model / full spatial-grid solver.
- **Frozen weights / resolution:** trained parameters held unchanged / the number of spatial intervals along each axis.
- **Paired GPU timing / pooled median:** results and time from the same device invocation / the central value across all retained timing repetitions.
- **Development selection / final confirmation:** choosing among methods on previously opened cases / evaluating frozen choices on independent unopened cases.
- **Stationarity / stall / target:** a sufficiently small objective gradient / small progress without that guarantee / the declared physical-error threshold and numerical checks.

The presentation's closing glossary defines every method and table term used on the slides.
