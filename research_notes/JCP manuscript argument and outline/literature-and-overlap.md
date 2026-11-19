# Literature and previous-manuscript placement notes

Prepared November 19, 2026. This preserves the focused research memo returned by `/root/jcp_literature` for the manuscript-outline task, condensed for source tracing. Publication metadata was checked against primary records; most detailed analyses were inspected in author versions, not independently reproduced.

The recommended consolidated-extension thesis is a controlled explanation of approximation, fixed weak-residual dynamics, tuning, integration and complete-query cost. The bank/head architecture and several application/scaling results are already in the conference source. Their reuse needs attribution and an explicit account of the additional journal evidence; a changed title or redrawn plot is not a new contribution. Current journal/conference submission eligibility was not rechecked for this outline.

## Published precedents

| Publication | Verified primary source | Contribution relevant to this outline |
|---|---|---|
| Lee and Carlberg, JCP 404, 108973 (2020) | https://doi.org/10.1016/j.jcp.2019.108973 ; https://arxiv.org/html/1812.08373v3 | Manifold Galerkin/LSPG properties and discrete error bounds; projection versus evolved error. Author sections 4 and 7 motivate precise assumptions and approximation/dynamics separation. |
| Kim, Choi, Widemann and Zohdi, JCP 451, 110841 (2022) | https://doi.org/10.1016/j.jcp.2021.110841 ; https://arxiv.org/html/2009.11990 | Sampled decoder/Jacobian work, stencil dependencies, dimension/sampling/data studies and hyper-reduced accuracy. Small latent dimension alone does not establish speed. |
| Barnett and Farhat, JCP 464, 111348 (2022) | https://doi.org/10.1016/j.jcp.2022.111348 ; https://arxiv.org/html/2204.02462 | Quadratic manifolds supply a credible non-neural nonlinear comparator, including physical flow outputs and offline/online resources. |
| Barnett, Farhat and Maday, JCP 492, 112420 (2023) | https://doi.org/10.1016/j.jcp.2023.112420 ; https://www.sciencedirect.com/science/article/pii/S0021999123005156 ; https://arxiv.org/html/2212.08939 | PROM-ANN combines a nonlinear correction with an outer basis and LSPG/ECSW. The inspected arXiv version is an AIAA SciTech precursor, so its section numbers and numerical outcomes must not be relabeled verified final-journal details. |
| Romor, Stabile and Rozza, JCP 524, 113729 (2025) | https://doi.org/10.1016/j.jcp.2025.113729 ; https://arxiv.org/html/2308.03396 | Reconstruction versus evolution, sampling, generalization and cost regimes. Nonlinear coefficient compression in an outer space is established precedent. |
| Puri, Prakash, Kara and Zhang, JCP 532, 113957 (2025) | https://doi.org/10.1016/j.jcp.2025.113957 ; https://arxiv.org/html/2405.14890v2 | SNF-ROM establishes smooth neural-field projection-based dynamics, derivative considerations, and coupled time/sampling effects. The inspected author version predates journal publication. |
| Chang, Chen, Wang, Chiaramonte, Carlberg and Grinspun, SIGGRAPH Asia Conference Papers (2023) | https://doi.org/10.1145/3610548.3618158 ; https://arxiv.org/html/2310.15907 ; https://dgp.toronto.edu/~changyue/LiCROM/ | LiCROM establishes continuous neural basis fields, separate cubature and cached sampled basis values. This is a conference paper, not a journal paper. |
| Hernández, Caicedo and Ferrer, CMAME (2017) | https://doi.org/10.1016/j.cma.2016.10.022 ; https://www.sciencedirect.com/science/article/pii/S004578251631355X | Empirical cubature supplies fitted integration sites and positive weights. Metadata, abstract and section excerpts were accessible; a later full-page request returned 403. |

The closest combined precedent is LiCROM for the bank, PROM-ANN/Romor for nonlinear coefficient maps, Lee–Carlberg for manifold projection, and empirical cubature/ECSW for sampled evaluation. SNF-ROM is a particularly close JCP neural-field reference. Unsupported priority language includes “first continuous NM-ROM,” “first bank-times-head decoder,” and “first hyper-reduced neural field.” The paper should state its controlled numerical findings without those claims.

## Own-manuscript overlap

Actual source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex`.

The research memo verified methodology around lines 311–582, including bank/head, boundary lift, ordering, fixed tests, preassembly and quadrature. The harder Navier–Stokes and cylinder results begin around line 640; the box problems/comparators and tunability discussion begin around line 767. The off-mesh scaling/output figure `fig:meshflat` occurs around line 2765. The ordering appendix is around line 1363. Line numbers are navigation aids for this saved version, not permanent scholarly citations.

The primary outline uses those elements as inherited background or scoped supporting evidence. Newly completed matched learned-bank head dynamics, weak-test/width analysis, and complete-query cost form the main additional evidence. If a distinct companion is necessary, restrict it to classical quadrature for nonlinear advection, target-operator attribution, rule-size qualification, storage/contraction and complete-cost limits; retain the broad head/application story with the original article.

## Glossary

**Primary source** is the publisher or author record. **Author version** is a public manuscript that can differ from the final article. **LSPG** is least-squares Petrov–Galerkin projection. **ECSW** is an energy-conserving sampling and weighting hyper-reduction method. **Consolidated extension** means a journal article that explicitly builds on the earlier framework and identifies its additional findings.
