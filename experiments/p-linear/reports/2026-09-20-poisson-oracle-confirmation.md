# Corrected Poisson augmented representation fits

This supplement confirms the query-metric projector repair on the complete original development cohort. The corrected untimed fits have passed an independent NumPy field and gradient audit; historical timed results remain unchanged.

Source `5a8d7bc8dea9fcdfcdc74de197541ccdec4a886e`; job `3996320`; GPU `NVIDIA A100 80GB PCIe`. Audit checked 240 case/rank combinations. Maximum independent decoded-field error discrepancy: 1.005258e-10.

Retained evidence: [complete raw-result archive](../artifacts/plorc01/README.md), [independent local audit](../artifacts/plorc01/audit.json), [source audit](../artifacts/plorc01/provenance-audit.json), [archive restore proof](../artifacts/plorc01/restore-audit.json), and [result/audit hashes and machine-readable tables](oracle-confirmation.json).

The historical nonzero-rank oracle values remain retracted under DESIGN A10. The corrected values below come from a new run and have their own provenance. The original nearest-code starts and the additional safeguards are reported separately. Additional starts use previously retained online solutions at the same rank and the preceding rank’s best fit. These are truth-informed representation diagnostics, with no inference-time cost or generalization claim.

For a query-mesh QR factor $R_G$ and an orthonormal basis $Q_q$ for $\operatorname{span}(R_G C_q)$, the fitted residual is

$$r_q(z) = (I-Q_qQ_q^\top)(R_Gh_\theta(z)-T).$$

The finite best-found value is an upper bound on the unknown global minimum representation error. The free-bank floor is a lower bound. At $q=R$, a direct free-coefficient projection returns that floor, with the redundant nonlinear head removed; the original-starts column is therefore marked direct at that endpoint.

## Frozen checkpoint `new_K32`

| Intervals | Correction rank | Original starts, worst (%) | Safeguarded best found, worst (%) | Median (%) | Bank floor, worst (%) | Stationary / cases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0 | 3.121122 | 3.121122 | 0.505002 | 0.745860 | 12 / 12 |
| 256 | 32 | 2.455447 | 2.455447 | 0.442024 | 0.745860 | 12 / 12 |
| 256 | 64 | 2.078654 | 2.078654 | 0.412625 | 0.745860 | 12 / 12 |
| 256 | 128 | 1.549105 | 1.549105 | 0.319424 | 0.745860 | 12 / 12 |
| 256 | 256 | 0.968819 | 0.968819 | 0.131488 | 0.745860 | 12 / 12 |
| 256 | 512 | Direct endpoint | 0.745860 | 0.058037 | 0.745860 | 12 / 12 |
| 1024 | 0 | 3.113946 | 3.113946 | 0.504893 | 0.742127 | 12 / 12 |
| 1024 | 32 | 2.449682 | 2.449682 | 0.442061 | 0.742127 | 12 / 12 |
| 1024 | 64 | 2.073797 | 2.073797 | 0.412685 | 0.742127 | 12 / 12 |
| 1024 | 128 | 1.545275 | 1.545275 | 0.319436 | 0.742127 | 12 / 12 |
| 1024 | 256 | 0.964757 | 0.964757 | 0.131455 | 0.742127 | 12 / 12 |
| 1024 | 512 | Direct endpoint | 0.742127 | 0.058010 | 0.742127 | 12 / 12 |

## Frozen checkpoint `incumbent`

| Intervals | Correction rank | Original starts, worst (%) | Safeguarded best found, worst (%) | Median (%) | Bank floor, worst (%) | Stationary / cases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0 | 6.101346 | 6.101346 | 1.242299 | 2.322648 | 12 / 12 |
| 256 | 32 | 4.671509 | 4.671509 | 0.936493 | 2.322648 | 12 / 12 |
| 256 | 64 | 3.433584 | 3.433584 | 0.786310 | 2.322648 | 12 / 12 |
| 256 | 128 | Direct endpoint | 2.322648 | 0.368446 | 2.322648 | 12 / 12 |
| 1024 | 0 | 6.093048 | 6.093048 | 1.242114 | 2.315180 | 12 / 12 |
| 1024 | 32 | 4.664376 | 4.664376 | 0.936174 | 2.315180 | 12 / 12 |
| 1024 | 64 | 3.427089 | 3.427089 | 0.786022 | 2.315180 | 12 / 12 |
| 1024 | 128 | Direct endpoint | 2.315180 | 0.368161 | 2.315180 | 12 / 12 |

## Retraction history

| Intervals | Checkpoint | Rank | Historical value (%) | Historical status | Corrected value (%) |
| ---: | --- | ---: | ---: | --- | ---: |
| 256 | `new_K32` | 32 | 2.455494 | Retracted: unnormalized query projector | 2.455447 |
| 256 | `new_K32` | 64 | 2.078735 | Retracted: unnormalized query projector | 2.078654 |
| 256 | `new_K32` | 128 | 1.549257 | Retracted: unnormalized query projector | 1.549105 |
| 256 | `new_K32` | 256 | 0.969138 | Retracted: unnormalized query projector | 0.968819 |
| 256 | `new_K32` | 512 | 0.746241 | Retracted: unnormalized query projector | 0.745860 |
| 1024 | `new_K32` | 32 | 24.224570 | Retracted: unnormalized query projector | 2.449682 |
| 1024 | `new_K32` | 64 | 32.320667 | Retracted: unnormalized query projector | 2.073797 |
| 1024 | `new_K32` | 128 | 38.148244 | Retracted: unnormalized query projector | 1.545275 |
| 1024 | `new_K32` | 256 | 43.769427 | Retracted: unnormalized query projector | 0.964757 |
| 1024 | `new_K32` | 512 | 45.749621 | Retracted: unnormalized query projector | 0.742127 |
| 256 | `incumbent` | 32 | 4.671612 | Retracted: unnormalized query projector | 4.671509 |
| 256 | `incumbent` | 64 | 3.433815 | Retracted: unnormalized query projector | 3.433584 |
| 256 | `incumbent` | 128 | 2.323071 | Retracted: unnormalized query projector | 2.322648 |
| 1024 | `incumbent` | 32 | 50.830796 | Retracted: unnormalized query projector | 4.664376 |
| 1024 | `incumbent` | 64 | 74.178551 | Retracted: unnormalized query projector | 3.427089 |
| 1024 | `incumbent` | 128 | 85.280996 | Retracted: unnormalized query projector | 2.315180 |

## Glossary

- **Intervals**: subdivisions along each side of the square spatial mesh.
- **Checkpoint**: frozen neural bank and latent-to-coefficient head parameters.
- **Rank $R$**: number of learned bank functions; **latent dimension $K$**: number of head inputs.
- **Correction rank $q$**: number of additional nested linear bank directions allowed during fitting.
- **Worst / median**: largest / middle relative Euclidean field error over the original development sources, normalized by the same-grid exact finite-difference solution.
- **Original starts**: the unchanged nearest-training-code multistart prescription, with the corrected projector.
- **Safeguarded best found**: smallest achieved objective after adding feasible starts from retained solves and the previous correction rank; a finite optimizer result, not a proven global minimum.
- **Bank floor**: the smallest field error obtainable with unrestricted coefficients in the learned linear bank.
- **Stationary / cases**: cases meeting the recorded normalized-gradient stopping criterion, out of all evaluated sources; the direct full-bank endpoint is stationary by construction.
- **Query metric / QR factor**: the Euclidean field geometry induced by evaluating bank functions at the query mesh, represented by a triangular matrix.
- **Projector**: an operation that removes the component along the permitted linear correction directions.
- **Development cohort**: already opened source cases; these data do not form a new sealed final test.
- **Independent audit**: separate NumPy/SciPy equations checking saved fields, objectives, derivatives, summaries and numerical provenance.
