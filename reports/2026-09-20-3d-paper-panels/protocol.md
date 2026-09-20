# Experimental setup and operator training for the 3D panels

These generated tables describe the selected audited development panels. They are provisional; final testing and remaining tuning are tracked in the canonical lab log.

The original run configuration and saved training record supply every numeric cell. Training records may be reused from earlier attempts; this table does not claim that training occurred inside the query-timing allocation or that different architectures received equal compute.

| PDE | Training cases | Fields / training case | Outputs / query | Components | Training grid | Bank R | Heads K | Weak tests M | Correction ranks q |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| Burgers 3D | 512 | 8 | 51 | 1 | 33 nodes per axis | 256 | 32 | 642 | 0, 64, 128, 192 |
| Heat 3D | 128 | 6 | 6 | 1 | 32 intervals per axis | 128 | 8, 16 | 512 | 0, 8, 32, 64, 96 |
| Poisson 3D | 512 | 1 | 1 | 1 | 32 intervals per axis | 128 | 8, 16 | 512 | 0, 16, 32, 64, 96 |
| Navier–Stokes 3D | 512 | 6 | 6 | 3 | 32 periodic points per axis | 512 | 16 | 1024 | 0, 16, 32, 64, 128 |

A field means a complete spatial state; a vector state includes all velocity components. Time-zero fields are included in the counts when requested. Operators may pass through the supplied initial field and interpolate trained output times, as specified by the corresponding panel. The counts describe training membership, not statistically independent state samples.

| PDE | Operator | Parameters | Requested updates | Completed updates | Selected update | Batch | Initial learning rate | Training seed | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Burgers 3D | `fno3d` | 1771431 | 20000 | 20000 | 16100 | 2 | 0.001 | 920310 | steps |
| Burgers 3D | `unet3d` | 89431 | 20000 | 20000 | 17500 | 2 | 0.001 | 920311 | steps |
| Heat 3D | `fno3d_w16_m6` | 1771349 | 20000 | 20000 | 19500 | 4 | 0.001 | 920321 | steps |
| Heat 3D | `unet3d_w8` | 89197 | 20000 | 20000 | 20000 | 4 | 0.001 | 920322 | steps |
| Poisson 3D | `fno3d_w24_m8` | 9440953 | 30000 | 30000 | 26750 | 2 | 0.0005 | 920450 | steps |
| Poisson 3D | `unet3d_w16` | 354577 | 30000 | 30000 | 28750 | 2 | 0.0005 | 920451 | steps |
| Poisson 3D | `deeponet3d_r128_w16` | 791697 | 30000 | 30000 | 29000 | 2 | 0.001 | 920452 | steps |
| Poisson 3D | `transolver3d_w48_s32` | 561272 | 30000 | 30000 | 29750 | 2 | 0.001 | 920453 | steps |
| Navier–Stokes 3D | `fno3d` | 4196511 | 8000 | 8000 | 7100 | 2 | 0.001 | 202609205 | steps |
| Navier–Stokes 3D | `unet3d` | 89287 | 8000 | 8000 | 7900 | 2 | 0.001 | 202609206 | steps |

Checkpoint selection uses the recorded development criterion. Reaching an update or elapsed-time budget is not evidence that training converged. The adjacent JSON retains architecture settings, normalization, original training metadata and source hashes.

## Glossary

- **PDE / grid:** differential equation / number of spatial nodes or intervals along each axis, using the stated convention.
- **Training case / field / output:** one sampled physical input / one spatial state / one requested state returned by a query.
- **Components:** scalar channels in a physical field; velocity is vector valued.
- **Bank R / head K / weak tests M / correction q:** learned spatial basis size / nonlinear code size / number of smooth residual tests / added linear correction directions.
- **Operator / parameters:** a trained solution-map model / its recorded trainable parameter count.
- **Requested / completed / selected updates:** configured optimizer budget / actual updates executed / update supplying the retained validation-selected checkpoint.
- **Batch / initial learning rate / seed:** examples per optimizer update / starting step-size setting before its recorded schedule / reproducible random initializer.
- **Exit:** recorded reason training stopped; an update limit or wall-time limit is not a convergence certificate.
- **Development / provisional:** inputs available during selection / evidence still awaiting final confirmation.
