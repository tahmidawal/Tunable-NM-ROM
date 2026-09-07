# Fixed spatial bank: head optimization versus training coverage

This implements the approved `NEXT-ACCURACY-PILOT.md` in the existing heat tree.
All settings and seeds are pinned in `config-head.json`; no results are asserted.
The original checkpoint is a frozen control. Each of the original-cohort and
expanded-cohort arms starts from that same head and runs both paired sampling
seeds. Only head weights, the head's linear skip and training snapshot codes
are optimized. Spatial-network parameters, Fourier lift and boundary/output
scale are checked for exact equality before every checkpoint save.

Let the unchanged bank factor as $G=QR$. For a training field $u_s$, define
$b_s=Q^Tu_s$, $n_s=\|u_s\|^2$ and
$p_s=\|u_s\|^2-\|b_s\|^2$. The minibatch objective is exactly

$$L=\frac{1}{|\mathcal B|}\sum_{s\in\mathcal B}
\frac{\|Rh(z_s)-b_s\|^2+p_s}{n_s}.$$

Loss and parameter-gradient parity against full fields is tested. All matrices
and targets are explicit JIT arguments. The added snapshot codes initialize from
the nearest original decoded **training** snapshot in this same field metric.
No validation field or Gaussian descriptor enters training or the code library.
The validation cohort is first evaluated after all four training endpoints are
saved. Matched update counts, batch sizes, architecture, starting head and paired
sampling seeds separate more optimization from more training coverage. Actual
offline time and every endpoint are retained, not a best validation checkpoint.

Every model then receives the same strict field-fit protocol: nearest training
code starts and the mean training code, with a fixed LM budget and stationarity
tolerance. Initial states and later times are reported separately. All starts,
selected indices, gradients, stop reasons, Jacobian singular values and resulting
fields are saved. The unrestricted bank projection and spatial singular values
are invariant controls. No local optimum is labeled globally optimal.

A refined arm qualifies for rollout only if every validation initial fit is
finite, stationary and below the pinned initial-error threshold. A failed arm
still has its checkpoint and full reconstruction diagnostics saved. If any arm
passes, it and the original frozen control run the verified compiled full query
at both declared solver tolerances, paired with the direct DST FOM in the same
job. The two original meshes are fixed; no larger resolution is added in this
pilot. Compiled/modular field, latent and counter parity is checked anew for
every timed case, model, tolerance and mesh before measurement.

Complete query cost covers host full-field initial input, fitting from both
online starts, weak Crank–Nicolson evolution, all requested dense field outputs
and host output transfer. Raw repetitions and those invocations' fields, errors
and solver statuses are saved. Compression happens between timing blocks, never
between subjects within a block. Fresh reference verification precedes training;
every physical comparison uses one shared observation grid. The reference
refinement estimate is empirical and the rigorous physical bound remains null.

This remains the pinned restricted single-bump family. It does not establish
multi-component heat coverage, independent final confirmation or a speed benefit.

## Plain-language glossary

- **Bank / head / code:** frozen spatial functions / trainable coefficient map /
  compressed coordinate for one training snapshot.
- **QR / norm / perpendicular term:** orthonormal–triangular factorization /
  field magnitude / part of a field outside the frozen bank's span.
- **Minibatch / endpoint / coverage:** sampled training subset / final scheduled
  model / variety of training states.
- **LM / stationarity / singular values:** damped least-squares solve / sufficiently
  small objective gradient / measured strengths of independent matrix directions.
- **Rollout gate / query / parity:** predeclared condition for testing evolution /
  complete input-to-output solve / agreement of fields and solver counters.
- **DST / FOM / empirical bound:** sine transform / direct full-grid solver /
  observed refinement estimate, not a mathematically proved error bound.
