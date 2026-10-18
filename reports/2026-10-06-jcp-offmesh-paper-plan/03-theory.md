# 03 — Theory: what error analysis the JCP paper can rigorously contain

Planning note (agent 3 of 5, 2026-10-06). It sets out the statements the paper can prove, the ones that are only heuristic, and the numerical checks that each statement needs. Status: a draft for planning. Every number quoted here comes from the two lane reports (`reports/2026-10-01-burgers{2d,3d}-offmesh-quadrature.md`) or from Hari's `SUMMARY.md`/`DECODERS.md`. They are used only to sanity-check the predicted rates. None of them is a new measurement.

## 0. Setting and notation

The notation follows the discarded manuscript (`paper_latex/main.tex`). Domain $\Omega=[0,1]^d$ with $d\in\{2,3\}$, $L$ intervals per axis, $h=1/L$ and $N=L^d$ nodes. The bank (eq:bank, eq:coordnet) is
$$\tilde u(x)=\widehat G_{R'}(x)\,c,\qquad \widehat G(x)=\mu(x)\,g_\phi(x)^{\mathsf T}T,\qquad g_\phi=\mathrm{MLP}_\phi\big(\gamma(x)\big),\quad \gamma(x)=[\cos 2\pi\Omega x,\ \sin 2\pi\Omega x],$$
with SiLU activations and the polynomial Dirichlet factor $\mu=4^d\prod_j x_j(1-x_j)$. The tests are $P=\Phi^{\mathsf T}$, the $M$ lowest sines, with continuum versions $\psi_a(x)=2^{d/2}\prod_j\sin(a_j\pi x_j)$. The span uses $c\in\mathbb R^{R'}$. With the head, replace $c$ by $h(z)$ and $\widehat G_{R'}$ by $J_{\mathcal D}$. One backward-Euler step solves
$$r(c)=D\big(Ac-p+\Delta t\,(N(c)+\nu\Lambda Ac)\big)=Ac-Dp+\Delta t\,D\,N(c),\qquad D=(I+\Delta t\nu\Lambda)^{-1},\ A=\Phi^{\mathsf T}\widehat G'_{R'}. \tag{0.1}$$
The second form uses $D(I+\Delta t\nu\Lambda)=I$, and it is the identity that makes the analysis below clean. I write $\mathcal A(u)$ for the manuscript's advection operator, to avoid a clash with the matrix $A$. The three tested advections are
- the continuum one, $N_a(c)=L^{d/2}\int_\Omega \psi_a\,f$ with $f=u\,(\mathbf 1\cdot\nabla u)$;
- a rule $(x_q,w_q)_{q\le m}$, giving $N_{m,a}(c)=L^{d/2}\sum_q w_q\psi_a(x_q)f(x_q)$;
- the mesh stencil, $N_{h}(c)=\Phi^{\mathsf T}\mathcal A_h(u)$ (eq:upwind). EQ and the mesh lattice approximate this one.

$\rho$ is eq:rho, the relative error of a tested advection against a stated target. Throughout, $a_{\max}$ denotes the largest per-axis test frequency. If the $M$ tests are ordered by eigenvalue, $a_{\max}\approx(4M/\pi)^{1/2}$ in 2D and $(6M/\pi)^{1/3}$ in 3D: about 51 for $M=2048$ in 2D and about 16 for $M=2048$ in 3D.

**Standing assumptions.** (S1) The activations are analytic: SiLU, sine and Gaussian qualify, ReLU does not. (S2) $\mu$ is a polynomial vanishing to first order on $\partial\Omega$; the L-shape and other non-box masks are *not* covered. (S3) The reached coefficients lie in a bounded set $\mathcal C$. All the quadrature statements hold uniformly for $c\in\mathcal C$, and they also hold for $\partial N/\partial c$, because the Jacobian integrand $\psi_a(\hat g_j\,\mathbf 1\cdot\nabla \hat g_k+\hat g_k\,\mathbf 1\cdot\nabla\hat g_j)$ lies in the same function class as $f$.

---

## 1. Quadrature error of the tested advection

### Lemma 1.1 (analyticity strip of the bank). *Provable, but pessimistic.*
Fix an axis $k$ and complexify only $x_k\mapsto x_k+i y$ with $|y|\le\eta$, keeping the other coordinates real. Let $\omega_k=\max_j|\Omega_{jk}|$, and let $W_\ell$ and $B_\ell$ be the layer weights and the bounds on the real pre-activations over $\Omega$. Then $g_\phi$, and therefore $\tilde u$ and $\nabla\tilde u$, are analytic and bounded on the strip $|y|<\eta_*$, with
$$\eta_*\ \ge\ \frac{1}{2\pi\omega_k}\,\operatorname{arcsinh}\!\Big(\frac{\pi/2}{\prod_\ell \|W_\ell\|_\infty\,(1+2B_\ell+\pi)}\Big).$$
*Sketch.* SiLU $\sigma(z)=z/(1+e^{-z})$ is meromorphic with poles at $z=i\pi(2j+1)$. For $|\operatorname{Im}z|\le\pi/2$ one has $|1+e^{-z}|\ge1$, which gives $|\sigma(z)|\le|z|$ and $|\sigma'(z)|\le1+2|z|$. The first layer has $|\operatorname{Im}\gamma_j|\le\sinh(2\pi\omega_k\eta)$. Inducting layer by layer, every pre-activation stays inside $|\operatorname{Im}|\le\pi/2$, so no pole is reached. $\mu$, $\psi_a$ and $\gamma$ are entire.

**Consequence.** The strip shrinks like $1/(\text{RFF scale})$, and the poles exist only because of SiLU. A bank built from SIREN, Gaussian RBFs or fixed sines is *entire*. This matches `DECODERS.md`. Over Gauss $32^2\to48^2\to64^2$, the error of the RFF+SiLU bank falls by about $173\times$ then $115\times$, which is geometric. The sine, RBF and SIREN banks accelerate instead, which is super-geometric: sine $5.5\text{e-}2\to2.3\text{e-}8\to$ roundoff.

### Theorem 1.2 (tensor Gauss–Legendre, $p$ points per axis). *Provable.*
For any $\varrho>1$ with $\eta(\varrho):=(\varrho-\varrho^{-1})/4<\eta_*$,
$$\big|N_a(c)-N_{p^d,a}(c)\big|\ \le\ L^{d/2}\,d\,\frac{32}{15}\,\frac{\varrho^{-2p}}{\varrho^2-1}\;e^{\pi a_{\max}\eta(\varrho)}\;\Gamma_{\eta(\varrho)}\,\|c\|^2,$$
where $\Gamma_\eta=\sup\|\widehat G_{R'}(z)\|\,\|\mathbf 1\cdot\nabla\widehat G_{R'}(z)\|$ is taken over the one-variable-complexified Bernstein ellipses.
*Sketch.* Telescope $I^{\otimes d}-Q^{\otimes d}=\sum_k Q^{\otimes(k-1)}\otimes(I-Q)\otimes I^{\otimes(d-k)}$. The Gauss weights are positive and sum to 1, so each term needs analyticity in only one variable. Then apply the 1D Bernstein-ellipse bound (Trefethen, *ATAP* Thm 19.3, rescaled to $[0,1]$) together with $|\psi_a(x+iy)|\le e^{\pi a|y|}$.

**Corollary 1.3 (two regimes).** These are rigorous upper bounds; the onset value is not sharp.
- (i) *Test-frequency limited.* Take $\eta_*=\infty$ (an entire bank) and optimise $\varrho$. With $\omega=\pi a_{\max}/2$ plus the bank's own growth rate, this gives $|{\rm err}|\lesssim(e\,\omega/4p)^{2p}$. The convergence is super-geometric once $p\gtrsim e\omega/4\approx1.07\,a_{\max}$. The true onset is nearer $p\approx(\pi/4)a_{\max}$, and the bound overestimates it.
- (ii) *Decoder limited.* For large $p$ the rate is $\varrho_*^{-2p}$ with $\varrho_*=2\eta_*+\sqrt{4\eta_*^2+1}$, so measuring $\varrho$ infers the strip. Hari's 2D RFF bank gives $\varrho\approx1.17$, hence $\eta\approx0.08$ (scale 1.5, 32 features). Our 2D bank (scale 4, 128 features, width 1024) gives only about $10\times$ from $64^2$ to $128^2$, so $\varrho\approx1.018$ and $\eta\approx0.009$. Lemma 1.1 predicts this ordering. **This is a candidate explanation for the "cause not isolated" slow convergence in the 2D report. It is a hypothesis to test (§6, C1), not a result.**

In 3D with $a_{\max}\approx16$, Corollary 1.3(i) puts the onset around $p\approx13$–$17$. The measured `gl16` is not converged ($\rho=0.62$) and `gl24` is ($\rho=1.0\text{e-}2$). This is consistent.

### Theorem 1.4 (randomly shifted rank-1 lattice). *Provable for the class; constants are not explicit.*
**Key structural fact (Lemma 1.5).** $f=\psi_a\,u\,(\mathbf 1\cdot\nabla u)$ vanishes to **exactly second order** on each face. At $x_1=0$: $\psi_a=O(x_1)$ and $u=O(x_1)$, while $\partial_1u=\mu_{x_1}g=O(1)$ and $\partial_ju|_{x_1=0}=0$ for $j\neq1$. Its periodic extension is therefore $C^1$, but $\partial_1^2f$ jumps across the face. The flux form $-\nabla\psi_a\cdot\mathbf 1\,u^2/2$ behaves the same way.

Integrating by parts three times per coordinate leaves face terms. Each face term is again a smooth function vanishing to order $\ge2$ on the lower-dimensional faces; for example, $\partial_1^2f|_{x_1=0}=2\,\partial_1\psi_a\,(\partial_1u)^2=O(x_2^3)$. Recursing gives the product decay
$$|\hat f(\mathbf h)|\le C_f\prod_j\bar h_j^{-3},\qquad \bar h=\max(1,|h|),$$
which places $f$ in the Korobov class with $\alpha=3$ (Sloan–Joe). For any shift $\Delta$, the lattice error is $\big|\sum_{0\neq\mathbf h\in L^\perp}\hat f(\mathbf h)e^{2\pi i\mathbf h\cdot\Delta}\big|\le C_fP_3(\mathbf z,m)$. CBC lattices achieve $P_\alpha=O(m^{-\alpha+\varepsilon})$ (Kuo 2003; Dick 2004), and Fibonacci lattices achieve $O(m^{-\alpha}\log m)$. Hence
$$|N_a-N_{m,a}|=O(m^{-3+\varepsilon}).$$

**This corrects Hari's claim of super-algebraic lattice convergence.** The lattice rate is set by the **boundary vanishing order**, not by the decoder's smoothness. A fixed sine bank (an entire function) is still only $m^{-3}$ on Fibonacci lattices. The data agree:

| source | ladder | observed exponent |
|---|---|---|
| Hari 2D, RFF bank | Fibonacci $1597\to4181\to6765$ | $\approx m^{-2.6}$, then $m^{-3.4}$ |
| Hari 2D, sine bank | Fibonacci $4181\to6765$ | $\approx m^{-2.7}$ |
| ours 3D, $R'=512$ | lattice, per doubling of $m$ | $8.3\times$, $9.1\times$, $12.8\times$ (the $m^{-3}$ prediction is $8\times$) |

Before the asymptotic regime, the dual lattice has to avoid the integrand's frequency box. That box has half-width $K\approx a_{\max}/2$ plus the bank's band per axis, which requires $m\gtrsim(K+1)^d$.

**Corollary 1.6 (Gauss vs lattice, and why 3D favours lattices).**
- *Onset cost.* Gauss needs about $(\pi a_{\max}/4)^d$ points before it resolves the tests; a lattice or trapezoid needs about $(a_{\max}/2)^d$. The ratio is $(\pi/2)^d$: about $2.5$ in 2D and $3.9$ in 3D.
- *Asymptotic rate.* Gauss falls like $\exp(-2\ln\varrho\,m^{1/d})$, which degrades with $d$. The lattice falls like $m^{-3}$ up to logarithmic factors.
- *Crossover.* Gauss overtakes the lattice at $m_\times$ solving $2\ln\varrho\,m^{1/d}\approx3\ln m+{\rm const}$. In 2D that point is near $10^3$ (Hari: Gauss $32^2$ and Fibonacci 987 are equal; Gauss wins by 4096). In 3D with $\varrho\approx1.2$ it is above $10^6$, consistent with ours: the lattice beats Gauss at $m=4096$ and $32768$. This explains "lattice 2D-second, 3D-first" as a theorem plus two fitted constants.

### Proposition 1.7 (the tent transform). *Partly provable; one observed discrepancy is unexplained.*
The tent-transformed integrand $g=f\circ\varphi$, with $\varphi(t)=1-|2t-1|$, is the reflection of $f$. The relevant derivatives at the reflection points behave as follows:
- $g'$ is continuous, because $f'=0$ at the walls (Lemma 1.5);
- $g''$ is continuous, because reflection makes the even derivatives match;
- $g'''$ jumps.

So $g\in\alpha=4$, which is asymptotically *better* than $\alpha=3$. The common explanation that the tent "adds a kink" **does not apply to this integrand**. What the transform does provably is compress each axis by a factor of 2. This doubles the test band per axis and multiplies the onset size $m$ by $2^d$, which accounts for a 1–2 order loss at fixed $m$ before the asymptotic regime. **Open discrepancy:** Hari's tent ladder from 2584 to 4181 falls only like $m^{-1.1}$, slower than either prediction. The paper should either extend that ladder or audit the implementation before saying anything beyond "the tent transform doubles the band".

### Proposition 1.8 (why Smolyak fails). *The exactness lower bound is provable; the error lower bound is heuristic.*
**(a) Rigorous.** Any rule, with any weights, that is exact on the tensor space $Q^d_{2n-1}$ (degree $\le2n-1$ in each variable) has at least $n^d$ nodes. *Proof.* If there were fewer, some nonzero $q\in Q^d_{n-1}$ (dimension $n^d$) would vanish at every node. Then $q^2\in Q^d_{2n-2}$ has $\int q^2>0$ while the rule returns 0. Gauss attains the bound. A sparse grid saves points *only* by giving up exactness on the "corner" of $Q^d$, i.e. on the products $T_k(x)T_l(y)$ with both $k$ and $l$ large.

**(b) The test set lives in that corner.** Under eigenvalue ordering the tests include diagonal modes $(a,a)$ and $(a,a,a)$ with $a\approx a_{\max}/\sqrt d$. In Chebyshev form, $\sin(\omega t)$ has coefficients $2|J_k(\omega)|$: of size $\sim\omega^{-1/2}$ for $k\lesssim\omega$, and super-exponentially small beyond $\omega+O(\omega^{1/3})$. So $\psi_{(a,a)}$ carries $O(\omega^{-1})$ coefficients across the *whole* square $k,l\lesssim\omega=a\pi/2$. A level-$\ell$ Smolyak rule is exact only on $\sum_{i+j\le\ell+1}P_{n_i}\otimes P_{n_j}$. Its error on $\psi_{(a,a)}f$ is a sum of $O(\omega^{-1})$ coefficients multiplied by the non-zero Smolyak errors on omitted $T_kT_l$, and the error stays $O(1)$ unless these terms cancel. Proving there is no cancellation needs an explicit computation; the check is **C5**.

**(c)** CC-Smolyak weights have mixed signs and $\sum|w_q|$ grows with level. This amplifies the $\varepsilon_1$ term of Lemma 3.1 and explains the observed LM damping/budget exits. The prediction to test: Smolyak $\rho$ is small for anisotropic tests ($a$ or $b$ small) and $O(1)$ for diagonal ones. Smolyak could work only with tests of low mixed order, which would mean changing $P$.

**Other rules (cite, don't prove).** Scrambled Sobol has RMSE $O(m^{-3/2+\varepsilon})$ over scrambles (Owen), but a single scramble realisation is observed at about $1/m$. Plain Monte Carlo converges at $m^{-1/2}$.

---

## 2. Consistency decomposition: mesh target vs continuum target

### Proposition 2.1 (numerical-diffusion expansion of the sign-upwind stencil). *Provable, for $u\in C^3(\bar\Omega)$; this holds by Lemma 1.1.*
At every interior node, whichever way the sign switch goes,
$$u\,\delta_ku=u\,\partial_ku-\tfrac h2|u|\,\partial_k^2u+O\big(h^2|u|\,\|\partial_k^3u\|_\infty\big).$$
For $u>0$ the backward difference gives $u_x-\tfrac h2u_{xx}$. For $u\le0$ the forward difference gives $u_x+\tfrac h2u_{xx}$, and multiplying by $u<0$ gives the same expression. The zero ghost values equal $u|_{\partial\Omega}=0$ exactly, so the expansion also holds at nodes next to a wall. Summing against the mesh-sampled tests gives
$$N_h(c)=N(c)-h\,E_1(c)+O(h^2),\qquad E_{1,a}(c)=\tfrac{L^{d/2}}2\int_\Omega\psi_a\,|u|\,\Delta u\,dx. \tag{2.1}$$
The $O(h^2)$ term also absorbs the composite-sum error of the mesh. That error is $O(h^3)$ or smaller for $\psi_af$ by Lemma 1.5, and $O(h^2)$ for the Lipschitz integrand $\psi_a|u|\Delta u$. It requires $h\,a_{\max}$ small enough that the mesh resolves the tests.

### Corollary 2.2 (the two plateaus and mesh invariance). *Provable.*
Let $Q_m$ be any rule family with $N_m\to N$, as in §1. Then:
- An **off-mesh rule** converges to the continuum target, so $\rho_{\rm cont}\to0$, while against the mesh target
$$\rho_{\rm mesh}\to\frac{h\|E_1(c)\|}{\|N_h(c)\|}+O(h^2).$$
That limit halves under mesh doubling. Observed: 0.19, 0.10, 0.053 in our 3D lane; 0.25, 0.12, 0.065 in Hari's 2D study.
- A **mesh rule** (EQ, the mesh lattice) converges to $N_h$, so $\rho_{\rm cont}\to h\|E_1\|/\|N\|$. Neither family can go below the other's floor, so each must be certified against its own target.
- **Mesh invariance.** $N_m(c)$ does not depend on $h$. In (0.1), the only $h$-dependence of an off-mesh step is in the discrete linear terms $A$ and $\Lambda$. The relative eigenvalue error is $\approx(\pi a h)^2/12$ and the sampled Gram error is $O(h^3)$ or smaller. The off-mesh ROM is therefore mesh-invariant to $O(h^2)$, while a mesh-rule ROM carries an $O(h)$ shift of its tested residual.

**Caveat for the paper.** The fact that the off-mesh ROM beats the same-mesh FOM against the refined reference is *not* implied by the theorem. It requires the projection error $e_\Pi$ (§3) to be smaller than the FOM's $O(h)$ error, which holds only because the bank was trained on finer meshes. Separately, the refined reference is itself an upwind solution, so it is biased toward mesh discretisations.

---

## 3. Perturbation and a-priori bound for the LSPG / Gauss–Newton solve

The reference problem is $c_*=\arg\min\|r_*(c)\|$ with exact continuum advection. The rule problem is $c_m=\arg\min\|r_m\|$, where $r_m=r_*+\delta$. From (0.1),
$$\delta(c)=\Delta t\,D\,\big(N_m(c)-N(c)\big),\qquad \|D\|\le1,$$
so $\varepsilon_0:=\sup_{\mathcal B}\|\delta\|\le\Delta t\,\varepsilon_Q$ and $\varepsilon_1:=\sup_{\mathcal B}\|\partial\delta\|\le\Delta t\,\varepsilon_Q'$. Here $\varepsilon_Q$ and $\varepsilon'_Q$ are the sup-norm quadrature errors of $N$ and $\partial N$, which §1 bounds.

### Lemma 3.1 (one step). *Provable; it is a nonlinear-least-squares perturbation result via Newton–Kantorovich.*
Assume:
- (A1) $J_*=\partial r_*(c_*)$ has $\sigma_{\min}(J_*)=s>0$;
- (A2) the residual is small, $\|r_*(c_*)\|=\theta$ with $\theta\,\|\partial^2r_*\|_{\mathcal B}\le s^2/4$;
- (A3) $\varepsilon_1\le s/4$, and the resulting bound fits in the ball $\mathcal B$.

Then there is a unique stationary point $c_m\in\mathcal B$ and
$$\|c_m-c_*\|\le2\Big(\frac{\varepsilon_0}{s}+\frac{\theta\,\varepsilon_1}{s^2}\Big)\big(1+o(1)\big).$$
*Sketch.* $g_m(c)=\partial r_m^{\mathsf T}r_m$, and one Newton step from $c_*$ gives $-(J^{\mathsf T}J+E)^{-1}\big(J^{\mathsf T}\delta+\partial\delta^{\mathsf T}r_*\big)$ with $\|E\|\le s^2/2$. Then use $\|(J^{\mathsf T}J+E)^{-1}J^{\mathsf T}\|\le2/s$ (from $\|J^+\|=1/s$) and $\|(J^{\mathsf T}J+E)^{-1}\|\le2/s^2$. LM converges to the same stationary point. Its stopping tolerance $\tau_{\rm solve}$ adds a term $\le2\tau_{\rm solve}/s$.

**Explicit stability constant.** From (0.1), $J(c)=A+\Delta t\,D\,\partial N(c)$. With mesh-orthonormal $\Phi$ and $\widehat G_{R'}$,
$$s\ \ge\ \beta-\Delta t\,\sup_{\mathcal B}\|\partial N\|,\qquad \beta=\sigma_{\min}(\Phi^{\mathsf T}\widehat G_{R'}).$$
$\beta$ is the cosine of the largest principal angle between the bank span and the test span. It is an inf–sup constant that can be computed offline and does not depend on $m$.

### Theorem 3.2 (accumulation over time steps). *Conditional on (A4), which is an assumption to be verified numerically, not proved.*
(A4) The reference step map $S_*:c_n\mapsto c_{n+1}$ is Lipschitz with constant $e^{\Lambda_S\Delta t}$ on a tube around $\{c^*_n\}$, and (A1)–(A3) hold uniformly on that tube. Write $\varepsilon_Q$ and $\varepsilon'_Q$ for the sup-norm errors over the tube. Then, as long as the bound stays inside the tube,
$$\|c^m_n-c^*_n\|\ \le\ 2\Big(\frac{\varepsilon_Q}{s}+\frac{\theta\,\varepsilon_Q'}{s^2}\Big)\,\frac{e^{\Lambda_St_n}-1}{\Lambda_S}\ \xrightarrow{\Lambda_S\to0}\ 2t_n\Big(\frac{\varepsilon_Q}{s}+\frac{\theta\varepsilon_Q'}{s^2}\Big).$$
*Sketch.* Split
$$e_{n+1}\le\|S_m(c^m_n)-S_*(c^m_n)\|+\|S_*(c^m_n)-S_*(c^*_n)\|.$$
Lemma 3.1 bounds the first term and (A4) the second; then apply discrete Gronwall. Because $\delta$ carries a factor $\Delta t$, the accumulated quadrature error scales with $t_n$, not with the number of steps $n$. **The bound is therefore robust to the time step.** The hard part is $\Lambda_S$: Hari's stiff Allen–Cahn run ($1/\tau=20$) shows the exponential factor becoming dominant.

### Theorem 3.3 (a-priori ROM error, Strang-type). *Conditional on (A1)–(A4) and $u\in C^2([0,T];C^3)$.*
Let $\Pi$ be the mesh-orthogonal projector onto $\operatorname{span}\widehat G_{R'}$, and let $e_\Pi:=\sup_t\big(\|(I-\Pi)u\|+\|\partial_t(I-\Pi)u\|\big)$. Inserting $\Pi u(t_n)$ into the reference scheme leaves a local residual of size
$$\le\Delta t\big(C_\Pi e_\Pi+C_t\Delta t+C_hh^2\big),$$
from the projection error, backward Euler and the discrete linear terms. LSPG quasi-optimality, $\|c-\pi\|\le(2/s)\|r(\pi)\|$ to first order, and Gronwall then give
$$\boxed{\ \|u(t_n)-\widehat G_{R'}c^m_n\|\le \underbrace{e_\Pi}_{\text{manifold}}+\frac{2\,\Gamma(t_n)}{s}\Big[\underbrace{C_\Pi e_\Pi}_{\text{LSPG}}+\underbrace{C_t\Delta t}_{\text{time}}+\underbrace{C_hh^2}_{\text{linear terms}}+\underbrace{\varepsilon_Q\big(1+\tfrac{\theta\varepsilon'_Q}{s\,\varepsilon_Q}\big)}_{\text{quadrature}}+\underbrace{h\|E_1\|}_{\text{mesh rules only}}\Big]+\frac{2\tau_{\rm solve}}{s}\ }$$
with $\Gamma(t)=(e^{\Lambda_St}-1)/\Lambda_S$. For mesh rules, $\varepsilon_Q$ is measured against $N_h$, and the $h\|E_1\|$ term enters through (2.1).

**What is honest to say.** Each term is computable or measurable, but the bound is sufficient and probably pessimistic, by the Gronwall factor and by $1/s$. Its value for the paper is the **structure**: quadrature error enters additively, scaled by $\Gamma/s$, and it decays geometrically (Gauss) or as $m^{-3}$ (lattice). The mesh rules carry an irreducible $O(h)$ term that the off-mesh rules do not. The current reports state "not decomposed"; Theorem 3.3 tells the experiments which terms to measure (C8).

---

## 4. Cost model

Per evaluation, with real arithmetic in f64 (8 B). $u$ and $\mathbf 1\cdot\nabla u$ need only one directional derivative, so the off-mesh rule stores $B,\,B_\nabla\in\mathbb R^{m\times R'}$ and $\Psi=[w_q\psi_a(x_q)]\in\mathbb R^{m\times M}$.

| arm | storage (numbers) | residual flops | Jacobian flops | Jacobian kernel | $N$-dependence online |
|---|---|---|---|---|---|
| mesh-dense | $N(2d{+}1)R'$ + $NM$ (or separable sine transforms) | $\approx2NR'+2NM$ | $2MNR'$ | GEMM | linear |
| EQ ($N_{\rm eq}$ nodes) | $\le(2d{+}1)N_{\rm eq}R'+N_{\rm eq}M$ | $\approx2(2d{+}1)N_{\rm eq}R'+2N_{\rm eq}M$ | $2MN_{\rm eq}R'$ | GEMM | none; per-mesh NNLS fit offline |
| tensor | $MR'^2$ | $2MR'^2$ | $2MR'^2$ (symmetrised $T$) | GEMV, bandwidth-bound | none; $O(NMR'^2)$ offline |
| off-mesh ($m$ points) | $m(2R'+M)$ | $4mR'+2mM$ | $3mR'+2MmR'$ | GEMM, compute-bound | none; $O(m\cdot{\rm MLP})$ offline |

The model reproduces the 3D report exactly. With $R'=512$ and $M=2048$:
- tensor: $2MR'^2=1.07$ GFLOP and $8MR'^2=4.3$ GB;
- `lat4096`: $2MmR'=8.6$ GFLOP and $8m(2R'+M)=0.10$ GB;
- `gl24`: 29.0 GFLOP.

**Proposition 4.1 (roofline crossover).** The tensor Jacobian has arithmetic intensity $1/4$ flop/B, so its time is $\approx8MR'^2/\mathrm{BW}$. The off-mesh Jacobian has intensity $\approx MR'/\big(4(2R'+M)\big)$, which is $R'/6$ for $M=4R'$, so its time is $\approx2MmR'/F$. The off-mesh rule is faster when
$$m<m^\star=4R'\,F_{\rm eff}/\mathrm{BW}_{\rm eff},$$
and uses less memory when $m<MR'^2/(2R'+M)\approx\tfrac23R'^2$ (for $M=4R'$). With the measured H200 effective rates (about 48 TF/s f64 GEMM and about 4.1 TB/s), $m^\star\approx24$k at $R'=512$. The report's crossover lies between `gl24` (13.8k, faster) and `lat32768` (slower). **Prediction:** $m^\star\propto R'$, and the tensor's $R'^3$ memory makes it infeasible from about $R'\ge1024$ (34 GB). Below about 0.2 ms both kernels are latency-bound, which explains why the two are "comparable" at $R'=256$.

**Structural remark.** The off-mesh rule *is* a rank-$m$ factorisation of the continuum tensor, $T_a=B^{\mathsf T}\operatorname{diag}(w\odot\psi_a)B_\nabla$ (symmetrised). It wins when the analytic integrand admits $m\ll R'^2$.

Online cost is independent of $N$ for every row except mesh-dense. Two per-query costs stay $O(NR')$ and no advection rule removes them: projecting $u_0$, and decoding each output. The paper must state this; at $256^3$ these terms dominate the query.

---

## 5. Data-free a-posteriori certification

**Estimator.** Pair the rule $m$ with a refinement $m'$ and set $\hat\rho_m(c)=\|N_{m'}(c)-N_m(c)\|/\|N_{m'}(c)\|$. Suitable refinements:
- an embedded or extensible lattice, $m\to2m$, which is nested, so $N_{2m}$ costs $m$ extra points and the refined value comes for free;
- Gauss $p\to p+8$;
- a tensor Gauss–Kronrod extension, which is nested.

### Proposition 5.1 (two-sided effectivity under saturation). *Provable given (Sat).*
(Sat) $\|N-N_{m'}\|\le\kappa\|N-N_m\|$ with $\kappa<1$. Then
$$\frac{\|N_{m'}-N_m\|}{1+\kappa}\le\|N-N_m\|\le\frac{\|N_{m'}-N_m\|}{1-\kappa}.$$
In the asymptotic regime, §1 supplies $\kappa$: $\kappa\approx\varrho^{-16}$ for $p\to p+8$, and $\kappa\approx2^{-3}$ for lattice doubling. **(Sat) fails before the asymptotic regime.** For example, two unresolved rules can agree by accident, or a nested lattice pair can share an aliasing vector, since $L_{2m}^\perp\subset L_m^\perp$. The safeguard is a three-level test: accept only if $\hat\rho_m/\hat\rho_{m'}$ is within a factor of the predicted $\kappa^{-1}$.

### Corollary 5.2 (running a-posteriori trajectory bound). *Computable, conditional on $\Lambda_S$.*
Replace $\varepsilon_Q$ in Theorem 3.2 by the per-step $\hat\varepsilon_k=\|N_{m'}-N_m\|(c^m_k)/(1-\kappa)$. Replace $s$ by $s_k=\sigma_{\min}(J_k)$, which is cheap from the $R'\times R'$ LM normal matrix. For $\Lambda_S$, use either the online estimate $\|J_k^+DA\|$ or a declared bound. The result is a per-query, per-step certificate with no reference data and an $N$-independent cost of $O(m'(R'+M))$. It also gives an **adaptive rule**: refine $m$ whenever $\hat\rho_m$ exceeds the bar.

**Contrast with EQ (to state plainly in the paper).**
- EQ's error is a fit residual on the fit states. Nothing bounds it at other states.
- Certifying EQ requires held-out *reached* states and dense $O(N)$ evaluations of $N_h$. The result is a statistical statement over a state distribution: it was confirmed on five draws at $4096^2$ and failed the confirmation draw.
- EQ must be redone for every mesh.
- EQ has no rate in $m$, so there is no saturation property and no refinement estimator.
- An online EQ check would need $N_h$ at the reached state, i.e. $O(NR')$, which defeats the purpose.

The off-mesh certificate is deterministic, per state, mesh-free and online. Its weakness is (Sat) before the asymptotic regime, and it does not cover the $\Lambda_S$ factor.

---

## 6. Statements that need numerical verification (concrete checks)

| id | statement | check | pass criterion |
|---|---|---|---|
| C1 | Thm 1.2 / Cor 1.3(ii): Gauss rate is set by the decoder strip | Measure each bank's strip independently, from the Chebyshev-coefficient decay of $\hat g_j$ along axis lines. Predict $\varrho_*=2\eta+\sqrt{4\eta^2+1}$ and compare with the fitted slope of $\log\rho$ against $p$. Repeat for RFF scales 1.5, 2, 4 and for a SIREN/sine control. | predicted and fitted $\ln\varrho$ within a factor of 2; entire banks super-geometric |
| C2 | Cor 1.3(i): onset at $p\approx(\pi/4)\text{–}1.07\,a_{\max}$ | Vary $M$ (and hence $a_{\max}$) at a fixed bank and record the $p$ at which $\rho$ starts to fall. | onset scales linearly with $a_{\max}$ |
| C3 | Thm 1.4: lattice $\rho\sim m^{-3}$, independent of the decoder | Extend the CBC/Fibonacci ladders to 2–3 more doublings, with 10 random shifts; fit the exponent. Run a sine bank as an entire control. | exponent in $[2.6,3.4]$ for every bank |
| C4 | Prop 1.7: the tent penalty is band-doubling (pre-asymptotic) | Extend the tent ladder to $m\ge4\cdot10^4$ in 2D; audit the transform code. | tent slope tends to $-4$, or the discrepancy is reported |
| C5 | Prop 1.8(b): Smolyak fails on diagonal tests | Report Smolyak $\rho$ separately for each test $(a,b)$. | error concentrated at $a\approx b$; anisotropic modes small |
| C6 | Cor 2.2: the mesh-target plateau equals $h\|E_1\|/\|N_h\|$ | Compute $E_1(c)$ by fine Gauss quadrature at reached states and compare with the measured $\rho_{\rm mesh}$ plateaus (0.19, 0.10, 0.053). | agreement within 10 %; ratio $\to2$ per refinement |
| C7 | Lemma 3.1, Thm 3.2 | Inject a synthetic $\delta$ of size $\epsilon\in\{10^{-6},\dots,10^{-2}\}$; measure $\|c_m-c_*\|$ per step and over the trajectory. Record $s_k$, $\beta$ and $\theta_k$. | linear in $\epsilon$; per step $\le2\epsilon/s$; accumulation $\propto t_n$ |
| C8 | Thm 3.3 decomposition | Measure separately: $e_\Pi$ (truth projected on the span); the continuum-rollout error (Gauss $80^3$ / $640^2$); the quadrature distance; and the $\Delta t$-halving error. | the terms sum to at least the observed total, and no term is negative |
| C9 | Prop 4.1 cost model | Measure flops, bytes and time for $R'\in\{256,512,1024\}$ and an $m$ sweep; locate the measured $m^\star$. | $m^\star\propto R'$ within 30 % |
| C10 | Prop 5.1 effectivity | On all reached states, compute $\hat\rho_m/\rho_m$ against the true continuum. | ratio in $[1/(1+\kappa),1/(1-\kappa)]$ in the asymptotic regime; the three-level test flags every pre-asymptotic failure |
| C11 | Cor 2.2 mesh invariance $O(h^2)$ | Spread of the off-mesh rollout over $L$ against $h^2$. | slope 2 (the current spreads of 0.01–0.02 pp are consistent but not yet tested for slope) |

**Assumptions, all flagged:**
- (S1)–(S3);
- the small-residual condition (A2);
- uniform $s>0$ on the reached tube (A1);
- the discrete stability constant $\Lambda_S$ (A4), the weakest link, which is never proved, only measured;
- solution regularity $u\in C^2([0,T];C^3)$, which is reasonable for viscous Burgers but not checked;
- saturation (Sat) for the estimator.

**Not covered:** the head mode beyond a remark (replace $s$ by $\sigma_{\min}(J\,J_h)$; nonconvex); non-box domains; non-analytic activations; NS (whose tensor is exact, so no quadrature is involved).
