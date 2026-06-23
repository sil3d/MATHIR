# MATHIR V7: Formal Proof Sketches

**Companion to `THEORY_V7.md`**

This document collects, in a single self-contained reference, the formal proof sketches for the six theorems of MATHIR V7. It is intended for reviewers and graduate students who want to verify the argument structure without working through the full text. The full statements, assumptions, and discussion are in `THEORY_V7.md`; here we focus on the chain of inequalities and the key technical lemmas.

---

## Notation and conventions

| Symbol | Meaning |
|--------|---------|
| $\mathcal{X} \subseteq \mathbb{R}^d$ | Embedding space |
| $X_t$ | Observation at time $t$ |
| $M_t$ | Memory state at time $t$ |
| $\mathcal{N}(\mu, \Sigma)$ | Multivariate Gaussian |
| $D_{\mathrm{KL}}(P \| Q)$ | Kullback-Leibler divergence |
| $I(X; Y)$ | Mutual information |
| $D_M(x; \mu, \Sigma)$ | Mahalanobis distance |
| $\Delta_n$ | Probability simplex in $n$ dimensions |
| $d_{\mathbb{B}}(u,v)$ | Poincaré ball distance |
| $\mathcal{M}_{\mathrm{DS}}$ | Doubly-stochastic polytope |
| $\mathcal{S}_\omega$ | Sinkhorn-Knopp projection with relaxation $\omega$ |
| $\chi^2_{d, 1-\alpha}$ | $(1-\alpha)$-quantile of $\chi^2$ with $d$ d.f. |

Throughout, $\|A\|$ denotes the spectral norm, $\|A\|_F$ the Frobenius norm, and $\|v\|$ the Euclidean norm of a vector $v$.

---

## Theorem 1 — Information Capacity

**Statement.** Under the assumptions of `THEORY_V7.md` §3.1,
$$I(X; M_t) \;\le\; (N + W + I + 2V + P + s) \cdot d \cdot \log_2(1 + \mathrm{SNR}) \;+\; \tfrac{1}{2} \log_2 \det(I + D D^\top / d).$$

**Proof sketch.** The argument proceeds in four clean steps.

*Step 1. AWGN per-slot capacity.* Apply the Shannon-Hartley theorem to a single memory slot modelled as a length-$d$ real-valued AWGN channel. The capacity in bits per channel use is $\frac{1}{2}\log_2(1 + \mathrm{SNR})$; multiplying by $d$ channel uses gives the per-slot bound [Shannon, 1948; Cover & Thomas, 2006, Thm 9.1.1].

*Step 2. Tier summation.* Add the per-slot capacities across the four vector tiers, accounting for the doubled capacity of variational slots ($\mu$ and $\sigma$):
$$\text{bits}_{\mathrm{vector}} = (N + W + I + 2V + P)\cdot d \cdot \tfrac{1}{2}\log_2(1 + \mathrm{SNR}).$$

*Step 3. Dictionary volume.* Apply Donoho's sparse-representation theorem [Donoho, 2006, Thm 1.3] to the sparse-coding tier: the number of distinguishable atoms in $D$ is at most $\frac{1}{2}\log_2\det(I + D D^\top / d)$. With $s$-sparse codes, the realised contribution is $s \cdot d \cdot \frac{1}{2}\log_2(1+\mathrm{SNR})$ from the active atoms plus the dictionary-volume correction.

*Step 4. Composition by data processing.* The observed data $X$ passes through the encoder $\phi$ and the router $R$ before reaching the slot. By the data-processing inequality [Cover & Thomas, 2006, Thm 2.8.1],
$$I(X; M_t) \;\le\; I(\phi(X); M_t) \;\le\; \text{sum of per-slot capacities}.$$
The data-processing gap is $O(\sqrt{d/N})$ under sub-Gaussian concentration of empirical encoders. $\blacksquare$

**Tightness.** Equality requires (a) matched-filter encoders (jointly Gaussian slot distributions), (b) AWGN noise, and (c) statistically independent slots. The third condition is the binding constraint: with finite $N$, slot dependence introduces a $O(\sqrt{d/N})$ gap.

**Reference for the reader.** Cover & Thomas (2006), Chapters 9 and 2, for the AWGN and data-processing steps; Donoho (2006) for the dictionary volume.

---

## Theorem 2 — Retention Guarantee

**Statement.** Under Lipschitz encoder, $\eta$-stable router, Robbins-Monro prototypes, and sub-Gaussian keys, the recall accuracy for an item stored $K$ steps ago satisfies
$$\Pr\bigl(\mathrm{Accuracy}(K) \ge 1 - C K L \eta / N\bigr) \;\ge\; 1 - \exp(-N/2).$$

**Proof sketch.** The argument uses the three-step decomposition in `THEORY_V7.md` §3.2; here we highlight the key technical ingredients.

*Step 1. Lipschitz contraction of keys.* From Assumption A3,
$$\|k_t - k_{t+1}\| = \|\phi(x_t) - \phi(x_{t+1})\| \le L \|x_t - x_{t+1}\| \le 2LR,$$
where $R = \sup_t \|x_t\|$ is finite by Assumption A1. The keys therefore lie in a $2LR$-ball.

*Step 2. Prototype concentration by Robbins-Monro.* Apply [Kushner & Yin, 2003, Thm 2.1] to the prototype update
$$\pi_j^{(t+1)} = \pi_j^{(t)} + \beta_t (x_t - \pi_j^{(t)}).$$
The Robbins-Monro condition $\sum_t \beta_t = \infty$, $\sum_t \beta_t^2 < \infty$ implies
$$\mathrm{Var}(\pi_j^{(t)} - \pi_j^*) \;\le\; s^2 \sum_{i<t} \beta_i^2 \;\le\; \sigma_\pi^2,$$
uniformly in $t$.

*Step 3. Concentration of the empirical key average.* The key distribution at time $t$ is a mixture of $N$ sub-Gaussians with variance proxy at most $(2LR)^2 + \sigma_\pi^2$. The empirical mean $\bar k = \frac{1}{N}\sum_i k_i$ has variance $\sigma_\mathrm{key}^2 \le ((2LR)^2 + \sigma_\pi^2)/N$. By the standard sub-Gaussian concentration [Vershynin, 2018, Thm 2.6.3],
$$\Pr\bigl(\|\bar k - \mathbb{E}[k]\| > \varepsilon\bigr) \;\le\; 2\exp\!\Bigl(-\frac{N \varepsilon^2}{2 \sigma_\mathrm{key}^2}\Bigr).$$

*Step 4. Translate key perturbation to accuracy loss.* The encoder's inverse-Lipschitz constant in a small ball is at most $1/L$, so a key perturbation of size $\varepsilon$ corresponds to an embedding-space error of $\varepsilon/L$. The accuracy bound follows by setting $\varepsilon = K L \eta / N$ and simplifying the exponential.

**Constant $C$.** From the explicit constants in the concentration step, $C = 2\sigma_\mathrm{key} \sqrt 2 / s^2$, which depends only on the sub-Gaussian proxy and the Lipschitz constant.

**Reference for the reader.** Vershynin (2018) Chapter 2 for sub-Gaussian concentration; Kushner & Yin (2003) Chapter 2 for Robbins-Monro; Anderson (2003) for the cognitive-science motivation of the Ebbinghaus curve.

---

