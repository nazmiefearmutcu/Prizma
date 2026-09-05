# Theoretical Analysis of `quad2` Monomial Selection, Gradient Stability, and Convergence

This document provides a rigorous mathematical and theoretical analysis of the `quad2` and `quad2_lowrank` feature maps used in the PRISM / Prizma-Seq architecture. We focus on two primary theoretical results:
1. **Gradient Stability under BPTT**: Proving that the recurrence Jacobian is contractive in spectral norm, bounded by the decay gate $\alpha_t$.
2. **Convergence Rates on Associative Recall**: Formulating the delta rule recurrence as gradient descent on a least-squares objective and proving that the convergence rate and associative capacity are directly bounded by the key crosstalk metric.

---

## 1. Architectural Context and Feature Maps

In the Global Delta Memory (GDM) framework of Prizma-Seq, the model maintains a rectangular associative state $S_t \in \mathbb{R}^{d_v \times d_{\phi}}$, where the value dimension is $d_v$ and the feature-mapped key/query dimension is $d_{\phi} \gg d_h$ (with $d_h$ being the head dimension). 

Let $x_t \in \mathbb{R}^{d_h}$ be the input vector at time step $t$, which is $L_2$-normalized to the unit sphere: $\|x_t\|_2 = 1$. The feature map $\phi: \mathbb{S}^{d_h-1} \to \mathbb{S}^{d_{\phi}-1}$ expands this input space to increase capacity without adding trainable parameters. We compare four settings:

1. **`none` (Linear Baseline)**:
   $$\phi(x) = x \quad \implies \quad d_{\phi} = d_h$$
2. **`quad2` (Random Monomial Selection)**:
   $$\phi(x) = \text{L2-Norm}([x \ ; \ \psi_{rand}(x)]) \quad \implies \quad d_{\phi} = d_h + d_{n2}$$
   where $\psi_{rand}(x) \in \mathbb{R}^{d_{n2}}$ is a vector of random monomial pairs:
   $$\psi_{rand}(x)_k = x[I_k] \cdot x[J_k]$$
   for fixed, random, seeded indices $I_k, J_k \in \{1, \dots, d_h\}$ for $k = 1, \dots, d_{n2}$.
3. **`quad2_lowrank` (Structured Low-Rank Monomial Selection)**:
   First project $x$ to a lower-dimensional subspace $z = x P \in \mathbb{R}^r$ using a fixed, random, seeded projection matrix $P \in \mathbb{R}^{d_h \times r}$. Then compute all $r(r+1)/2$ upper-triangular monomials of $z$:
   $$\phi(x) = \text{L2-Norm}([x \ ; \ z[I_{lr}] \cdot z[J_{lr}]]) \quad \implies \quad d_{\phi} = d_h + \frac{r(r+1)}{2}$$
   where $I_{lr}, J_{lr}$ iterate over all pairs $1 \le i \le j \le r$.
4. **`rand_linear` (Control Group)**:
   $$\phi(x) = \text{L2-Norm}(x W_{rand}) \quad \implies \quad d_{\phi} = d_h + d_{n2}$$
   where $W_{rand} \in \mathbb{R}^{d_h \times d_{\phi}}$ is a fixed random projection. Because $\text{rank}(W_{rand}) \le d_h$, this map cannot lift the representation out of the $d_h$-dimensional subspace, proving that any capacity gain is due to the non-linear monomials rather than the dimensional expansion.

---

## 2. Mathematical Formulation of the State Recurrence

For a sequence of queries, keys, and values $(q_t, k_t, v_t)$, where $q_t, k_t$ are mapped and $L_2$-normalized to the unit sphere ($\|q_t\|_2 = \|k_t\|_2 = 1$ in $\mathbb{R}^{d_{\phi}}$), the sequential state update for $n_{\delta} = 1$ is:
$$S_t = \alpha_t S_{t-1} + u_t k_t^T$$
where $u_t \in \mathbb{R}^{d_v}$ is the update vector. In the standard erasing delta-rule write mode:
$$u_t = \beta_t (v_t - \alpha_t S_{t-1} k_t)$$
where $\beta_t \in [0, \beta_{cap}]$ is the input-dependent write gate, and $\alpha_t \in [0.5, 1.0]$ is the decay gate. Substituting $u_t$ into the state update yields:
$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^T$$
Factoring out $S_{t-1}$ gives the linear matrix recurrence:
$$S_t = S_{t-1} M_t + B_t$$
where the transition matrix $M_t$ and input matrix $B_t$ are defined as:
$$M_t = \alpha_t (I - \beta_t k_t k_t^T) \in \mathbb{R}^{d_{\phi} \times d_{\phi}}$$
$$B_t = \beta_t v_t k_t^T \in \mathbb{R}^{d_v \times d_{\phi}}$$

---

## 3. Proof of Gradient Stability under BPTT

In training, backpropagation through time (BPTT) computes the gradient of a scalar loss function $\mathcal{L}$ with respect to the hidden state $S_t$. We establish that gradients cannot explode by proving the contractiveness of the state transition Jacobian.

### Theorem 1 (Jacobian Contractiveness)
*Let the state $S_t$ follow the recurrence $S_t = S_{t-1} M_t + B_t$. The Jacobian operator $J_t = \frac{\partial \text{vec}(S_t)}{\partial \text{vec}(S_{t-1})}$ satisfies:*
$$\| J_t \|_2 \le \alpha_t \max(1, |1 - \beta_t|)$$
*If the write gate is bounded by $0 \le \beta_t \le 1$ and the decay gate satisfies $0.5 \le \alpha_t \le 1.0$, then:*
$$\| J_t \|_2 \le \alpha_t \le 1.0$$
*which proves that the recurrence is strictly contractive in spectral norm.*

### Proof
We begin by vectorizing the recurrence relation. Using the Kronecker product vectorization identity $\text{vec}(A Y C) = (C^T \otimes A) \text{vec}(Y)$, we vectorize the transition $S_t = I_{d_v} S_{t-1} M_t + B_t$:
$$\text{vec}(S_t) = (M_t^T \otimes I_{d_v}) \text{vec}(S_{t-1}) + \text{vec}(B_t)$$
The Jacobian matrix of this linear transformation is:
$$J_t = \frac{\partial \text{vec}(S_t)}{\partial \text{vec}(S_{t-1})} = M_t^T \otimes I_{d_v}$$
The spectral norm (operator $L_2$ norm) of a Kronecker product is the product of the spectral norms of its components:
$$\| J_t \|_2 = \| M_t^T \otimes I_{d_v} \|_2 = \| M_t^T \|_2 \| I_{d_v} \|_2 = \| M_t \|_2$$
Since $M_t = \alpha_t (I - \beta_t k_t k_t^T)$ is symmetric, its singular values are the absolute values of its eigenvalues. Let $v \in \mathbb{R}^{d_{\phi}}$ be an eigenvector of the term $I - \beta_t k_t k_t^T$ with eigenvalue $\lambda$. We analyze the eigenvalues:
1. **Parallel Subspace**: Let $v = k_t$. Since $\|k_t\|_2 = 1$:
   $$(I - \beta_t k_t k_t^T) k_t = k_t - \beta_t (k_t^T k_t) k_t = (1 - \beta_t) k_t$$
   Thus, $\lambda_1 = 1 - \beta_t$ is an eigenvalue with eigenvector $k_t$.
2. **Orthogonal Subspace**: Let $v \perp k_t$, meaning $k_t^T v = 0$.
   $$(I - \beta_t k_t k_t^T) v = v - \beta_t (k_t^T v) k_t = v$$
   Thus, $\lambda = 1$ is an eigenvalue with multiplicity $d_{\phi} - 1$.

The spectrum of $I - \beta_t k_t k_t^T$ is therefore $\{1 - \beta_t, 1, \dots, 1\}$. The spectral norm is the maximum absolute eigenvalue:
$$\| I - \beta_t k_t k_t^T \|_2 = \max(1, |1 - \beta_t|)$$
Scaling by the scalar decay factor $\alpha_t \ge 0$, we obtain the spectral norm of $M_t$:
$$\| M_t \|_2 = \alpha_t \max(1, |1 - \beta_t|)$$
which yields the bound on the Jacobian:
$$\| J_t \|_2 = \alpha_t \max(1, |1 - \beta_t|)$$
When the write gate is bounded in the stable regime $0 \le \beta_t \le 1$, we have $|1 - \beta_t| \le 1$, simplifying the norm to:
$$\| J_t \|_2 = \alpha_t$$
Since $\alpha_t \le 1.0$, the Jacobian is bounded by $\alpha_t \le 1.0$, making the state recurrence contractive. $\blacksquare$

### Implications for Exploding and Vanishing Gradients
1. **Prevention of Exploding Gradients**:
   Under BPTT, the gradient of the loss $\mathcal{L}$ at time step $T$ with respect to the state $S_t$ ($t < T$) is:
   $$\frac{\partial \mathcal{L}}{\partial \text{vec}(S_t)} = \frac{\partial \mathcal{L}}{\partial \text{vec}(S_T)} \prod_{\tau=t+1}^T J_{\tau}$$
   Applying the submultiplicativity of matrix norms:
   $$\left\| \frac{\partial \mathcal{L}}{\partial \text{vec}(S_t)} \right\|_2 \le \left\| \frac{\partial \mathcal{L}}{\partial \text{vec}(S_T)} \right\|_2 \prod_{\tau=t+1}^T \| J_{\tau} \|_2 \le \left\| \frac{\partial \mathcal{L}}{\partial \text{vec}(S_T)} \right\|_2 \prod_{\tau=t+1}^T \alpha_{\tau}$$
   Because $\alpha_{\tau} \le 1$ for all $\tau$, the product $\prod_{\tau=t+1}^T \alpha_{\tau} \le 1$. The gradient norm is bounded by the final gradient norm at step $T$, which theoretically guarantees that gradients cannot explode.
2. **Mitigation of Vanishing Gradients**:
   The decay gate $\alpha_t$ acts as a selective memory controller.
   - For directions orthogonal to the current key $k_t$, the transition eigenvalue is exactly $\alpha_t$. When $\alpha_t = 1.0$ (no decay), the gradient is propagated backwards with a norm multiplier of $1.0$, ensuring long-term memory persistence without decay.
   - For the direction parallel to the current key $k_t$, the eigenvalue is $\alpha_t(1 - \beta_t)$. When $\beta_t \approx 1$, this eigenvalue approaches $0$. Mathematically, this corresponds to the selective erasure of the old value associated with key $k_t$ to overwrite it with the new value $v_t$, preventing memory saturation and interference.

---

## 4. Theoretical Analysis of Monomial Selection and Capacity

The role of the feature map is to reduce off-diagonal correlation (crosstalk) between keys, thereby increasing the storage capacity of the state matrix. We define the **key crosstalk** metric as:
$$\text{cross}(\phi) = \mathbb{E}_{k_i, k_j \sim \mathbb{S}^{d_h-1}} \left[ \left| \phi(k_i)^T \phi(k_j) \right| \right] \quad \text{for } i \neq j$$

### Analysis of the Quadratic Kernel and Random Monomials
Let $x, y \in \mathbb{R}^{d_h}$ be unit vectors ($x^T x = y^T y = 1$) with correlation $\rho = x^T y \in [-1, 1]$.
A full quadratic feature map $\Phi_{full}(x) = \text{vec}(x x^T) \in \mathbb{R}^{d_h^2}$ has the inner product:
$$\Phi_{full}(x)^T \Phi_{full}(y) = \sum_{a,b=1}^{d_h} x[a] x[b] y[a] y[b] = (x^T y)^2 = \rho^2$$
For off-diagonal keys where $|\rho| < 1$, the quadratic projection squares the correlation, yielding $\rho^2 \ll |\rho|$. For example, an off-diagonal similarity of $\rho = 0.5$ is suppressed to $\rho^2 = 0.25$.

To avoid the $O(d_h^2)$ dimension expansion, `quad2` samples a subset of $d_{n2}$ random monomial pairs. Let $\psi(x) \in \mathbb{R}^{d_{n2}}$ be the unnormalized monomial vector. The expectation of its inner product is:
$$\mathbb{E}[\psi(x)^T \psi(y)] = \sum_{k=1}^{d_{n2}} \mathbb{E}[x[I_k] y[I_k] x[J_k] y[J_k]]$$
Assuming $I_k \neq J_k$, the terms are independent:
$$\mathbb{E}[\psi(x)^T \psi(y)] = d_{n2} \mathbb{E}[x[I] y[I]] \mathbb{E}[x[J] y[J]] = d_{n2} \left( \frac{x^T y}{d_h} \right) \left( \frac{x^T y}{d_h} \right) = \frac{d_{n2}}{d_h^2} \rho^2$$
For a random unit vector, the squared norm of the monomial vector is:
$$\|\psi(x)\|_2^2 = \sum_{k=1}^{d_{n2}} x[I_k]^2 x[J_k]^2 \approx \frac{d_{n2}}{d_h^2}$$
For the concatenated, normalized feature map $\phi(x) = \text{L2-Norm}([x \ ; \ \psi(x)])$:
$$\phi(x)^T \phi(y) \approx \frac{x^T y + \psi(x)^T \psi(y)}{1 + \frac{d_{n2}}{d_h^2}}$$
Taking the expectation over the random monomial selection:
$$\mathbb{E}[\phi(x)^T \phi(y)] \approx \frac{\rho + \frac{d_{n2}}{d_h^2} \rho^2}{1 + \frac{d_{n2}}{d_h^2}} = (1 - \lambda) \rho + \lambda \rho^2$$
where $\lambda = \frac{d_{n2}/d_h^2}{1 + d_{n2}/d_h^2} \in [0, 1]$ represents the quadratic mixing ratio. This shows that the inner product of the feature map is a convex combination of the linear and quadratic terms. Since $|\rho| < 1$, the mixing of the quadratic term strictly compresses the off-diagonal similarity:
$$\left| (1 - \lambda) \rho + \lambda \rho^2 \right| < |\rho| \quad \text{for } 0 < |\rho| < 1$$
proving that monomial selection suppresses key crosstalk.

### Analysis of Structured Low-Rank Monomials (`quad2_lowrank`)
In `quad2_lowrank`, we first project $x \in \mathbb{R}^{d_h}$ to a lower-dimensional space $z = x P \in \mathbb{R}^r$ using a fixed random projection $P \in \mathbb{R}^{d_h \times r}$.
By the Johnson-Lindenstrauss lemma, if the entries of $P$ are drawn from $\mathcal{N}(0, 1/d_h)$, the pairwise inner products are preserved in expectation:
$$\mathbb{E}[z_x^T z_y] = x \mathbb{E}[P P^T] y = x^T y = \rho$$
Taking *all* $r(r+1)/2$ monomials of $z$ forms a complete basis for quadratic polynomials in the $r$-dimensional projected subspace. This structured design eliminates the sampling variance associated with sparse random monomial selection, enabling `quad2_lowrank` to achieve competitive crosstalk suppression at a fraction of the dimensionality:
- `quad2` (d_phi = 256): crosstalk $\approx 0.076$
- `quad2_lowrank` (d_phi = 137, $r=14$): crosstalk $\approx 0.085$
- `none` (d_phi = 32): crosstalk $\approx 0.142$

---

## 5. Convergence Rates on Associative Recall Tasks

We analyze Multi-Query Associative Recall (MQAR) with $N$ key-value pairs $(k_i, v_i)$ stored at steps $t_i$. We show that the delta rule recurrence acts as an online optimizer, and its convergence rate is limited by the key crosstalk.

### Theorem 2 (Convergence and Capacity)
*Let the state $S_t$ follow the recurrence with $\beta_t = 1$ and $\alpha_t = 1$ (no decay). Let $\Phi \in \mathbb{R}^{d_{\phi} \times N}$ be the matrix of feature-mapped keys $\Phi = [\phi(k_1), \dots, \phi(k_N)]$, and $V \in \mathbb{R}^{d_v \times N}$ be the matrix of target values. Let the key Gram matrix be $G = \Phi^T \Phi = I + E$, where $E$ contains the off-diagonal similarities.*
*The residual reconstruction error $R^{(k)} = S^{(k)} \Phi - V$ after $k$ passes of BPTT / recurrent steps satisfies:*
$$\| R^{(k)} \|_F \le \| V \|_F \| E \|_2^k$$
*where $\| E \|_2$ is the spectral norm of the crosstalk matrix. If $\| E \|_2 < 1$, the error converges exponentially to 0 with rate $\|E\|_2$.*

### Proof
The delta rule state update with write gate $\beta_t = \beta$ is equivalent to a gradient step on the least-squares error objective:
$$E(S) = \frac{1}{2} \sum_{i=1}^N \| S \phi(k_i) - v_i \|_2^2 = \frac{1}{2} \| S \Phi - V \|_F^2$$
The gradient of $E(S)$ is:
$$\nabla_S E(S) = (S \Phi - V) \Phi^T$$
The gradient descent recurrence is:
$$S^{(k+1)} = S^{(k)} - \beta (S^{(k)} \Phi - V) \Phi^T$$
Post-multiplying by $\Phi$ and subtracting $V$ yields the recurrence for the residual matrix $R^{(k)} = S^{(k)} \Phi - V$:
$$R^{(k+1)} = R^{(k)} - \beta R^{(k)} \Phi^T \Phi = R^{(k)} (I - \beta G)$$
Choosing the optimal step size $\beta = 1$ and substituting $G = I + E$:
$$I - \beta G = I - (I + E) = -E$$
which simplifies the residual recurrence to:
$$R^{(k+1)} = -R^{(k)} E$$
Assuming the state is initialized to zero ($S^{(0)} = 0$, so $R^{(0)} = -V$), solving the recurrence yields:
$$R^{(k)} = R^{(0)} (-E)^k = -V (-E)^k$$
Taking the Frobenius norm and applying the matrix norm submultiplicativity:
$$\| R^{(k)} \|_F \le \| V \|_F \| E^k \|_2 \le \| V \|_F \| E \|_2^k$$
proving that the reconstruction error decays exponentially at rate $\|E\|_2$. $\blacksquare$

### Relationship between Crosstalk, Sequence Length, and Capacity
By the Gershgorin Circle Theorem, the eigenvalues of the symmetric crosstalk matrix $E$ are bounded by its maximum row absolute sum:
$$\| E \|_2 \le \max_i \sum_{j \neq i} |E_{ij}|$$
Taking the expectation of this bound over the key distribution:
$$\mathbb{E}[\| E \|_2] \approx (N - 1) \mathbb{E}[|E_{ij}|] = (N - 1) \text{cross}(\phi)$$
For the reconstruction error to converge to zero (implying perfect recall of all associations), we require $\|E\|_2 < 1$. This yields a theoretical bound on the sequence capacity $N$:
$$N < 1 + \frac{1}{\text{cross}(\phi)}$$

Evaluating this capacity bound for each feature map:
1. **`none`** ($\text{cross} \approx 0.142$):
   $$N_{none} < 1 + \frac{1}{0.142} \approx 8.0$$
2. **`quad2_lowrank`** ($\text{cross} \approx 0.085$):
   $$N_{lowrank} < 1 + \frac{1}{0.085} \approx 12.8$$
3. **`quad2`** ($\text{cross} \approx 0.076$):
   $$N_{quad2} < 1 + \frac{1}{0.076} \approx 14.2$$

This mathematically explains why the linear baseline fails on harder MQAR tasks (such as $D=128$), whereas both `quad2` and `quad2_lowrank` successfully solve them. By reducing the key crosstalk, the feature maps shrink the spectral norm of the error transition matrix, driving the reconstruction error to zero and expanding the associative recall capacity of the global delta memory.

---

## Addendum 2026-09-03 — scope correction (Flags A–C)

*Drafted 2026-09-04 from the regenerated probe artifact `results/feat_map_probe.json` (the
2026-06-08 artifact's legacy fields reproduce bit-exactly except one last-ULP float
[`quad2_lowrank.std` …011 → …016, relative 3.2e-16 — reduction-order noise of the regenerating
machine's BLAS]; source: committee report 11,
`committee/brainstorm_2026-09-03/11_theory_capacity_expressivity.md`). The original text above is
preserved verbatim; this addendum corrects its **scope**, not its algebra. Theorem 1 and Theorem 2
stand as proven norm statements. What does not survive is what §4–§5 *implied*.*

### Number provenance first: the §4 crosstalk values do not match this repo's own committed artifact

Section 4 quotes `quad2 ≈ 0.076` and `quad2_lowrank ≈ 0.085`. The committed raw artifact
`results/feat_map_probe.json` (written by `feat_map_probe.py`, single commit 55e03ac, regenerated
2026-09-04 with legacy values reproduced to 1 ULP — see the header note) has **always** measured:

| map | §4 quotes | committed artifact (D=128, d_h=32, 256 draws, seed 42) |
|---|---|---|
| `none` | ≈ 0.142 | **0.14228** (matches) |
| `quad2` | ≈ 0.076 | **0.11699** (does **not** match; ratio 1.54) |
| `quad2_lowrank` | ≈ 0.085 | **0.12648** (does **not** match; ratio 1.49) |

The 0.076/0.085 figures trace to the "prior code path" the probe's own docstring flags as the
origin of the 0.085 absolute bar; no committed script or artifact reproduces them. Everything below
therefore uses the artifact's numbers, and the §4 bullet's relative ordering (none > lowrank > quad2)
remains correct even though its absolute values are unreproducible.

### Flag A — the Gershgorin capacity bound cannot explain the repo's own MQAR D=128 PASS

What $N$ means here, carefully: §5 states the corollary for "MQAR with $N$ key–value pairs"; in the
repo's harness the MQAR rung label $D$ **is** the pair count (`seq/tasks.py`, `MQAR(vocab, num_pairs=D)`),
and reads are issued after all $D$ writes (dense-query protocol), so the hard rung stores
$N = D = 128$ associations that must be *simultaneously* held. Explaining the D=128 PASS therefore
requires an effective capacity $N^* \ge 128$.

The Gershgorin corollary of §5, evaluated at the artifact's measured crosstalk, gives
$N_{quad2} < 1 + 1/0.11699 \approx \mathbf{9.55}$ (and $N_{lowrank} \approx 8.91$; $N_{none} \approx 8.03$) — even
*stronger* failure than the $N<14.2$ computed from the unreproducible 0.076. A bound that caps
capacity at $N<10$ **cannot** explain a pass at $N=128$; at most it is consistent with the
*observation* that quad2 passes where none fails, but the bound itself is violated by the repo's own
headline result (README: "MQAR (D=128) PASS", raw artifact `results/gpu_bench.json`). The bound
explains the **linear baseline's failure** only (and there the simpler rank cap $d_\phi = 32 < 128$
suffices). Any text reading §5 as "resolving the capacity block on MQAR D=128" is an overclaim by
implication.

### Flag B — §5 conflates the crosstalk *mean* with the fluctuation that limits recall

The step $\mathbb{E}[\|E\|_2] \approx (N-1)\,\mathbb{E}[|E_{ij}|]$ is a worst-case row-sum bound: it
is tight only if all off-diagonal entries in a row align in sign. They are approximately zero-mean
random variables; random-matrix reality (Wigner-type row noise) gives $\|E\|_2 \approx 2\sigma_2\sqrt{N}$
for the *spectral* quantity, and the *per-row recall noise* that MQAR scores grows as
$\sigma_2\sqrt{N-1}$ — not $(N-1)\,\text{cross}(\phi)$. Theorem 2 itself is correct (it is a norm
bound and its proof is untouched); only the *capacity corollary* built on it is loose by a factor
growing with $N$. The candidate replacement law and its derivation sketch live in
`docs/crosstalk_capacity_law.md`.

### Flag C — the correct capacity comparator is the second moment, not the mean; the law is a CANDIDATE, not yet fitted or validated

`feat_map_probe.py` now additionally reports (exact definitions in its module docstring, tested in
`tests/test_crosstalk_metrics.py`): the common mode $\mu_{\text{signed}}$ of the split
$E = \mu\,11^\top + W$, the fluctuation $\sigma_2$ of $W$ (both as std of $|\cos|$ — the
pre-registered probe definition — and of signed cosines), the crosstalk efficiency
$\eta = 1/(\sigma_2^2 d_\phi)$, and $N^* = \min(1 + 1/\sigma_2^2,\ d_\phi)$ at $\varepsilon=1$.
Measured (random keys, seed 42):

| map | $\sigma_2$ (\|cos\| def) | $\sigma_2^{W}$ (signed) | $\mu_{\text{signed}}$ | $\eta^{W}$ | $N^*_{\|cos\|}$, $\varepsilon{=}1$ |
|---|---|---|---|---|---|
| `none` | 0.1052 | **0.1769** (floor $1/\sqrt{32}=0.1768$) | +0.00003 | **0.998** (floor 1) | 32.0 (rank-capped) |
| `quad2` | 0.0872 | 0.1458 | **+0.00639** | 0.184 | 132.5 |
| `quad2_lowrank` | 0.0952 | 0.1575 | +0.01601 | 0.294 | 111.4 |

Three honesty notes against committee report 11's arithmetic (the repo pays for accuracy, not for
agreeing with the commission):

1. **Conversion factor slip.** Report 11 converts mean→fluctuation with $\sigma_2 \approx
   \text{cross}\cdot\pi/2$; the correct half-normal factor is $\sqrt{\pi/2} \approx 1.2533$. Verified
   on `none`: $0.14228\times\sqrt{\pi/2} = 0.17833$ vs measured 0.17692 (−0.8%); the $\pi/2$ factor
   would overestimate by +26%.
2. **The $\eta$ numbers in report 11 inherit the stale 0.076.** With measured values, quad2's
   $\eta^W = 0.184$ (report 11 claimed ≈ 0.27), and `none` sits **at** the random floor
   $\eta^W \approx 1.0$ (report 11 claimed ≈ 0.63 — an artifact of the wrong conversion).
3. **The common-mode prediction is validated.** Report 11 predicts $\mu \approx \lambda/d_h \approx
   5.6\times10^{-3}$ for quad2; measured $\mu_{\text{signed}} = +6.39\times10^{-3}$ (ratio 1.14).
   The mean/fluctuation split itself is real and measured.

**And the law at $\varepsilon=1$ does not yet explain the D=128 PASS either.** With random-key
$\sigma_2$: the $|\cos|$-definition gives $N^* = 132.5$ (borderline vs 128), but the SNR-consistent
signed second moment $\sqrt{\mathbb{E}[c^2]} = \sqrt{\sigma_2^{W\,2} + \mu^2} = 0.1459$ gives
$N^* = 48.06 \ll 128$. Report 11's "consistent with $\varepsilon \approx 0.9$–$1.2$" relied on the
unreproducible 0.076; the measured numbers imply $\varepsilon \approx 1.64$ would be needed. The
law is therefore a **candidate** with an O(1) tolerance constant that must be *fitted once at D=64*
and *frozen before* the deferred D-frontier run — protocol, pass/fail bars, and kill conditions are
pre-registered in `docs/crosstalk_capacity_law.md` (registry id PR-2026-09-03-02, status IN-WRITE).
If the fitted $\varepsilon$ lands far from 1, the law is demoted, and Flags A/B stand alone: the
honest current statement is *"no bound in this document explains the quad2 D=128 PASS; explaining it
is the open job of the capacity-law program."*
