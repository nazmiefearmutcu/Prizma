# PRISM-Seq — Implementable Architecture Specification

**Status:** Committee-agreed build contract. This is the single source of truth the architects implement against. It synthesizes the three candidate designs and the three judge reports into ONE buildable PyTorch/MPS architecture, then states the exact recurrence, the falsifiable bar, the experiment matrix, the causal ablations, and the honest failure section.

**Winning design (committee verdict):** **single-state Predictive-Coding Gated-DeltaNet** mixer (Designs 2/3 core — confirmed by all three judges as the only variant that fits the minutes-per-run MPS budget and reuses the existing `seq/` harness), built with **Design 3's exact free-energy derivation** as the canonical novelty defense, **L2-normalized keys** (Designs 1/3, for an exact projection erase and crispest recall — NOT Design 2's kernel feature map, which two judges flagged as recall-blurring), and **Design 1's multi-expert precision routing carried as an OPTIONAL later axis only** (the buildability judge measured M=4 at ~34 min/run, which breaks the budget, and identified a chunked-parallel/differentiability hazard in the per-token router). The continual-learning result is a clearly separated secondary axis.

**The claim we license (verbatim, never inflated):**
> *PRISM-Seq is a credible efficient-attention-replacement candidate at small scale, in the tested regime: parameter- and FLOP-matched against a tuned Transformer it clears the field-standard attention-diagnostic suite, is competitive on a small char-LM within an explicit margin, demonstrates a measured linear/constant-memory inference advantage, and its named mechanism is causally responsible (not "a bigger RNN"). Large-scale LM parity and backprop-free parity are NOT claimed and are stated as open frontiers.*

---

## 0. Notation and conventions

| Symbol | Meaning | Shape |
|---|---|---|
| `B` | batch | scalar |
| `T`, `n` | sequence length / context position | scalar |
| `V` | vocabulary | scalar |
| `d` | model width (`d_model`) | scalar |
| `H` | number of heads | scalar |
| `d_h` | per-head width = `d/H` (this IS the recall-state rank `k`) | scalar |
| `C` | training chunk length (default 64) | scalar |
| `w` | exact local-window width (default 16; ≤64) | scalar |
| `x_t` | per-token hidden (pre-mixer, post-LN) | `[B,d]` |
| `q_t,k_t,v_t` | per-head query/key/value | `[B,H,d_h]` |
| `S_t` | carried associative workspace state (per head) | `[B,H,d_h,d_h]` |
| `β_t` | precision/surprise write gate (per head) | `[B,H,1]` |
| `α_t` | data-dependent forget gate (per head) | `[B,H,1]` |
| `ε_t` | prediction error / reconstruction surprise | `[B,H,d_h]` |

Model interface (frozen by `seq/common.py`, do not change): every model is an `nn.Module` with
`forward(inputs: LongTensor[B,T]) -> logits: FloatTensor[B,T,V]`. Tasks emit `(inputs[B,T], targets[B,T], mask[B,T])`; loss/accuracy are masked CE / masked token accuracy. Target hardware: Apple Silicon MPS, 16 GB, **float32, no autocast**.

---

## PART 1 — The mixing operator (the precision-routed workspace that replaces attention)

### 1.1 What is replaced and why

Self-attention forms an `n×n` score matrix: every query attends to every key (O(n²) compute, O(n) KV-cache). PRISM-Seq replaces this with a **carried associative workspace state** `S_t ∈ R^{d_h×d_h}` per head — the cortical workspace `a` of `PRISM.md §1`, finally implemented as a key→value-bound matrix instead of a flat latent (it was never implemented in `src/prism.py`; this is the C1 linchpin gap). Token `t` mixes with all earlier tokens **only through `S`**, never through a pairwise matrix. Mixing is two state operations per token:

1. **READ** (recognition-by-reconstruction): retrieve the value bound to the current query.
2. **WRITE** (precision-gated targeted erase-and-write): bind the current (key, value), erasing any stale binding at that key first.

Cost: **O(n·d_h²)** total training, **O(d_h²) per step / O(d_h²) state** inference — independent of `n`. This is the linear-cost, content-based, input-dependent mixing class attention belongs to, but with a **targeted overwrite** that pure-decay SSMs (S4/RetNet/early-Mamba) lack and that makes MQAR solvable.

### 1.2 Per-token encoder (the input-dependent, per-token, content-based gate)

Every routing/write/read decision is a function of the **current token's content, recomputed every step** (mandatory per C1; PRISM's per-domain recognition-routing recast to per-token):

```
x_t   = RMSNorm( h_t )                         # h_t = residual stream input, [B,d]
qkv_t = W_qkv x_t                              # fused, [B, 3*d]  -> reshape [B,H,3*d_h]
q_t   = L2norm( RoPE_t( qkv_t[...,0] ) )       # [B,H,d_h]   (RoPE on q,k — eq 1.7)
k_t   = L2norm( RoPE_t( qkv_t[...,1] ) )       # [B,H,d_h]   ||k_t||=1 -> exact projection erase
v_t   = qkv_t[...,2]                            # [B,H,d_h]   (NOT normalized)
```

`W_qkv = nn.Linear(d, 3d, bias=False)` — this is **exactly attention's 4·d² projection budget** minus the output proj, so param-matching is trivial-by-construction (Design 2/judge-3 observation). **L2-normalization of `k` is non-negotiable** (`‖k_t‖=1`): it makes the Householder erase `(I − β k kᵀ)` an exact rank-1 projection → exact last-binding overwrite → crispest MQAR recall (judge 1's decisive point). We use L2-norm, **not** Design 2's positive kernel feature map `elu+1` (which blurs key separation and reduces effective slot count; kept only as a documented stability fallback in §1.10).

### 1.3 Precision-weighted write gate (PRISM's bid, per-token) — the unified precision signal

The single precision/surprise driver, read three ways (the named novelty C7.2):

```
ε_t = v_t − S_{t-1} k_t                         # [B,H,d_h]  reconstruction error at key k_t
e_t = ‖ε_t‖²                                    # [B,H,1]    surprise energy  (= PRISM b_m = ½εᵀΠε)
β_t = sigmoid( a_β · e_t + W_β x_t + b_β )       # [B,H,1]    write strength, clamped to (0, 0.99]
α_t = sigmoid( W_α x_t )                         # [B,H,1]    data-dependent forget (Gated-DeltaNet)
```

- `ε_t = v_t − S_{t-1} k_t` **is literally PRISM's error neuron** `ε_m = x_m − W_m f(z_m)` (`PRISM.md §2.1`), read in value-space. This is the load-bearing PC identity, not an analogy.
- `β_t` is PRISM's metaplastic gate `β(E)` with `dβ/dE>0`: **surprising (novel-key) tokens write hard; mastered/familiar tokens decay `β` toward a floor, protecting existing bindings** (`PRISM.md §2.3`).
- `α_t` is the data-dependent decay (PRISM's PGM/ω consolidation as a per-token forget). Decay-alone under-recalls, write-alone under-forgets — both are needed (C1).
- **One driver, two opposite-sign reads:** write-rate `β` (rises with surprise) and read-precision `π` (rises with mastery) are the same `e_t` read with opposite sign (`PRISM.md §2.3`). Competitors compute query-gating and forget-gating from **separate learned projections**; PRISM-Seq derives both from one error signal. B6 must show the read-gate is causally equivalent in function to the write-`β`, or the unification claim is dropped honestly.

`W_β, W_α = nn.Linear(d, H, bias=True)` (one scalar gate per head; `a_β, b_β` learned scalars). Negligible params, fully counted in the ledger.

### 1.4 The workspace WRITE — targeted erase-and-write (DERIVED, not borrowed)

```
S_t = α_t · S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
    = α_t · S_{t-1} + β_t ( v_t − α_t S_{t-1} k_t ) k_tᵀ                   (with ‖k_t‖=1)
```

**Derivation (the killer reframe — Design 3, the strongest defense against "just Gated-DeltaNet with a PC relabel"):** the write is ONE gradient step on PRISM's per-token free-energy functional. Define the per-token free energy
```
F_t(S) = ½ ‖ v_t − S k_t ‖²_{Π_t}              (the §2.1 module-error term, in state-space)
∂F_t/∂S = −( v_t − S k_t ) k_tᵀ = ε_t k_tᵀ      (the open error neuron times the trace)
S_t = α_t S_{t-1} − β_t · ∂F_t/∂S |_{S=S_{t-1}}  = α_t S_{t-1} + β_t ε_t k_tᵀ
```
So the DeltaNet/Householder erase-and-write **falls out of predictive coding**: `β_t Π_t ε_t k_tᵀ` is exactly `PRISM.md §2.4`'s local rule `dW ∝ (Π ε) ⊗ r` with `r = k_t`, and the eligibility trace `Tr_m ~ f(z)ε` IS the outer product `ε_t k_tᵀ`. The `(I − β k kᵀ)` factor ERASES the old binding at `k_t` before writing `v_t` — a later key overwrites an earlier one. **Additive/EMA accumulation is FORBIDDEN as the recall substrate** (C1): it is the decay-memory family that fails MQAR.

### 1.5 The workspace READ — recognition-by-reconstruction + local-window safety net

```
o_S_t   = S_t q_t                               # [B,H,d_h]  long-range associative readout
o_win_t = WindowAttn_w( q_t, {k,v}_{t-w..t} )    # [B,H,d_h]  exact local recall (C3, w≤64)
o_t     = W_o( concat_heads( o_S_t + o_win_t ) ) # [B,d]
h_{t}'  = h_t + o_t                              # residual; then standard FFN sublayer
```

- `o_S_t = S_t q_t = Σ_j (k_jᵀ q_t) v_j` (up to erases) is the numpy code's argmin-surprise routing (`prism.py:99–107`) recast as a one-shot associative read with **frozen weights, per token**. For near-orthonormal keys `k_jᵀ q_t ≈ δ_{ij}`, so the read returns the last-bound value — the `softmax-argmax(k·q)` analogue.
- `WindowAttn_w` is the field-standard small exact-attention window (Based/Griffin/Gated-DeltaNet) guaranteeing precise local recall while `S` carries long range (C3). **Disclosed in the diagram and the param/FLOP ledger.** `w=16` default (kept small so long-range recall provably needs `S` — B6 gate). It shares `q,k,v` projections with the mixer (no extra projection params; only the FLOPs are counted).
- `W_o = nn.Linear(d, d, bias=False)` — the attention-output-proj analogue.

### 1.6 The FFN sublayer (identical to the baseline — Design 3 discipline)

After the mixer residual, an FFN sublayer **byte-identical to the Transformer baseline's** (`seq/transformer.py`: RMSNorm pre-norm + SwiGLU). This guarantees that **only the mixing operator differs** between PRISM-Seq and the Transformer, making the comparison an unambiguous attention-replacement test and the param-match unimpeachable (judges 2 and 3, graft).

### 1.7 Position and causality

- **Causality is structural and free.** `S_t` is built strictly left-to-right; token `t` reads only `S_{t-1}` (bindings from tokens `<t`). No causal mask matrix is needed (unlike attention). The chunked training form (§2.2) uses strictly-lower-triangular masks so within-chunk reads never see the current/future token's own write.
- **Implicit position** comes free from write order (a later key overwrites an earlier one — order-sensitive by design).
- **Explicit position** is RoPE applied to `q_t, k_t` in the encoder (eq 1.2; parameter-free, composes with the read and aids the window head). This is the C1-allowed bolt-on. The window head also injects exact relative position over the last `w` tokens. (Fallback if RoPE×state degrades recall: apply RoPE only to the window head, leave delta keys position-free — §1.10.)
- **Length extrapolation** is reported honestly (B4b): train at seq=256, report accuracy-vs-length at 512/1024. A drop is allowed and scoped; claiming "solves induction" from train-length-only numbers is forbidden.

### 1.8 The associative-recall mechanism (the make-or-break, fully wired)

MQAR within ONE forward pass, weights frozen, sequence-resident:

1. **Binding presented** `(key_i, value_i)`: encoder emits `k_i` (L2-normed) and `v_i`. The token is novel ⇒ `ε_i = v_i − S_{i-1}k_i` is large ⇒ `β_i ≈ 1` ⇒ WRITE stamps the outer product `v_i k_iᵀ` into `S` after erasing any stale binding at `k_i`. This IS the eligibility-trace Hebbian bind `Tr ~ f(z)ε`; ART vigilance "novel key → fresh direction in `S`, known key → overwrite existing".
2. **Query presented** later: encoder emits `q ≈ k_j`; READ `o_S = S q = Σ_i (k_iᵀ q) v_i ≈ v_j` because `k_jᵀ q ≈ 1` and all other `k_iᵀ q ≈ 0` (near-orthonormal L2 keys). **No weight update at query time** — recall lives entirely in the carried activity `S`.
3. **Re-binding correctness:** if a key is re-bound, the `(I − β k kᵀ)` erase removes the old value first, so recall returns the LAST binding (correct MQAR semantics), not a decayed superposition. This is exactly why additive/EMA fails and delta-rule passes.

**Induction** (`… [A][B] … [A]→[B]`) is the same mechanism: bind successor `B` at key `A`, read at `A` later; gap-distance-invariant because `S` has no positional decay (passes the 64–256 gap bin, B2). **Selective copying** is the same: `β_t` is content-dependent ⇒ writes data tokens (high surprise), skips blanks (low surprise), then reads them out in order. **One mechanism, three tasks — no special-casing** (C2).

**Recall capacity = rank of `S` = `d_h` near-orthogonal bindings per head, `H·d_h` effective across heads** (e.g. `d=64,H=2` → `d_h=32` × 2 heads). This is the structural ceiling attention lacks; it is the **B1b knob**, swept and reported, with a pre-registered `D*` (§Part 5). The window head backstops local recall; B6 must show long-range (64–256 gap) recall survives window-head ablation.

### 1.9 (Optional, gated-OFF by default) Multi-expert precision routing — Design 1, LATER axis only

`M` parallel workspace experts `{S_t^{(m)}}`, precision bid `b_m = π_m ‖ε_t^{(m)}‖²`, router `g_t = softmax_m(−b_m/τ)`, aggregated read `r_t = Σ_m g_{t,m} S_t^{(m)} q_t`. This preserves PRISM's genuine differentiator (precision-routed mixture-of-recurrent-memories) and raises effective rank to `M·d_h`.

**It is NOT in the gating build.** The buildability judge measured `M=4` at ~34 min/run (breaks the minutes budget) and identified that the per-token router's dependence on each expert's carried state is a sequential within-chunk dependency the chunked WY/UT transform assumes away. **Therefore:** the multi-expert axis is implemented ONLY after the single-state model clears the minimal bar, and ONLY as **top-1 hard routing evaluated in the O(1) recurrent decode path** (where the sequential dependency is free), never inside the chunked-parallel training form. If it is built, B6 must verify the router sends a query to the SAME expert that holds its key (else routing splits bindings and breaks MQAR — judge 1's flagged failure mode); if that verification is shaky, fall back to the single shared state. The single-state model is the deliverable that licenses the claim.

### 1.10 Numerical fallbacks (documented, do not silently engage)

- If the chunked triangular solve goes ill-conditioned on MPS fp32: reduce `C` to 32, or fall back to forward-substitution; verify against the sequential reference (§2.3).
- If RoPE×state degrades recall: apply RoPE only to the window head, leave delta keys position-free.
- If L2-key recall is unstable at a config: the Design 2 L1-normalized kernel read `φ̂=elu(u)+1` is the documented stability fallback (accepting the recall-crispness cost, disclosed).

---

## PART 2 — Exact recurrence: O(n) train / O(1)-step inference

### 2.1 The per-step recurrence (inference; O(1) state, O(d_h²) compute)

For one token, per head, carrying `S ∈ R^{d_h×d_h}`:

```
1. project:  q,k,v = encode(x_t);  k,q = L2norm(RoPE(·))           # O(d²)
2. read err: ε   = v − S k                                          # O(d_h²)
3. gates:    β   = σ(a_β‖ε‖² + W_β x_t + b_β) ∈ (0,0.99];  α = σ(W_α x_t)
4. write:    S   = α·S + β·ε·kᵀ                                     # O(d_h²) rank-1 update
5. read:     o_S = S q                                              # O(d_h²)
6. window:   o_w = WindowAttn_w(q, ring[-w:])                       # O(w·d_h), w constant
7. out:      o   = W_o(concat_heads(o_S + o_w))                     # O(d²)
8. ffn:      h   = h + o;  h = h + SwiGLU(RMSNorm(h))               # O(d·d_ff)
```

Per-step cost and memory are **constant in `n`** because `S` is fixed-size `d_h²` and the window ring is fixed-size `w`. There is **no inner settling loop** in the trained recurrence — the PC settling is amortized into the single delta step (one gradient step per token). The "settling-iteration constant" the bar (B5) requires be counted is exactly these fixed per-step FLOPs. State carried at decode = `H·d_h²` floats per head per layer (e.g. `H=2, d_h=32, L=2` ≈ 4K floats) plus the `w`-token window ring — both constant in `n`.

### 2.2 The chunked-parallel TRAINING form (O(n), matmul-bound, minutes on MPS)

A purely sequential settling loop over `n` Python steps is too slow to tune. Use the DeltaNet **WY/UT-transform**: re-express the rank-1 updates over a chunk of length `C` as matmuls + one `C×C` triangular solve, carrying `S` across `⌈n/C⌉` chunks in a short Python loop (e.g. seq=256, C=64 → **4 chunk iterations, not 256 token steps**).

Within a chunk, let `K,V,Q ∈ R^{C×d_h}`, `β ∈ R^C`, `α ∈ R^C` (per head, per batch):
```
# strictly-lower-triangular intra-chunk interaction (mask diag and future)
A    = tril( diag(β) · (K Kᵀ) , −1)                 # C×C
T    = ( I + A )^{-1}                                # one triangular solve (solve_triangular)
W    = T · ( diag(β) V − diag(β) (K S_startᵀ) )      # C×d_h   corrected writes (decay-folded)
S_end = decay_fold(α) · S_start + Wᵀ K               # carry state to next chunk
O_intra = tril( Q Kᵀ , −1) · W                       # C×d_h   intra-chunk reads
O_inter = Q · S_startᵀ                               # C×d_h   reads from carried state
O_chunk = O_inter + O_intra                          # then + window-head, + W_o
```
(`decay_fold` folds the per-token `α` cumulative products into the WY matrices per the Gated-DeltaNet recipe; RoPE is applied to `Q,K` before the products.) Training is **O(n·d_h²/C + n·C·d_h)** — fully matmul-parallel within chunks, no Python token loop. The `C×C` solve (`C=64`) is tiny; the cost is dominated by the `d×d` projection matmuls MPS runs natively. The whole form is **autograd-differentiable** (the triangular solve has a stable backward on MPS) — no custom backward needed at this scale.

### 2.3 Sequential reference (correctness gate, not for training)

A naive Python per-token recurrence (eq 2.1) is implemented as `_delta_reference(...)` and used in tests to verify the chunked form matches it to `< 1e-4` on a tiny case (B,H,T,d_h small), forward AND gradient. This is the numerical-stability guard the failure audit requires.

---

## PART 3 — Training modes (backprop primary + local/DFA bonus + expected tax)

### 3.1 PRIMARY — backprop, end-to-end (the gating path, everything scored on this)

All of `{W_qkv, W_β, W_α, W_o, the chunked delta recurrence, the window head, FFN, embeddings, tied head}` are differentiable torch modules. Optimizer: **AdamW (betas 0.9/0.95, wd 0.1)**, cosine + warmup (the `seq/common.TrainConfig` recipe), **identical to the Transformer baseline**. Per-task LR sweep `lr ∈ {1e-3, 3e-4, 1e-4}`, pick best val. The DeltaNet/Householder structure with L2-normed keys gives the **wide trainable-LR window** (pure-decay memory's razor-thin LR window is avoided precisely because the erase keeps `S` well-conditioned).

### 3.2 BONUS (non-gating, C4) — local / PC / DFA mode with a quantified tax

- The **state write is already local by construction**: the delta step is a one-step PC gradient on `ε_t k_tᵀ` — **no backprop-through-time for the state path**.
- The per-token projections + readout are trained with **fixed-random feedback (DFA)** instead of `Wᵀ`, inherited from the numpy code's `Bdec/Bcls` feedback matrices (`PRISM.md §2.5`).
- **Expected tax:** DFA lags most on long-credit tasks (MQAR rung3, large-gap induction), which is also where the razor-thin-LR instability bites. The local mode gets an **extra LR sweep** and the **trainable-LR window is itself reported as a metric**.
- **Reported as a NUMBER with architecture held constant** (local-PRISM-Seq vs backprop-PRISM-Seq, the SAME model) — NEVER local-PRISM-Seq vs backprop-Transformer (C4 anti-confound). Backprop-free parity is stated as an open frontier, not claimed; this axis is never a gate.

### 3.3 Param/FLOP honesty (C5, hard rule)

Count **EVERY tensor in the forward pass** in a disclosed ledger via `seq/common.count_by_module`, including the fixed window-head and any fixed-random DFA feedback matrices (disclosed as **non-trainable-but-not-free** — memory + FLOPs counted). The numpy doc's "sabit FA matrisleri sayılmaz" stance (`PRISM.md §6.5`) is **FORBIDDEN** here. FLOPs/step (inference) and FLOPs/token (training) are audited and reported for both models so a win cannot come from secretly spending more compute.

---

## PART 4 — PyTorch module breakdown (targets MPS, drops into `seq/`)

All in `seq/prism_seq.py` (~300 lines). Reuses `seq/common.py` (train/eval/param ledger/latency) and `seq/transformer.py` (FFN/RMSNorm/baseline). Float32, no autocast.

```python
@dataclass
class PRISMSeqConfig:
    vocab: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2            # d_h = d_model // n_heads  (= recall-state rank k)
    chunk: int = 64             # C: chunk-parallel training granularity
    window: int = 16            # w: exact local-window head (<=64), disclosed in ledger
    d_ff: int = None            # SwiGLU inner; default matches transformer.TFConfig
    max_len: int = 1024
    rope: bool = True
    beta_cap: float = 0.99
    # multi-expert (OFF by default; later axis only — §1.9)
    n_experts: int = 1

class PRISMSeqBlock(nn.Module):
    """Pre-LN -> precision-routed workspace mixer -> residual -> SwiGLU FFN -> residual."""
    def __init__(self, cfg): ...
        # self.norm1 = RMSNorm(d)                         # reuse transformer.RMSNorm
        # self.W_qkv = nn.Linear(d, 3*d, bias=False)
        # self.W_beta = nn.Linear(d, H, bias=True); self.a_beta, self.b_beta = scalars
        # self.W_alpha = nn.Linear(d, H, bias=True)
        # self.W_o   = nn.Linear(d, d, bias=False)
        # self.window = WindowAttn(w, H, d_h)
        # self.norm2 = RMSNorm(d); self.mlp = SwiGLU(cfg)  # byte-identical to baseline
    def forward(self, h):                  # h: [B,T,d]  -> [B,T,d]   (TRAIN path, chunked)
        ...

    def step(self, h_t, state):            # h_t: [B,1,d]; state carries (S[B,H,d_h,d_h], ring)
        ...                                # returns (o_t[B,1,d], new_state)   O(1) recurrence

def chunked_delta_forward(q, k, v, beta, alpha, S0, chunk):
    # q,k,v: [B,H,T,d_h]; beta,alpha: [B,H,T,1]; S0: [B,H,d_h,d_h]
    # returns reads O: [B,H,T,d_h] and final state S_end: [B,H,d_h,d_h]   (eq 2.2, autograd)
    ...

def _delta_reference(q, k, v, beta, alpha, S0):
    # naive per-token recurrence (eq 2.1) for the < 1e-4 correctness gate (eq 2.3)
    ...

class WindowAttn(nn.Module):
    # exact causal attention restricted to last w tokens via unfold/mask; shares q,k,v
    def forward(self, q, k, v): ...        # [B,H,T,d_h] -> [B,H,T,d_h]

class PRISMSeqLM(nn.Module):
    """TokEmb (+optional learned pos for char-LM parity) -> L x PRISMSeqBlock -> RMSNorm -> tied head."""
    def __init__(self, cfg): ...
        # self.tok = nn.Embedding(V, d); self.blocks = ModuleList(...); self.head tied to tok
    def forward(self, idx):                # idx: [B,T] long -> logits: [B,T,V]   (frozen interface)
        ...
    def init_state(self, batch, device):   # for seq/common.autoregressive_latency(step_api=True)
        ...                                # returns per-layer (S zeros, empty window ring)
    def step(self, tok, state):            # tok: [B,1] -> (logits[B,1,V], new_state)   O(1)/step
        ...
```

Tensor-shape contract (diagnostic config `d=64,H=2,d_h=32`): `q,k,v` `[B,2,T,32]`; `S` `[B,2,32,32]`; `β,α` `[B,2,T,1]`; chunked `O` `[B,2,T,32]`; `logits` `[B,T,V]`. **MPS tricks:** `C≤64`, `H≤4`, `d≤192`; fp32; `solve_triangular` (forward-sub fallback); `torch.mps.synchronize()` around B5 timing; one deterministic driver per bar item → `results.json` with per-seed numbers + param/FLOP ledger + git hash.

---

## PART 5 — The parameter-matched Transformer baseline

**Use `seq/transformer.py` verbatim** (the confirmed non-strawman): decoder-only, RMSNorm pre-norm, causal MHA via `scaled_dot_product_attention(is_causal=True)`, **SwiGLU** MLP (`d_ff ≈ 8/3·d`), learned absolute positions, tied head. PRISM-Seq reuses the SAME RMSNorm/SwiGLU/embedding/head code (§1.6) so **only the mixer differs**.

- **Tuned as hard as PRISM-Seq:** per-task LR sweep `{1e-3,3e-4,1e-4}`, AdamW (0.9/0.95, wd 0.1), cosine+warmup, dropout where it helps. Must clear the well-posedness floor on every task (≥0.98 induction, solves MQAR rung1) BEFORE comparison; else re-tune the task config.
- **±5% trainable-param match at every config**, `d_model/L/vocab/seq` identical. Mixer projections are `4·d²` in both models (PRISM-Seq `W_qkv 3d² + W_o d²` = attention's `qkv 3d² + proj d²`); FFN identical; gates add `O(d·H)` — so ±5% is hit by a single `d_model` nudge.
- **B5 gives the Transformer its best honest path (KV-cache)**, never a denied one.
- Matched sizes: diagnostics (B1–B3, B6) ~110–120K params; char-LM (B4) ~1.4M params.

---

## PART 6 — The FINAL agreed falsifiable bar (numbered pass/fail)

All comparisons are PRISM-Seq vs the param/FLOP-matched, individually-tuned Transformer, same data/seeds/hardware. Every number is mean over **≥3 seeds {0,1,2}** (≥5 for B4) with **95% CIs**. A verdict is declared only when the CI supports it. The discriminating rung is the gate; report ALL seeds and ALL rungs.

1. **B1 — MQAR (decisive gate).** Zoology/Based protocol (`seq/tasks.MQAR`), keys distinct per sequence, loss masked to answer positions, eval = 2000 fresh sequences on a disjoint seed. Run rung1 (seq=128,D=16,Q=4), rung2 (seq=256,D=32,Q=8), **rung3 (seq=512,D=64,Q=16) — the gate.** **PASS:** rung1 & rung2 ≥ 0.95 AND ≥ T−0.02; **rung3 ≥ T−0.03** with CIs not favoring T by >0.03. **FAIL:** collapses to chance on any rung T solves, or trails by >0.03 on rung3.
2. **B1b — MQAR capacity sweep (pre-registered, run FIRST).** Vary `D ∈ {8,16,32,64,128}`; sweep PRISM-Seq recall-state knob (`d_h`/effective rank) ∈ {8,16,32,64}. **PASS:** stays within B1 margins up to a **pre-registered `D*`** (register `D*` and the `d_h` achieving it, with effective near-orthogonal slot count ≥ ~1.5×D*, BEFORE seeing results); recall-vs-state curve reported. **FAIL:** falls below `D*` at every feasible state size ⇒ rescope claim to achieved `D*` (honest) or B1 fails.
3. **B2 — Induction (sanity, subsumed by B1).** `seq/tasks.Induction`, seq=256, scored at induction positions, gap-binned (1–16 / 16–64 / 64–256). **PASS:** ≥ 0.98 AND ≥ T−0.01; **64–256 gap bin ≥ 0.95** (T itself ≥0.98). **FAIL:** high accuracy only in the 1–16 bin (local shortcut, not true induction).
4. **B3 — Selective copying (input-dependent-gating gate).** `seq/tasks.SelectiveCopy`, seq=256, 16 data tokens at RANDOM positions; run BOTH the fixed-spacing control and the selective variant. **PASS:** selective per-token ≥ 0.97 AND ≥ T−0.02; exact-sequence ≥ 0.90; control gate: ≥0.97 on fixed too, AND `PRISM_noGate` (B6) drops on the SELECTIVE variant while staying high on the fixed variant. **FAIL:** cannot exceed a time-invariant-mixing baseline on the selective variant.
5. **B4 — Char-LM (closest, scale-sensitive call).** tiny-shakespeare (contiguous 90/5/5, no boundary leak) + text8 first 5 MB; seq=256; early-stop on val BPC; evaluate FROZEN on TEST. **PASS ("competitive"):** test BPC ≤ T + 0.05 on BOTH corpora, ≥3 seeds (≥5 for the closest), CI not worse by >0.05; must beat a 5-gram / small-LSTM floor. **BONUS:** ≤ T (true parity/win). **Minimal-viable note:** B4 may be reported as "competitive within margin OR explicitly the closest gap" without invalidating the minimal claim, provided B1/B1b/B3/B5/B6 pass cleanly.
6. **B4b — Length extrapolation.** Train B2/B3 at seq=256, test at seq=512/1024; report the curve. **PASS:** report honestly; a scoped drop is allowed; claiming "solves" from train-length-only numbers is forbidden.
7. **B5 — Structural advantage (the "alternative" differentiator, MEASURED).** Same B4 checkpoint, sweep `n ∈ {128,256,512,1024,2048,4096}`. PRISM-Seq via `init_state/step` (O(1)) vs Transformer KV-cache; measure per-token decode latency vs n, peak state/cache memory vs n, prefill+generate-512 time. `torch.mps` memory APIs + `mps.synchronize()`, 20-iter median, settling constant counted. **PASS:** PRISM-Seq latency statistically FLAT in n (slope not >0; within 20% across n=128→4096) while T grows ≥linearly; AND PRISM-Seq peak state memory CONSTANT (within 10%) while T's KV-cache grows ≥4×; AND measured crossover **n\* ≤ 2048**. **FAIL:** cannot run O(1)-state, OR latency/memory also grow with n.
8. **B6 — Mechanism-causal ablations (anti-strawman gate).** On B1-rung3 + B3-selective, all variants param-matched within ±5%. **6.1 internal:** `PRISM_noPrecision` (uniform input-independent gate), `PRISM_noWorkspace` (plain causal linear recurrence, no state broadcast), `PRISM_noDelta` (additive/EMA write — the make-or-break), `PRISM_noRouteReadout` (no content-based read). **6.2 family + controls:** vanilla GRU, minimal linear-attn block, single Mamba/SSM block (param+FLOP matched); uniform-write / learned-softmax-gate / random-gate controls. **PASS:** `noPrecision` and `noDelta` drop selective and/or MQAR-rung3 by ≥0.15; `noWorkspace` trails full on MQAR-rung3 by ≥0.10; precision-gating beats uniform/softmax/random with non-overlapping CIs; full model ≥ GRU/linear-attn/Mamba family. **FAIL:** any internal ablation matches full within CI on both tasks (mechanism not load-bearing ⇒ it is just a recurrent net).

**Minimal Viable Bar (honestly licenses the claim):** **B1 (incl. rung3) + B1b + B3 (with control) + B5 + B6.** B2 is a cheap sanity pass; B4 should clear but may be the closest-gap item. **If MQAR-rung3 (B1) fails OR B5 shows no real structural advantage, the alternative claim is NOT supported — regardless of the other items.**

---

## PART 7 — Experiment matrix (concrete MPS configs)

Serialize GPU jobs (single MPS); parallelize only CPU data-prep across 10 cores. Total ≈ 2.5–3 h serialized, dominated by B4. **Run order: B1b FIRST** (capacity ceiling is the highest-likelihood failure; pre-register `D*` before results).

| # | Experiment | Models compared | Config (d / L / H / seq / steps) | Seeds | Primary metric & pass | Est. time |
|---|---|---|---|---|---|---|
| 0 | **B1b capacity sweep** (FIRST) | PRISM-Seq (d_h∈{8,16,32,64}) vs Transformer | 64 / 2 / 2 / 512 / 4000; D∈{8,16,32,64,128} | 3 | recall-vs-D curve; pass to pre-registered D* | ~30 min |
| 1 | **B1 MQAR** (rung1/2/3) | PRISM-Seq vs Transformer | 64 / 2 / 2 / {128,256,512} / 4000 | 3 | answer-acc; rung3 ≥ T−0.03 | ~30 min |
| 2 | **B2 Induction** (+gap bins) | PRISM-Seq vs Transformer | 64 / 2 / 2 / 256 / 3000 | 3 | ≥0.98, 64–256 bin ≥0.95 | ~12 min |
| 3 | **B3 Selective copy** (selective+fixed) | PRISM-Seq vs Transformer (+noGate) | 64 / 2 / 2 / 256 / 4000 | 3 | selective per-token ≥0.97; control gate | ~25 min |
| 4 | **B4 Char-LM** | PRISM-Seq vs Transformer | 192 / 3 / 4 / 256 / 4000 (val early-stop) | 5 | test BPC ≤ T+0.05, both corpora | ~1.4 h |
| 5 | **B4b length extrapolation** | PRISM-Seq, Transformer (B2/B3 ckpts) | test seq {512,1024} | 3 | report curve honestly | ~5 min |
| 6 | **B5 inference advantage** | PRISM-Seq (step) vs Transformer (KV-cache) | reuse B4 ckpt; n∈{128…4096}, gen 512, 20-iter median | n/a | flat latency + constant mem + n*≤2048 | ~10 min |
| 7 | **B6.1 internal ablations** | full vs noPrecision/noWorkspace/noDelta/noRouteReadout | 64 / 2 / 2; rung3 + selective | 3 | each drops ≥ stated Δ, non-overlap CI | ~45 min |
| 8 | **B6.2 family baselines + gate controls** | PRISM-Seq vs GRU/linear-attn/Mamba; uniform/softmax/random gate | matched params+FLOPs; rung3 + selective | 3 | precision-gate beats controls; ≥ family | ~40 min |

**Per-item deliverable:** deterministic driver + `results.json` (per-seed raw + config + param/FLOP ledger + commit hash). **Suite deliverable:** one-command re-run, honest-limits ledger, borrowed-vs-new table, explicit losses/open-frontiers statement.

---

## PART 8 — Ablations proving the PRISM mechanism is causal (detail for B6)

Each ablation removes exactly one load-bearing PRISM piece, param-matched within ±5% so a drop cannot be blamed on capacity:

| Ablation | What it removes | Predicted effect (the causal signature) |
|---|---|---|
| `PRISM_noDelta` | targeted erase-and-write → additive/EMA `S_t = α S_{t-1} + β v_t k_tᵀ` (no `(I−βkkᵀ)`) | MQAR-rung3 and/or selective drop ≥0.15 — re-bindings superpose, last-binding recall breaks. **The make-or-break write-rule ablation.** |
| `PRISM_noPrecision` | content-based surprise gate → fixed/uniform input-independent gate | selective-copy drops ≥0.15 — cannot skip blanks / write data selectively |
| `PRISM_noWorkspace` | carried-state broadcast → plain causal linear recurrence of equal params | trails full on MQAR-rung3 by ≥0.10 — no associative binding store |
| `PRISM_noRouteReadout` | content-based read `S q` → position/identity readout | MQAR collapses — query no longer routes to its bound key |
| **gate controls** | precision-`β` → uniform-write / learned-softmax-gate / random-gate | precision-gating beats all three with non-overlapping CIs |
| **family baselines** | whole mixer → GRU / linear-attn / Mamba block | full PRISM-Seq ≥ family; if tied, honestly reported as "linear-RNN family member, competitive with Mamba — NOT a Transformer alternative" |
| **window-head ablation** | shrink/remove `WindowAttn_w` with MQAR bindings/induction gaps spaced PAST `w` | long-range (64–256 gap) recall SURVIVES → proves `S`, not the window, carries long range (C3 honesty gate) |
| **precision-unification test** | compare read-gate (π↑) vs write-`β` functional roles | read-gate causally equivalent to write-`β`, or the one-precision-signal novelty is DROPPED honestly |

---

## PART 9 — Honest predicted-failure section

These are the highest-probability failure modes, each with its mitigation and its honest downgrade if the mitigation fails.

1. **Capacity ceiling (the #1 overclaim risk).** Recall caps at `H·d_h` near-orthogonal bindings. At `d=64,H=2` that is ~64 slots, and MQAR rung3 has D=64 — comfortable but not generous. *Mitigation:* B1b runs FIRST, `D*` pre-registered, window head backstops local recall, multi-expert (§1.9) raises effective rank if needed. *Downgrade:* rescope the claim to the achieved `D*` rather than hide the ceiling.
2. **Window-head leakage (C3 honesty risk).** If the local-attention head silently carries the recall the workspace was supposed to, the novelty collapses to "attention with extra steps." *Mitigation:* keep `w=16`; B6 must show 64–256-gap recall survives window ablation; ledger discloses `w`.
3. **"Just Gated-DeltaNet with a PC relabel" refutation.** Linear-cost content-based delta mixing is already owned by DeltaNet/Gated-DeltaNet/Based. *Defensible novelty (must be DEMONSTRATED, not asserted):* (a) the PC free-energy DERIVATION of the delta write (§1.4); (b) the one-precision-signal unification shown causal in B6; (c) task-free continual SEQUENCE modeling (no SSM/linear-attn competitor offers it). *Downgrade:* if B6's unification test fails AND the CL axis is weak, the honest verdict is "a member of the linear-RNN family with novel gating, competitive with Mamba — NOT a Transformer alternative." Linear cost / O(1) decode are MEASURED bonuses, never the novelty (shared with all SSMs).
4. **B4 char-LM lands just outside +0.05 BPC.** The closest, scale-sensitive call. *Mitigation:* tune as hard as the baseline, both corpora + CIs. *Downgrade:* minimal-viable rule reports it as the closest gap without sinking the minimal claim.
5. **Chunked-solve numerical instability on MPS fp32.** `solve_triangular` / `(I+A)^{-1}` can be touchy if `β→1` and keys collinear. *Mitigation:* L2-normed keys keep the factor a clean projection; clamp `β∈(0,0.99]`; `C≤64`; forward-substitution fallback; verify vs `_delta_reference` to <1e-4.
6. **Local/DFA mode tax may read as unusable** on long-credit recall (DFA's known weakness) with a razor-thin LR window. *Mitigation:* strictly a bonus axis, never a gate; report the tax as a number (architecture held constant) and the trainable-LR window as a metric.
7. **Length extrapolation (B4b) may degrade past train length** despite the position-free state. *Mitigation/honesty:* report the curve; scope it as an open limit; never hide it.
8. **Multi-expert routing split (only if §1.9 is built).** A query may route to a different expert than the one holding its key, breaking MQAR — a failure single-state DeltaNet does not have. *Mitigation:* keep §1.9 OFF for the gating build; if built, top-1 in the decode path only, and B6 must verify same-expert routing or fall back to single state.

**The verdict to aim for — and never exceed — is exactly Part 1's claim string.** Large-scale LM parity and backprop-free parity remain UNPROVEN open frontiers. The continual-learning result (PRISM's proven strength) is carried as a clearly separated secondary axis (task-free continual SEQUENCE modeling via recognition-surprise expert recruit/freeze over a non-stationary token stream), with the same honest limits as `PRISM.md §8` (needs input-distinguishable, temporally-contiguous domains), never conflated with sequence competence.

---

### Files referenced
- Brief: `/Users/nazmi/Desktop/Projeler/proje/PRISM/docs/TRANSFORMER_ALTERNATIVE_BRIEF.md`
- PRISM writeup: `/Users/nazmi/Desktop/Projeler/proje/PRISM/docs/PRISM.md`
- Numpy ancestor (no workspace dynamics / token mixing / position — NOT the perf substrate): `/Users/nazmi/Desktop/Projeler/proje/PRISM/src/prism.py`
- Existing harness (reuse): `/Users/nazmi/Desktop/Projeler/proje/PRISM/seq/common.py`, `seq/tasks.py`, `seq/transformer.py`
- Build target (new): `/Users/nazmi/Desktop/Projeler/proje/PRISM/seq/prism_seq.py` + per-bar driver scripts
