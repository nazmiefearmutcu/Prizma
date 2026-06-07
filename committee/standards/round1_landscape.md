# Council 3 — Standards & Landscape Council · Round 1

**Mandate:** Judge whether the Prizma-Seq / PRISM-Seq claim of being "DRAMATICALLY Pareto-dominant over a tuned Transformer" is *real against the actual state of the field*, and set the bar that prevents a strawman victory.

**Scope of the project under review (recap, no shared context assumed):** A non-Transformer sequence architecture in the linear-attention / DeltaNet / state-space family — a Gated-DeltaNet-style token mixer with a parameter-free quadratic feature map giving a *rectangular* associative "delta" state, plus a verified O(1)/constant-memory streaming inference path. Target: dramatic Pareto-dominance over a param/FLOP-matched tuned Transformer at SMALL scale (≤~2M params; char-level LM + attention-diagnostic tasks: MQAR / induction / selective-copy), with ONE scale-up confirmation at ~10–50M.

Date: 2026-06-07. All citations 2024–2026. Evidence, not vibes.

---

## PART A — THE THREE META-QUESTIONS

### Q1. HAS the Transformer actually been surpassed in the field?

**Short answer: NO — not "surpassed" in the strong sense at scale. It has been *matched* at small/medium scale on LM perplexity, *beaten* on a narrow set of efficiency/long-context axes, and *decisively out-competed in production only by HYBRIDS that keep ~7–25% attention.* Pure linear-attention / SSM has not dethroned attention; it has forced attention into a smaller, cheaper role.**

By axis:

| Axis | State of play (2024–2026) | Who leads |
|---|---|---|
| **LM perplexity / downstream avg, small–medium scale (≤~3B, ≤~3T tok)** | Linear-attention/SSM now **matches** a tuned Transformer. Gated DeltaNet (ICLR'25) reaches Wiki ppl **16.42 vs Mamba2 16.56** and commonsense avg **55.32 vs 54.89** at 1.3B. RWKV-7 (2.9B) **matches 3B English SoTA and sets multilingual 3B SoTA on fewer tokens**. | **Tie** (alternatives caught up) |
| **LM quality at *frontier* scale (≥30B, agentic/reasoning/knowledge)** | No *pure* non-attention model is at the frontier. Every frontier-competitive "alternative" (Jamba, Nemotron-H/Nemotron-3, Falcon-H1, Qwen 3.6) is a **hybrid that retains attention**. | **Attention (inside hybrids)** |
| **Associative recall / copying / in-context retrieval** | Attention wins **decisively and provably**. "Repeat After Me" (Jelassi et al., ICML'24) proves a 2-layer Transformer copies exp-length strings with params **logarithmic** in length, while *any* fixed-state model needs state **linear** in length; empirically Pythia-410M beats Mamba-2.8B on phone-book lookup once L≥70. This is the single hardest, most durable Transformer moat — and it is exactly PRISM's target axis. | **Attention (decisive, theory-backed)** |
| **State tracking / "stateful" reasoning (S5, parity, automata, NC¹)** | Subtle. "Illusion of State" (Merrill et al., ICML'24) shows *diagonal* SSMs (Mamba/S4/S6) are in TC⁰ — **no better than Transformers** and provably can't track some automata. But newer **delta-rule** models with *negative eigenvalues* (DeltaProduct, RWKV-7) provably **recognize all regular languages / NC¹**, exceeding Transformers under standard conjectures. | **Split — new linear RNNs lead on formal state-tracking; diagonal SSMs do not** |
| **Decode throughput / KV-cache memory** | Alternatives win **cleanly**. Mamba ~**4–5× decode throughput** of a same-size Transformer (no KV cache → bigger batches); Nemotron-H **up to 3× faster** at 65K context by replacing ~92% of attention with Mamba. | **Alternatives / hybrids (decisive)** |
| **Long-context (≥128K)** | Hybrids win on cost. Jamba-1.5 = only model with effective 256K on RULER while cutting KV cache ~**10×** (~9GB vs order-of-magnitude more). Pure SSM extends *length* cheaply but *degrades on retrieval* in long context. | **Hybrids (cost+length); attention still needed for retrieval quality** |
| **Length extrapolation** | Mixed. Gated DeltaNet/RWKV extrapolate beyond train length on LM ppl; but on *copying* GSSMs drop to ~0 immediately beyond train length while Transformers (ALiBi/NoPE) generalize to 20× (≤50→1000). | **Task-dependent** |

**Bottom line for Q1:** "Surpassed" is true only on **throughput/decode-memory** and (newly) **formal state-tracking via delta-rule RNNs**. On the headline LM axis it is **parity at small/medium scale, not dominance at scale**. On **recall/copying it is the Transformer that dominates, provably** — and that is PRISM's chosen battlefield, which raises the bar.

### Q2. BY HOW MUCH? (numerical deltas)

**Where alternatives lead:**
- **Decode throughput:** ~**4–5×** (Mamba vs same-size Transformer); operator-level studies show the crossover is sequence-length dependent — Transformers ~1.8× faster at short seq, SSMs up to **~4× faster at ~57K** and can run **~220K tokens on a 24GB GPU (~4× longer)**.
- **KV-cache / decode memory:** **~10× smaller** (Jamba-1.5 at 256K ~9GB) → enables far larger batch / longer context at fixed VRAM. This is **O(L) → O(1)** in state, the structural win PRISM also claims.
- **LM perplexity at parity scale:** the lead is **small and within-noise** — Gated DeltaNet beats Mamba2 by ~**0.14 Wiki ppl** and ~**0.4 pts** commonsense at 1.3B. These are *peer-vs-peer* (linear vs linear) gains, **not** "beat a tuned Transformer by a wide margin."
- **Formal expressivity:** DeltaProduct/RWKV-7 cross from TC⁰ into NC¹ / all-regular-languages — a *qualitative* (not %) advantage on synthetic state-tracking that attention cannot match under standard conjectures.

**Where alternatives trail:**
- **Copying/recall, long inputs:** the gap widens to **~100 points** (near-perfect Transformer vs ~0% GSSM) once input length exceeds the fixed-state capacity; phone-book lookup: **410M Transformer > 2.8B Mamba** at L≥70 — a ~**7× param disadvantage overcome by architecture**.
- **5-shot MMLU / in-context learning at scale:** pure Mamba/Mamba-2 "lag significantly"; hybrids *partially* close it by re-adding attention but **pure architectures don't catch up**. (Qualitative in the lit; the fix is *adding attention back*, which is the tell.)
- **State-tracking for *diagonal* SSMs:** **zero** advantage over Transformers (both TC⁰) — Mamba provably can't do some automata/chess-state tasks.

**Net:** alternatives win **multiplicatively (4–10×) on efficiency/memory/length**, win **qualitatively on formal state-tracking (delta-rule only)**, lose **by up to ~100 points on recall/copying at length** and trail on **ICL/MMLU at scale** — which is why the field went hybrid rather than pure.

### Q3. INDUSTRY DIRECTION — what is a credible "Transformer alternative" in 2026?

**The field has converged on HYBRIDS, not pure alternatives.** The 2026 production frontier (Jamba/Jamba-1.5, NVIDIA Nemotron-H → Nemotron-3 incl. a 550B MoE hybrid, Falcon-H1, Qwen 3.6 "linear attention + sparse MoE") is **attention+linear hybrids**. The empirical consensus recipe:
- **~7–8% of layers are attention** (Nemotron-H fixes 4 attention layers; Jamba uses 1:7 attention:Mamba; "7–8% attention-to-total is reasonable"). The rest are linear/SSM.
- Token-mixer of choice is moving from diagonal SSM (Mamba2) toward **gated delta-rule** linear attention (Gated DeltaNet, RWKV-7) for better recall + state-tracking.
- **Test-time-memory** (Titans, Behrouz et al., NeurIPS'25) is the live research frontier — deep neural memory updated at inference, 2M-token contexts, beats GPT-4 on BABILong — but it is *additive memory*, still typically paired with a short attention window.
- Theory is unifying the two ("How Many Heads Make an SSM?", Dec'25): attention and SSM as instances of one operator; choice is task-dependent, **not binary**.

**A credible 2026 "Transformer alternative" therefore looks like ONE OF:**
1. A **hybrid** that hits frontier quality at materially lower decode cost (the production-proven path), OR
2. A **pure** linear/delta-rule token mixer that **provably matches attention on recall/state-tracking diagnostics** AND keeps the O(1) decode win — i.e., it closes the *recall moat* without re-adding attention.

PRISM is attempting path 2 (with an optional bounded-hybrid Pareto-partner). That is the **hardest** path and the one the field has *not* yet completed at scale — which is both why it is interesting and why the bar must be high.

---

## PART B — THE STANDARD (the bar that prevents a strawman win)

PRISM's claim is deliberately *scoped*: "dramatically Pareto-dominant **in the tested regime**" (≤~2M params, char-LM + diagnostics, +1 confirmation at 10–50M). That scoping is legitimate **only if** the regime claim is airtight and not won by handicapping the baseline. The standard below is what makes the scoped claim credible.

### (a) What counts as a NON-STRAWMAN matched Transformer baseline at small scale

A baseline TF is non-strawman only if **all** hold:

1. **Architecture is modern-standard, not 2017-vanilla.** Pre-LN, RoPE *and* an ALiBi/NoPE variant tested (length-extrapolation-friendly PE matters at this scale — see Repeat-After-Me), SwiGLU or GeLU MLP, RMSNorm, no learned abs-pos. Report the exact config.
2. **Per-model tuning, symmetric effort.** Each model trained at *its own* best LR + warmup + schedule from the **same grid** (the project's own logs show the TF was previously under-trained and LR-fragile on MQAR; the fix — per-model gentle-warmup, generous budget, plateau-stop — must be applied **identically to every TF arm at every D**, including the hard D=128 arm). Document the grid and the selected point per model.
3. **Param-matched AND FLOP-matched arms both reported.** Param-match alone is insufficient because PRISM-quad2 currently costs **~2.1–2.7× TF FLOPs/token** (rectangular state buffers). The honest comparison needs **(i)** a param-matched TF and **(ii)** an *iso-FLOP* TF (bigger TF at PRISM's FLOP budget). A win that evaporates at iso-FLOP is not Pareto-dominance.
4. **Adequate training budget to the phase transition.** MQAR/induction have late, sharp transitions; the TF must be trained to its own plateau, not stopped near PRISM's faster convergence. Under-budgeting the TF = strawman (this was the project's recurring false-fail).
5. **Frozen, disjoint, leakage-checked eval set**, identical across models; verify the TF *can* score low-and-high on the *same* frozen set the alternative scores high on (already done at D=128: TF 0.016 vs quad2 0.999 on the same set — keep this discipline).
6. **The "did it LEARN vs CAN'T represent" disambiguation.** Before claiming a recall win, run the decisive flip-test: can the *matched* TF memorize a single hard batch / cross threshold under a constant-high-LR long run? If any TF arm crosses ~0.5 on the hard rung, the failure was optimization, not capacity, and the strong "attention fails" framing is **forbidden** — the honest claim becomes "the matched TF did not learn this under tuned recipes," which is weaker.

### (b) Which axes a win on is MEANINGFUL vs TRIVIAL

| Win | Verdict | Why |
|---|---|---|
| **char-LM BPC margin** (≥0.03) at iso-param **and** iso-FLOP, multi-seed | **Meaningful** if it survives iso-FLOP; **trivial** if it only holds at iso-param while costing 2–3× FLOPs | Quality must not be bought with compute |
| **O(1)/constant-memory decode** (verified step==forward, flat memory vs length) | **Meaningful & structural** | This is the real, durable differentiator (matches the field's 10× KV-cache / 4–5× throughput story); must be *measured*, not asserted |
| **All-n wall-clock latency win** (short *and* long seq) | **Meaningful** — and harder than it sounds: literature shows TF is ~1.8× faster at *short* seq; an *all-n* win including short sequences is a strong claim that must be measured on fixed hardware with warmup, not FLOP-extrapolated | Crossover is length-dependent; "all-n" is the non-trivial version |
| **MQAR/induction/selective-copy parity (incl. the hard recall rung)** | **The decisive axis** — parity here is what separates PRISM from "just another efficient-but-forgetful SSM"; a *win* over a properly-tuned TF here would be genuinely notable given Repeat-After-Me | This is the Transformer's provable moat; parity is necessary for the whole thesis |
| **per-FLOP parity (≤1.0×)** | **Meaningful as a guardrail**, but note current state is 2.1–2.7×; ≤1.0× is an *open* engineering target (Lever D/F), not yet achieved | Without it, "Pareto" is false on the FLOP axis |
| Beating a *vanilla/untuned/under-trained* TF | **Trivial / forbidden** | Strawman |
| Win at iso-param while losing at iso-FLOP | **Not Pareto-dominance** | Pareto requires no axis worse |

### (c) Scale / seed / statistics evidence required

- **Seeds:** ≥10 seeds per arm per task (project already targets ≥10 + TOST — good; TOST is the *right* tool for parity claims, not just NHST). Report mean ± 95% CI and the **solve-rate** (fraction of seeds crossing a task threshold), because MQAR is bimodal at this scale — averages hide the bimodality.
- **Equivalence testing:** for *parity* claims use **TOST** with a pre-registered equivalence margin; for *superiority* claims (BPC win, latency win) use a one-sided test powered to detect the claimed effect (report the power calc and the MDE).
- **Pre-registration:** fix the metric, margin, eval set, budget, and stopping rule *before* the runs (prevents garden-of-forking-paths). The project's plateau-stop + frozen-eval discipline is the right foundation; write it down as a protocol.
- **Ablations (causal, required):** `use_window=False`, random-linear-projection control vs the quadratic feature map, intermediate-D recall frontier (not just D=64 and D=128), and a feature-map-off (`none`) arm — to attribute the recall lift to the rectangular-delta-state mechanism, not incidentals.
- **FLOP ledger published** (the project has `flop_ledger.py`): every comparison states param count AND FLOPs/token for every arm.
- **Scale-up confirmation:** one 10–50M run is the **minimum**; it must show the small-scale Pareto picture **does not invert** (i.e., the BPC margin and the recall parity persist or improve, and the latency/memory win holds). A single seed at scale-up is acceptable for *confirmation* (not claim-establishment) **only if** small-scale is fully powered; ≥3 seeds at scale-up would be stronger and is recommended.

### (d) VERDICT on whether the stated success constitutes a CREDIBLE "dramatically Pareto-dominant in the tested regime" claim

**The project's §3 target — char-LM ≥0.03 BPC win + all-n latency win + per-FLOP ≤1.0× + retained memory/param-efficiency, ≥10 seeds + TOST, + one 10–50M confirmation — IF FULLY MET AS WRITTEN, CLEARS THE BAR for a *scoped, honest* claim, with three mandatory framing/condition guardrails:**

**It clears the bar because** it already encodes the things that usually make such claims fraudulent: a *measured* O(1) advantage (not asserted), a *per-FLOP ≤1.0×* condition (which, if achieved, kills the "won by spending compute" objection — the single biggest current hole, given today's 2.1–2.7× FLOP gap), powered multi-seed stats with the *correct* test (TOST for parity), and a scale-up that guards against small-scale-only artifacts. The combination "O(1) memory + all-n latency win + per-FLOP parity + char-LM margin" is, on the efficiency/structural axes, *genuinely* Pareto-dominance in the tested regime — and it matches the kind of advantage the field actually credits (the 4–10× efficiency story).

**But it needs RAISING / TIGHTENING on these points, or the word "dramatically" is not earned:**

1. **The recall/diagnostic axis must be an explicit pass/fail GATE, not folded into "char-LM win."** Given Repeat-After-Me's *provable* Transformer moat on copying/recall, a "Pareto-dominance" claim that wins on char-LM + efficiency but only achieves *parity (or worse)* on MQAR/induction/selective-copy is **Pareto-dominant only if recall is genuinely ≥ parity**. Require: **MQAR (incl. the hard rung) + induction + selective-copy at ≥ tuned-TF parity (TOST), with the hard-rung result protected by the optimization-vs-capacity flip-test.** If recall is merely "not much worse," the honest word is **"competitive/Pareto-efficient," not "dramatically dominant."**

2. **"Dramatically" must be reserved for a *multi-axis simultaneous* win, and the per-FLOP ≤1.0× condition is currently UNMET (2.1–2.7×).** As written, ≤1.0× is a *target*; until Levers D/F actually deliver it, the claim is **conditional**. The bar: "dramatic" requires **(BPC win ≥0.03 at iso-FLOP) AND (all-n latency win) AND (O(1) memory) AND (recall ≥ parity)** *all at once, all powered*. Drop any one and the correct word downgrades to "Pareto-efficient in the tested regime."

3. **The scope must be stated in the same breath as the claim.** "Dramatically Pareto-dominant" is only defensible with the rider **"at ≤~2M params on char-LM + attention diagnostics, with one 10–50M confirmation — NOT a claim about frontier-scale LM quality, MMLU/ICL, or natural-language long-context retrieval, where attention and hybrids still lead."** Without that rider the claim over-reaches what the *field* has established and becomes a strawman-of-scale (winning small while implying large).

**One genuinely missing piece the field would demand and §3 omits:** a **head-to-head against a strong *small* HYBRID baseline** (e.g., a tiny Samba/Gated-DeltaNet-H-style "mostly-linear + 1–2 attention layers" at the same param/FLOP budget). The field's verdict is that hybrids dominate *both* pure-attention and pure-linear. If PRISM (pure-O(1)) is "dramatically dominant," it should at minimum be **Pareto-competitive with a matched tiny hybrid** on the diagnostics — otherwise the honest framing is "best *pure*-O(1) point on the frontier," not "dramatically beats the Transformer." Recommend adding a matched tiny-hybrid arm as a third reference point (the project already contemplates a bounded-hybrid Pareto-partner — make it a *baseline*, not just an alternative product).

---

## PART C — THE BAR (checklist form)

A claim of "dramatically Pareto-dominant in the tested regime" is credible **iff every box is checked**:

- [ ] **Baseline is modern** (pre-LN, RoPE+ALiBi/NoPE, SwiGLU/RMSNorm), config published
- [ ] **Per-model tuning, symmetric grid**, selected point documented per arm, *applied to the hard recall rung too*
- [ ] **Both iso-param AND iso-FLOP TF arms** reported; FLOP ledger published per arm
- [ ] **TF trained to its own plateau** (no under-budgeting), frozen disjoint leakage-checked eval set, same set scored by all arms
- [ ] **Optimization-vs-capacity flip-test** run on the hard rung; framing matches result (no "attention fails" if a TF arm crosses threshold)
- [ ] **char-LM BPC win ≥0.03 survives at ISO-FLOP** (not just iso-param)
- [ ] **MQAR (incl. hard rung) + induction + selective-copy ≥ tuned-TF parity (TOST)** — explicit gate
- [ ] **O(1)/constant-memory decode measured** (step==forward, flat mem vs length)
- [ ] **All-n latency win measured** on fixed hardware (incl. short seq, with warmup) — not FLOP-extrapolated
- [ ] **per-FLOP ≤1.0×** actually achieved (not just targeted) for the headline arm
- [ ] **≥10 seeds/arm/task, mean±95% CI + solve-rate; TOST for parity, powered one-sided test (with MDE) for superiority; pre-registered protocol**
- [ ] **Causal ablations:** feature-map-off, random-projection control, window-off, intermediate-D frontier
- [ ] **Matched tiny-HYBRID reference arm** included (Samba/GatedDeltaNet-H-style), PRISM Pareto-competitive with it
- [ ] **10–50M confirmation** shows the small-scale Pareto picture does **not invert** (margin + recall parity + latency/memory hold)
- [ ] **Scope rider stated with the claim** (small-scale char-LM + diagnostics; NOT frontier quality / MMLU / NL long-context retrieval)
- [ ] **Word discipline:** "dramatically dominant" only if BPC(iso-FLOP) + all-n latency + O(1) mem + recall-parity hold *simultaneously, powered*; else "Pareto-efficient/competitive in the tested regime"

---

## SOURCES (title — arXiv id — 1-line takeaway)

- **Repeat After Me: Transformers are Better than State Space Models at Copying** — arXiv:2402.01032 (ICML'24) — *Proves Transformers copy exp-length strings with log-params while fixed-state models need state linear in length; Pythia-410M beats Mamba-2.8B on phone-book lookup. The recall moat is real and provable.*
- **The Illusion of State in State-Space Models** — arXiv:2404.08819 (ICML'24) — *Diagonal SSMs (Mamba/S4/S6) are in TC⁰, no more expressive than Transformers for state-tracking; can't track some automata/chess state.*
- **Gated Delta Networks: Improving Mamba2 with Delta Rule** — arXiv:2412.06464 (ICLR'25) — *Gating + delta rule beats Mamba2 (Wiki ppl 16.42 vs 16.56; commonsense 55.32 vs 54.89 at 1.3B); hybrids with sliding-window attention / Mamba2 are best. Peer-vs-peer gains, small margins.*
- **RWKV-7 "Goose" with Expressive Dynamic State Evolution** — arXiv:2503.14456 — *Generalized delta rule w/ vector gating; 2.9B matches 3B English SoTA + multilingual 3B SoTA on fewer tokens; recognizes all regular languages / state-tracking beyond Transformers under standard conjectures; constant memory/time per token.*
- **DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products** — arXiv:2502.10297 — *Multi-step Householder delta updates push linear RNNs into stronger state-tracking; up to 3 layers beats RWKV-7 on expressivity.*
- **Titans: Learning to Memorize at Test Time** — arXiv:2501.00663 (NeurIPS'25) — *Deep neural long-term memory updated at test time; 2M-token context; beats GPT-4 on BABILong. The test-time-memory frontier (still paired with short attention).*
- **Jamba: A Hybrid Transformer-Mamba Language Model** — arXiv:2403.19887 — *First production Attention-SSM hybrid >7B; Jamba-1.5 = only model w/ effective 256K on RULER, ~10× smaller KV cache; 1:7 attention:Mamba ratio.*
- **NVIDIA Nemotron Nano 2 / Nemotron-H: Accurate, Efficient Hybrid Mamba-Transformer** — arXiv:2508.14444 — *~92% of attention replaced by Mamba (≈4 attention layers, 7–8% ratio); up to 3× faster at 65K context while matching/exceeding MMLU/GSM8K/HumanEval/MATH.*
- **Falcon-H1: A Family of Hybrid-Head Language Models** — arXiv:2507.22448 — *Parallel attention+SSM hybrid; 34B matches/beats 70B-class (Qwen2.5-72B, Llama3.3-70B). 2026 production hybrid.*
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — arXiv:2312.00752 — *Selective SSM; ~5× decode throughput vs same-size Transformer via no-KV-cache larger batches; the canonical O(1)-state efficiency claim.*
- **Characterizing SSM and Hybrid LM Performance with Long Context** — arXiv:2507.12442 — *Operator-level: Transformer ~1.8× faster at short seq, SSM up to ~4× faster at ~57K, ~220K tokens on a 24GB GPU (~4× longer). The latency crossover is length-dependent — relevant to PRISM's "all-n latency win" claim.*
- **(context) "How Many Heads Make an SSM?" / unifying attention & SSM (Dec'25)** — *Attention and SSM as one input-dependent operator; choice is task-dependent, not binary — supports a hybrid/unified field direction.*
