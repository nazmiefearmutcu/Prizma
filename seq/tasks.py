"""
Canonical attention-diagnostic synthetic tasks — the suite the field uses to decide whether a
non-attention architecture can really stand in for a Transformer.

All tasks share one causal next-token interface:
    sample(batch, device) -> (inputs[B,T] long, targets[B,T] long, mask[B,T] {0,1} float)
Loss/accuracy are computed only on masked positions. Token id 0 is a reserved filler/pad.

  * MQAR          Multi-Query Associative Recall (Arora et al. 2023, Zoology/Based). THE test that
                  separates real attention-alternatives from impostors: in-context key->value
                  lookup for many queries, over a gap. Linear models with fixed state struggle as
                  #pairs grows; attention solves it trivially.
  * AssocRecall   single-query AR (Ba et al.; Hyena) — easier sanity version of MQAR.
  * SelectiveCopy Mamba's selective-copy: copy the data tokens in order, ignoring interspersed
                  fillers. Requires INPUT-DEPENDENT gating (content-selective memory).
  * Induction     in-context induction-head probe: [.. A B .. A] -> predict B. The ICL primitive.
"""
from __future__ import annotations

import torch


def _distinct_per_row(B, n, lo, hi, device):
    """n distinct ints in [lo,hi) for each of B rows -> (B,n) long. hi-lo must be >= n."""
    span = hi - lo
    perm = torch.argsort(torch.rand(B, span, device=device), dim=1)[:, :n]
    return perm + lo


class MQAR:
    """Multi-Query Associative Recall (Zoology/Based standard, dense). vocab in [1,V); 0 filler.
    Sequence = [k0 v0 ... k_{D-1} v_{D-1}]  (+optional filler gap)  [q_1 q_2 ... q_M], where each
    query q is one of the bound keys. The TARGET at each query position is that key's bound value
    (NOT a next token — the answer never appears in the input), masked to query positions only.
    Dense queries give rich supervision; capacity is set by the #bindings D (the B1b axis)."""

    def __init__(self, vocab=64, num_pairs=8, num_queries=None, gap=0, key_frac=0.5):
        # DISJOINT token ranges: keys in [1, n_key), values in [n_key, vocab). This removes
        # key/value collisions so recall is unambiguous and the capacity ceiling (B1b) is clean.
        self.n_key = max(num_pairs + 1, int(vocab * key_frac))
        assert self.n_key - 1 >= num_pairs, "need >= num_pairs distinct keys in the key range"
        assert vocab - self.n_key >= 2, "need a value range"
        self.vocab = vocab
        self.D = num_pairs
        self.M = num_queries if num_queries is not None else max(2 * num_pairs, 8)
        self.gap = gap
        self.seq_len = 2 * num_pairs + gap + self.M
        self.name = f"MQAR(V={vocab},keys<{self.n_key},pairs={num_pairs},q={self.M},gap={gap})"

    def sample(self, B, device):
        D, M, V = self.D, self.M, self.vocab
        keys = _distinct_per_row(B, D, 1, self.n_key, device)     # (B,D) distinct keys
        vals = torch.randint(self.n_key, V, (B, D), device=device)  # (B,D) values (disjoint range)
        ctx = torch.empty(B, 2 * D, dtype=torch.long, device=device)
        ctx[:, 0::2] = keys
        ctx[:, 1::2] = vals
        qi = torch.randint(0, D, (B, M), device=device)           # which binding each query hits
        qk = torch.gather(keys, 1, qi)                            # query keys (the input)
        qa = torch.gather(vals, 1, qi)                            # bound values (the target)
        if self.gap > 0:
            filler = torch.zeros(B, self.gap, dtype=torch.long, device=device)
            inp = torch.cat([ctx, filler, qk], dim=1)
        else:
            inp = torch.cat([ctx, qk], dim=1)
        T = inp.shape[1]
        tgt = torch.zeros(B, T, dtype=torch.long, device=device)
        mask = torch.zeros(B, T, device=device)
        qstart = 2 * D + self.gap
        tgt[:, qstart:] = qa                                      # recall target at each query pos
        mask[:, qstart:] = 1.0
        return inp, tgt, mask


class AssocRecall:
    """Single-query associative recall: many KV pairs, one query at the end."""

    def __init__(self, vocab=64, num_pairs=16, gap=0):
        self.inner = MQAR(vocab=vocab, num_pairs=num_pairs, num_queries=1, gap=gap)
        self.vocab = vocab
        self.seq_len = self.inner.seq_len
        self.name = f"AssocRecall(V={vocab},pairs={num_pairs},gap={gap})"

    def sample(self, B, device):
        return self.inner.sample(B, device)


class SelectiveCopy:
    """Copy the K data tokens (in order) out of an L-slot memory region full of fillers.
    vocab: filler=0, data in [1, vocab-1), marker=vocab-1.
    input = [memory(L) with K data tokens scattered, marker, d1, d2, ..., dK]
    predict d1..dK at positions marker..d_{K-1} (next-token), masked there."""

    def __init__(self, vocab=32, mem_len=64, n_data=16, fixed=False):
        assert n_data <= mem_len
        self.vocab = vocab
        self.L = mem_len
        self.K = n_data
        self.fixed = fixed     # True -> data at fixed evenly-spaced positions (control variant)
        self.marker = vocab - 1
        self.seq_len = mem_len + 1 + n_data
        self.name = f"SelectiveCopy(V={vocab},mem={mem_len},k={n_data},{'fixed' if fixed else 'selective'})"

    def sample(self, B, device):
        L, K, V = self.L, self.K, self.vocab
        mem = torch.zeros(B, L, dtype=torch.long, device=device)            # fillers
        if self.fixed:
            base = torch.linspace(0, L - 1, K, device=device).long()        # fixed spacing (control)
            pos = base[None].expand(B, K).clone()
        else:
            pos = torch.argsort(torch.rand(B, L, device=device), dim=1)[:, :K]  # K random slots/row
        pos, _ = torch.sort(pos, dim=1)                                     # left-to-right order
        data = torch.randint(1, V - 1, (B, K), device=device)              # data tokens
        mem.scatter_(1, pos, data)
        marker = torch.full((B, 1), self.marker, dtype=torch.long, device=device)
        inp = torch.cat([mem, marker, data], dim=1)                        # teacher-forced output
        T = inp.shape[1]
        tgt = torch.zeros_like(inp)
        tgt[:, :-1] = inp[:, 1:]
        mask = torch.zeros(B, T, device=device)
        mask[:, L:L + K] = 1.0     # positions: marker (predict d1) .. d_{K-1} (predict dK)
        return inp, tgt, mask


class Induction:
    """In-context induction: prefix contains a unique bigram [A,B]; the final token is A; predict B.
    vocab tokens in [1, vocab); 0 filler. mask=1 only at the final position."""

    def __init__(self, vocab=32, seq_len=64):
        self.vocab = vocab
        self.seq_len = seq_len + 1   # +1 for the trailing query A
        self.name = f"Induction(V={vocab},len={seq_len})"
        self._L = seq_len

    def sample(self, B, device):
        L, V = self._L, self.vocab
        seq = torch.randint(1, V, (B, L), device=device)
        A = torch.randint(1, V, (B,), device=device)
        B_tok = torch.randint(1, V, (B,), device=device)
        # ensure B != A so the answer is informative
        clash = B_tok == A
        B_tok[clash] = (B_tok[clash] % (V - 1)) + 1
        B_tok[B_tok == A] = (A[B_tok == A] % (V - 1)) + 1
        # remove any existing A from the prefix (so A is unique once we place the bigram)
        seq[seq == A[:, None]] = 0
        # also keep the bigram-following slot clean: place [A,B] at a random early position
        ppos = torch.randint(0, L - 2, (B,), device=device)
        ar = torch.arange(B, device=device)
        seq[ar, ppos] = A
        seq[ar, ppos + 1] = B_tok
        # any filler 0 left from removal -> replace with a token guaranteed != A
        filler = (A % (V - 1)) + 1
        zero = seq == 0
        seq = torch.where(zero, filler[:, None].expand_as(seq), seq)
        # but that replacement might have re-introduced A if filler==A (filler!=A by construction)
        # re-place bigram in case a filler overwrote it (positions fixed, so re-assert)
        seq[ar, ppos] = A
        seq[ar, ppos + 1] = B_tok
        query = A[:, None]
        inp = torch.cat([seq, query], dim=1)            # (B, L+1)
        T = inp.shape[1]
        tgt = torch.zeros(B, T, dtype=torch.long, device=device)
        tgt[:, -1] = B_tok
        mask = torch.zeros(B, T, device=device)
        mask[:, -1] = 1.0
        return inp, tgt, mask


class MixedMQAR:
    """MQAR trained over a SPECTRUM of difficulties: each training batch samples the number of KV
    pairs d ~ U[min_pairs, max_pairs] (the standard Zoology training distribution). The easy
    instances supply the gradient that BOOTSTRAPS the recall circuit, so high-D recall becomes
    learnable in feasible compute — where FIXED-high-D training stalls at chance (no foothold for the
    sharp phase transition). EVALUATION is fixed at the target (max_pairs), so the reported number is
    target-D recall. Identical for both models -> fair. n_key is stable across d (= vocab*key_frac
    while d < that), so key/value ranges don't shift with difficulty."""

    def __init__(self, vocab=256, max_pairs=64, num_queries=128, gap=0, min_pairs=1):
        self.vocab = vocab
        self.max_pairs = max_pairs
        self.M = num_queries
        self.gap = gap
        self.min_pairs = min_pairs
        self.seq_len = 2 * max_pairs + gap + num_queries          # max layout (for max_len sizing)
        self.name = f"MixedMQAR(V={vocab},pairs={min_pairs}-{max_pairs},q={num_queries},gap={gap})"

    def _mqar(self, d):
        return MQAR(vocab=self.vocab, num_pairs=d, num_queries=self.M, gap=self.gap)

    def sample(self, B, device):            # TRAINING: a random difficulty per batch
        # Sample the scalar difficulty on CPU: distribution-identical (U[min,max]) but avoids a
        # per-step GPU->CPU sync that serializes the (tiny) model on CUDA. Batch tensors below
        # are still generated on `device`.
        d = int(torch.randint(self.min_pairs, self.max_pairs + 1, (1,)).item())
        return self._mqar(d).sample(B, device)

    def eval_sample(self, B, device):       # EVAL: fixed at the TARGET difficulty (max_pairs)
        return self._mqar(self.max_pairs).sample(B, device)


TASK_REGISTRY = {
    "mqar": MQAR,
    "mixed_mqar": MixedMQAR,
    "assoc": AssocRecall,
    "selcopy": SelectiveCopy,
    "induction": Induction,
}


if __name__ == "__main__":
    dev = torch.device("cpu")
    for name, cls in TASK_REGISTRY.items():
        t = cls()
        x, y, m = t.sample(4, dev)
        print(f"{t.name:<40} x{tuple(x.shape)} masked={int(m.sum().item())}")
        # sanity: an oracle that knows the mapping would score 1.0; random ~1/vocab
