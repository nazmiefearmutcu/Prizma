"""
Pure-numpy isolated recall-capacity probe of the delta-rule state S in R^{d_h x d_h}.
Replicates delta._delta_reference exactly (alpha=1, write_mode delta vs additive). NO learning.
Writes D key->value bindings (random L2-unit keys, random values), reads each back PRE-write with
the stored key, measures cosine + hard-recall (nearest stored value == self, the MQAR decision).

This is the architectural memory ceiling. Learned keys can be MORE orthogonal than random gaussian
(an MLP/conv can decorrelate them), so random-key hard-recall is a conservative-ish estimate; the
'ortho' column is the best-case ceiling when keys are exactly orthonormal (only possible for D<=d_h).
"""
import numpy as np


def delta_write_read(K, V, beta=1.0, write_mode="delta"):
    """K: (D,d), unit rows. V: (D,d). Returns recalled (D,d) reading each stored key PRE-write
    AFTER all D writes. S_t = S_{t-1} + beta*(v - S_{t-1} k) k^T (delta) or + beta*v k^T (additive)."""
    D, d = K.shape
    S = np.zeros((d, d))
    for i in range(D):
        k = K[i]; v = V[i]
        if write_mode == "delta":
            u = beta * (v - S @ k)
        else:
            u = beta * v
        S = S + np.outer(u, k)
    # read each stored key back from the FINAL state
    read = (S @ K.T).T                              # (D,d): row i = S k_i
    return read, S


def metrics(read, V):
    rn = read / (np.linalg.norm(read, axis=1, keepdims=True) + 1e-9)
    vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    cos = (rn * vn).sum(1).mean()
    sim = rn @ vn.T
    pred = sim.argmax(1)
    hard = (pred == np.arange(len(V))).mean()
    return cos, hard


def probe(D, d_h, write_mode="delta", key_mode="random", beta=1.0, trials=128, seed=0):
    rng = np.random.default_rng(seed)
    cs, hs = [], []
    for _ in range(trials):
        if key_mode == "ortho" and D <= d_h:
            A = rng.standard_normal((d_h, d_h))
            Q, _ = np.linalg.qr(A)
            K = Q[:D]
        else:
            K = rng.standard_normal((D, d_h))
            K = K / np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.standard_normal((D, d_h))
        read, _ = delta_write_read(K, V, beta=beta, write_mode=write_mode)
        c, h = metrics(read, V)
        cs.append(c); hs.append(h)
    return float(np.mean(cs)), float(np.mean(hs))


def two_head_hard(D, d_h, key_mode="random", trials=128, seed=0):
    """Model with H=2 heads: a single MQAR query token is read by BOTH heads and the outputs summed
    (then projected). The most charitable read of '2 heads helps capacity' is that the model could
    learn to STRIPE bindings across heads (binding i -> head i%2), halving per-head load. We test the
    striped best case: each head stores ~D/2 bindings; recall a binding from its assigned head only.
    Returns hard-recall under perfect striping (an upper bound on the 2-head benefit)."""
    rng = np.random.default_rng(seed)
    hs = []
    for _ in range(trials):
        K = rng.standard_normal((D, d_h)); K = K / np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.standard_normal((D, d_h))
        assign = np.arange(D) % 2
        readall = np.zeros_like(V)
        for head in range(2):
            idx = np.where(assign == head)[0]
            if len(idx) == 0:
                continue
            r, _ = delta_write_read(K[idx], V[idx], write_mode="delta")
            readall[idx] = r
        # decision: nearest among ALL D stored values (cross-head interference is zero by construction
        # in the read, but the decoder still has to disambiguate against all D)
        rn = readall / (np.linalg.norm(readall, axis=1, keepdims=True) + 1e-9)
        vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        pred = (rn @ vn.T).argmax(1)
        hs.append((pred == np.arange(D)).mean())
    return float(np.mean(hs))


if __name__ == "__main__":
    print("=== Delta-state recall ceiling: hard-recall = P(nearest stored value is correct) ===")
    print("    MQAR succeeds (~1.0) only while hard-recall ~ 1.0.\n")
    for d_h in [32, 64]:
        print(f"-- d_h={d_h}  (rank ceiling {d_h}) --")
        hdr = f"{'D':>5} | {'delta cos':>9} {'delta hard':>10} | {'add hard':>8} | {'ortho hard':>10} | {'2head striped':>13}"
        print(hdr)
        for D in [4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160]:
            dc, dh_ = probe(D, d_h, "delta", "random")
            _, ah = probe(D, d_h, "additive", "random")
            oh = probe(D, d_h, "delta", "ortho")[1] if D <= d_h else float("nan")
            th = two_head_hard(D, d_h)
            print(f"{D:>5} | {dc:>9.3f} {dh_:>10.3f} | {ah:>8.3f} | {oh:>10.3f} | {th:>13.3f}")
        print()
    # Define empirical D*: largest D with delta-hard-recall >= 0.97 (random keys).
    print("--- empirical D* (random-key hard-recall >= 0.97) ---")
    for d_h in [16, 32, 64, 128]:
        ds = None
        for D in range(2, 4 * d_h):
            _, h = probe(D, d_h, "delta", "random", trials=64)
            if h < 0.97:
                ds = D - 1; break
        print(f"  d_h={d_h:>3}:  D* ~ {ds}   ( ~ {ds/d_h:.2f} * d_h )")
