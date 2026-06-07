import torch
from seq.common import build_and_train, TrainConfig, get_device
from seq.transformer import Transformer, TFConfig

class _TinyTask:
    vocab = 16
    def sample(self, B, device):
        x = torch.randint(0, self.vocab, (B, 12), device=device)
        return x, x, torch.ones_like(x)

def _fac(vocab, max_len):
    return Transformer(TFConfig(vocab=vocab, d_model=32, n_layers=1, n_heads=2, max_len=max_len))

def test_seed_pinned_init_is_bit_reproducible():
    dev = get_device()
    cfg = TrainConfig(steps=3, eval_every=3, min_steps=0, batch_size=4, log=False)
    r1 = build_and_train(_fac, _TinyTask(), cfg, dev, seed=7, vocab=16, max_len=16)
    r2 = build_and_train(_fac, _TinyTask(), cfg, dev, seed=7, vocab=16, max_len=16)
    assert abs(r1.final_loss - r2.final_loss) < 1e-6, (r1.final_loss, r2.final_loss)
