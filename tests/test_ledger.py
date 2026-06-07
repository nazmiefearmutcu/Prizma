from seq.ledger import param_match_report
from seq.transformer import Transformer, TFConfig
from seq.prizma_seq import PrizmaSeqLM, PrizmaSeqConfig

def test_param_match_within_2pct_at_matched_config():
    tf = Transformer(TFConfig(vocab=64, d_model=128, n_layers=4, n_heads=4))
    pz = PrizmaSeqLM(PrizmaSeqConfig(vocab=64, d_model=128, n_layers=4, n_heads=4, feat_map="quad2"))
    rep = param_match_report(tf, pz)
    assert rep["matched"], rep            # |Δparams| / tf < 0.02
    assert rep["feat_map_added_params"] == 0   # quad2 is buffers -> 0 trainable params
