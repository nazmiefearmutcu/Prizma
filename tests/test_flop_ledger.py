"""R4 d_phi reconciliation — the per-config FLOP ledger emitter must cover the FOUR
canonical (feat_map, feat_n2/feat_rank, d_phi) configs and tie every FLOP number to a REAL
param-matched PrizmaSeqLM/Transformer pair (not just an analytical d_phi)."""
import flop_ledger

EXPECTED = {
    "none_d32": 32,
    "quad2_d128_codedefault": 128,
    "quad2_d256_v1ref": 256,
    "quad2_lowrank_d137_v2lean": 137,
}


def test_emit_per_config_ledger_has_four_configs_and_param_match():
    out = flop_ledger.emit_per_config_ledger(d=128, H=4, L=4, verbose=False)
    # exactly the four expected config keys
    assert set(out.keys()) == set(EXPECTED.keys()), out.keys()
    for label, exp_dphi in EXPECTED.items():
        rec = out[label]
        # the analytical d_phi pinned to the config is the expected one
        assert rec["d_phi"] == exp_dphi, (label, rec["d_phi"])
        # constructed from REAL modules -> param-match within 2%
        assert rec["param_match"]["matched"], (label, rec["param_match"])
        assert rec["param_match"]["rel"] < 0.02, (label, rec["param_match"])
        # the per-token FLOP figures + ratios are present and positive
        for k in ("tf_per_tok", "ps_ascoded_per_tok", "ps_ideal_per_tok",
                  "ratio_ascoded", "ratio_ideal"):
            assert rec[k] > 0, (label, k, rec[k])
        # matched-TF search candidates are recorded
        assert "matched_tf" in rec


def test_config_labels_pin_the_documented_dphi_values():
    """Lock the R4 framing: code default = d_phi=128, v1 published = d_phi=256, v2 lean = d_phi=137."""
    out = flop_ledger.emit_per_config_ledger(d=128, H=4, L=4, verbose=False)
    assert out["quad2_d128_codedefault"]["feat_map"] == "quad2"
    assert out["quad2_d128_codedefault"]["feat_n2"] == 96
    assert out["quad2_d256_v1ref"]["feat_n2"] == 224
    assert out["quad2_lowrank_d137_v2lean"]["feat_map"] == "quad2_lowrank"
    assert out["quad2_lowrank_d137_v2lean"]["feat_rank"] == 14
    assert out["none_d32"]["feat_map"] == "none"
