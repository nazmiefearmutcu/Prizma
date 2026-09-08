"""Regression test for review 2026-09-08 M-11: seq.blockdrift_claim must import AS A PACKAGE from
the repo root and resolve Transformer from seq.transformer. The WINDOW-TF leg carried a local bare
`from transformer import Transformer` that only resolved when the file was run as a bare script —
`python -m seq.blockdrift_claim` trained all primaries and then crashed at that leg BEFORE writing
any verdict. The runner now uses its module-level `from seq.transformer import TFConfig,
Transformer` (the sys.path bootstrap at the top of the file covers the bare-script case).
"""
import seq.blockdrift_claim as bc
import seq.transformer as tf


def test_blockdrift_package_import_resolves_transformer():
    """Package import (repo root) must bind the same Transformer/TFConfig as seq.transformer."""
    assert bc.Transformer is tf.Transformer
    assert bc.TFConfig is tf.TFConfig
