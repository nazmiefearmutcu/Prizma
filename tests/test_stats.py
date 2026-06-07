import numpy as np
from seq.stats import summarize, tost_equivalence, superiority_test, solve_rate

def test_summarize_reports_t_ci_median_solverate():
    xs = [0.95, 0.97, 0.99, 0.96, 0.98, 0.94, 0.99, 0.97, 0.96, 0.98]
    s = summarize(xs, solve_thresh=0.9)
    assert s["n"] == 10
    assert abs(s["median"] - 0.97) < 1e-9
    assert s["solve_rate"] == 1.0
    assert s["ci95"][0] < s["mean"] < s["ci95"][1]

def test_superiority_detects_real_gap():
    a = list(np.full(10, 0.80) + np.array([0.01,-0.01]*5))   # ~0.80
    b = list(np.full(10, 0.70) + np.array([0.01,-0.01]*5))   # ~0.70
    res = superiority_test(a, b)            # H1: mean(a) > mean(b)
    assert res["p_value"] < 0.05 and res["significant"]

def test_tost_equivalence_within_margin():
    a = list(np.full(10, 0.900) + np.array([0.002,-0.002]*5))
    b = list(np.full(10, 0.901) + np.array([0.002,-0.002]*5))
    res = tost_equivalence(a, b, margin=0.02)
    assert res["equivalent"]
