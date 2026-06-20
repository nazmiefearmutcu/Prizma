"""
Tests for neuromorphic and analog aspects of Prizma:
1. Analog noise impact on inference and training robustness.
2. Weight and activation quantization (low-bit representations).
3. Predictive coding (PC) settling dynamics comparing exact vs random feedback,
   and Langevin neural sampling.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from src.prizma import Expert, Prizma, softmax
from src.data import make_base_dataset


# =====================================================================
# 1. Analog Noise Verification
# =====================================================================

def test_analog_noise_inference():
    """Verify that adding noise to inputs or activations monotonically increases reconstruction error/surprise."""
    X, y = make_base_dataset(n_samples=100, d=20, n_classes=2, seed=42)
    expert = Expert(d=20, h=10, K=2, seed=42)
    
    # Base surprise without noise
    surp_clean = expert.recon_error(X)
    mean_surp_clean = float(surp_clean.mean())
    
    # Verify that adding Gaussian noise to inputs increases surprise
    noise_levels = [0.05, 0.1, 0.2, 0.5]
    prev_surp = mean_surp_clean
    for sigma in noise_levels:
        rng = np.random.default_rng(42)
        X_noisy = X + rng.normal(0, sigma, X.shape).astype(np.float32)
        surp_noisy = float(expert.recon_error(X_noisy).mean())
        assert surp_noisy > prev_surp, (
            f"Noise std={sigma} surprise {surp_noisy} should be larger than previous {prev_surp}"
        )
        prev_surp = surp_noisy


def test_analog_noise_training():
    """Verify that a Prizma expert can still learn under activation and weight update noise, but reaches a higher asymptotic loss."""
    X, y = make_base_dataset(n_samples=200, d=20, n_classes=2, seed=42)
    Y = np.eye(2, dtype=np.float32)[y]
    
    expert_clean = Expert(d=20, h=10, K=2, seed=42)
    expert_noisy = Expert(d=20, h=10, K=2, seed=42)
    
    epochs = 30
    lr = 0.1
    lr_cls = 0.1
    
    # 1. Clean training loop
    for epoch in range(epochs):
        Z, EPS, logits = expert_clean.forward(X)
        P = softmax(logits, axis=1)
        dZ = 1.0 - Z ** 2
        D = (P - Y)
        
        # Update weights using exact PC rule
        expert_clean.Wdec += lr * (EPS.T @ Z) / len(X)
        expert_clean.bdec += lr * EPS.mean(0)
        expert_clean.Wcls -= lr_cls * (D.T @ Z) / len(X)
        expert_clean.bcls -= lr_cls * D.mean(0)
        
        g_rec = (EPS @ expert_clean.Bdec.T) * dZ
        g_cls = (D @ expert_clean.Bcls.T) * dZ
        g_lat = g_rec + g_cls
        expert_clean.Wenc += lr * (g_lat.T @ X) / len(X)
        expert_clean.benc += lr * g_lat.mean(0)
        
    # 2. Noisy training loop (adds analog noise to activation Z and updates dW)
    rng = np.random.default_rng(100)
    for epoch in range(epochs):
        Z = expert_noisy.encode(X)
        # Add moderate analog noise to latent activations Z
        Z_noisy = Z + rng.normal(0, 0.05, Z.shape).astype(np.float32)
        
        # Generative path with weight noise
        Wdec_noisy = expert_noisy.Wdec + rng.normal(0, 0.01, expert_noisy.Wdec.shape).astype(np.float32)
        Xhat = Z_noisy @ Wdec_noisy.T + expert_noisy.bdec
        EPS = X - Xhat
        
        logits = Z_noisy @ expert_noisy.Wcls.T + expert_noisy.bcls
        P = softmax(logits, axis=1)
        
        dZ = 1.0 - Z_noisy ** 2
        D = (P - Y)
        
        # Updates with noise
        dWdec = (EPS.T @ Z_noisy) / len(X) + rng.normal(0, 0.01, expert_noisy.Wdec.shape).astype(np.float32)
        dWcls = (D.T @ Z_noisy) / len(X) + rng.normal(0, 0.01, expert_noisy.Wcls.shape).astype(np.float32)
        
        expert_noisy.Wdec += lr * dWdec
        expert_noisy.bdec += lr * EPS.mean(0)
        expert_noisy.Wcls -= lr_cls * dWcls
        expert_noisy.bcls -= lr_cls * D.mean(0)
        
        g_rec = (EPS @ expert_noisy.Bdec.T) * dZ
        g_cls = (D @ expert_noisy.Bcls.T) * dZ
        g_lat = g_rec + g_cls
        
        dWenc = (g_lat.T @ X) / len(X) + rng.normal(0, 0.01, expert_noisy.Wenc.shape).astype(np.float32)
        expert_noisy.Wenc += lr * dWenc
        expert_noisy.benc += lr * g_lat.mean(0)
        
    init_expert = Expert(d=20, h=10, K=2, seed=42)
    init_err = float(init_expert.recon_error(X).mean())
    
    clean_err = float(expert_clean.recon_error(X).mean())
    
    # Evaluate noisy model under the same analog non-idealities
    rng_eval = np.random.default_rng(200)
    Z_noisy_eval = expert_noisy.encode(X) + rng_eval.normal(0, 0.05, (len(X), 10)).astype(np.float32)
    Wdec_noisy_eval = expert_noisy.Wdec + rng_eval.normal(0, 0.01, expert_noisy.Wdec.shape).astype(np.float32)
    Xhat_noisy_eval = Z_noisy_eval @ Wdec_noisy_eval.T + expert_noisy.bdec
    noisy_err = float(((X - Xhat_noisy_eval)**2).mean())
    
    # Assert that training decreased surprise in both cases
    assert clean_err < init_err, f"Clean training failed to decrease reconstruction error: {clean_err} vs {init_err}"
    assert noisy_err < init_err, f"Noisy training failed to decrease reconstruction error: {noisy_err} vs {init_err}"
    
    # Assert that clean model achieved lower reconstruction error than noisy model
    assert clean_err < noisy_err, f"Clean model reconstruction error ({clean_err}) should be lower than noisy model ({noisy_err})"




# =====================================================================
# 2. Quantization Verification
# =====================================================================

def test_quantization_inference():
    """Verify that quantizing weights to low-bit representations degrades performance, with lower precision causing higher error."""
    X, y = make_base_dataset(n_samples=200, d=20, n_classes=2, seed=42)
    Y = np.eye(2, dtype=np.float32)[y]
    
    expert = Expert(d=20, h=10, K=2, seed=42)
    lr, lr_cls = 0.1, 0.1
    for _ in range(40):
        Z = expert.encode(X)
        Xhat = Z @ expert.Wdec.T + expert.bdec
        EPS = X - Xhat
        logits = Z @ expert.Wcls.T + expert.bcls
        P = softmax(logits, axis=1)
        dZ = 1.0 - Z ** 2
        D = (P - Y)
        expert.Wdec += lr * (EPS.T @ Z) / len(X)
        expert.bdec += lr * EPS.mean(0)
        expert.Wcls -= lr_cls * (D.T @ Z) / len(X)
        expert.bcls -= lr_cls * D.mean(0)
        g_rec = (EPS @ expert.Bdec.T) * dZ
        g_cls = (D @ expert.Bcls.T) * dZ
        g_lat = g_rec + g_cls
        expert.Wenc += lr * (g_lat.T @ X) / len(X)
        expert.benc += lr * g_lat.mean(0)
        
    Z_clean = expert.encode(X)
    Xhat_clean = Z_clean @ expert.Wdec.T + expert.bdec
    recon_clean = float(((X - Xhat_clean)**2).mean())
    
    # Symmetric quantization helper
    def quantize(w, bits):
        vmax = np.max(np.abs(w))
        if vmax == 0:
            return w
        qmin = -(2**(bits-1))
        qmax = (2**(bits-1)) - 1
        scale = qmax / vmax
        qw = np.round(w * scale)
        qw = np.clip(qw, qmin, qmax)
        return qw / scale

    # Quantize Wdec to different bitwidths
    bitwidths = [2, 4, 8]
    recon_errors = []
    
    for b in bitwidths:
        Wdec_q = quantize(expert.Wdec, b)
        Xhat_q = Z_clean @ Wdec_q.T + expert.bdec
        err_q = float(((X - Xhat_q)**2).mean())
        recon_errors.append(err_q)
        
    # 2-bit error >= 4-bit error, and both are strictly greater than clean error
    assert recon_errors[0] >= recon_errors[1], f"2-bit error {recon_errors[0]} should be >= 4-bit error {recon_errors[1]}"
    assert recon_errors[0] > recon_clean, f"2-bit error {recon_errors[0]} should be > clean error {recon_clean}"
    assert recon_errors[1] > recon_clean, f"4-bit error {recon_errors[1]} should be > clean error {recon_clean}"
    # 8-bit error should be extremely close to clean error
    assert abs(recon_errors[2] - recon_clean) < 1e-3, f"8-bit error {recon_errors[2]} should be close to clean error {recon_clean}"



def test_quantization_aware_training():
    """Verify that training with quantized weights (e.g. 6-bit) still converges, reducing loss/surprise."""
    X, y = make_base_dataset(n_samples=200, d=20, n_classes=2, seed=42)
    Y = np.eye(2, dtype=np.float32)[y]
    
    def quantize(w, bits=6):
        vmax = np.max(np.abs(w))
        if vmax == 0:
            return w
        qmax = (2**(bits-1)) - 1
        scale = qmax / vmax
        return np.clip(np.round(w * scale), -qmax, qmax) / scale
        
    expert = Expert(d=20, h=10, K=2, seed=42)
    
    # Record initial reconstruction error
    Z_init = np.tanh(X @ quantize(expert.Wenc).T + expert.benc)
    Xhat_init = Z_init @ quantize(expert.Wdec).T + expert.bdec
    init_err = float(((X - Xhat_init)**2).mean())
    
    # Train under quantization (weights quantized in forward pass)
    lr, lr_cls = 0.1, 0.1
    for _ in range(40):
        Wenc_q = quantize(expert.Wenc)
        Wdec_q = quantize(expert.Wdec)
        Wcls_q = quantize(expert.Wcls)
        
        Z = np.tanh(X @ Wenc_q.T + expert.benc)
        Xhat = Z @ Wdec_q.T + expert.bdec
        EPS = X - Xhat
        logits = Z @ Wcls_q.T + expert.bcls
        P = softmax(logits, axis=1)
        
        dZ = 1.0 - Z ** 2
        D = (P - Y)
        
        expert.Wdec += lr * (EPS.T @ Z) / len(X)
        expert.bdec += lr * EPS.mean(0)
        expert.Wcls -= lr_cls * (D.T @ Z) / len(X)
        expert.bcls -= lr_cls * D.mean(0)
        
        g_rec = (EPS @ expert.Bdec.T) * dZ
        g_cls = (D @ expert.Bcls.T) * dZ
        g_lat = g_rec + g_cls
        expert.Wenc += lr * (g_lat.T @ X) / len(X)
        expert.benc += lr * g_lat.mean(0)
        
    # Verify final error using quantized weights is lower than initial error
    Wenc_final = quantize(expert.Wenc)
    Wdec_final = quantize(expert.Wdec)
    Z_final = np.tanh(X @ Wenc_final.T + expert.benc)
    Xhat_final = Z_final @ Wdec_final.T + expert.bdec
    final_err = float(((X - Xhat_final)**2).mean())
    
    assert final_err < init_err, f"Quantization-aware training failed to decrease reconstruction error: {final_err} vs {init_err}"


# =====================================================================
# 3. PC Settling Dynamics Verification
# =====================================================================

def test_pc_settling_dynamics_energy_minimization():
    """Verify that deterministic PC settling dynamics minimizes the local free energy monotonically under exact feedback."""
    X, y = make_base_dataset(n_samples=5, d=20, n_classes=2, seed=42)
    expert = Expert(d=20, h=10, K=2, seed=42)
    
    steps = 150
    dt = 0.01
    lambda_z = 0.5
    
    z_ff = expert.encode(X)
    z = z_ff.copy()
    energies = []
    
    for _ in range(steps):
        Xhat = z @ expert.Wdec.T + expert.bdec
        eps_x = X - Xhat
        
        # Local free energy: 0.5 * ||X - z Wdec.T - bdec||^2 + 0.5 * lambda_z * ||z - z_ff||^2
        energy = 0.5 * np.sum(eps_x ** 2, axis=1) + 0.5 * lambda_z * np.sum((z - z_ff) ** 2, axis=1)
        energies.append(float(energy.mean()))
        
        # Gradient dF/dz = -eps_x @ Wdec + lambda_z * (z - z_ff)
        grad_z = -eps_x @ expert.Wdec + lambda_z * (z - z_ff)
        z = z - dt * grad_z
        
    # Verify final energy is lower than initial energy
    assert energies[-1] < energies[0], f"Final energy {energies[-1]} should be lower than initial energy {energies[0]}"
    
    # Verify that energy generally decreases in first half vs second half
    assert energies[steps // 2] < energies[0]
    assert energies[-1] < energies[steps // 2]


def test_pc_settling_exact_vs_random_feedback():
    """Verify that exact feedback settles to a lower free energy state more efficiently than random feedback."""
    X, y = make_base_dataset(n_samples=10, d=20, n_classes=2, seed=42)
    expert = Expert(d=20, h=10, K=2, seed=42)
    
    steps = 100
    dt = 0.01
    lambda_z = 0.5
    
    # Exact feedback settling
    z_ff = expert.encode(X)
    z_exact = z_ff.copy()
    energies_exact = []
    for _ in range(steps):
        eps_x = X - (z_exact @ expert.Wdec.T + expert.bdec)
        energy = 0.5 * np.sum(eps_x ** 2, axis=1) + 0.5 * lambda_z * np.sum((z_exact - z_ff) ** 2, axis=1)
        energies_exact.append(float(energy.mean()))
        grad_z = -eps_x @ expert.Wdec + lambda_z * (z_exact - z_ff)
        z_exact = z_exact - dt * grad_z
        
    # Random feedback settling (using Bdec.T as the feedback path)
    z_random = z_ff.copy()
    energies_random = []
    for _ in range(steps):
        eps_x = X - (z_random @ expert.Wdec.T + expert.bdec)
        energy = 0.5 * np.sum(eps_x ** 2, axis=1) + 0.5 * lambda_z * np.sum((z_random - z_ff) ** 2, axis=1)
        energies_random.append(float(energy.mean()))
        # Replace Wdec in the gradient path with Bdec.T
        grad_z = -eps_x @ expert.Bdec.T + lambda_z * (z_random - z_ff)
        z_random = z_random - dt * grad_z
        
    # Exact feedback is mathematically aligned with the gradient of F,
    # so it should achieve lower final free energy than random feedback.
    assert energies_exact[-1] < energies_random[-1], (
        f"Exact feedback final energy {energies_exact[-1]} "
        f"should be lower than random feedback final energy {energies_random[-1]}"
    )


def test_langevin_neural_sampling():
    """Verify that Langevin settling with T > 0 behaves stochastically, with sample variance increasing with temperature."""
    X, y = make_base_dataset(n_samples=5, d=20, n_classes=2, seed=42)
    expert = Expert(d=20, h=10, K=2, seed=42)
    
    steps = 1500
    dt = 0.01
    lambda_z = 0.5
    
    temperatures = [0.0, 0.01, 0.1]
    histories = []
    
    for T in temperatures:
        rng = np.random.default_rng(42)
        z_ff = expert.encode(X)
        z = z_ff.copy()
        z_samples = []
        
        # Settle for 1300 steps to reach equilibrium, then sample
        for step in range(steps):
            eps_x = X - (z @ expert.Wdec.T + expert.bdec)
            grad_z = -eps_x @ expert.Wdec + lambda_z * (z - z_ff)
            
            noise = 0.0
            if T > 0.0:
                noise = np.sqrt(2.0 * T * dt) * rng.normal(0.0, 1.0, z.shape)
                
            z = z - dt * grad_z + noise
            if step >= 1300:
                z_samples.append(z.copy())
                
        histories.append(np.array(z_samples))
        
    # Calculate sample variance over time steps
    vars_over_time = [float(np.var(h, axis=0).mean()) for h in histories]
    
    assert vars_over_time[0] < 1e-6, f"Deterministic settling should have near zero variance, got {vars_over_time[0]}"
    assert vars_over_time[1] > 1e-6, f"Low temp settling should have positive variance, got {vars_over_time[1]}"
    assert vars_over_time[2] > vars_over_time[1], (
        f"Higher temp settling variance {vars_over_time[2]} "
        f"should be greater than low temp variance {vars_over_time[1]}"
    )

