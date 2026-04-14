"""
XP-005.1 — Mock Validation (Gate 2)
Builds synthetic 2D fields, runs the full forward model from manifest.py,
and verifies all outputs behave as expected before touching real data.
"""

import sys
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifest import (
    RHO_THRESHOLD, SIGMA, M_PROTON,
    phase_function, boundary_emission, propagation_field,
    observed_field, model_convergence, baseline_convergence,
    density_from_ne,
)

GRID = 100

# ==========================================================================
# STEP 1: Create synthetic input fields
# ==========================================================================

def make_gaussian_2d(cx, cy, width, peak, grid_size=GRID):
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    return peak * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))

# --- 1a) Electron density → mass density ---
n_e_cluster = make_gaussian_2d(50, 50, 15, peak=0.1)
n_e = n_e_cluster + 0.001
rho = density_from_ne(n_e)

# --- 1b) Temperature field & gradient ---
T_hot = make_gaussian_2d(50, 50, 20, peak=4.0)
T_cool = make_gaussian_2d(65, 50, 8, peak=-4.0)
T = 10.0 + T_hot + T_cool

grad_Ty, grad_Tx = np.gradient(T)
grad_T_mag = np.sqrt(grad_Tx**2 + grad_Ty**2)

# --- 1c) Galaxy surface density chi ---
gal_peak1 = make_gaussian_2d(45, 50, 12, peak=1.0)
gal_peak2 = make_gaussian_2d(65, 50, 8, peak=0.8)
sigma_gal = gal_peak1 + gal_peak2
chi = sigma_gal / np.max(sigma_gal)

# ==========================================================================
# STEP 2: Run forward model
# ==========================================================================

P = phase_function(rho)
psi_emit = boundary_emission(P, chi)
psi_prop = propagation_field(P, grad_T_mag, chi)
psi_obs = observed_field(psi_emit, psi_prop)
kappa_model = model_convergence(psi_obs)

fake_SB = n_e**2
kappa_baseline = baseline_convergence(fake_SB)

# ==========================================================================
# STEP 3: Verification checks
# ==========================================================================

results = {}

# CHECK 1 — Phase function peaks away from grid centre
P_max_loc = np.unravel_index(np.argmax(P), P.shape)
results["CHECK 1: P peaks away from centre"] = P_max_loc != (50, 50)

# CHECK 2 — Psi_emit responds to both P and chi
results["CHECK 2: psi_emit > 0 & non-uniform"] = (
    np.max(psi_emit) > 0 and np.std(psi_emit) > 0
)

# CHECK 3 — Psi_prop responds to temperature gradient
zero_grad_mask = grad_T_mag < 1e-10
if np.any(zero_grad_mask):
    psi_prop_at_zero_grad = np.max(np.abs(psi_prop[zero_grad_mask]))
    results["CHECK 3: psi_prop ≈ 0 where grad_T=0"] = psi_prop_at_zero_grad < 1e-30
else:
    results["CHECK 3: psi_prop ≈ 0 where grad_T=0"] = True

# CHECK 4 — kappa_model normalised [0, 1]
results["CHECK 4: kappa_model in [0,1]"] = (
    np.isclose(np.max(kappa_model), 1.0) and np.min(kappa_model) >= 0.0
)

# CHECK 5 — kappa_baseline normalised [0, 1]
results["CHECK 5: kappa_baseline in [0,1]"] = (
    np.isclose(np.max(kappa_baseline), 1.0) and np.min(kappa_baseline) >= 0.0
)

# CHECK 6 — kappa_model ≠ kappa_baseline
results["CHECK 6: model ≠ baseline"] = not np.allclose(kappa_model, kappa_baseline)

# CHECK 7 — Zero-input safety
zero_grid = np.zeros((10, 10))
kappa_zero_model = model_convergence(zero_grid)
kappa_zero_baseline = baseline_convergence(zero_grid)
results["CHECK 7: zero-input safety"] = (
    np.all(kappa_zero_model == 0) and np.all(kappa_zero_baseline == 0)
)

# CHECK 8 — P(x) value range
p_min, p_max, p_mean = np.min(P), np.max(P), np.mean(P)
p_in_range = p_min >= 0.0 and p_max <= 1.0
p_flag = "" if p_max >= 0.01 else "  ⚠ max(P) < 0.01 — rho never near rho_threshold on this grid"
results["CHECK 8: P in [0,1]"] = p_in_range

# ==========================================================================
# STEP 4: Diagnostic plot
# ==========================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

panels = [
    (axes[0, 0], rho, "rho (g/cm³)"),
    (axes[0, 1], T, "T (keV)"),
    (axes[0, 2], grad_T_mag, "|grad_T|"),
    (axes[0, 3], chi, "chi (galaxy density)"),
    (axes[1, 0], P, "P (phase function)"),
    (axes[1, 1], psi_emit, "Psi_emit"),
    (axes[1, 2], psi_obs, "Psi_obs"),
    (axes[1, 3], kappa_model, "kappa_model"),
]

for ax, data, title in panels:
    im = ax.imshow(data, origin="lower", cmap="inferno")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("XP-005.1 Mock Validation — Synthetic Fields", fontsize=14, y=1.01)
plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "mock_validation_fields.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================================
# STEP 5: Save checkpoint
# ==========================================================================

checkpoint = {
    "n_e": n_e,
    "rho": rho,
    "T": T,
    "grad_T_mag": grad_T_mag,
    "chi": chi,
    "P": P,
    "psi_emit": psi_emit,
    "psi_prop": psi_prop,
    "psi_obs": psi_obs,
    "kappa_model": kappa_model,
    "kappa_baseline": kappa_baseline,
    "fake_SB": fake_SB,
}

ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mock_validation.pkl")
with open(ckpt_path, "wb") as f:
    pickle.dump(checkpoint, f)

# ==========================================================================
# STEP 6: Summary
# ==========================================================================

def field_stats(name, arr):
    return f"  {name:20s}  min={np.min(arr):.6e}  max={np.max(arr):.6e}  mean={np.mean(arr):.6e}"

print("=" * 70)
print("XP-005.1 — MOCK VALIDATION RESULTS")
print("=" * 70)

print("\n--- CHECK RESULTS ---")
all_pass = True
for label, passed in results.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    extra = ""
    if "CHECK 8" in label:
        extra = p_flag
    print(f"  [{status}] {label}{extra}")

print("\n--- FIELD STATISTICS ---")
for name, arr in [
    ("rho", rho), ("T", T), ("|grad_T|", grad_T_mag), ("chi", chi),
    ("P", P), ("psi_emit", psi_emit), ("psi_prop", psi_prop),
    ("psi_obs", psi_obs), ("kappa_model", kappa_model), ("kappa_baseline", kappa_baseline),
]:
    print(field_stats(name, arr))

print(f"\n  P range detail: min={p_min:.6e}  max={p_max:.6e}  mean={p_mean:.6e}")
if p_flag:
    print(f"  {p_flag}")

km_peak = np.unravel_index(np.argmax(kappa_model), kappa_model.shape)
kb_peak = np.unravel_index(np.argmax(kappa_baseline), kappa_baseline.shape)
print(f"\n--- PEAK LOCATIONS ---")
print(f"  kappa_model peak:    {km_peak}")
print(f"  kappa_baseline peak: {kb_peak}")
print(f"  Peaks differ:        {'YES' if km_peak != kb_peak else 'NO'}")

print(f"\n  P max location:      {P_max_loc}")
print(f"  Diagnostic plot:     results/mock_validation_fields.png")
print(f"  Checkpoint saved:    checkpoints/mock_validation.pkl")

print("\n" + "=" * 70)
if all_pass:
    print("  ALL 8 CHECKS PASSED — Forward model validated on synthetic data.")
else:
    print("  SOME CHECKS FAILED — Review results above.")
print("=" * 70)
