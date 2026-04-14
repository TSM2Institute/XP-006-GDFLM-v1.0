"""
XP-005.1 — Module 3: Model Computation
Runs all locked manifest equations on real fields to produce kappa_model
and kappa_baseline.
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
    boundary_emission, propagation_field, observed_field,
    model_convergence, baseline_convergence,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — MODULE 3: MODEL COMPUTATION")
    print("=" * 70)

    print("\n--- STEP 1: Load all fields ---")
    P = np.load(os.path.join(DATA_PROC, "P_field.npy"))
    chi = np.load(os.path.join(DATA_PROC, "chi_field.npy"))
    grad_T_mag = np.load(os.path.join(DATA_PROC, "grad_T_magnitude.npy"))
    SB = np.load(os.path.join(DATA_PROC, "SB_field.npy"))
    rho = np.load(os.path.join(DATA_PROC, "rho_field.npy"))
    print(f"  P: {P.shape}, chi: {chi.shape}, |grad_T|: {grad_T_mag.shape}, SB: {SB.shape}")

    print("\n--- STEP 2: Compute model fields ---")
    psi_emit = boundary_emission(P, chi)
    print(f"  psi_emit computed")

    psi_prop = propagation_field(P, grad_T_mag, chi)
    print(f"  psi_prop computed")

    psi_obs = observed_field(psi_emit, psi_prop)
    print(f"  psi_obs computed")

    kappa_model = model_convergence(psi_obs)
    print(f"  kappa_model computed")

    kappa_baseline = baseline_convergence(SB)
    print(f"  kappa_baseline computed")

    print("\n--- STEP 3: Save fields ---")
    for name, arr in [
        ("psi_emit", psi_emit),
        ("psi_prop", psi_prop),
        ("psi_obs", psi_obs),
        ("kappa_model", kappa_model),
        ("kappa_baseline", kappa_baseline),
    ]:
        np.save(os.path.join(DATA_PROC, f"{name}.npy"), arr)
    print("  Saved: psi_emit.npy, psi_prop.npy, psi_obs.npy, kappa_model.npy, kappa_baseline.npy")

    print("\n--- STEP 4: Verification ---")

    def peak_loc(arr):
        return np.unravel_index(np.argmax(arr), arr.shape)

    header = f"  {'Field':<16s} | {'Shape':>9s} | {'Min':>14s} | {'Max':>14s} | Peak location"
    print(header)
    print("  " + "-" * (len(header) - 2))
    fields_info = [
        ("P(x,y)", P),
        ("chi(x,y)", chi),
        ("Psi_emit", psi_emit),
        ("Psi_prop", psi_prop),
        ("Psi_obs", psi_obs),
        ("kappa_model", kappa_model),
        ("kappa_baseline", kappa_baseline),
    ]
    for name, arr in fields_info:
        shape_str = f"{arr.shape[0]}x{arr.shape[1]}"
        pk = peak_loc(arr)
        print(f"  {name:<16s} | {shape_str:>9s} | {np.min(arr):>14.6e} | {np.max(arr):>14.6e} | {pk}")

    print("\n--- KEY CHECKS ---")
    checks = {}
    checks["1. kappa_model max == 1.0"] = np.isclose(np.max(kappa_model), 1.0)
    checks["2. kappa_model min >= 0.0"] = np.min(kappa_model) >= 0.0
    checks["3. kappa_baseline max == 1.0"] = np.isclose(np.max(kappa_baseline), 1.0)
    checks["4. kappa_baseline min >= 0.0"] = np.min(kappa_baseline) >= 0.0

    km_peak = peak_loc(kappa_model)
    kb_peak = peak_loc(kappa_baseline)
    checks["5. model peak != baseline peak"] = km_peak != kb_peak
    checks["6. psi_obs not all zeros"] = not np.all(psi_obs == 0)
    checks["7. psi_emit not all zeros"] = not np.all(psi_emit == 0)
    checks["8. psi_prop not all zeros"] = not np.all(psi_prop == 0)

    all_pass = True
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    print(f"\n  kappa_model peak:    {km_peak}")
    print(f"  kappa_baseline peak: {kb_peak}")
    print(f"  Peaks differ:        {'YES' if km_peak != kb_peak else 'NO'}")

    print("\n--- STEP 5: Save checkpoint ---")
    checkpoint = {
        "P": P,
        "chi": chi,
        "grad_T_mag": grad_T_mag,
        "SB": SB,
        "psi_emit": psi_emit,
        "psi_prop": psi_prop,
        "psi_obs": psi_obs,
        "kappa_model": kappa_model,
        "kappa_baseline": kappa_baseline,
    }
    ckpt_path = os.path.join(CKPT_DIR, "module3_model.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved: {ckpt_path}")

    print("\n--- STEP 6: Diagnostic plot ---")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    panels = [
        (axes[0, 0], P, "P(x,y)"),
        (axes[0, 1], chi, "chi(x,y)"),
        (axes[0, 2], grad_T_mag, "|grad_T|(x,y)"),
        (axes[0, 3], psi_emit, "Psi_emit"),
        (axes[1, 0], psi_prop, "Psi_prop"),
        (axes[1, 1], psi_obs, "Psi_obs"),
        (axes[1, 2], kappa_model, "kappa_model"),
        (axes[1, 3], kappa_baseline, "kappa_baseline"),
    ]

    for ax, data, title in panels:
        im = ax.imshow(data, origin="lower", cmap="inferno")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("XP-005.1 Module 3 — Model Outputs", fontsize=14, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "module3_model_outputs.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    print("\n" + "=" * 70)
    if all_pass:
        print("  MODULE 3 COMPLETE — All 8 checks PASSED.")
    else:
        print("  MODULE 3 COMPLETE — Some checks FAILED. Review above.")
    print("=" * 70)
