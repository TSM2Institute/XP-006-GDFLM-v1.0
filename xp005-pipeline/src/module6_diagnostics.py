"""
XP-005.1 — Module 6: Diagnostic Plots + Verdict
Produces all final figures and the markdown verdict file.
"""

import sys
import os
import json
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MANIFEST_SHA256_SHORT = "b1ca5dc7..."


# ==========================================================================
# Helpers
# ==========================================================================

def radial_profile(image, center, max_r=None):
    y, x = np.indices(image.shape)
    r = np.hypot(x - center[1], y - center[0])
    if max_r is None:
        max_r = int(np.max(r))
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), image.ravel())
    nr = np.bincount(r_int.ravel())
    nr_safe = np.where(nr == 0, 1, nr)
    radial = tbin / nr_safe
    return np.arange(len(radial))[:max_r], radial[:max_r]


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — MODULE 6: DIAGNOSTICS + VERDICT")
    print("=" * 70)

    # Load everything
    results_json_path = os.path.join(RESULTS_DIR, "XP005_results.json")
    if not os.path.exists(results_json_path):
        print(f"\n  ERROR: {results_json_path} not found. Run module5_comparison.py first.")
        sys.exit(1)

    with open(results_json_path) as f:
        results = json.load(f)

    with open(os.path.join(DATA_PROC, "grid_params.json")) as f:
        grid_params = json.load(f)
    pixel_scale = grid_params["pixel_scale_arcsec"]

    kappa_obs = np.load(os.path.join(DATA_PROC, "kappa_obs.npy"))
    kappa_model = np.load(os.path.join(DATA_PROC, "kappa_model.npy"))
    kappa_baseline = np.load(os.path.join(DATA_PROC, "kappa_baseline.npy"))
    psi_emit = np.load(os.path.join(DATA_PROC, "psi_emit.npy"))
    psi_prop = np.load(os.path.join(DATA_PROC, "psi_prop.npy"))
    psi_obs = np.load(os.path.join(DATA_PROC, "psi_obs.npy"))
    P = np.load(os.path.join(DATA_PROC, "P_field.npy"))

    # --- STEP 1: 6-panel diagnostic ---
    print("\n--- STEP 1: 6-panel diagnostic figure ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for ax, data, title in [
        (axes[0, 0], kappa_obs, "κ_obs (Clowe 2006)"),
        (axes[0, 1], kappa_model, "κ_model (XP-005.1)"),
        (axes[0, 2], kappa_baseline, "κ_baseline (X-ray SB)"),
    ]:
        im = ax.imshow(data, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    res_model = kappa_model - kappa_obs
    res_baseline = kappa_baseline - kappa_obs
    vmax = max(np.abs(res_model).max(), np.abs(res_baseline).max())

    for ax, data, title in [
        (axes[1, 0], res_model, "Residual (model − obs)"),
        (axes[1, 1], res_baseline, "Residual (baseline − obs)"),
    ]:
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    ax.imshow(np.zeros_like(kappa_obs), origin="lower", cmap="gray", vmin=0, vmax=1)
    levels_obs = [0.2, 0.4, 0.6, 0.8]
    levels_model = [0.2, 0.4, 0.6, 0.8]
    ax.contour(kappa_obs, levels=levels_obs, colors="lime", linewidths=1.5)
    ax.contour(kappa_model, levels=levels_model, colors="red", linewidths=1.5, linestyles="--")
    ax.set_title("Contour Comparison: obs (green) vs model (red dashed)", fontsize=11)

    plt.suptitle(f"XP-005.1 Diagnostic — VERDICT: {results['verdict']}", fontsize=14, y=1.01)
    plt.tight_layout()
    panel_path = os.path.join(RESULTS_DIR, "XP005_diagnostic_6panel.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {panel_path}")

    # --- STEP 2: residual map ---
    print("\n--- STEP 2: Residual map ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    rmax = np.abs(res_model).max()
    im = ax.imshow(res_model, origin="lower", cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    ax.set_title("κ_model − κ_obs (Residual Map)", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δκ (normalised)")
    res_path = os.path.join(RESULTS_DIR, "XP005_residual_map.png")
    plt.savefig(res_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {res_path}")

    # --- STEP 3: radial profiles ---
    print("\n--- STEP 3: Radial profile comparison ---")
    obs_peak = np.unravel_index(np.argmax(kappa_obs), kappa_obs.shape)
    r_obs, prof_obs = radial_profile(kappa_obs, obs_peak, max_r=80)
    r_mod, prof_mod = radial_profile(kappa_model, obs_peak, max_r=80)
    r_bl, prof_bl = radial_profile(kappa_baseline, obs_peak, max_r=80)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_obs * pixel_scale, prof_obs, "g-", lw=2, label="κ_obs (Clowe 2006)")
    ax.plot(r_mod * pixel_scale, prof_mod, "r--", lw=2, label="κ_model (XP-005.1)")
    ax.plot(r_bl * pixel_scale, prof_bl, "b:", lw=2, label="κ_baseline (X-ray SB)")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Normalised κ")
    ax.set_title(f"Radial Profiles around κ_obs peak {obs_peak}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    rad_path = os.path.join(RESULTS_DIR, "XP005_radial_profiles.png")
    plt.savefig(rad_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {rad_path}")

    # --- STEP 4: Component plot ---
    print("\n--- STEP 4: Component contribution plot ---")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, data, title in [
        (axes[0], P, "P(x,y)"),
        (axes[1], psi_emit, "Psi_emit"),
        (axes[2], psi_prop, "Psi_prop"),
        (axes[3], psi_obs, "Psi_obs"),
    ]:
        im = ax.imshow(data, origin="lower", cmap="inferno")
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("XP-005.1 Component Fields (real data)", fontsize=14, y=1.02)
    plt.tight_layout()
    comp_path = os.path.join(RESULTS_DIR, "XP005_components.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {comp_path}")

    # --- STEP 5: Verdict markdown ---
    print("\n--- STEP 5: Verdict markdown ---")
    m = results["metrics"]

    def row(label, threshold, measured, ok):
        return f"| {label} | {threshold} | {measured} | {'PASS' if ok else 'FAIL'} |"

    md = f"""# XP-005.1 VERDICT

**Test:** Boundary–Propagation Lensing Model
**Target:** Bullet Cluster (1E 0657-56)
**Date:** {results['date']}
**Build path:** {results['build_path']}

## Result: {results['verdict']}

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
{row("χ²/dof improvement", "≥ 25%", f"{m['chi2_improvement_percent']:.2f}%", m['chi2_improvement_pass'])}
{row("Peak offset", "< 30″", f"{m['peak_offset_arcsec']:.2f}″", m['peak_offset_pass'])}
{row("Residual RMS", "≤ 0.2", f"{m['residual_rms']:.4f}", m['residual_rms_pass'])}
{row("Stability", "< 10%", f"{m['stability_variation_percent']:.2f}%", m['stability_pass'])}

## Notes
- Comparison region: {results['comparison_region_pixels']} pixels where κ_obs > 0
- Error weighting: uniform σ = {results['sigma_uniform']} ({results['sigma_justification']})
- Manifest SHA-256: `{results['manifest_sha256']}`
- chi2_dof model: {m['chi2_dof_model']:.4f}
- chi2_dof baseline: {m['chi2_dof_baseline']:.4f}
- Stability subsample chi2_dof values: {m['stability_chi2_values']}

## Verdict Rule
{results['verdict_rule']}
"""
    verdict_path = os.path.join(RESULTS_DIR, "XP005_VERDICT.md")
    with open(verdict_path, "w") as f:
        f.write(md)
    print(f"  Saved: {verdict_path}")

    # --- STEP 6: Console summary ---
    print("\n" + "=" * 70)
    print(f"  XP-005.1 FINAL VERDICT: {results['verdict']}")
    print("=" * 70)
    print(f"\n  | Criterion             | Threshold | Measured       | Result |")
    print(f"  |-----------------------|-----------|----------------|--------|")
    print(f"  | χ²/dof improvement    | ≥ 25%     | {m['chi2_improvement_percent']:>8.2f}%      | {'PASS' if m['chi2_improvement_pass'] else 'FAIL'} |")
    print(f"  | Peak offset           | < 30″     | {m['peak_offset_arcsec']:>7.2f}″        | {'PASS' if m['peak_offset_pass'] else 'FAIL'} |")
    print(f"  | Residual RMS          | ≤ 0.2     | {m['residual_rms']:>8.4f}        | {'PASS' if m['residual_rms_pass'] else 'FAIL'} |")
    print(f"  | Stability             | < 10%     | {m['stability_variation_percent']:>8.2f}%      | {'PASS' if m['stability_pass'] else 'FAIL'} |")

    print(f"\n  Output files:")
    for p in [panel_path, res_path, rad_path, comp_path, verdict_path, results_json_path]:
        print(f"    {p}")
    print("=" * 70)
