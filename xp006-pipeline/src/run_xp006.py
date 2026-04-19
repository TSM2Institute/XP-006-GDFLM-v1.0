"""
run_xp006.py — XP-006 GDFLM v1.0 — Live Run (Gate 4)

Applies the GDFLM model K(x) = α·ρ̂ + β·|∇T̂| + γ·|∇ρ̂| to the real
Bullet Cluster fields, computes the four pre-registered metrics,
runs bootstrap stability with seeds (1, 2, 3), and applies the sealed
PASS/FAIL thresholds.

NO post-hoc rescue. The numbers are the numbers.

Run from the xp006-pipeline directory:
    python src/run_xp006.py

Outputs:
    results/XP006_results.json          — full machine-readable result
    results/XP006_VERDICT.md            — markdown verdict report
    results/XP006_diagnostic_6panel.png — κ_obs vs κ_model vs κ_baseline + residuals
    results/XP006_residual_map.png      — model − obs residual
    results/XP006_components.png        — ρ, |∇T|, |∇ρ|, K (normalised)
    checkpoints/run_xp006_state.pkl     — full state for reproducibility

Exit codes:
    0  — pipeline ran to completion (verdict recorded; PASS/FAIL is in results)
    1  — pipeline error (data missing, schema mismatch, etc.)
"""

import os
import sys
import json
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manifest import (
    GRID_SIZE, PIXEL_SCALE_ARCSEC,
    KAPPA_OBS_FILE, KAPPA_OBS_PIXEL_COUNT,
    COMPARISON_SIGMA,
    BOOTSTRAP_FRACTION, BOOTSTRAP_N_RUNS, BOOTSTRAP_SEEDS,
    CHI2_IMPROVEMENT_THRESHOLD, PEAK_OFFSET_THRESHOLD,
    RESIDUAL_RMS_THRESHOLD, STABILITY_THRESHOLD,
    DATA_PATHS,
    normalise_minmax, gradient_magnitude,
    K_model, model_convergence_xp006, baseline_convergence,
    bootstrap_subsample_indices, stability,
    manifest_signature,
)


BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
DATA_DIG    = os.path.join(BASE_DIR, "data", "digitised")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ==========================================================================
# κ_OBS RECONSTRUCTION (load pre-computed XP-005.1 inheritance .npy)
# ==========================================================================

def reconstruct_kappa_obs(csv_path):
    """Load the κ_obs grid inherited byte-for-byte from XP-005.1.

    The source CSV (clowe2006_kappa.csv) stores nine digitised contours in a
    multi-column format (pairs of X/Y columns per contour) with coordinates in
    normalised/arcsec space — not in 0-199 pixel coordinates.  XP-005.1
    performed the full digitisation, coordinate transformation, and cubic
    griddata interpolation including manual closure of the kappa_0.16_ur gap
    (sealed warning, carried forward).  The result was saved as
    data/processed/kappa_obs.npy (peak-normalised, 7487 non-zero pixels).

    Loading that .npy directly is the byte-for-byte inheritance path and
    avoids re-implementing the coordinate system that XP-005.1 established.
    The CSV path argument is retained for interface compatibility and
    provenance logging.
    """
    npy_path = os.path.join(DATA_PROC, "kappa_obs.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"kappa_obs.npy not found at {npy_path}. "
            f"This file is the XP-005.1 inheritance artefact and must exist.")
    kappa = np.load(npy_path)
    # Floor negatives (should be none, but defensive)
    kappa = np.where(kappa < 0, 0.0, kappa)
    return kappa


# ==========================================================================
# METRICS
# ==========================================================================

def chi2_dof(kappa_obs, kappa_pred, mask, sigma):
    """χ² / dof on pixels where mask is True. dof = N_pix (uniform σ, no fit)."""
    diff = (kappa_obs[mask] - kappa_pred[mask]) ** 2
    chi2 = np.sum(diff / (sigma ** 2))
    dof = int(mask.sum())
    return float(chi2 / dof), float(chi2), dof


def peak_offset_arcsec(kappa_obs, kappa_pred, pixel_scale=PIXEL_SCALE_ARCSEC):
    ro, co = np.unravel_index(np.argmax(kappa_obs),  kappa_obs.shape)
    rp, cp = np.unravel_index(np.argmax(kappa_pred), kappa_pred.shape)
    px_dist = float(np.hypot(ro - rp, co - cp))
    return px_dist * pixel_scale, (int(ro), int(co)), (int(rp), int(cp))


def residual_rms(kappa_obs, kappa_pred, mask):
    return float(np.sqrt(np.mean((kappa_obs[mask] - kappa_pred[mask]) ** 2)))


def chi2_improvement(chi2_dof_baseline, chi2_dof_model):
    """Fractional reduction (positive = improvement). Matches XP-005.1 convention."""
    if chi2_dof_baseline == 0:
        return float("nan")
    return float((chi2_dof_baseline - chi2_dof_model) / chi2_dof_baseline)


# ==========================================================================
# DIAGNOSTIC PLOTS
# ==========================================================================

def plot_six_panel(kappa_obs, kappa_model, kappa_baseline,
                   peak_obs, peak_model, peak_baseline, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    panels = [
        (kappa_obs,                          "κ_obs (Clowe 2006)",         peak_obs),
        (kappa_model,                        "κ_model (GDFLM v1.0)",       peak_model),
        (kappa_baseline,                     "κ_baseline (X-ray SB)",      peak_baseline),
        (kappa_model - kappa_obs,            "κ_model − κ_obs",            None),
        (kappa_baseline - kappa_obs,         "κ_baseline − κ_obs",         None),
        ((kappa_obs > 0).astype(float),      "Comparison region (κ_obs>0)", None),
    ]
    for ax, (data, title, peak) in zip(axes.flat, panels):
        if "−" in title:
            im = ax.imshow(data, origin="lower", cmap="RdBu_r",
                           vmin=-np.abs(data).max(), vmax=np.abs(data).max())
        else:
            im = ax.imshow(data, origin="lower", cmap="viridis")
        if peak is not None:
            ax.plot(peak[1], peak[0], "rx", markersize=12, mew=2)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "XP-006 GDFLM v1.0 — Bullet Cluster diagnostic\n"
        f"Manifest SHA-256: 2faf4b6c…ccfc4",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_components(rho, grad_T, grad_rho, K, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (data, title) in zip(axes, [
        (normalise_minmax(rho),       "ρ̂ (normalised)"),
        (normalise_minmax(grad_T),    "|∇T̂| (normalised)"),
        (normalise_minmax(grad_rho),  "|∇ρ̂| (normalised)"),
        (K,                           "K = ρ̂ + |∇T̂| + |∇ρ̂|"),
    ]):
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("XP-006 GDFLM v1.0 — Component fields", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_residual(kappa_obs, kappa_model, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    resid = kappa_model - kappa_obs
    vmax = np.abs(resid).max()
    im = ax.imshow(resid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("XP-006 residual: κ_model − κ_obs")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("=" * 72)
    print("XP-006 GDFLM v1.0 — LIVE RUN (Gate 4)")
    print("=" * 72)

    # --- Verify seal at runtime ---
    sig = manifest_signature()
    expected_sig = "2faf4b6c5e1c2ee40000d6f4a3b3811df151aff8f582036b2d75a97b708ccfc4"
    assert sig == expected_sig, f"MANIFEST DRIFT: {sig} != {expected_sig}"
    print(f"Manifest seal verified at runtime: {sig}")
    print()

    # --- Load fields ---
    print("--- LOADING INPUTS ---")
    rho        = np.load(os.path.join(DATA_PROC, "rho_field.npy"))
    T          = np.load(os.path.join(DATA_PROC, "T_field.npy"))
    grad_T_mag = np.load(os.path.join(DATA_PROC, "grad_T_magnitude.npy"))
    SB         = np.load(os.path.join(DATA_PROC, "SB_field.npy"))
    print(f"  rho:        shape={rho.shape},        peak={np.unravel_index(np.argmax(rho), rho.shape)}")
    print(f"  T:          shape={T.shape},          peak={np.unravel_index(np.argmax(T), T.shape)}")
    print(f"  grad_T_mag: shape={grad_T_mag.shape}, peak={np.unravel_index(np.argmax(grad_T_mag), grad_T_mag.shape)}")
    print(f"  SB:         shape={SB.shape},         peak={np.unravel_index(np.argmax(SB), SB.shape)}")
    print()

    # --- Compute |∇ρ| (NEW — XP-006 only) ---
    print("--- COMPUTING |∇ρ| ---")
    grad_rho_mag = gradient_magnitude(rho)
    print(f"  grad_rho_mag: shape={grad_rho_mag.shape}, "
          f"min={grad_rho_mag.min():.3e}, max={grad_rho_mag.max():.3e}, "
          f"peak={np.unravel_index(np.argmax(grad_rho_mag), grad_rho_mag.shape)}")
    np.save(os.path.join(DATA_PROC, "grad_rho_magnitude.npy"), grad_rho_mag)
    print()

    # --- Apply the model ---
    print("--- APPLYING GDFLM MODEL ---")
    K = K_model(rho, grad_T_mag, grad_rho_mag)
    kappa_model    = model_convergence_xp006(K)
    kappa_baseline = baseline_convergence(SB)
    print(f"  K:              min={K.min():.4f},          max={K.max():.4f}, "
          f"peak={np.unravel_index(np.argmax(K), K.shape)}")
    print(f"  kappa_model:    min={kappa_model.min():.4f},    max={kappa_model.max():.4f}")
    print(f"  kappa_baseline: min={kappa_baseline.min():.4f}, max={kappa_baseline.max():.4f}")
    print()

    # --- Reconstruct κ_obs ---
    print("--- RECONSTRUCTING κ_obs ---")
    kappa_obs_path = os.path.join(BASE_DIR, KAPPA_OBS_FILE)
    kappa_obs = reconstruct_kappa_obs(kappa_obs_path)
    n_positive = int((kappa_obs > 0).sum())
    print(f"  kappa_obs: shape={kappa_obs.shape}, "
          f"non-zero pixels={n_positive} (manifest expects {KAPPA_OBS_PIXEL_COUNT})")
    if n_positive != KAPPA_OBS_PIXEL_COUNT:
        print(f"  ⚠ WARNING: pixel count differs from sealed manifest constant.")
        print(f"     This is the inheritance-side check. Recording but proceeding;")
        print(f"     the numbers are the numbers — no rescue.")
    print()

    mask = kappa_obs > 0
    pos_idx = np.flatnonzero(mask.ravel())

    # --- METRICS (full comparison region) ---
    print("--- COMPUTING METRICS ON FULL COMPARISON REGION ---")
    chi2dof_model,    chi2_model,    dof_model    = chi2_dof(kappa_obs, kappa_model,    mask, COMPARISON_SIGMA)
    chi2dof_baseline, chi2_baseline, dof_baseline = chi2_dof(kappa_obs, kappa_baseline, mask, COMPARISON_SIGMA)
    chi2_improv = chi2_improvement(chi2dof_baseline, chi2dof_model)
    offset_arcsec, peak_obs, peak_model_pix = peak_offset_arcsec(kappa_obs, kappa_model)
    _,             _,        peak_baseline_pix = peak_offset_arcsec(kappa_obs, kappa_baseline)
    rms_full = residual_rms(kappa_obs, kappa_model, mask)

    print(f"  χ²/dof model    = {chi2dof_model:.4f}   (χ²={chi2_model:.2f}, dof={dof_model})")
    print(f"  χ²/dof baseline = {chi2dof_baseline:.4f}   (χ²={chi2_baseline:.2f}, dof={dof_baseline})")
    print(f"  χ²/dof improvement = {chi2_improv:+.4f}   (threshold ≥ {CHI2_IMPROVEMENT_THRESHOLD:+.2f})")
    print(f"  Peak κ_obs   at pixel {peak_obs}")
    print(f"  Peak κ_model at pixel {peak_model_pix}")
    print(f"  Peak κ_base  at pixel {peak_baseline_pix}")
    print(f"  Peak offset (model vs obs) = {offset_arcsec:.2f}\"   (threshold < {PEAK_OFFSET_THRESHOLD}\")")
    print(f"  Residual RMS (full)        = {rms_full:.4f}    (threshold ≤ {RESIDUAL_RMS_THRESHOLD})")
    print()

    # --- BOOTSTRAP STABILITY ---
    print("--- BOOTSTRAP STABILITY (seeds 1, 2, 3) ---")
    boot_chi2dof = []
    boot_details = []
    for seed in BOOTSTRAP_SEEDS:
        sub_idx = bootstrap_subsample_indices(pos_idx, seed=seed)
        sub_mask = np.zeros_like(mask)
        sub_mask.ravel()[sub_idx] = True
        c, raw_chi2, dof_b = chi2_dof(kappa_obs, kappa_model, sub_mask, COMPARISON_SIGMA)
        boot_chi2dof.append(c)
        boot_details.append({"seed": int(seed), "n_pixels": int(dof_b),
                             "chi2": float(raw_chi2), "chi2_dof": float(c)})
        print(f"  seed={seed}: n={dof_b}, χ²={raw_chi2:.2f}, χ²/dof={c:.4f}")
    stability_value = stability(boot_chi2dof)
    print(f"  Stability metric = (max-min)/mean = {stability_value:.4f}   "
          f"(threshold < {STABILITY_THRESHOLD})")
    print()

    # --- PRE-REGISTERED PASS/FAIL ---
    print("--- APPLYING PRE-REGISTERED PASS/FAIL THRESHOLDS ---")
    crit_chi2     = chi2_improv >= CHI2_IMPROVEMENT_THRESHOLD
    crit_offset   = offset_arcsec < PEAK_OFFSET_THRESHOLD
    crit_rms      = rms_full <= RESIDUAL_RMS_THRESHOLD
    crit_stability = stability_value < STABILITY_THRESHOLD

    criteria = [
        ("χ²/dof improvement ≥ 25%",  crit_chi2,
         f"{chi2_improv:+.4f}", f"≥ {CHI2_IMPROVEMENT_THRESHOLD:+.2f}"),
        ("Peak offset < 30 arcsec",   crit_offset,
         f"{offset_arcsec:.2f}\"",     f"< {PEAK_OFFSET_THRESHOLD}\""),
        ("Residual RMS ≤ 0.2",        crit_rms,
         f"{rms_full:.4f}",           f"≤ {RESIDUAL_RMS_THRESHOLD}"),
        ("Stability < 10%",           crit_stability,
         f"{stability_value:.4f}",     f"< {STABILITY_THRESHOLD}"),
    ]
    n_pass = sum(1 for _, p, _, _ in criteria if p)
    overall_verdict = "PASS" if n_pass == 4 else "FAIL"

    print()
    print(f"  {'Criterion':<32} {'Measured':<14} {'Threshold':<12} {'Result':<6}")
    print(f"  {'-'*32} {'-'*14} {'-'*12} {'-'*6}")
    for name, p, m, thr in criteria:
        print(f"  {name:<32} {m:<14} {thr:<12} {'PASS' if p else 'FAIL'}")
    print()
    print(f"  Criteria passed: {n_pass}/4")
    print(f"  OVERALL VERDICT: {overall_verdict}")
    print()

    # --- SAVE OUTPUTS ---
    print("--- SAVING OUTPUTS ---")

    # Plots
    plot_six_panel(kappa_obs, kappa_model, kappa_baseline,
                   peak_obs, peak_model_pix, peak_baseline_pix,
                   os.path.join(RESULTS_DIR, "XP006_diagnostic_6panel.png"))
    plot_components(rho, grad_T_mag, grad_rho_mag, K,
                    os.path.join(RESULTS_DIR, "XP006_components.png"))
    plot_residual(kappa_obs, kappa_model,
                  os.path.join(RESULTS_DIR, "XP006_residual_map.png"))
    print(f"  Saved: results/XP006_diagnostic_6panel.png")
    print(f"  Saved: results/XP006_components.png")
    print(f"  Saved: results/XP006_residual_map.png")

    # Results JSON
    result = {
        "pipeline": "XP-006_GDFLM_v1.0",
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "manifest_sha256": sig,
        "kappa_obs": {
            "source_csv": KAPPA_OBS_FILE,
            "n_positive_pixels": n_positive,
            "manifest_expected_pixel_count": KAPPA_OBS_PIXEL_COUNT,
            "pixel_count_matches_manifest": (n_positive == KAPPA_OBS_PIXEL_COUNT),
            "peak_pixel_row_col": list(peak_obs),
        },
        "kappa_model": {
            "peak_pixel_row_col": list(peak_model_pix),
        },
        "kappa_baseline": {
            "peak_pixel_row_col": list(peak_baseline_pix),
        },
        "metrics_full_region": {
            "chi2_model": chi2_model,
            "chi2_dof_model": chi2dof_model,
            "chi2_baseline": chi2_baseline,
            "chi2_dof_baseline": chi2dof_baseline,
            "chi2_dof_improvement_fraction": chi2_improv,
            "peak_offset_arcsec": offset_arcsec,
            "residual_rms": rms_full,
            "dof": dof_model,
            "comparison_sigma": COMPARISON_SIGMA,
        },
        "bootstrap_stability": {
            "seeds": list(BOOTSTRAP_SEEDS),
            "fraction": BOOTSTRAP_FRACTION,
            "n_runs": BOOTSTRAP_N_RUNS,
            "per_run": boot_details,
            "chi2_dof_values": [float(c) for c in boot_chi2dof],
            "stability_metric": float(stability_value),
            "stability_metric_definition": "(max - min) / mean of the 3 chi2/dof values",
        },
        "pass_fail": {
            "thresholds": {
                "chi2_improvement_min": CHI2_IMPROVEMENT_THRESHOLD,
                "peak_offset_max_arcsec": PEAK_OFFSET_THRESHOLD,
                "residual_rms_max": RESIDUAL_RMS_THRESHOLD,
                "stability_max": STABILITY_THRESHOLD,
            },
            "criteria": [
                {"name": name, "passed": bool(p), "measured": m, "threshold": thr}
                for name, p, m, thr in criteria
            ],
            "n_pass": n_pass,
            "overall_verdict": overall_verdict,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "XP006_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")

    # Verdict markdown
    md = []
    md.append(f"# XP-006 GDFLM v1.0 — Verdict\n")
    md.append(f"**Date:** {dt.datetime.utcnow().isoformat()}Z  ")
    md.append(f"**Manifest SHA-256:** `{sig}`\n")
    md.append(f"## Overall Verdict: **{overall_verdict}** ({n_pass}/4 criteria passed)\n")
    md.append(f"## κ_obs reconstruction\n")
    md.append(f"- Source: `{KAPPA_OBS_FILE}`")
    md.append(f"- Non-zero pixels: {n_positive} (manifest expects {KAPPA_OBS_PIXEL_COUNT}) — "
              f"{'match' if n_positive == KAPPA_OBS_PIXEL_COUNT else 'mismatch'}")
    md.append(f"- Peak pixel: {peak_obs}\n")
    md.append(f"## Pre-registered criteria\n")
    md.append("| Criterion | Threshold | Measured | Result |")
    md.append("|-----------|-----------|----------|--------|")
    for name, p, m, thr in criteria:
        md.append(f"| {name} | {thr} | {m} | **{'PASS' if p else 'FAIL'}** |")
    md.append("")
    md.append("## Peak locations\n")
    md.append(f"- κ_obs    peak: pixel {peak_obs}")
    md.append(f"- κ_model  peak: pixel {peak_model_pix}   (offset {offset_arcsec:.2f}\")")
    md.append(f"- κ_base   peak: pixel {peak_baseline_pix}")
    md.append("")
    md.append("## Bootstrap (seeds 1, 2, 3)\n")
    md.append("| Seed | n_pixels | χ²/dof |")
    md.append("|------|---------:|-------:|")
    for d in boot_details:
        md.append(f"| {d['seed']} | {d['n_pixels']} | {d['chi2_dof']:.4f} |")
    md.append(f"\nStability metric: **{stability_value:.4f}** "
              f"(threshold < {STABILITY_THRESHOLD})\n")
    md.append("## Outputs\n")
    md.append("- `results/XP006_results.json` — full machine-readable result")
    md.append("- `results/XP006_diagnostic_6panel.png`")
    md.append("- `results/XP006_components.png`")
    md.append("- `results/XP006_residual_map.png`")
    md.append("- `checkpoints/run_xp006_state.pkl`")
    md_path = os.path.join(RESULTS_DIR, "XP006_VERDICT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved: {md_path}")

    # Checkpoint pickle
    state = {
        "rho": rho, "T": T, "grad_T_mag": grad_T_mag, "SB": SB,
        "grad_rho_mag": grad_rho_mag, "K": K,
        "kappa_model": kappa_model, "kappa_baseline": kappa_baseline,
        "kappa_obs": kappa_obs, "mask": mask,
        "result": result,
    }
    ckpt_path = os.path.join(CKPT_DIR, "run_xp006_state.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(state, f)
    print(f"  Saved: {ckpt_path}")
    print()
    print("=" * 72)
    print(f"GATE 4 COMPLETE — VERDICT: {overall_verdict} ({n_pass}/4)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
