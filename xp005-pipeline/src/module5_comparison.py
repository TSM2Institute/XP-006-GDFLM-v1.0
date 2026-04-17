"""
XP-005.1 — Module 5: Comparison + PASS/FAIL
Loads the digitised Clowe 2006 κ_obs map, runs the pre-registered comparison
against kappa_model and kappa_baseline, and applies all four PASS/FAIL criteria.
"""

import sys
import os
import json
import pickle
import csv
import datetime

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifest import (
    SIGMA_GAL_ARCSEC,
    CHI2_IMPROVEMENT_THRESHOLD,
    PEAK_OFFSET_THRESHOLD,
    RESIDUAL_RMS_THRESHOLD,
    STABILITY_THRESHOLD,
    phase_function,
    boundary_emission,
    propagation_field,
    observed_field,
    model_convergence,
)
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
DATA_DIG = os.path.join(BASE_DIR, "data", "digitised")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MANIFEST_SHA256 = "b1ca5dc7900fa7f04330d72804f62b499fecdacbfd0aac2105feb8bea5524b2e"

# Calibration anchors for digitised CSV → RA/Dec
X1_DIG, RA1 = -5.6078e-3, 104.675
X2_DIG, RA2 = -4.9944,    104.550
Y1_DIG, DEC1 = -2.8039e-2, -55.9667
Y2_DIG, DEC2 = -5.5508e-2, -55.9333

# Standard Bullet Cluster centre (midpoint of Clowe figure)
RA_CENTRE = 104.625
DEC_CENTRE = -55.950

SIGMA_UNIFORM = 0.14
SIGMA_JUSTIFICATION = "Bradac et al. 2006 ~14% uncertainty over ACS field"


# ==========================================================================
# PART A — Load and transform digitised κ_obs
# ==========================================================================

def parse_digitised_csv(filepath):
    """Parse WebPlotDigitizer CSV with paired X,Y columns per contour."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError("Digitised CSV has fewer than 3 rows; cannot parse.")

    contour_header = rows[0]
    n_cols = len(contour_header)

    # Build contour name list (each contour spans 2 columns: X, Y)
    contours = []
    col = 0
    while col < n_cols:
        name = contour_header[col].strip()
        if not name:
            col += 1
            continue
        contours.append({"name": name, "x_col": col, "y_col": col + 1, "points": []})
        col += 2

    # Parse data rows
    for r in rows[2:]:
        for c in contours:
            if c["x_col"] >= len(r) or c["y_col"] >= len(r):
                continue
            xs = r[c["x_col"]].strip()
            ys = r[c["y_col"]].strip()
            if xs == "" or ys == "":
                continue
            try:
                x = float(xs)
                y = float(ys)
                c["points"].append((x, y))
            except ValueError:
                continue

    return contours


def transform_to_radec(x_dig, y_dig):
    ra = RA1 + (x_dig - X1_DIG) * (RA2 - RA1) / (X2_DIG - X1_DIG)
    dec = DEC1 + (y_dig - Y1_DIG) * (DEC2 - DEC1) / (Y2_DIG - Y1_DIG)
    return ra, dec


def radec_to_pixel(ra, dec, grid_params):
    pixel_scale = grid_params["pixel_scale_arcsec"]
    cx, cy = grid_params["centre_pixel"]
    cos_dec = np.cos(np.radians(DEC_CENTRE))
    col = cx + (RA_CENTRE - ra) * 3600.0 * cos_dec / pixel_scale
    row = cy - (dec - DEC_CENTRE) * 3600.0 / pixel_scale
    return col, row


def kappa_value_from_name(name):
    """Extract numerical kappa level from contour name like 'kappa_0.23_main'."""
    parts = name.split("_")
    for p in parts[1:]:
        try:
            return float(p)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse kappa value from contour name: {name}")


def reconstruct_kappa_obs(contours, grid_params):
    grid_size = grid_params["grid_size"]
    kappa_obs = np.zeros((grid_size, grid_size))

    # Group contours by kappa level
    levels = {}
    for c in contours:
        if not c["points"]:
            continue
        level = kappa_value_from_name(c["name"])
        levels.setdefault(level, []).append(c)

    print(f"  Found contour levels: {sorted(levels.keys())}")

    inside_counts = []
    centroids = {}

    for level in sorted(levels.keys()):
        for c in levels[level]:
            radec_pts = [transform_to_radec(x, y) for (x, y) in c["points"]]
            pixel_pts = [radec_to_pixel(ra, dec, grid_params) for (ra, dec) in radec_pts]
            cols = np.array([p[0] for p in pixel_pts])
            rows = np.array([p[1] for p in pixel_pts])

            inside = (
                (cols >= 0) & (cols < grid_size) &
                (rows >= 0) & (rows < grid_size)
            )
            inside_counts.append((c["name"], int(np.sum(inside)), int(np.sum(~inside))))

            # Centroid in RA/Dec for verification
            ra_c = np.mean([p[0] for p in radec_pts])
            dec_c = np.mean([p[1] for p in radec_pts])
            centroids[c["name"]] = (ra_c, dec_c)

            # Close polygon if needed
            if len(pixel_pts) >= 3:
                first = pixel_pts[0]
                last = pixel_pts[-1]
                gap = np.hypot(first[0] - last[0], first[1] - last[1])
                if gap > 5.0:
                    print(f"  WARNING: contour '{c['name']}' not closed (gap={gap:.2f} px); closing manually")
                pixel_pts_closed = pixel_pts + [pixel_pts[0]]

                poly = MplPath(pixel_pts_closed)

                yy, xx = np.mgrid[0:grid_size, 0:grid_size]
                pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
                inside_mask = poly.contains_points(pts_grid).reshape(grid_size, grid_size)

                # Fill with this kappa level (overwrites any lower level)
                kappa_obs[inside_mask] = level

    return kappa_obs, inside_counts, centroids


# ==========================================================================
# PART B — Comparison + stability
# ==========================================================================

def build_chi_from_subsample(galaxies_sub, grid_params):
    grid_size = grid_params["grid_size"]
    pixel_scale = grid_params["pixel_scale_arcsec"]
    cx, cy = grid_params["centre_pixel"]
    cos_dec = np.cos(np.radians(-55.9))

    x_pixel = cx - (galaxies_sub["MatchRA"].values - 104.6) * 3600.0 * cos_dec / pixel_scale
    y_pixel = cy + (galaxies_sub["MatchDec"].values - (-55.9)) * 3600.0 / pixel_scale

    inside = (
        (x_pixel >= 0) & (x_pixel < grid_size) &
        (y_pixel >= 0) & (y_pixel < grid_size)
    )
    raw_hist, _, _ = np.histogram2d(
        y_pixel[inside], x_pixel[inside],
        bins=grid_size,
        range=[[0, grid_size], [0, grid_size]],
    )
    sigma_pix = SIGMA_GAL_ARCSEC / pixel_scale
    smoothed = gaussian_filter(raw_hist, sigma=sigma_pix)
    max_val = np.max(smoothed)
    return smoothed / max_val if max_val > 0 else np.zeros_like(smoothed)


def chi2_dof(model_field, obs_field, mask):
    diff = (model_field[mask] - obs_field[mask]) / SIGMA_UNIFORM
    return np.sum(diff**2) / np.sum(mask)


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — MODULE 5: COMPARISON + PASS/FAIL")
    print("=" * 70)

    digitised_path = os.path.join(DATA_DIG, "clowe2006_kappa.csv")
    if not os.path.exists(digitised_path):
        print("\n  ERROR: Digitised κ_obs CSV not found.")
        print(f"  Expected at: {digitised_path}")
        print("\n  Place the WebPlotDigitizer CSV (Clowe 2006 Figure 1 contour traces)")
        print("  at the path above before re-running this module.")
        print("\n  Module 5 cannot proceed without κ_obs. Halting.")
        sys.exit(1)

    with open(os.path.join(DATA_PROC, "grid_params.json")) as f:
        grid_params = json.load(f)

    # --- PART A ---
    print("\n--- PART A: Load and reconstruct κ_obs from digitised CSV ---")
    contours = parse_digitised_csv(digitised_path)
    print(f"  Parsed {len(contours)} contour traces")
    for c in contours:
        print(f"    {c['name']}: {len(c['points'])} points")

    kappa_obs_unnorm, inside_counts, centroids = reconstruct_kappa_obs(contours, grid_params)

    print("\n  Contour centroids (RA, Dec):")
    for name, (ra, dec) in centroids.items():
        print(f"    {name:30s}  RA={ra:.4f}°  Dec={dec:.4f}°")

    print("\n  Pixels inside/outside grid per contour:")
    for name, n_in, n_out in inside_counts:
        print(f"    {name:30s}  inside={n_in:5d}  outside={n_out:5d}")

    max_unnorm = np.max(kappa_obs_unnorm)
    if max_unnorm == 0:
        print("\n  ERROR: kappa_obs is all zero after reconstruction. Halting.")
        sys.exit(1)
    kappa_obs = kappa_obs_unnorm / max_unnorm

    np.save(os.path.join(DATA_PROC, "kappa_obs.npy"), kappa_obs)
    np.save(os.path.join(DATA_PROC, "kappa_obs_unnorm.npy"), kappa_obs_unnorm)

    obs_peak = np.unravel_index(np.argmax(kappa_obs), kappa_obs.shape)
    print(f"\n  κ_obs (unnormalised): min={np.min(kappa_obs_unnorm):.4f}  max={max_unnorm:.4f}")
    print(f"  κ_obs (normalised):   min={np.min(kappa_obs):.4f}  max={np.max(kappa_obs):.4f}")
    print(f"  Non-zero pixels:      {int(np.sum(kappa_obs > 0))}")
    print(f"  Peak pixel:           {obs_peak}")

    # --- PART B ---
    print("\n--- PART B: Comparison statistics ---")
    kappa_model = np.load(os.path.join(DATA_PROC, "kappa_model.npy"))
    kappa_baseline = np.load(os.path.join(DATA_PROC, "kappa_baseline.npy"))

    mask = kappa_obs > 0
    n_pixels = int(np.sum(mask))
    print(f"  Comparison region: {n_pixels} pixels (where κ_obs > 0)")
    print(f"  σ_uniform = {SIGMA_UNIFORM} ({SIGMA_JUSTIFICATION})")

    # METRIC 1 — chi2/dof improvement
    chi2_dof_model = chi2_dof(kappa_model, kappa_obs, mask)
    chi2_dof_baseline = chi2_dof(kappa_baseline, kappa_obs, mask)
    improvement = (chi2_dof_baseline - chi2_dof_model) / chi2_dof_baseline * 100.0
    chi2_pass = improvement >= (CHI2_IMPROVEMENT_THRESHOLD * 100.0)
    print(f"\n  METRIC 1 — χ²/dof:")
    print(f"    chi2_dof_model    = {chi2_dof_model:.4f}")
    print(f"    chi2_dof_baseline = {chi2_dof_baseline:.4f}")
    print(f"    improvement       = {improvement:.2f}%   threshold ≥ 25%   [{'PASS' if chi2_pass else 'FAIL'}]")

    # METRIC 2 — Peak offset
    model_peak = np.unravel_index(np.argmax(kappa_model), kappa_model.shape)
    obs_peak = np.unravel_index(np.argmax(kappa_obs), kappa_obs.shape)
    dpix = np.hypot(model_peak[0] - obs_peak[0], model_peak[1] - obs_peak[1])
    offset_arcsec = dpix * grid_params["pixel_scale_arcsec"]
    offset_pass = offset_arcsec < PEAK_OFFSET_THRESHOLD
    print(f"\n  METRIC 2 — Peak offset:")
    print(f"    model peak = {model_peak}, obs peak = {obs_peak}")
    print(f"    distance   = {dpix:.2f} px = {offset_arcsec:.2f} arcsec   threshold < 30″   [{'PASS' if offset_pass else 'FAIL'}]")

    # METRIC 3 — Residual RMS
    residuals = kappa_model[mask] - kappa_obs[mask]
    rms = float(np.sqrt(np.mean(residuals**2)))
    rms_pass = rms <= RESIDUAL_RMS_THRESHOLD
    print(f"\n  METRIC 3 — Residual RMS:")
    print(f"    rms = {rms:.4f}   threshold ≤ 0.2   [{'PASS' if rms_pass else 'FAIL'}]")

    # METRIC 4 — Stability across galaxy subsamples
    print(f"\n  METRIC 4 — Stability across 3 subsamples (2/3 each):")
    cat_path = os.path.join(DATA_PROC, "galaxy_catalogue_clean.csv")
    galaxies = pd.read_csv(cat_path)
    rho = np.load(os.path.join(DATA_PROC, "rho_field.npy"))
    grad_T_mag = np.load(os.path.join(DATA_PROC, "grad_T_magnitude.npy"))
    P_full = phase_function(rho)

    chi2_dof_subs = []
    for i in [1, 2, 3]:
        rng = np.random.default_rng(seed=42 + i)
        n_sub = int(2 * len(galaxies) / 3)
        idx = rng.choice(len(galaxies), size=n_sub, replace=False)
        sub = galaxies.iloc[idx]

        chi_sub = build_chi_from_subsample(sub, grid_params)
        psi_emit_sub = boundary_emission(P_full, chi_sub)
        psi_prop_sub = propagation_field(P_full, grad_T_mag, chi_sub)
        psi_obs_sub = observed_field(psi_emit_sub, psi_prop_sub)
        kappa_model_sub = model_convergence(psi_obs_sub)

        c2 = chi2_dof(kappa_model_sub, kappa_obs, mask)
        chi2_dof_subs.append(c2)
        print(f"    subsample {i} (seed={42+i}): chi2_dof = {c2:.4f}")

    mean_c2 = float(np.mean(chi2_dof_subs))
    variation_percent = (max(chi2_dof_subs) - min(chi2_dof_subs)) / mean_c2 * 100.0
    stability_pass = variation_percent < (STABILITY_THRESHOLD * 100.0)
    print(f"    variation = {variation_percent:.2f}%   threshold < 10%   [{'PASS' if stability_pass else 'FAIL'}]")

    # --- VERDICT ---
    all_pass = chi2_pass and offset_pass and rms_pass and stability_pass
    verdict = "PASS" if all_pass else "FAIL"

    # --- PART C: Compile results JSON ---
    results = {
        "test_id": "XP-005.1",
        "target": "Bullet Cluster (1E 0657-56)",
        "date": datetime.date.today().isoformat(),
        "manifest_sha256": MANIFEST_SHA256,
        "build_path": "A (parametric + digitised)",
        "comparison_region_pixels": n_pixels,
        "sigma_uniform": SIGMA_UNIFORM,
        "sigma_justification": SIGMA_JUSTIFICATION,
        "metrics": {
            "chi2_dof_model": float(chi2_dof_model),
            "chi2_dof_baseline": float(chi2_dof_baseline),
            "chi2_improvement_percent": float(improvement),
            "chi2_improvement_threshold": 25.0,
            "chi2_improvement_pass": bool(chi2_pass),
            "peak_offset_arcsec": float(offset_arcsec),
            "peak_offset_threshold": 30.0,
            "peak_offset_pass": bool(offset_pass),
            "residual_rms": rms,
            "residual_rms_threshold": 0.2,
            "residual_rms_pass": bool(rms_pass),
            "stability_chi2_values": [float(v) for v in chi2_dof_subs],
            "stability_variation_percent": float(variation_percent),
            "stability_threshold": 10.0,
            "stability_pass": bool(stability_pass),
        },
        "verdict": verdict,
        "verdict_rule": "All four criteria must pass for PASS. Any failure = FAIL.",
    }

    results_path = os.path.join(RESULTS_DIR, "XP005_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    ckpt = {
        "kappa_obs": kappa_obs,
        "kappa_obs_unnorm": kappa_obs_unnorm,
        "mask": mask,
        "results": results,
        "chi2_dof_subs": chi2_dof_subs,
    }
    with open(os.path.join(CKPT_DIR, "module5_results.pkl"), "wb") as f:
        pickle.dump(ckpt, f)
    print(f"  Saved: checkpoints/module5_results.pkl")

    print("\n" + "=" * 70)
    print(f"  VERDICT: {verdict}")
    print("=" * 70)
