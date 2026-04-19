"""
XP-005.1 — Module 2: Field Construction
Builds chi(x,y) from the galaxy catalogue and computes P(x,y) from rho.
"""

import sys
import os
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifest import SIGMA_GAL_ARCSEC, phase_function

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RA_CENTRE = 104.6
DEC_CENTRE = -55.9

# ==========================================================================
# STEP 1: Build galaxy surface density chi(x,y)
# ==========================================================================

def build_chi_field(grid_params):
    grid_size = grid_params["grid_size"]
    pixel_scale = grid_params["pixel_scale_arcsec"]
    cx, cy = grid_params["centre_pixel"]

    cat_path = os.path.join(DATA_PROC, "galaxy_catalogue_clean.csv")
    df = pd.read_csv(cat_path)
    print(f"  Loaded galaxy catalogue: {len(df)} galaxies")

    dec_centre_rad = np.radians(DEC_CENTRE)
    cos_dec = np.cos(dec_centre_rad)

    x_pixel = cx - (df["MatchRA"].values - RA_CENTRE) * 3600.0 * cos_dec / pixel_scale
    y_pixel = cy + (df["MatchDec"].values - DEC_CENTRE) * 3600.0 / pixel_scale

    inside = (
        (x_pixel >= 0) & (x_pixel < grid_size) &
        (y_pixel >= 0) & (y_pixel < grid_size)
    )
    n_inside = int(np.sum(inside))
    n_outside = len(df) - n_inside
    print(f"  Galaxies inside grid:  {n_inside}")
    print(f"  Galaxies outside grid: {n_outside}")

    x_in = x_pixel[inside]
    y_in = y_pixel[inside]

    raw_hist, _, _ = np.histogram2d(
        y_in, x_in,
        bins=grid_size,
        range=[[0, grid_size], [0, grid_size]],
    )

    sigma_pix = SIGMA_GAL_ARCSEC / pixel_scale
    print(f"  Smoothing kernel sigma: {sigma_pix:.4f} pixels (SIGMA_GAL_ARCSEC={SIGMA_GAL_ARCSEC:.2f}, scale={pixel_scale})")

    smoothed = gaussian_filter(raw_hist, sigma=sigma_pix)

    max_val = np.max(smoothed)
    if max_val == 0:
        chi = np.zeros_like(smoothed)
    else:
        chi = smoothed / max_val

    return chi, raw_hist, x_pixel, y_pixel, inside, n_inside, n_outside, sigma_pix


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — MODULE 2: FIELD CONSTRUCTION")
    print("=" * 70)

    with open(os.path.join(DATA_PROC, "grid_params.json")) as f:
        grid_params = json.load(f)

    print("\n--- STEP 1: Build chi(x,y) from galaxy catalogue ---")
    chi, raw_hist, x_pix, y_pix, inside, n_inside, n_outside, sigma_pix = build_chi_field(grid_params)
    np.save(os.path.join(DATA_PROC, "chi_field.npy"), chi)
    print(f"  Saved: chi_field.npy")

    print("\n--- STEP 2: Load Module 1 fields ---")
    rho = np.load(os.path.join(DATA_PROC, "rho_field.npy"))
    T = np.load(os.path.join(DATA_PROC, "T_field.npy"))
    grad_T_mag = np.load(os.path.join(DATA_PROC, "grad_T_magnitude.npy"))
    SB = np.load(os.path.join(DATA_PROC, "SB_field.npy"))
    print(f"  Loaded: rho {rho.shape}, T {T.shape}, |grad_T| {grad_T_mag.shape}, SB {SB.shape}")

    print("\n--- STEP 3: Compute phase function P(x,y) ---")
    P = phase_function(rho)
    np.save(os.path.join(DATA_PROC, "P_field.npy"), P)
    print(f"  Saved: P_field.npy")

    print("\n--- STEP 4: Verification ---")
    chi_peak = np.unravel_index(np.argmax(chi), chi.shape)
    P_peak = np.unravel_index(np.argmax(P), P.shape)

    print(f"  chi(x,y): min={np.min(chi):.6e}  max={np.max(chi):.6e}  mean={np.mean(chi):.6e}  peak={chi_peak}")
    print(f"  P(x,y):   min={np.min(P):.6e}  max={np.max(P):.6e}  mean={np.mean(P):.6e}  peak={P_peak}")
    print(f"  Smoothing sigma: {sigma_pix:.4f} pixels (expected ~2.27)")

    print("\n--- STEP 5: Save checkpoint ---")
    checkpoint = {
        "chi": chi,
        "P": P,
        "raw_hist": raw_hist,
        "galaxy_x_pixel": x_pix,
        "galaxy_y_pixel": y_pix,
        "inside_mask": inside,
        "n_inside": n_inside,
        "n_outside": n_outside,
        "sigma_pix": sigma_pix,
        "grid_params": grid_params,
    }
    ckpt_path = os.path.join(CKPT_DIR, "module2_fields.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved: {ckpt_path}")

    print("\n--- STEP 6: Diagnostic plot ---")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ax = axes[0]
    ax.scatter(x_pix[inside], y_pix[inside], s=0.5, alpha=0.3, c="cyan")
    ax.set_xlim(0, grid_params["grid_size"])
    ax.set_ylim(0, grid_params["grid_size"])
    ax.set_title("Galaxy positions on grid")
    ax.set_aspect("equal")

    ax = axes[1]
    im = ax.imshow(raw_hist, origin="lower", cmap="inferno")
    ax.set_title("Raw galaxy density (histogram)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im = ax.imshow(chi, origin="lower", cmap="inferno")
    ax.set_title(f"Smoothed chi(x,y) [σ={sigma_pix:.2f} px]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[3]
    im = ax.imshow(P, origin="lower", cmap="inferno")
    ax.set_title("P(x,y) — Phase function")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("XP-005.1 Module 2 — chi & P Fields", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "module2_chi_P_fields.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    print("\n" + "=" * 70)
    print("  MODULE 2 COMPLETE")
    print("=" * 70)
