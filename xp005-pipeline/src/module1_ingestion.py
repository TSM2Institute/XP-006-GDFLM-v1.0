"""
XP-005.1 — Module 1: Data Ingestion
Downloads HSC v3 galaxy catalogue, builds parametric density and temperature
fields from published literature, and creates a placeholder κ map loader.
"""

import sys
import os
import json
import pickle
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifest import M_PROTON, KPC_PER_ARCSEC

# ==========================================================================
# LITERATURE PARAMETERS (fully cited)
# ==========================================================================

LITERATURE_PARAMS = {
    "n0_main": {
        "value": 3.0e-3,
        "units": "cm^-3",
        "source": "Markevitch et al. 2002, ApJ 567, L27",
    },
    "rc_main": {
        "value": 250.0,
        "units": "kpc",
        "source": "Markevitch et al. 2002, ApJ 567, L27",
    },
    "beta_main": {
        "value": 0.70,
        "units": "dimensionless",
        "source": "Markevitch et al. 2002, ApJ 567, L27",
    },
    "n0_bullet": {
        "value": 4.0e-2,
        "units": "cm^-3",
        "source": "Markevitch 2006, ESA SP-604",
    },
    "rc_bullet": {
        "value": 50.0,
        "units": "kpc",
        "source": "Markevitch 2006, ESA SP-604",
    },
    "beta_bullet": {
        "value": 0.60,
        "units": "dimensionless",
        "source": "Markevitch 2006, ESA SP-604",
    },
    "T_main": {
        "value": 14.1,
        "units": "keV",
        "source": "Markevitch 2006, ESA SP-604",
    },
    "T_bullet": {
        "value": 6.5,
        "units": "keV",
        "source": "Markevitch 2006, ESA SP-604",
    },
    "T_background": {
        "value": 10.0,
        "units": "keV",
        "source": "Markevitch 2006, ESA SP-604",
    },
}

# ==========================================================================
# GRID SETUP
# ==========================================================================

GRID_SIZE = 200
PIXEL_SCALE_ARCSEC = 2.0
CENTRE_PIXEL = (100, 100)
BULLET_OFFSET_ARCSEC = 65.0
BULLET_OFFSET_PIXELS = int(round(BULLET_OFFSET_ARCSEC / PIXEL_SCALE_ARCSEC))
BULLET_PIXEL = (CENTRE_PIXEL[0] + BULLET_OFFSET_PIXELS, CENTRE_PIXEL[1])

GRID_PARAMS = {
    "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
    "grid_size": GRID_SIZE,
    "centre_pixel": list(CENTRE_PIXEL),
    "bullet_pixel": list(BULLET_PIXEL),
    "bullet_offset_pixels": BULLET_OFFSET_PIXELS,
    "kpc_per_arcsec": KPC_PER_ARCSEC,
}

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
DATA_DIG = os.path.join(BASE_DIR, "data", "digitised")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ==========================================================================
# PART A: Galaxy Catalogue (HSC v3) from MAST
# ==========================================================================

def download_hsc_catalogue():
    url = (
        "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/summary/magauto.csv"
        "?ra=104.6&dec=-55.9&radius=0.083&pagesize=50000"
        "&columns=[MatchRA,MatchDec,MatchID,NumImages,NumFilters,NumVisits,"
        "A_F606W,A_F775W,A_F814W,A_F850LP,W3_F110W,W3_F160W,CI]"
    )
    raw_path = os.path.join(DATA_RAW, "bullet_cluster_hsc_v3.csv")

    print("Downloading HSC v3 catalogue from MAST...")
    try:
        urllib.request.urlretrieve(url, raw_path)
        print(f"  Saved raw download to: {raw_path}")
    except Exception as e:
        print(f"  WARNING: Download failed ({e})")
        print("  Attempting to continue with existing file if available...")
        if not os.path.exists(raw_path):
            print("  No existing file found. Galaxy catalogue unavailable.")
            return None

    df = pd.read_csv(raw_path)
    total = len(df)
    print(f"  Total sources downloaded: {total}")

    df = df.dropna(subset=["CI"])
    after_nan = len(df)
    print(f"  After removing CI=NaN: {after_nan}")

    stars = df[df["CI"] < 1.4]
    galaxies = df[df["CI"] >= 1.4].copy()
    print(f"  Stars removed (CI < 1.4): {len(stars)}")
    print(f"  Galaxies retained (CI >= 1.4): {len(galaxies)}")

    clean_path = os.path.join(DATA_PROC, "galaxy_catalogue_clean.csv")
    galaxies.to_csv(clean_path, index=False)
    print(f"  Saved cleaned catalogue to: {clean_path}")

    ra_min, ra_max = galaxies["MatchRA"].min(), galaxies["MatchRA"].max()
    dec_min, dec_max = galaxies["MatchDec"].min(), galaxies["MatchDec"].max()
    print(f"  RA range:  {ra_min:.4f} — {ra_max:.4f} deg")
    print(f"  Dec range: {dec_min:.4f} — {dec_max:.4f} deg")

    return galaxies


# ==========================================================================
# PART B: Parametric Density and Temperature Fields
# ==========================================================================

def beta_model(x, y, cx, cy, n0, rc_kpc, beta):
    dx = (x - cx) * PIXEL_SCALE_ARCSEC * KPC_PER_ARCSEC
    dy = (y - cy) * PIXEL_SCALE_ARCSEC * KPC_PER_ARCSEC
    r_kpc = np.sqrt(dx**2 + dy**2)
    return n0 * (1.0 + (r_kpc / rc_kpc) ** 2) ** (-3.0 * beta / 2.0)


def build_density_field():
    y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]

    n_e_main = beta_model(
        x, y,
        CENTRE_PIXEL[0], CENTRE_PIXEL[1],
        LITERATURE_PARAMS["n0_main"]["value"],
        LITERATURE_PARAMS["rc_main"]["value"],
        LITERATURE_PARAMS["beta_main"]["value"],
    )
    n_e_bullet = beta_model(
        x, y,
        BULLET_PIXEL[0], BULLET_PIXEL[1],
        LITERATURE_PARAMS["n0_bullet"]["value"],
        LITERATURE_PARAMS["rc_bullet"]["value"],
        LITERATURE_PARAMS["beta_bullet"]["value"],
    )
    n_e_total = n_e_main + n_e_bullet
    rho = n_e_total * M_PROTON
    return n_e_total, rho


def make_gaussian_2d(cx, cy, width_pix, amplitude, grid_size=GRID_SIZE):
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width_pix ** 2))


def build_temperature_field():
    T_bg = LITERATURE_PARAMS["T_background"]["value"]
    T_main_amp = LITERATURE_PARAMS["T_main"]["value"] - T_bg
    T_bullet_amp = LITERATURE_PARAMS["T_bullet"]["value"] - T_bg

    T = np.full((GRID_SIZE, GRID_SIZE), T_bg)
    T += make_gaussian_2d(CENTRE_PIXEL[0], CENTRE_PIXEL[1], 60, T_main_amp)
    T += make_gaussian_2d(BULLET_PIXEL[0], BULLET_PIXEL[1], 20, T_bullet_amp)

    grad_Ty, grad_Tx = np.gradient(T)
    grad_T_mag = np.sqrt(grad_Tx**2 + grad_Ty**2)
    return T, grad_T_mag


def build_surface_brightness(n_e_total):
    return n_e_total**2


# ==========================================================================
# PART C: κ Map Placeholder
# ==========================================================================

def load_kappa_obs(filepath=None):
    if filepath is None:
        filepath = os.path.join(DATA_DIG, "clowe2006_kappa.csv")

    if not os.path.exists(filepath):
        print(
            f"\n  κ_obs map not yet digitised. "
            f"Place digitised CSV at: data/digitised/clowe2006_kappa.csv"
        )
        return None

    df = pd.read_csv(filepath)
    xi = np.arange(GRID_SIZE)
    yi = np.arange(GRID_SIZE)
    grid_x, grid_y = np.meshgrid(xi, yi)

    kappa = griddata(
        (df["x"].values, df["y"].values),
        df["kappa"].values,
        (grid_x, grid_y),
        method="cubic",
        fill_value=0.0,
    )
    print(f"  Loaded κ_obs from {filepath}: shape={kappa.shape}")
    return kappa


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — MODULE 1: DATA INGESTION")
    print("=" * 70)

    print("\n--- LITERATURE PARAMETERS ---")
    for key, info in LITERATURE_PARAMS.items():
        print(f"  {key:16s} = {info['value']}  [{info['units']}]  — {info['source']}")

    # --- PART A ---
    print("\n--- PART A: HSC v3 Galaxy Catalogue ---")
    galaxies = download_hsc_catalogue()

    # --- PART B ---
    print("\n--- PART B: Parametric Fields ---")
    n_e_total, rho = build_density_field()
    T, grad_T_mag = build_temperature_field()
    SB = build_surface_brightness(n_e_total)

    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, pixel scale = {PIXEL_SCALE_ARCSEC} arcsec/pixel")
    print(f"  Main cluster centre: pixel {CENTRE_PIXEL}")
    print(f"  Bullet centre:       pixel {BULLET_PIXEL} ({BULLET_OFFSET_PIXELS} px offset)")

    # --- PART C ---
    print("\n--- PART C: κ Map Placeholder ---")
    kappa_obs = load_kappa_obs()

    # --- Save outputs ---
    print("\n--- SAVING OUTPUTS ---")
    np.save(os.path.join(DATA_PROC, "rho_field.npy"), rho)
    np.save(os.path.join(DATA_PROC, "T_field.npy"), T)
    np.save(os.path.join(DATA_PROC, "grad_T_magnitude.npy"), grad_T_mag)
    np.save(os.path.join(DATA_PROC, "SB_field.npy"), SB)

    chi_placeholder = np.zeros((GRID_SIZE, GRID_SIZE))
    np.save(os.path.join(DATA_PROC, "chi_field.npy"), chi_placeholder)

    with open(os.path.join(DATA_PROC, "grid_params.json"), "w") as f:
        json.dump(GRID_PARAMS, f, indent=2)

    print("  Saved: rho_field.npy, T_field.npy, grad_T_magnitude.npy, SB_field.npy")
    print("  Saved: chi_field.npy (placeholder — computed in Module 2)")
    print("  Saved: grid_params.json")

    checkpoint = {
        "n_e_total": n_e_total,
        "rho": rho,
        "T": T,
        "grad_T_mag": grad_T_mag,
        "SB": SB,
        "chi_placeholder": chi_placeholder,
        "grid_params": GRID_PARAMS,
        "literature_params": LITERATURE_PARAMS,
        "galaxies": galaxies if galaxies is not None else None,
        "kappa_obs": kappa_obs,
    }
    ckpt_path = os.path.join(CKPT_DIR, "module1_ingestion.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Checkpoint saved: {ckpt_path}")

    # --- Verification ---
    print("\n--- VERIFICATION TABLE ---")
    header = f"  {'Field':<15s} | {'Shape':>9s} | {'Min':>14s} | {'Max':>14s} | {'Units'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, arr, units in [
        ("rho(x,y)", rho, "g/cm³"),
        ("T(x,y)", T, "keV"),
        ("|grad_T|(x,y)", grad_T_mag, "keV/pixel"),
        ("SB(x,y)", SB, "arb"),
    ]:
        shape_str = f"{arr.shape[0]}x{arr.shape[1]}"
        print(f"  {name:<15s} | {shape_str:>9s} | {np.min(arr):>14.6e} | {np.max(arr):>14.6e} | {units}")
    if galaxies is not None:
        print(f"  {'Galaxy cat':<15s} | {str(len(galaxies))+'x2':>9s} | {'—':>14s} | {'—':>14s} | deg")

    print("\n--- SPOT CHECKS ---")
    cx, cy = CENTRE_PIXEL
    bx, by = BULLET_PIXEL
    print(f"  rho at grid centre ({cx},{cy}):  {rho[cy, cx]:.4e} g/cm³  (expect ~5e-27)")
    print(f"  rho at bullet ({bx},{by}):       {rho[by, bx]:.4e} g/cm³  (expect ~6.7e-26)")
    print(f"  T at grid centre:              {T[cy, cx]:.2f} keV     (expect ~14 keV)")
    print(f"  T at bullet:                   {T[by, bx]:.2f} keV     (expect ~6.5 keV)")

    grad_T_peak = np.unravel_index(np.argmax(grad_T_mag), grad_T_mag.shape)
    print(f"  max(|grad_T|) location:        {grad_T_peak}  (expect between main & bullet)")
    print(f"  max(|grad_T|) value:           {np.max(grad_T_mag):.6e} keV/pixel")

    if galaxies is not None:
        print(f"  Number of galaxies:            {len(galaxies)}")

    # --- Diagnostic plot ---
    print("\n--- SAVING DIAGNOSTIC PLOT ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    panels = [
        (axes[0, 0], n_e_total, "n_e (cm⁻³)"),
        (axes[0, 1], rho, "ρ (g/cm³)"),
        (axes[0, 2], T, "T (keV)"),
        (axes[1, 0], grad_T_mag, "|∇T| (keV/pixel)"),
        (axes[1, 1], SB, "Surface Brightness (arb)"),
    ]
    for ax, data, title in panels:
        im = ax.imshow(data, origin="lower", cmap="inferno")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_gal = axes[1, 2]
    if galaxies is not None:
        ax_gal.scatter(
            galaxies["MatchRA"], galaxies["MatchDec"],
            s=1, alpha=0.4, c="cyan",
        )
        ax_gal.set_xlabel("RA (deg)")
        ax_gal.set_ylabel("Dec (deg)")
        ax_gal.set_title("Galaxy Positions (HSC v3)")
        ax_gal.invert_xaxis()
    else:
        ax_gal.text(0.5, 0.5, "No galaxy data", ha="center", va="center",
                    transform=ax_gal.transAxes, fontsize=14)
        ax_gal.set_title("Galaxy Positions")

    plt.suptitle("XP-005.1 Module 1 — Data Ingestion Fields", fontsize=14, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "module1_fields.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    print("\n" + "=" * 70)
    print("  MODULE 1 COMPLETE — Data ingestion finished.")
    print("=" * 70)
