"""
module1_xp006.py — XP-006 GDFLM v1.0 — Slim Field Regeneration

Regenerates the four input fields needed by the GDFLM model
(rho_field, T_field, grad_T_magnitude, SB_field) parametrically
from XP-005.1 literature constants. No HSC galaxy download.
No phase-function constants. No chi field.

Formulas mirror XP-005.1 src/module1_ingestion.py (Parts B and C only)
byte-for-byte to preserve continuity. Imports cleanly from the sealed
XP-006 manifest.

Outputs (saved to ../data/processed/):
    rho_field.npy         — gas density (g/cm^3)
    T_field.npy           — temperature (keV)
    grad_T_magnitude.npy  — |grad T| (keV / pixel, same convention as XP-005.1)
    SB_field.npy          — surface brightness proxy (n_e^2)

Run from the xp006-pipeline directory:
    python src/module1_xp006.py
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manifest import (
    GRID_SIZE,
    PIXEL_SCALE_ARCSEC,
    KPC_PER_ARCSEC,
    DATA_PATHS,
)


# ==========================================================================
# PHYSICAL CONSTANT (not a model parameter — defined locally with citation)
# ==========================================================================

M_PROTON = 1.6726e-24   # g   (CODATA proton mass; matches XP-005.1 manifest)


# ==========================================================================
# LITERATURE PARAMETERS — Markevitch 2002 / 2006 (mirror XP-005.1)
# ==========================================================================

LITERATURE_PARAMS = {
    "n0_main":      {"value": 3.0e-3, "units": "cm^-3",
                     "source": "Markevitch et al. 2002, ApJ 567, L27"},
    "rc_main":      {"value": 250.0,  "units": "kpc",
                     "source": "Markevitch et al. 2002, ApJ 567, L27"},
    "beta_main":    {"value": 0.70,   "units": "dimensionless",
                     "source": "Markevitch et al. 2002, ApJ 567, L27"},
    "n0_bullet":    {"value": 4.0e-2, "units": "cm^-3",
                     "source": "Markevitch 2006, ESA SP-604"},
    "rc_bullet":    {"value": 50.0,   "units": "kpc",
                     "source": "Markevitch 2006, ESA SP-604"},
    "beta_bullet":  {"value": 0.60,   "units": "dimensionless",
                     "source": "Markevitch 2006, ESA SP-604"},
    "T_main":       {"value": 14.1,   "units": "keV",
                     "source": "Markevitch 2006, ESA SP-604"},
    "T_bullet":     {"value": 6.5,    "units": "keV",
                     "source": "Markevitch 2006, ESA SP-604"},
    "T_background": {"value": 10.0,   "units": "keV",
                     "source": "Markevitch 2006, ESA SP-604"},
}


# ==========================================================================
# GEOMETRY (mirror XP-005.1)
# ==========================================================================

CENTRE_PIXEL          = (100, 100)                      # main cluster
BULLET_OFFSET_ARCSEC  = 65.0
BULLET_OFFSET_PIXELS  = int(round(BULLET_OFFSET_ARCSEC / PIXEL_SCALE_ARCSEC))
BULLET_PIXEL          = (CENTRE_PIXEL[0] + BULLET_OFFSET_PIXELS,
                         CENTRE_PIXEL[1])  # → (132, 100)


# ==========================================================================
# FIELD CONSTRUCTION (mirror XP-005.1 Part B exactly)
# ==========================================================================

def beta_model(x, y, cx, cy, n0, rc_kpc, beta):
    """Isothermal beta-model: n_e(r) = n0 * (1 + (r/rc)^2)^(-3 beta / 2)"""
    dx = (x - cx) * PIXEL_SCALE_ARCSEC * KPC_PER_ARCSEC
    dy = (y - cy) * PIXEL_SCALE_ARCSEC * KPC_PER_ARCSEC
    r_kpc = np.sqrt(dx**2 + dy**2)
    return n0 * (1.0 + (r_kpc / rc_kpc) ** 2) ** (-3.0 * beta / 2.0)


def make_gaussian_2d(cx, cy, width_pix, amplitude, grid_size=GRID_SIZE):
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    return amplitude * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * width_pix ** 2)
    )


def build_density_field():
    """Sum of two beta-models (main + bullet), converted to mass density."""
    y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    n_e_main = beta_model(
        x, y, CENTRE_PIXEL[0], CENTRE_PIXEL[1],
        LITERATURE_PARAMS["n0_main"]["value"],
        LITERATURE_PARAMS["rc_main"]["value"],
        LITERATURE_PARAMS["beta_main"]["value"],
    )
    n_e_bullet = beta_model(
        x, y, BULLET_PIXEL[0], BULLET_PIXEL[1],
        LITERATURE_PARAMS["n0_bullet"]["value"],
        LITERATURE_PARAMS["rc_bullet"]["value"],
        LITERATURE_PARAMS["beta_bullet"]["value"],
    )
    n_e_total = n_e_main + n_e_bullet
    rho = n_e_total * M_PROTON
    return n_e_total, rho


def build_temperature_field():
    """T_bg + 60-px Gaussian at main + 20-px Gaussian at bullet."""
    T_bg         = LITERATURE_PARAMS["T_background"]["value"]
    T_main_amp   = LITERATURE_PARAMS["T_main"]["value"] - T_bg
    T_bullet_amp = LITERATURE_PARAMS["T_bullet"]["value"] - T_bg

    T = np.full((GRID_SIZE, GRID_SIZE), T_bg)
    T += make_gaussian_2d(CENTRE_PIXEL[0], CENTRE_PIXEL[1], 60, T_main_amp)
    T += make_gaussian_2d(BULLET_PIXEL[0], BULLET_PIXEL[1], 20, T_bullet_amp)

    # IMPORTANT: np.gradient called WITHOUT pixel spacing argument,
    # mirroring XP-005.1 exactly. Units are keV / pixel.
    # This is intentional for byte-for-byte continuity with XP-005.1.
    grad_Ty, grad_Tx = np.gradient(T)
    grad_T_mag = np.sqrt(grad_Tx**2 + grad_Ty**2)
    return T, grad_T_mag


def build_surface_brightness(n_e_total):
    """SB ∝ n_e² (line-of-sight integrated thermal bremsstrahlung proxy)."""
    return n_e_total ** 2


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-006 — MODULE 1 (slim): PARAMETRIC FIELD REGENERATION")
    print("=" * 70)

    # Resolve absolute output paths
    base_dir   = os.path.join(os.path.dirname(__file__), "..")
    data_proc  = os.path.join(base_dir, "data", "processed")
    os.makedirs(data_proc, exist_ok=True)

    print("\n--- LITERATURE PARAMETERS ---")
    for key, info in LITERATURE_PARAMS.items():
        print(f"  {key:14s} = {info['value']:>8}  [{info['units']:<13}]  — {info['source']}")

    print(f"\n--- GEOMETRY ---")
    print(f"  Grid:                 {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Pixel scale:          {PIXEL_SCALE_ARCSEC} arcsec/pixel")
    print(f"  Main cluster centre:  pixel {CENTRE_PIXEL}")
    print(f"  Bullet centre:        pixel {BULLET_PIXEL} (offset {BULLET_OFFSET_PIXELS} px)")

    print("\n--- BUILDING FIELDS ---")
    n_e_total, rho = build_density_field()
    T, grad_T_mag  = build_temperature_field()
    SB             = build_surface_brightness(n_e_total)

    # Save outputs (paths defined in manifest DATA_PATHS, but we resolve
    # against the module's parent directory for portability)
    out = {
        "rho_field.npy":        rho,
        "T_field.npy":          T,
        "grad_T_magnitude.npy": grad_T_mag,
        "SB_field.npy":         SB,
    }
    print("\n--- SAVING OUTPUTS ---")
    for fname, arr in out.items():
        path = os.path.join(data_proc, fname)
        np.save(path, arr)
        print(f"  {path}   shape={arr.shape}   dtype={arr.dtype}")

    print("\n--- SANITY CHECKS ---")
    rho_peak_idx = np.unravel_index(np.argmax(rho), rho.shape)
    T_peak_idx   = np.unravel_index(np.argmax(T),   T.shape)
    SB_peak_idx  = np.unravel_index(np.argmax(SB),  SB.shape)
    grad_T_peak_idx = np.unravel_index(np.argmax(grad_T_mag), grad_T_mag.shape)

    print(f"  rho        peak pixel = {rho_peak_idx}    (expect near bullet (100,132) or main (100,100))")
    print(f"             min={rho.min():.3e}  max={rho.max():.3e}  mean={rho.mean():.3e}  g/cm^3")
    print(f"  T          peak pixel = {T_peak_idx}    (expect near main (100,100))")
    print(f"             min={T.min():.3f}    max={T.max():.3f}    mean={T.mean():.3f}    keV")
    print(f"             T at bullet pixel (100,132) = {T[100, 132]:.3f} keV   "
          f"(XP-005.1 produced 10.06)")
    print(f"  grad_T     peak pixel = {grad_T_peak_idx}")
    print(f"             min={grad_T_mag.min():.3e}  max={grad_T_mag.max():.3e}  mean={grad_T_mag.mean():.3e}")
    print(f"  SB         peak pixel = {SB_peak_idx}    (expect near bullet (100,132))")
    print(f"             min={SB.min():.3e}  max={SB.max():.3e}  mean={SB.mean():.3e}")
    print(f"             SB strictly non-negative: {(SB >= 0).all()}")

    print("\n--- DONE ---")
    print("All four .npy fields written to data/processed/.")
    print("Next: Build Prompt 3 — mock validation of K(x) summation (Gate 2).")
