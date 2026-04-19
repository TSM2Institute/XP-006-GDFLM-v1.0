"""
manifest.py — XP-006 Gradient-Dominant Field Lensing Model (GDFLM v1.0)

TSM2 Institute for Cosmology Ltd
Pipeline: XP-006
Target:   Bullet Cluster (1E 0657-558)

Single source of truth. Every module imports from here. Nothing
hardcoded elsewhere. SHA-256 verification at file end — both platforms
must report the same hash before any build prompt is issued.

INHERITANCE: Data path, grid, κ_obs, σ, and PASS/FAIL thresholds are
inherited from XP-005.1 (sealed manifest SHA-256
b1ca5dc7900fa7f04330d72804f62b499fecdacbfd0aac2105feb8bea5524b2e).
Same files at the same paths. Phase function constants (RHO_THRESHOLD,
SIGMA) and galaxy χ field intentionally REMOVED — not used by GDFLM.

MODEL FORM (zero free parameters):
    K(x) = α·ρ̂(x) + β·|∇T̂(x)| + γ·|∇ρ̂(x)|
    α = β = γ = 1.0
    Each input field is min-max normalised independently to [0,1]
    before summation. K is then peak-normalised for comparison
    against κ_obs.
"""

import hashlib
import numpy as np


# ============================================================
# MODEL CONSTANTS (XP-006 specific) — DO NOT MODIFY
# ============================================================

ALPHA = 1.0   # weight on normalised ρ
BETA  = 1.0   # weight on normalised |∇T|
GAMMA = 1.0   # weight on normalised |∇ρ|


# ============================================================
# GRID & TARGET (inherited from XP-005.1)
# ============================================================

GRID_SIZE          = 200       # 200×200 pixels
PIXEL_SCALE_ARCSEC = 2.0       # arcsec/pixel
CLUSTER_Z          = 0.296     # Bullet Cluster redshift
KPC_PER_ARCSEC     = 4.413     # at z=0.296 (H0=70, Om=0.3, OL=0.7)


# ============================================================
# COMPARISON CONFIGURATION (inherited from XP-005.1)
# ============================================================

KAPPA_OBS_SOURCE        = "digitised_clowe2006"
KAPPA_OBS_FILE          = "data/digitised/clowe2006_kappa.csv"
KAPPA_OBS_PIXEL_COUNT   = 7487   # non-zero pixels after reconstruction
KAPPA_OBS_KNOWN_WARNING = ("kappa_0.16_ur contour: 17.9 px gap, "
                           "manual closure carried forward from XP-005.1")
COMPARISON_SIGMA        = 0.14   # uniform error, Bradač et al. 2006


# ============================================================
# BOOTSTRAP STABILITY (XP-006 specific — replaces galaxy resampling)
# ============================================================

BOOTSTRAP_FRACTION = 2.0 / 3.0    # 2/3 of κ_obs > 0 pixels per run
BOOTSTRAP_N_RUNS   = 3            # three subsample runs
BOOTSTRAP_SEEDS    = (1, 2, 3)    # fixed for cross-platform reproducibility
STABILITY_METRIC   = "(max - min) / mean of the 3 chi2/dof values"


# ============================================================
# PASS/FAIL THRESHOLDS (inherited from XP-005.1, identical)
# ============================================================

CHI2_IMPROVEMENT_THRESHOLD = 0.25   # K must achieve ≥25% χ²/dof improvement vs baseline
PEAK_OFFSET_THRESHOLD      = 30.0   # arcsec — K peak vs κ_obs peak
RESIDUAL_RMS_THRESHOLD     = 0.2    # normalised κ units
STABILITY_THRESHOLD        = 0.10   # < 10% across the 3 bootstrap χ²/dof values


# ============================================================
# CORE FUNCTIONS (locked — no modification permitted)
# ============================================================

def normalise_minmax(field):
    """
    Min-max normalisation to [0, 1]:
        x_norm = (x - min(x)) / (max(x) - min(x))
    Applied independently to ρ, |∇T|, and |∇ρ| before summation.
    Returns zeros if max == min.
    """
    field = np.asarray(field, dtype=float)
    fmin = np.min(field)
    fmax = np.max(field)
    if fmax == fmin:
        return np.zeros_like(field)
    return (field - fmin) / (fmax - fmin)


def gradient_magnitude(field, pixel_scale_arcsec=PIXEL_SCALE_ARCSEC):
    """
    Magnitude of 2D spatial gradient:
        |∇f| = sqrt((df/dx)^2 + (df/dy)^2)
    Computed via numpy.gradient with explicit pixel spacing.

    Used to compute |∇ρ| on the fly in XP-006.
    |∇T| is loaded pre-computed from XP-005.1 (grad_T_magnitude.npy)
    and is NOT recomputed here, to preserve byte-for-byte continuity
    with the prior pipeline.
    """
    gy, gx = np.gradient(np.asarray(field, dtype=float),
                         pixel_scale_arcsec)
    return np.sqrt(gx**2 + gy**2)


def K_model(rho_field, grad_T_magnitude_field, grad_rho_magnitude_field):
    """
    XP-006 model equation:
        K(x) = α·ρ̂ + β·|∇T̂| + γ·|∇ρ̂|

    Each input is min-max normalised independently to [0,1], then
    summed with α = β = γ = 1.0. Output lies in [0, 3] before
    peak-normalisation in model_convergence_xp006.
    """
    rho_n     = normalise_minmax(rho_field)
    gradT_n   = normalise_minmax(grad_T_magnitude_field)
    gradRho_n = normalise_minmax(grad_rho_magnitude_field)
    return ALPHA * rho_n + BETA * gradT_n + GAMMA * gradRho_n


def model_convergence_xp006(K_field):
    """
    Peak-normalise K for comparison against κ_obs.
    Matches XP-005.1 model_convergence convention (peak = 1).
    """
    fmax = np.max(K_field)
    if fmax == 0:
        return np.zeros_like(K_field)
    return K_field / fmax


def baseline_convergence(surface_brightness):
    """
    Baseline = X-ray surface brightness, peak-normalised.
    Identical to XP-005.1 baseline_convergence.
    χ²/dof improvement is measured against this κ_baseline.
    """
    fmax = np.max(surface_brightness)
    if fmax == 0:
        return np.zeros_like(surface_brightness)
    return surface_brightness / fmax


def bootstrap_subsample_indices(positive_pixel_indices, seed):
    """
    Generate one bootstrap subsample of the comparison region.
        seed -> np.random.default_rng(seed)
        select 2/3 of positive_pixel_indices WITHOUT replacement
    Returns the selected indices.
    """
    rng = np.random.default_rng(seed)
    n_total = len(positive_pixel_indices)
    n_select = int(round(BOOTSTRAP_FRACTION * n_total))
    return rng.choice(positive_pixel_indices,
                      size=n_select, replace=False)


def stability(chi2_dof_values):
    """
    Bootstrap stability metric:
        (max - min) / mean of the three χ²/dof values
    Pre-registered threshold: < STABILITY_THRESHOLD (0.10).
    """
    arr = np.asarray(chi2_dof_values, dtype=float)
    return (arr.max() - arr.min()) / arr.mean()


# ============================================================
# DATA PATHS (inherited from XP-005.1 — files must already exist)
# ============================================================

DATA_PATHS = {
    "rho_field":        "data/processed/rho_field.npy",
    "T_field":          "data/processed/T_field.npy",
    "grad_T_magnitude": "data/processed/grad_T_magnitude.npy",
    "SB_field":         "data/processed/SB_field.npy",
    "grid_params":      "data/processed/grid_params.json",
    "kappa_obs_csv":    "data/digitised/clowe2006_kappa.csv",
}


# ============================================================
# SHA-256 VERIFICATION
# ============================================================

def manifest_signature():
    """
    Concatenate all locked constants and return SHA-256.
    Run `python manifest.py` to print the signature.
    Both platforms must report the same hash before any build
    prompt is issued.
    """
    payload = "|".join([
        f"PIPELINE=XP-006_GDFLM_v1.0",
        f"ALPHA={ALPHA}", f"BETA={BETA}", f"GAMMA={GAMMA}",
        f"GRID_SIZE={GRID_SIZE}",
        f"PIXEL_SCALE_ARCSEC={PIXEL_SCALE_ARCSEC}",
        f"CLUSTER_Z={CLUSTER_Z}",
        f"KPC_PER_ARCSEC={KPC_PER_ARCSEC}",
        f"KAPPA_OBS_SOURCE={KAPPA_OBS_SOURCE}",
        f"KAPPA_OBS_PIXEL_COUNT={KAPPA_OBS_PIXEL_COUNT}",
        f"COMPARISON_SIGMA={COMPARISON_SIGMA}",
        f"BOOTSTRAP_FRACTION={BOOTSTRAP_FRACTION}",
        f"BOOTSTRAP_N_RUNS={BOOTSTRAP_N_RUNS}",
        f"BOOTSTRAP_SEEDS={BOOTSTRAP_SEEDS}",
        f"STABILITY_METRIC={STABILITY_METRIC}",
        f"CHI2_IMPROVEMENT_THRESHOLD={CHI2_IMPROVEMENT_THRESHOLD}",
        f"PEAK_OFFSET_THRESHOLD={PEAK_OFFSET_THRESHOLD}",
        f"RESIDUAL_RMS_THRESHOLD={RESIDUAL_RMS_THRESHOLD}",
        f"STABILITY_THRESHOLD={STABILITY_THRESHOLD}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    sig = manifest_signature()
    print("XP-006 GDFLM v1.0 manifest signature (SHA-256):")
    print(sig)
