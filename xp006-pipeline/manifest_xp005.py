"""
XP-005.1 — Boundary–Propagation Lensing Model
MANIFEST — Single source of truth for all constants, thresholds, and core equations.

All constants are LOCKED. No modifications permitted.
Every module in the pipeline imports from this file. Nothing is hardcoded elsewhere.
"""

import hashlib
import numpy as np

# ==============================================================================
# LOCKED PHYSICAL CONSTANTS
# ==============================================================================

RHO_THRESHOLD = 1.0e-25          # g/cm³ — density transition scale
SIGMA = 0.3 * RHO_THRESHOLD      # g/cm³ — phase function width (= 3.0e-26)
SIGMA_GAL_KPC = 20.0             # kpc — galaxy smoothing kernel (physical)
M_PROTON = 1.6726e-24            # g — proton mass (CODATA)
CLUSTER_Z = 0.296                # Bullet Cluster redshift
KPC_PER_ARCSEC = 4.413           # at z=0.296 (H0=70, Om=0.3, OL=0.7)
SIGMA_GAL_ARCSEC = SIGMA_GAL_KPC / KPC_PER_ARCSEC  # ≈ 4.53 arcsec

# ==============================================================================
# PASS/FAIL THRESHOLDS
# ==============================================================================

CHI2_IMPROVEMENT_THRESHOLD = 0.25   # ≥ 25% reduction over baseline
PEAK_OFFSET_THRESHOLD = 30.0        # arcsec
RESIDUAL_RMS_THRESHOLD = 0.2        # normalised κ units
STABILITY_THRESHOLD = 0.10          # < 10% variation across subsamples

# ==============================================================================
# CORE EQUATIONS (locked — implement exactly as specified)
# ==============================================================================

def phase_function(rho):
    """P(x) = exp(-((rho - rho_threshold)^2 / sigma^2))"""
    return np.exp(-((rho - RHO_THRESHOLD)**2 / SIGMA**2))


def boundary_emission(P, chi):
    """Psi_emit = P(x) * chi(x)"""
    return P * chi


def propagation_field(P, grad_T, chi):
    """Psi_prop = P(x) * |grad_T(x)| * chi(x)"""
    return P * np.abs(grad_T) * chi


def observed_field(psi_emit, psi_prop):
    """Psi_obs = Psi_emit * Psi_prop"""
    return psi_emit * psi_prop


def model_convergence(psi_obs):
    """kappa_model = Psi_obs / max(Psi_obs)"""
    max_val = np.max(psi_obs)
    if max_val == 0:
        return np.zeros_like(psi_obs)
    return psi_obs / max_val


def baseline_convergence(surface_brightness):
    """kappa_baseline = SB / max(SB)"""
    max_val = np.max(surface_brightness)
    if max_val == 0:
        return np.zeros_like(surface_brightness)
    return surface_brightness / max_val


def density_from_ne(n_e):
    """rho = n_e * m_p (pure hydrogen approximation)"""
    return n_e * M_PROTON


# ==============================================================================
# SHA-256 INTEGRITY VERIFICATION
# ==============================================================================

def compute_manifest_hash():
    """Compute SHA-256 of all locked constant values concatenated as a string."""
    constant_string = (
        f"{RHO_THRESHOLD}"
        f"{SIGMA}"
        f"{SIGMA_GAL_KPC}"
        f"{M_PROTON}"
        f"{CLUSTER_Z}"
        f"{KPC_PER_ARCSEC}"
        f"{SIGMA_GAL_ARCSEC}"
        f"{CHI2_IMPROVEMENT_THRESHOLD}"
        f"{PEAK_OFFSET_THRESHOLD}"
        f"{RESIDUAL_RMS_THRESHOLD}"
        f"{STABILITY_THRESHOLD}"
    )
    return hashlib.sha256(constant_string.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print("=" * 70)
    print("XP-005.1 — Boundary–Propagation Lensing Model — MANIFEST")
    print("=" * 70)

    print("\n--- PHYSICAL CONSTANTS ---")
    print(f"  RHO_THRESHOLD      = {RHO_THRESHOLD:.6e}  g/cm³")
    print(f"  SIGMA              = {SIGMA:.6e}  g/cm³  (= 0.3 × RHO_THRESHOLD)")
    print(f"  SIGMA_GAL_KPC      = {SIGMA_GAL_KPC}            kpc")
    print(f"  M_PROTON           = {M_PROTON:.4e}  g  (CODATA)")
    print(f"  CLUSTER_Z          = {CLUSTER_Z}               (Bullet Cluster redshift)")
    print(f"  KPC_PER_ARCSEC     = {KPC_PER_ARCSEC}             at z=0.296")
    print(f"  SIGMA_GAL_ARCSEC   = {SIGMA_GAL_ARCSEC:.4f}          arcsec (≈ 4.53)")

    print("\n--- PASS/FAIL THRESHOLDS ---")
    print(f"  CHI2_IMPROVEMENT_THRESHOLD = {CHI2_IMPROVEMENT_THRESHOLD}  (≥ 25% reduction)")
    print(f"  PEAK_OFFSET_THRESHOLD      = {PEAK_OFFSET_THRESHOLD}   arcsec")
    print(f"  RESIDUAL_RMS_THRESHOLD     = {RESIDUAL_RMS_THRESHOLD}    normalised κ units")
    print(f"  STABILITY_THRESHOLD        = {STABILITY_THRESHOLD}   (< 10% variation)")

    print("\n--- VERIFICATION CHECKS ---")
    pf_at_threshold = phase_function(np.array([RHO_THRESHOLD]))[0]
    pf_at_zero = phase_function(np.array([0.0]))[0]
    print(f"  phase_function(RHO_THRESHOLD) = {pf_at_threshold:.6f}  [expected: 1.0]")
    print(f"  phase_function(0)             = {pf_at_zero:.6e}  [expected: ~0]")
    sigma_check = "PASS" if abs(SIGMA - 3.0e-26) < 1e-40 else "FAIL"
    arcsec_check = "PASS" if abs(SIGMA_GAL_ARCSEC - 4.53) < 0.01 else "FAIL"
    peak_check = "PASS" if abs(pf_at_threshold - 1.0) < 1e-10 else "FAIL"
    zero_check = "PASS" if pf_at_zero < 1e-3 else "FAIL"
    print(f"  SIGMA == 3.0e-26              [{sigma_check}]")
    print(f"  SIGMA_GAL_ARCSEC ≈ 4.53       [{arcsec_check}]")
    print(f"  phase_function peak == 1.0    [{peak_check}]")
    print(f"  phase_function(0) ≈ 0         [{zero_check}]")

    manifest_hash = compute_manifest_hash()
    print("\n--- SHA-256 INTEGRITY HASH ---")
    print(f"  {manifest_hash}")
    print("\n  Store this hash to verify constants have not been modified.")
    print("=" * 70)
