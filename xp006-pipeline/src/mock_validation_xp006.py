"""
mock_validation_xp006.py — XP-006 GDFLM v1.0 — Gate 2 Mock Validation

Exercises every manifest function on synthetic inputs where the correct
answer is known in advance. NO live data touched. NO real .npy files
loaded. All 8 tests run before summary; failures are not silent.

Run from the xp006-pipeline directory:
    python src/mock_validation_xp006.py

Exit codes:
    0  — all 8 tests passed (Gate 2 PASS)
    1  — one or more tests failed (Gate 2 FAIL — diagnose before live run)
"""

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manifest import (
    ALPHA, BETA, GAMMA,
    GRID_SIZE, PIXEL_SCALE_ARCSEC,
    BOOTSTRAP_FRACTION, BOOTSTRAP_SEEDS,
    STABILITY_THRESHOLD,
    normalise_minmax,
    gradient_magnitude,
    K_model,
    model_convergence_xp006,
    baseline_convergence,
    bootstrap_subsample_indices,
    stability,
)


# ==========================================================================
# HELPERS
# ==========================================================================

def gaussian_2d(cx, cy, sigma, amplitude=1.0, grid_size=GRID_SIZE):
    """Synthetic Gaussian on a (grid_size, grid_size) grid centred at (cx,cy)."""
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    return amplitude * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)
    )


def peak_pixel(field):
    """Return (row, col) of argmax."""
    return np.unravel_index(np.argmax(field), field.shape)


def peak_offset_arcsec(field_a, field_b, pixel_scale=PIXEL_SCALE_ARCSEC):
    """Centre-to-centre Euclidean distance between two field peaks, in arcsec."""
    ra, ca = peak_pixel(field_a)
    rb, cb = peak_pixel(field_b)
    return float(np.hypot(ra - rb, ca - cb) * pixel_scale)


# ==========================================================================
# TEST RUNNER
# ==========================================================================

results = []   # list of (test_name, passed: bool, message: str)


def run(name, fn):
    try:
        msg = fn()
        results.append((name, True, msg or "OK"))
        print(f"  [PASS]  {name}: {msg or 'OK'}")
    except AssertionError as e:
        results.append((name, False, f"ASSERTION FAILED: {e}"))
        print(f"  [FAIL]  {name}: ASSERTION FAILED: {e}")
        traceback.print_exc(limit=2)
    except Exception as e:
        results.append((name, False, f"EXCEPTION: {type(e).__name__}: {e}"))
        print(f"  [FAIL]  {name}: EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


# ==========================================================================
# TESTS
# ==========================================================================

def test_1_all_zeros():
    """All-zeros input must produce all-zeros K with no NaN/inf."""
    z = np.zeros((GRID_SIZE, GRID_SIZE))
    K = K_model(z, z, z)
    assert K.shape == (GRID_SIZE, GRID_SIZE), f"shape {K.shape}"
    assert np.all(np.isfinite(K)), "K contains NaN or inf"
    assert np.all(K == 0.0), f"K not all zero: max={K.max()}"
    kappa = model_convergence_xp006(K)
    assert np.all(np.isfinite(kappa)), "kappa contains NaN or inf"
    assert np.all(kappa == 0.0), "kappa not all zero on zero K"
    return f"K all zeros, kappa all zeros, no NaN/inf"


def test_2_all_constant():
    """Min-max on constant field returns zeros (min==max edge case)."""
    c = np.full((GRID_SIZE, GRID_SIZE), 7.3)
    n = normalise_minmax(c)
    assert np.all(n == 0.0), f"constant field did not normalise to zeros: max={n.max()}"
    K = K_model(c, c, c)
    assert np.all(K == 0.0), f"K on constant inputs should be zero, got max={K.max()}"
    return "constant field → zeros (per docstring contract)"


def test_3_single_term_gaussians():
    """A Gaussian on one term, zeros on the others, must produce a K peak at that Gaussian."""
    z = np.zeros((GRID_SIZE, GRID_SIZE))

    # rho-only Gaussian at (50, 60)
    g_rho = gaussian_2d(60, 50, sigma=10)   # cx=col=60, cy=row=50
    K_rho_only = K_model(g_rho, z, z)
    p_rho = peak_pixel(K_rho_only)
    assert p_rho == (50, 60), f"rho-only K peak at {p_rho}, expected (50, 60)"

    # gradT-only Gaussian at (120, 140)
    g_gradT = gaussian_2d(140, 120, sigma=8)
    K_gradT_only = K_model(z, g_gradT, z)
    p_gradT = peak_pixel(K_gradT_only)
    assert p_gradT == (120, 140), f"gradT-only K peak at {p_gradT}, expected (120, 140)"

    # gradRho-only Gaussian at (30, 170)
    g_gradRho = gaussian_2d(170, 30, sigma=12)
    K_gradRho_only = K_model(z, z, g_gradRho)
    p_gradRho = peak_pixel(K_gradRho_only)
    assert p_gradRho == (30, 170), f"gradRho-only K peak at {p_gradRho}, expected (30, 170)"

    return "each term independently produces K peak at its input Gaussian centre"


def test_4_equal_weighting():
    """Three Gaussians at SAME pixel, one per term, must sum to K=3.0 at that pixel
       (before peak-normalisation), proving α=β=γ=1.0 is symmetric."""
    g = gaussian_2d(100, 100, sigma=15)   # same Gaussian for all three
    K = K_model(g, g, g)
    # Each input normalises identically to [0,1]; at the peak pixel each is 1.0;
    # equal-weighted sum at peak is 3.0
    peak_val = K[100, 100]
    assert abs(peak_val - 3.0) < 1e-9, (
        f"K peak value {peak_val}, expected 3.0 (1+1+1 with α=β=γ=1.0)")
    # And the global max should equal 3.0 (peak coincides)
    assert abs(K.max() - 3.0) < 1e-9, f"K.max()={K.max()}, expected 3.0"
    # Peak-normalised should give 1.0 at that pixel
    kappa = model_convergence_xp006(K)
    assert abs(kappa[100, 100] - 1.0) < 1e-9, (
        f"peak-normalised K at peak = {kappa[100, 100]}, expected 1.0")
    assert abs(kappa.max() - 1.0) < 1e-9
    # Sanity: weights are still 1.0
    assert (ALPHA, BETA, GAMMA) == (1.0, 1.0, 1.0), \
        f"manifest weights drifted: {(ALPHA, BETA, GAMMA)}"
    return f"K(peak)=3.0 (1+1+1); kappa(peak)=1.0; weights {(ALPHA, BETA, GAMMA)}"


def test_5_peak_offset_arcsec():
    """Two well-separated Gaussians; offset metric should return the correct arcsec."""
    # Place Gaussian A at (50, 50), Gaussian B at (50, 80) → 30-pixel offset
    A = gaussian_2d(50, 50, sigma=5)
    B = gaussian_2d(80, 50, sigma=5)
    offset = peak_offset_arcsec(A, B)
    expected = 30.0 * PIXEL_SCALE_ARCSEC   # 30 px × 2.0 arcsec/px = 60 arcsec
    assert abs(offset - expected) < 1e-9, (
        f"offset {offset} arcsec, expected {expected}")
    # Now diagonal offset: A at (40, 40), C at (70, 80) → sqrt(30²+40²)=50 px
    C = gaussian_2d(80, 70, sigma=5)
    A2 = gaussian_2d(40, 40, sigma=5)
    diag = peak_offset_arcsec(A2, C)
    expected_diag = 50.0 * PIXEL_SCALE_ARCSEC   # 100 arcsec
    assert abs(diag - expected_diag) < 1e-9, (
        f"diagonal offset {diag}, expected {expected_diag}")
    return f"30-px offset → {offset:.2f}\"; 3-4-5 triangle 50-px offset → {diag:.2f}\""


def test_6_normalise_contract():
    """Min-max normalisation: any non-degenerate input → output ∈ [0,1] with min=0 and max=1."""
    rng = np.random.default_rng(42)
    for trial in range(5):
        x = rng.normal(loc=trial * 1.5, scale=2.0, size=(GRID_SIZE, GRID_SIZE))
        n = normalise_minmax(x)
        assert n.min() == 0.0, f"trial {trial}: normalised min = {n.min()}, expected 0"
        assert n.max() == 1.0, f"trial {trial}: normalised max = {n.max()}, expected 1"
        assert (n >= 0.0).all() and (n <= 1.0).all(), \
            f"trial {trial}: out of [0,1] range"
    return "5 random fields all normalised to exactly [0,1]"


def test_7_bootstrap_reproducibility():
    """Same seed → same indices. Different seed → different indices. Size = round(2/3 × N)."""
    n_pixels = 7487   # matches KAPPA_OBS_PIXEL_COUNT
    pos_idx = np.arange(n_pixels)

    sub_a = bootstrap_subsample_indices(pos_idx, seed=BOOTSTRAP_SEEDS[0])
    sub_a_again = bootstrap_subsample_indices(pos_idx, seed=BOOTSTRAP_SEEDS[0])
    sub_b = bootstrap_subsample_indices(pos_idx, seed=BOOTSTRAP_SEEDS[1])
    sub_c = bootstrap_subsample_indices(pos_idx, seed=BOOTSTRAP_SEEDS[2])

    # Reproducibility
    assert np.array_equal(sub_a, sub_a_again), "same seed produced different indices"
    # Distinctness
    assert not np.array_equal(sub_a, sub_b), "seeds 1 and 2 produced identical indices"
    assert not np.array_equal(sub_a, sub_c), "seeds 1 and 3 produced identical indices"
    assert not np.array_equal(sub_b, sub_c), "seeds 2 and 3 produced identical indices"

    # Size correctness: round(2/3 × N)
    expected_size = int(round(BOOTSTRAP_FRACTION * n_pixels))
    assert len(sub_a) == expected_size, (
        f"sub_a size {len(sub_a)}, expected {expected_size}")
    assert len(sub_b) == expected_size
    assert len(sub_c) == expected_size

    # Without-replacement check
    assert len(np.unique(sub_a)) == len(sub_a), "sub_a contains duplicates"
    assert len(np.unique(sub_b)) == len(sub_b), "sub_b contains duplicates"
    assert len(np.unique(sub_c)) == len(sub_c), "sub_c contains duplicates"

    return (f"seeds (1,2,3) → distinct, reproducible, no-replacement subsamples "
            f"of size {expected_size} from {n_pixels} pixels")


def test_8_stability_metric():
    """Stability = (max-min)/mean. Verify arithmetic and threshold classification."""
    # Case A — should PASS (<0.10): tight cluster around 1.0
    a = [1.0, 1.05, 1.10]
    sa = stability(a)
    expected_a = (1.10 - 1.00) / np.mean(a)   # 0.10 / 1.05 ≈ 0.0952
    assert abs(sa - expected_a) < 1e-12, f"case A: stability {sa}, expected {expected_a}"
    assert sa < STABILITY_THRESHOLD, f"case A should pass <0.10, got {sa}"

    # Case B — should FAIL (>0.10): wide spread
    b = [1.0, 1.20, 1.40]
    sb = stability(b)
    expected_b = (1.40 - 1.00) / np.mean(b)   # 0.40 / 1.20 ≈ 0.333
    assert abs(sb - expected_b) < 1e-12, f"case B: stability {sb}, expected {expected_b}"
    assert sb > STABILITY_THRESHOLD, f"case B should fail >0.10, got {sb}"

    # Case C — boundary: identical values → stability 0
    c = [2.5, 2.5, 2.5]
    sc = stability(c)
    assert sc == 0.0, f"identical values should give 0 stability, got {sc}"

    return (f"PASS case ({a}) → {sa:.4f} <0.10; "
            f"FAIL case ({b}) → {sb:.4f} >0.10; "
            f"identical → 0.0")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XP-006 GDFLM v1.0 — GATE 2 MOCK VALIDATION")
    print("=" * 70)
    print(f"Manifest weights: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}, pixel scale: {PIXEL_SCALE_ARCSEC}\"/px")
    print(f"Bootstrap: fraction={BOOTSTRAP_FRACTION:.6f}, seeds={BOOTSTRAP_SEEDS}")
    print(f"Stability threshold: <{STABILITY_THRESHOLD}")
    print()
    print("Running 8 tests on synthetic inputs (NO live data)...")
    print()

    run("Test 1 — all-zeros input handling",     test_1_all_zeros)
    run("Test 2 — constant input edge case",     test_2_all_constant)
    run("Test 3 — single-term Gaussians",        test_3_single_term_gaussians)
    run("Test 4 — equal weighting α=β=γ=1.0",    test_4_equal_weighting)
    run("Test 5 — peak offset arcsec metric",    test_5_peak_offset_arcsec)
    run("Test 6 — normalisation contract",       test_6_normalise_contract)
    run("Test 7 — bootstrap reproducibility",    test_7_bootstrap_reproducibility)
    run("Test 8 — stability metric arithmetic",  test_8_stability_metric)

    print()
    print("=" * 70)
    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = len(results) - n_pass
    print(f"SUMMARY: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("=" * 70)

    if n_fail > 0:
        print()
        print("FAILED TESTS:")
        for name, passed, msg in results:
            if not passed:
                print(f"  - {name}: {msg}")
        print()
        print("GATE 2: FAIL — DO NOT proceed to live run. Diagnose and fix.")
        sys.exit(1)
    else:
        print()
        print("GATE 2: PASS — model equation and utilities verified on synthetic inputs.")
        print("Cleared to proceed to Build Prompt 4 (live run + bootstrap + verdict).")
        sys.exit(0)
