# XP-005.1 VERDICT

**Test:** Boundary–Propagation Lensing Model
**Target:** Bullet Cluster (1E 0657-56)
**Date:** 2026-04-17
**Build path:** A (parametric + digitised)

## Result: FAIL

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| χ²/dof improvement | ≥ 25% | -4.03% | FAIL |
| Peak offset | < 30″ | 94.02″ | FAIL |
| Residual RMS | ≤ 0.2 | 0.5859 | FAIL |
| Stability | < 10% | 0.08% | PASS |

## Notes
- Comparison region: 7487 pixels where κ_obs > 0
- Error weighting: uniform σ = 0.14 (Bradac et al. 2006 ~14% uncertainty over ACS field)
- Manifest SHA-256: `b1ca5dc7900fa7f04330d72804f62b499fecdacbfd0aac2105feb8bea5524b2e`
- chi2_dof model: 17.5137
- chi2_dof baseline: 16.8346
- Stability subsample chi2_dof values: [17.502551079237957, 17.515700836613455, 17.50957246141482]

## Verdict Rule
All four criteria must pass for PASS. Any failure = FAIL.
