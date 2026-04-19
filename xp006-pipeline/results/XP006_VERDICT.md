# XP-006 GDFLM v1.0 — Verdict

**Date:** 2026-04-19T09:42:16.725369Z  
**Manifest SHA-256:** `2faf4b6c5e1c2ee40000d6f4a3b3811df151aff8f582036b2d75a97b708ccfc4`

## Overall Verdict: **FAIL** (2/4 criteria passed)

## κ_obs reconstruction

- Source: `data/digitised/clowe2006_kappa.csv`
- Non-zero pixels: 7487 (manifest expects 7487) — match
- Peak pixel: (92, 87)

## Pre-registered criteria

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| χ²/dof improvement ≥ 25% | ≥ +0.25 | +0.4833 | **PASS** |
| Peak offset < 30 arcsec | < 30.0" | 85.51" | **FAIL** |
| Residual RMS ≤ 0.2 | ≤ 0.2 | 0.4129 | **FAIL** |
| Stability < 10% | < 0.1 | 0.0195 | **PASS** |

## Peak locations

- κ_obs    peak: pixel (92, 87)
- κ_model  peak: pixel (100, 129)   (offset 85.51")
- κ_base   peak: pixel (100, 132)

## Bootstrap (seeds 1, 2, 3)

| Seed | n_pixels | χ²/dof |
|------|---------:|-------:|
| 1 | 4991 | 8.7045 |
| 2 | 4991 | 8.5866 |
| 3 | 4991 | 8.7559 |

Stability metric: **0.0195** (threshold < 0.1)

## Outputs

- `results/XP006_results.json` — full machine-readable result
- `results/XP006_diagnostic_6panel.png`
- `results/XP006_components.png`
- `results/XP006_residual_map.png`
- `checkpoints/run_xp006_state.pkl`