# XP-005.1 — Boundary–Propagation Lensing Model

**Test ID:** XP-005.1  
**Status:** FAIL  
**Date:** 17 April 2026  
**Target:** Bullet Cluster (1E 0657-56)  

## Result

XP-005.1 tested whether a boundary–propagation lensing model could 
reproduce the observed weak-lensing convergence (κ) of the Bullet Cluster.

**Verdict: FAIL** — Three of four pre-registered criteria not met.

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| χ²/dof improvement over baseline | ≥ 25% | -4.03% | FAIL |
| Peak offset from observed κ | < 30″ | 94.02″ | FAIL |
| Residual RMS (normalised κ) | ≤ 0.2 | 0.586 | FAIL |
| Stability across subsamples | < 10% | 0.08% | PASS |

The model places the convergence peak at the galaxy surface density 
maximum, not at the observed mass centre. The failure is systematic 
(stability = 0.08%), not stochastic.

## Model

The tested model computes:

```
P(x) = exp(-((ρ(x) - ρ_threshold)² / σ²))
Ψ_emit(x) = P(x) · χ(x)
Ψ_prop(x) = P(x) · |∇T(x)| · χ(x)
Ψ_obs(x)  = Ψ_emit(x) · Ψ_prop(x)
κ_model(x) = Ψ_obs(x) / max(Ψ_obs(x))
```

Where:
- ρ(x) = gas mass density from X-ray observations (g/cm³)
- ρ_threshold = 1.0 × 10⁻²⁵ g/cm³ (locked)
- σ = 3.0 × 10⁻²⁶ g/cm³ (locked)
- χ(x) = normalised galaxy surface density from HSC v3 catalogue
- T(x) = ICM temperature field (keV)
- ∇T(x) = temperature gradient

All constants pre-registered and locked before data analysis. 
No parameter tuning permitted or performed.

## Relationship to Other Pipelines

This is an **alternative lensing pathway** test, independent of 
SKIN-a-CAT v1.2.1 (which uses k_TSM × N_HI refractive lensing). 
The two pipelines share no inputs, constants, or equations. 
SKIN-a-CAT results are unaffected by this outcome.

## Data Sources

| Input | Source |
|-------|--------|
| Galaxy catalogue | Hubble Source Catalog v3 (MAST) — 4,113 galaxies |
| Electron density | Parametric β-model (Markevitch et al. 2002) |
| Temperature | Two-component model (Markevitch 2006) |
| Observed κ map | Digitised from Clowe et al. 2006, Figure 1 |
| X-ray baseline | Surface brightness ∝ n_e² |

Build path: A (parametric + digitised). See MANIFEST for details.

## Repository Structure

```
xp005-pipeline/
├── manifest.py                    # Locked constants and equations (SHA-256 verified)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── src/
│   ├── module1_ingestion.py       # Data download and field construction
│   ├── module2_fields.py          # Galaxy density and phase function
│   ├── module3_model.py           # Forward model computation
│   ├── module5_comparison.py      # Statistical comparison and PASS/FAIL
│   ├── module6_diagnostics.py     # Plots and reporting
│   └── mock_validation.py         # Synthetic data validation (Gate 2)
├── data/
│   ├── raw/                       # Downloaded files (not tracked — large)
│   ├── digitised/
│   │   └── clowe2006_kappa.csv    # Digitised Clowe 2006 κ contours
│   └── processed/                 # Computed fields (not tracked — large .npy)
├── results/
│   ├── XP005_results.json         # Machine-readable results
│   ├── XP005_VERDICT.md           # Human-readable verdict
│   ├── XP005_diagnostic_6panel.png
│   ├── XP005_residual_map.png
│   ├── XP005_radial_profiles.png
│   ├── XP005_components.png
│   └── mock_validation_fields.png
├── checkpoints/                   # Module checkpoints (not tracked — large .pkl)
└── docs/
    └── checkpoint_packages/       # Gate review materials
```

## Reproduction

```bash
pip install -r requirements.txt
python manifest.py                    # Verify constants (prints SHA-256)
python src/module1_ingestion.py       # Download data + build fields
python src/module2_fields.py          # Compute chi and P fields
python src/module3_model.py           # Run forward model
python src/module5_comparison.py      # Compare against Clowe 2006
python src/module6_diagnostics.py     # Generate plots and verdict
```

Requires: digitised κ map at `data/digitised/clowe2006_kappa.csv`

## Manifest Integrity

SHA-256 of locked constants: `b1ca5dc7900fa7f04330d72804f62b499fecdacbfd0aac2105feb8bea5524b2e`

Run `python manifest.py` to verify constants are unmodified.

## Licence

MIT

## Author

TSM2 Institute for Cosmology Ltd  
Test designed by Geoffrey E. Thwaites  
Pipeline built by Claude (Anthropic) with Grok (xAI) review  
Operated by Graham Hill
