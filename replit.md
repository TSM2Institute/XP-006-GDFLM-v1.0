# XP-005.1 — Boundary–Propagation Lensing Model

A Python scientific pipeline that tests whether a specific set of equations can reproduce the observed gravitational lensing convergence (κ) map of the Bullet Cluster.

## Project Type
Command-line Python pipeline. No web app, no frontend, no server.

## Structure

```
xp005-pipeline/
├── manifest.py              # Single source of truth — all locked constants & equations
├── src/
│   └── __init__.py
├── data/
│   ├── raw/                 # Raw observational data
│   ├── digitised/           # Digitised maps
│   └── processed/           # Pipeline-processed outputs
├── checkpoints/             # Run checkpoints
├── results/                 # Final results
└── docs/
    └── checkpoint_packages/ # Checkpoint documentation packages
```

## Running

```bash
cd xp005-pipeline
python manifest.py               # Print constants + SHA-256 hash
python src/mock_validation.py     # Gate 2: synthetic forward-model validation
python src/module1_ingestion.py   # Module 1: data ingestion (HSC catalogue + parametric fields)
python src/module2_fields.py     # Module 2: chi from galaxies + phase function P
python src/module3_model.py      # Module 3: forward model → kappa_model & kappa_baseline
python src/module5_comparison.py # Module 5: load κ_obs, run PASS/FAIL comparison
python src/module6_diagnostics.py # Module 6: diagnostic plots + verdict markdown
```

## Pipeline Stages
1. **manifest.py** — Locked constants, equations, SHA-256 integrity check
2. **src/mock_validation.py** — Synthetic 100x100 grid validation (8 checks, diagnostic plot, checkpoint)
3. **src/module1_ingestion.py** — Downloads HSC v3 galaxy catalogue from MAST, builds parametric density/temperature fields from literature β-model parameters, saves all fields as .npy
4. **src/module2_fields.py** — Builds chi(x,y) from galaxy catalogue (2D histogram + Gaussian smoothing), computes P(x,y) from phase_function(rho)
5. **src/module3_model.py** — Runs full forward model: psi_emit → psi_prop → psi_obs → kappa_model + kappa_baseline (8 verification checks)
6. **src/module5_comparison.py** — Loads digitised Clowe 2006 κ_obs, reconstructs filled map from contours, runs the four PASS/FAIL metrics (χ²/dof improvement, peak offset, residual RMS, stability across galaxy subsamples)
7. **src/module6_diagnostics.py** — 6-panel diagnostic figure, residual map, radial profiles, component plot, final verdict markdown

## Required External Data
Module 5 requires a WebPlotDigitizer CSV of the Clowe 2006 Figure 1 κ contours, placed at:
`xp005-pipeline/data/digitised/clowe2006_kappa.csv`

CSV format:
- Row 1: contour names paired (`kappa_0.16`, blank, `kappa_0.23_main`, blank, ...)
- Row 2: `X, Y, X, Y, ...`
- Following rows: digitised coordinate pairs in WebPlotDigitizer's internal system
- Calibration anchors (built into Module 5):
  - X1=-5.6078e-3 → RA=104.675°,  X2=-4.9944 → RA=104.550°
  - Y1=-2.8039e-2 → Dec=-55.9667°, Y2=-5.5508e-2 → Dec=-55.9333°

## Locked Constants (manifest.py)

| Constant | Value | Units |
|---|---|---|
| RHO_THRESHOLD | 1.0e-25 | g/cm³ |
| SIGMA | 3.0e-26 | g/cm³ |
| SIGMA_GAL_KPC | 20.0 | kpc |
| M_PROTON | 1.6726e-24 | g |
| CLUSTER_Z | 0.296 | — |
| KPC_PER_ARCSEC | 4.413 | — |
| SIGMA_GAL_ARCSEC | ≈4.53 | arcsec |

## SHA-256 Integrity Hash (constants, locked)
`b1ca5dc7900fa7f04330d72804f62b499fecdacbfd0aac2105feb8bea5524b2e`

## Dependencies
- Python 3.11
- numpy, scipy, matplotlib, astropy, pandas

## Design Rules
- `manifest.py` is the single source of truth. All modules import from it.
- Nothing is hardcoded outside `manifest.py`.
- Constants are locked and must not be modified.
