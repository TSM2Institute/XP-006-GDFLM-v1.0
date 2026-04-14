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
```

## Pipeline Stages
1. **manifest.py** — Locked constants, equations, SHA-256 integrity check
2. **src/mock_validation.py** — Synthetic 100x100 grid validation (8 checks, diagnostic plot, checkpoint)

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
