# XP-006 GDFLM v1.0 — Run Notes

## Schema deviation discovered during Build Prompt 4

When `run_xp006.py` was first executed (Build Prompt 4), the κ_obs reconstruction
function expected CSV columns named `x`, `y`, `kappa`. The actual file at
`data/digitised/clowe2006_kappa.csv` has a different schema: 9 contours, each
spanning a paired `X`, `Y` column block, with the contour level encoded in the
column header (e.g. `kappa_0.23_main`). The expected schema was drawn from a
simpler loader convention; the actual schema requires the polygon-fill logic
that lives in XP-005.1's `src/module5_comparison.py`.

Per the build prompt's "stop and report" instruction, the run was halted, the
issue was diagnosed, and a decision was made.

## Initial decision (Build Prompt 4)

The function body of `reconstruct_kappa_obs()` inside `run_xp006.py` was edited
in-line to load `data/processed/kappa_obs.npy` directly (a byte-for-byte
inheritance from XP-005.1's prior run on the same CSV). The function signature
was preserved. The run completed cleanly with this substitution.

Verdict produced: FAIL (2/4) — recorded in initial XP006_results.json.

## Provenance gap identified post-run

The initial `XP006_results.json` declared `kappa_obs.source_csv` as the CSV
path, but the run did not parse the CSV — it loaded the .npy. A skeptic
cloning the XP-006 repo at that point could not reproduce the verdict because
`kappa_obs.npy` was `.gitignore`-excluded and lived only in the local workspace
as a side-effect of XP-005.1.

## Provenance fix (Build Prompt 4.1)

1. Polygon-fill reconstruction logic from XP-005.1 `src/module5_comparison.py`
   was lifted into a new standalone module `src/kappa_obs_reconstruction.py`,
   imports adapted, calibration anchors copied verbatim.

2. `run_xp006.py` updated to call the new module instead of `np.load`. The
   function name `reconstruct_kappa_obs` is retained but now actually
   reconstructs.

3. The new parser was verified to reproduce the inherited
   `kappa_obs.npy` (XP-005.1 SHA-256
   `6b75561bde292b07b21f92c8de2ba59d09616aa3aed65b4314e576ac1c6a4831`)
   to numerical equality — confirming the inheritance is faithful and the
   in-repo regeneration is the canonical path.

4. `.gitignore` updated to permit `data/processed/kappa_obs.npy` to be
   committed as a sealed reference snapshot. All other `.npy` files
   (regenerable parametric fields) remain excluded.

5. `XP006_results.json` updated to declare both the CSV reconstruction path
   and the reference snapshot SHA, removing the misleading source declaration.

6. Pipeline re-run end-to-end. All four metric values reproduced to numerical
   equality.

## Final verdict (unchanged): FAIL (2/4)

The metric values are not affected by this fix. The kappa_obs array used in
the original run and the array produced by the new parser are numerically
identical — the only thing that changed is the audit trail.

## What this episode demonstrates

- Pre-flight CSV schema verification (independent of the "files exist" check)
  should be added to the workflow doc as a future requirement. Cowork would
  have caught this during Phase 2 if the schema-confirmation step had been
  explicit.
- The "stop and report" instruction worked as intended. The schema mismatch
  was caught before the verdict was committed.
- The provenance gap was caught and fixed before the Gate 4 review package
  went to Grok. The result is unchanged but the audit trail is now
  self-consistent.

## Workflow lesson recorded

Future pipeline pre-flight phases must include explicit schema validation
of any CSV inputs against the loader's expected column structure, not just
file existence checks. Build prompts must verify schema before issuing
the run prompt.
